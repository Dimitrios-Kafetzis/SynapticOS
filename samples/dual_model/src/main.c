/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file main.c
 * @brief SynapticOS - dual_model sample, CPU0 (AI runtime core)
 *
 * The AI-runtime side of the asymmetric dual-core demo:
 *  1. boots the SynapticOS runtime
 *  2. registers two models: face_detect (96x96x3 vision) and
 *     keyword_spot (49x10 MFCC audio)
 *  3. initializes IPC over the shared region and starts serving
 *     cross-core inference requests
 *  4. releases CPU1 and completes the STATUS_REQ/RESP handshake
 *
 * CPU1 (samples/dual_model/remote, flashed to flash bank 1) then
 * alternates inference requests between the two models over IPC.
 * If CPU1 does not respond, this core continues single-core with
 * the full shell available.
 */

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <synaptic/syn_api.h>
#include <string.h>

LOG_MODULE_REGISTER(dual_model, LOG_LEVEL_INF);

#ifdef CONFIG_SYNAPTIC_DUAL_CORE

#include <zephyr/devicetree.h>
#include "syn_boot_internal.h"
#include "syn_infer_remote.h"

#define SYN_SHARED_NODE DT_NODELABEL(syn_shared)

/* Dummy model binary: the Neutron NPU HAL runs its stub path until
 * real compiled models arrive in a later phase (honest labeling:
 * latencies measured with this blob exercise the full runtime path
 * but not real NPU math).
 */
static const uint8_t dummy_model[64] = {0};

static int register_model(const char *name, uint32_t input_size,
			  uint32_t output_size)
{
	syn_model_info_t info = {0};
	syn_model_handle_t handle;
	int ret;

	strncpy(info.name, name, sizeof(info.name) - 1);
	strncpy(info.version, "1.0.0", sizeof(info.version) - 1);
	info.input_size = input_size;
	info.output_size = output_size;
	info.flash_size = sizeof(dummy_model);
	info.sram_required = 4096;
	info.input_dtype = SYN_NPU_DTYPE_INT8;
	info.output_dtype = SYN_NPU_DTYPE_INT8;

	ret = syn_model_register(&info, &handle);
	if (ret != 0) {
		LOG_ERR("Register '%s' failed: %d", name, ret);
		return ret;
	}

	syn_model_load(handle);
	LOG_INF("Model '%s' ready (handle %u, in %u out %u)",
		name, handle, input_size, output_size);
	return 0;
}

#if defined(CONFIG_SYNAPTIC_LAYER_EXEC) && defined(CONFIG_SYNAPTIC_SHELL)
/* `demo preempt`: on-hardware demonstration of layer-boundary
 * preemption (Phase 5.1/5.2). Loads a synthetic layered DAG model,
 * runs an unpreempted baseline, then a NORMAL preemptible job that
 * a REALTIME job preempts mid-execution; verifies the resumed
 * output bit-exactly and reports timings. Cross-core serving is
 * drained for the duration (CPU1 requests get -EAGAIN) because the
 * NPU's resident model becomes the layered blob.
 */
#include <zephyr/shell/shell.h>
#include "syn_npu_layered.h"
#include "syn_infer_internal.h"

#define DEMO_IN_SIZE   32
#define DEMO_OUT_SIZE  10
#define DEMO_LAYERS    8
#define DEMO_WORK      2000  /* ~ms-scale per layer at 150 MHz */

static uint8_t demo_blob[SYN_LAYERED_HDR_SIZE +
			 DEMO_LAYERS * SYN_LAYERED_DDESC_SIZE];
static int demo_blob_size;
static syn_model_handle_t demo_model_handle;

static uint8_t demo_in_a[DEMO_IN_SIZE];
static uint8_t demo_in_b[DEMO_IN_SIZE];

static volatile int demo_done_count;
static uint32_t demo_done_order[2];
static uint32_t demo_rt_done_cyc;

static void demo_cb(syn_job_id_t job, const syn_tensor_t *output,
		    void *user_data)
{
	ARG_UNUSED(job);
	ARG_UNUSED(output);

	uint32_t tag = (uint32_t)(uintptr_t)user_data;

	if (tag == 2U) {
		demo_rt_done_cyc = k_cycle_get_32();
	}
	if (demo_done_count < 2) {
		demo_done_order[demo_done_count] = tag;
	}
	demo_done_count++;
}

static int demo_ensure_model(const struct shell *sh)
{
	if (demo_blob_size == 0) {
		/* Long skip: L4 also consumes L0's output, so the
		 * planner must keep it alive across L1-L4.
		 */
		syn_layered_dag_layer_t layers[DEMO_LAYERS] = {
			{ 64, DEMO_WORK, SYN_LAYERED_SRC_PREV,
			  SYN_LAYERED_SRC_NONE },
			{ 64, DEMO_WORK, SYN_LAYERED_SRC_PREV,
			  SYN_LAYERED_SRC_NONE },
			{ 48, DEMO_WORK, SYN_LAYERED_SRC_PREV,
			  SYN_LAYERED_SRC_NONE },
			{ 48, DEMO_WORK, SYN_LAYERED_SRC_PREV,
			  SYN_LAYERED_SRC_NONE },
			{ 48, DEMO_WORK, SYN_LAYERED_SRC_PREV, 0 },
			{ 32, DEMO_WORK, SYN_LAYERED_SRC_PREV,
			  SYN_LAYERED_SRC_NONE },
			{ 16, DEMO_WORK, SYN_LAYERED_SRC_PREV,
			  SYN_LAYERED_SRC_NONE },
			{ DEMO_OUT_SIZE, DEMO_WORK, SYN_LAYERED_SRC_PREV,
			  SYN_LAYERED_SRC_NONE },
		};

		demo_blob_size = syn_npu_layered_make_dag(
			demo_blob, sizeof(demo_blob), DEMO_IN_SIZE,
			DEMO_LAYERS, layers);
		if (demo_blob_size <= 0) {
			shell_error(sh, "make_dag failed: %d",
				    demo_blob_size);
			return -EINVAL;
		}

		syn_model_info_t info = {0};

		strncpy(info.name, "layered_demo", sizeof(info.name) - 1);
		strncpy(info.version, "5.1", sizeof(info.version) - 1);
		info.input_size = DEMO_IN_SIZE;
		info.output_size = DEMO_OUT_SIZE;
		info.input_dtype = SYN_NPU_DTYPE_INT8;
		info.output_dtype = SYN_NPU_DTYPE_INT8;

		int ret = syn_model_register(&info, &demo_model_handle);

		if (ret != 0) {
			shell_error(sh, "register failed: %d", ret);
			demo_blob_size = 0;
			return ret;
		}

		for (int i = 0; i < DEMO_IN_SIZE; i++) {
			demo_in_a[i] = (uint8_t)(i * 3 + 1);
			demo_in_b[i] = (uint8_t)(0xA5 - i * 7);
		}
	}
	return 0;
}

static int cmd_demo_preempt(const struct shell *sh, size_t argc,
			    char **argv)
{
	ARG_UNUSED(argc);
	ARG_UNUSED(argv);

	int ret = demo_ensure_model(sh);

	if (ret != 0) {
		return ret;
	}

	/* The layered blob becomes the resident NPU model: pause
	 * cross-core serving for the duration.
	 */
	ret = syn_remote_serve_drain(2000);
	if (ret != 0) {
		shell_error(sh, "serve drain failed: %d", ret);
		syn_remote_serve_resume();
		return ret;
	}

	ret = syn_hal_npu_load_model(demo_blob, (size_t)demo_blob_size);
	if (ret != 0) {
		shell_error(sh, "layered blob load failed: %d", ret);
		syn_remote_serve_resume();
		return ret;
	}

	uint32_t in_shape[1] = { DEMO_IN_SIZE };
	syn_tensor_t ta, tb;

	syn_mem_tensor_init(&ta, in_shape, 1, SYN_NPU_DTYPE_INT8);
	ta.data = demo_in_a;
	ta.lifetime = SYN_MEM_SHARED;
	syn_mem_tensor_init(&tb, in_shape, 1, SYN_NPU_DTYPE_INT8);
	tb.data = demo_in_b;
	tb.lifetime = SYN_MEM_SHARED;

	/* Unpreempted baseline */
	int8_t baseline[DEMO_OUT_SIZE];
	syn_tensor_t base_out = { .data = baseline,
				  .size = sizeof(baseline) };
	uint32_t t0 = k_cycle_get_32();

	ret = syn_infer_run_sync(demo_model_handle, &ta, &base_out,
				 SYN_PRIORITY_NORMAL);

	uint32_t base_us = k_cyc_to_us_ceil32(k_cycle_get_32() - t0);

	syn_mem_reset_ephemeral();
	if (ret != 0) {
		shell_error(sh, "baseline run failed: %d", ret);
		goto restore;
	}
	shell_print(sh, "baseline: %u us for %u layers (stub NPU, "
		    "synthetic model)", base_us, DEMO_LAYERS);

	uint32_t planned, naive, plan_us;

	syn_npu_layered_plan_info(&planned, &naive, &plan_us);
	shell_print(sh, "plan: peak %u vs all-live %u bytes (-%u%%), "
		    "planned in %u us", planned, naive,
		    (unsigned)(100U - planned * 100U / naive), plan_us);

	/* Preemption run */
	syn_infer_stats_t st0, st1;

	syn_infer_get_stats(&st0);
	demo_done_count = 0;

	syn_pipeline_t *pn = syn_pipeline_create("demo_n");
	syn_pipeline_t *pr = syn_pipeline_create("demo_r");

	if (pn == NULL || pr == NULL ||
	    syn_pipeline_add_model(pn, demo_model_handle) != 0 ||
	    syn_pipeline_add_model(pr, demo_model_handle) != 0 ||
	    syn_pipeline_build(pn) != 0 || syn_pipeline_build(pr) != 0) {
		shell_error(sh, "pipeline setup failed");
		syn_pipeline_destroy(pn);
		syn_pipeline_destroy(pr);
		ret = -EIO;
		goto restore;
	}

	syn_infer_params_t np = {
		.priority = SYN_PRIORITY_NORMAL,
		.preemptible = true,
		.callback = demo_cb,
		.user_data = (void *)1,
	};
	syn_job_id_t jn = syn_infer_submit(pn, &ta, &np);

	/* Land inside the NORMAL job (~8 layers x ~ms each) */
	k_msleep(3);

	syn_infer_params_t rp = {
		.priority = SYN_PRIORITY_REALTIME,
		.callback = demo_cb,
		.user_data = (void *)2,
	};
	uint32_t rt_submit_cyc = k_cycle_get_32();
	syn_job_id_t jr = syn_infer_submit(pr, &tb, &rp);

	if (jn == SYN_JOB_INVALID || jr == SYN_JOB_INVALID) {
		shell_error(sh, "submit failed");
		syn_pipeline_destroy(pn);
		syn_pipeline_destroy(pr);
		ret = -EIO;
		goto restore;
	}

	(void)syn_infer_wait(jr, 5000);
	(void)syn_infer_wait(jn, 5000);

	syn_tensor_t rn, rr;
	int ret_n, ret_r;

	ret_r = syn_infer_get_result(jr, &rr);
	ret_n = syn_infer_get_result(jn, &rn);
	syn_infer_get_stats(&st1);

	uint32_t rt_latency_us = k_cyc_to_us_ceil32(demo_rt_done_cyc -
						    rt_submit_cyc);
	bool order_ok = (demo_done_count == 2 &&
			 demo_done_order[0] == 2U);
	bool exact = (ret_n == 0 && rn.size == DEMO_OUT_SIZE &&
		      memcmp(rn.data, baseline, DEMO_OUT_SIZE) == 0);

	shell_print(sh, "rt job: completed %s the normal job, "
		    "%u us from submit to completion",
		    order_ok ? "BEFORE" : "AFTER", rt_latency_us);
	shell_print(sh, "normal job resumed: output %s the baseline",
		    exact ? "MATCHES" : "DIFFERS FROM");
	shell_print(sh, "preemptions +%u, resumes +%u, ctx save %u us "
		    "(max %u us)", st1.preemptions - st0.preemptions,
		    st1.resumes - st0.resumes, st1.last_save_us,
		    st1.max_save_us);
	shell_print(sh, "verdict: %s",
		    (order_ok && exact &&
		     st1.preemptions > st0.preemptions) ?
		    "PASS" : "FAIL");

	syn_pipeline_destroy(pn);
	syn_pipeline_destroy(pr);
	syn_mem_reset_ephemeral();
	ret = (ret_r == 0 && order_ok && exact) ? 0 : -EIO;

restore:
	/* Put the stub blob back and reopen cross-core serving */
	(void)syn_hal_npu_load_model(dummy_model, sizeof(dummy_model));
	syn_remote_serve_resume();
	return ret;
}

SHELL_STATIC_SUBCMD_SET_CREATE(sub_demo,
	SHELL_CMD(preempt, NULL,
		  "Layer-boundary preemption demo (drains CPU1 serving)",
		  cmd_demo_preempt),
	SHELL_SUBCMD_SET_END
);
SHELL_CMD_REGISTER(demo, &sub_demo, "dual_model demos", NULL);
#endif /* CONFIG_SYNAPTIC_LAYER_EXEC && CONFIG_SYNAPTIC_SHELL */

int main(void)
{
	int ret;
	void *shm_base = (void *)DT_REG_ADDR(SYN_SHARED_NODE);
	size_t shm_size = DT_REG_SIZE(SYN_SHARED_NODE);

	LOG_INF("SynapticOS %s dual_model (CPU0, AI runtime)",
		syn_version());

	ret = syn_init();
	if (ret != 0) {
		LOG_ERR("Runtime init failed: %d", ret);
		return ret;
	}

	/* Two models for CPU1 to alternate between */
	ret = register_model("face_detect", 96 * 96 * 3, 144);
	if (ret != 0) {
		return ret;
	}
	ret = register_model("keyword_spot", 49 * 10, 12);
	if (ret != 0) {
		return ret;
	}

	ret = syn_hal_npu_load_model(dummy_model, sizeof(dummy_model));
	if (ret != 0) {
		LOG_ERR("NPU load failed: %d", ret);
		return ret;
	}

	ret = syn_ipc_init(shm_base, shm_size);
	if (ret != 0) {
		LOG_ERR("IPC init failed: %d", ret);
		return ret;
	}

	ret = syn_remote_serve_init();
	if (ret != 0) {
		LOG_ERR("Inference serving init failed: %d", ret);
		return ret;
	}

	ret = syn_boot_secondary(shm_base, 200);
	if (ret != 0) {
		LOG_WRN("CPU1 not responding (%d): single-core mode", ret);
	} else {
		LOG_INF("Dual-core up: CPU1 boot %u us, handshake %u us",
			syn_boot_cpu1_boot_us(), syn_boot_handshake_us());
	}

	/* Shell owns the console from here. CPU1 traffic is served by
	 * the IPC dispatch thread; check with: syn ipc status
	 */
	return 0;
}

#else /* !CONFIG_SYNAPTIC_DUAL_CORE */

int main(void)
{
	LOG_INF("dual_model requires the FRDM-MCXN947 dual-core build");
	return 0;
}

#endif /* CONFIG_SYNAPTIC_DUAL_CORE */
