/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file test_layer_preempt.c
 * @brief Unit tests for layer-granular execution, deadline dispatch
 *        and layer-boundary preemption (Phase 5.1)
 *
 * Uses the synthetic layered model format executed by the stub NPU
 * path (results are stub-labeled by definition). The ztest thread is
 * cooperative: submitted jobs stay QUEUED until this thread blocks,
 * which makes dispatch-order assertions deterministic; k_msleep()
 * hands the CPU to the scheduler thread mid-test to reach RUNNING
 * and SUSPENDED states.
 */

#include <zephyr/ztest.h>
#include <synaptic/syn_infer.h>
#include <synaptic/syn_mem.h>
#include <synaptic/syn_model.h>
#include <synaptic/syn_hal_npu.h>
#include <string.h>

#include "syn_infer_internal.h"
#include "syn_npu_layered.h"

#define LP_INPUT_SIZE   32
#define LP_OUTPUT_SIZE  10
#define LP_LAYERS       8

/* Arena shared with test_scheduler.c (see there): the layered
 * pipeline only allocates small stage outputs from it, activations
 * live in the layered module's own context.
 */
extern uint8_t test_infer_arena[4096];
#define lp_arena test_infer_arena

/* 8 layers sized so one job spans several tens of ms and a test
 * sleeping 8 ms reliably lands mid-execution. The busy-work constant
 * is per-platform: QEMU runs under icount (instruction-counted time),
 * the FRDM runs 20x more iterations at 150 MHz for a similar wall
 * profile.
 */
#ifdef CONFIG_SOC_SERIES_MCXNX4X
#define LP_WORK 2000
#else
#define LP_WORK 200
#endif

static const uint16_t lp_out_sizes[LP_LAYERS] = {
	64, 64, 48, 48, 32, 24, 16, LP_OUTPUT_SIZE
};
static const uint16_t lp_work[LP_LAYERS] = {
	LP_WORK, LP_WORK, LP_WORK, LP_WORK,
	LP_WORK, LP_WORK, LP_WORK, LP_WORK
};

static uint8_t lp_blob[SYN_LAYERED_HDR_SIZE +
		       LP_LAYERS * SYN_LAYERED_DESC_SIZE];
static int lp_blob_size;
static syn_model_handle_t lp_model;

static uint8_t input_a[LP_INPUT_SIZE];
static uint8_t input_b[LP_INPUT_SIZE];
static syn_tensor_t tensor_a;
static syn_tensor_t tensor_b;

/* Completion-order recording */
static uint32_t done_order[4];
static volatile int done_count;

static void order_cb(syn_job_id_t job, const syn_tensor_t *output,
		     void *user_data)
{
	ARG_UNUSED(job);
	ARG_UNUSED(output);

	if (done_count < 4) {
		done_order[done_count] = (uint32_t)(uintptr_t)user_data;
	}
	done_count++;
}

static void make_inputs(void)
{
	uint32_t shape[1] = { LP_INPUT_SIZE };

	for (int i = 0; i < LP_INPUT_SIZE; i++) {
		input_a[i] = (uint8_t)(i * 3 + 1);
		input_b[i] = (uint8_t)(0xA5 - i * 7);
	}
	syn_mem_tensor_init(&tensor_a, shape, 1, SYN_NPU_DTYPE_INT8);
	tensor_a.data = input_a;
	tensor_a.lifetime = SYN_MEM_SHARED;
	syn_mem_tensor_init(&tensor_b, shape, 1, SYN_NPU_DTYPE_INT8);
	tensor_b.data = input_b;
	tensor_b.lifetime = SYN_MEM_SHARED;
}

static syn_pipeline_t *make_pipe(const char *name)
{
	syn_pipeline_t *pipe = syn_pipeline_create(name);

	zassert_not_null(pipe, "pipeline create failed");
	zassert_equal(syn_pipeline_add_model(pipe, lp_model), 0,
		      "add_model failed");
	zassert_equal(syn_pipeline_build(pipe), 0, "build failed");
	return pipe;
}

static void *lp_suite_setup(void)
{
	lp_blob_size = syn_npu_layered_make_model(lp_blob, sizeof(lp_blob),
						  LP_INPUT_SIZE, LP_LAYERS,
						  lp_out_sizes, lp_work);
	zassert_true(lp_blob_size > 0, "make_model failed: %d", lp_blob_size);

	if (syn_model_get_by_name("layered_test", &lp_model) != 0) {
		syn_model_info_t info = {0};

		strncpy(info.name, "layered_test", sizeof(info.name));
		strncpy(info.version, "1.0.0", sizeof(info.version));
		info.input_size = LP_INPUT_SIZE;
		info.output_size = LP_OUTPUT_SIZE;
		info.input_dtype = SYN_NPU_DTYPE_INT8;
		info.output_dtype = SYN_NPU_DTYPE_INT8;
		zassert_equal(syn_model_register(&info, &lp_model), 0,
			      "model register failed");
	}
	make_inputs();
	return NULL;
}

static void lp_before(void *fixture)
{
	ARG_UNUSED(fixture);
	syn_mem_init(lp_arena, sizeof(lp_arena));

	/* Make the layered blob the resident NPU model (other suites
	 * may have loaded something else in between).
	 */
	syn_hal_npu_deinit();
	zassert_equal(syn_hal_npu_init(), 0, "NPU init failed");
	zassert_equal(syn_hal_npu_load_model(lp_blob, (size_t)lp_blob_size),
		      0, "layered blob load failed");

	done_count = 0;
	memset(done_order, 0xFF, sizeof(done_order));
}

ZTEST_SUITE(syn_layer_preempt_suite, NULL, lp_suite_setup, lp_before,
	    NULL, NULL);

/** Layered model runs end-to-end and is deterministic. */
ZTEST(syn_layer_preempt_suite, test_layered_end_to_end)
{
	int8_t out1[LP_OUTPUT_SIZE], out2[LP_OUTPUT_SIZE];
	syn_tensor_t o1 = { .data = out1, .size = sizeof(out1) };
	syn_tensor_t o2 = { .data = out2, .size = sizeof(out2) };

	zassert_equal(syn_infer_run_sync(lp_model, &tensor_a, &o1,
					 SYN_PRIORITY_NORMAL), 0,
		      "layered run_sync failed");
	zassert_equal(o1.size, LP_OUTPUT_SIZE, "wrong output size %u",
		      (unsigned)o1.size);
	syn_mem_reset_ephemeral();

	zassert_equal(syn_infer_run_sync(lp_model, &tensor_a, &o2,
					 SYN_PRIORITY_NORMAL), 0,
		      "second run_sync failed");
	syn_mem_reset_ephemeral();
	zassert_mem_equal(out1, out2, LP_OUTPUT_SIZE,
			  "same input must give identical output");

	zassert_equal(syn_infer_run_sync(lp_model, &tensor_b, &o2,
					 SYN_PRIORITY_NORMAL), 0,
		      "run_sync with input B failed");
	syn_mem_reset_ephemeral();
	zassert_true(memcmp(out1, out2, LP_OUTPUT_SIZE) != 0,
		     "different input should change the output");
}

/** Mismatched input size is rejected by the layered session. */
ZTEST(syn_layer_preempt_suite, test_layered_bad_input)
{
	static uint8_t small[LP_INPUT_SIZE / 2];
	uint32_t shape[1] = { sizeof(small) };
	syn_tensor_t in;
	int8_t out_buf[LP_OUTPUT_SIZE];
	syn_tensor_t out = { .data = out_buf, .size = sizeof(out_buf) };

	syn_mem_tensor_init(&in, shape, 1, SYN_NPU_DTYPE_INT8);
	in.data = small;
	in.lifetime = SYN_MEM_SHARED;

	zassert_equal(syn_infer_run_sync(lp_model, &in, &out,
					 SYN_PRIORITY_NORMAL), -EINVAL,
		      "wrong input size should fail with -EINVAL");
	syn_mem_reset_ephemeral();
}

/** REALTIME preempts a preemptible NORMAL job at a layer boundary
 *  and the resumed job's output is bit-exact.
 */
ZTEST(syn_layer_preempt_suite, test_preempt_resume_bit_exact)
{
	int8_t baseline[LP_OUTPUT_SIZE];
	syn_tensor_t base_out = { .data = baseline, .size = sizeof(baseline) };

	/* Unpreempted reference run */
	zassert_equal(syn_infer_run_sync(lp_model, &tensor_a, &base_out,
					 SYN_PRIORITY_NORMAL), 0,
		      "baseline run failed");
	syn_mem_reset_ephemeral();

	syn_infer_stats_t st0, st1;

	syn_infer_get_stats(&st0);

	syn_pipeline_t *pn = make_pipe("preempt_n");
	syn_pipeline_t *pr = make_pipe("preempt_r");

	syn_infer_params_t normal_params = {
		.priority = SYN_PRIORITY_NORMAL,
		.preemptible = true,
		.callback = order_cb,
		.user_data = (void *)1,
	};
	syn_job_id_t jn = syn_infer_submit(pn, &tensor_a, &normal_params);

	zassert_not_equal(jn, SYN_JOB_INVALID, "NORMAL submit failed");

	/* Let the NORMAL job start and get a few layers in (~40 ms job) */
	k_msleep(8);

	syn_infer_params_t rt_params = {
		.priority = SYN_PRIORITY_REALTIME,
		.callback = order_cb,
		.user_data = (void *)2,
	};
	syn_job_id_t jr = syn_infer_submit(pr, &tensor_b, &rt_params);

	zassert_not_equal(jr, SYN_JOB_INVALID, "RT submit failed");

	zassert_equal(syn_infer_wait(jr, 5000), 0, "RT wait failed");
	zassert_equal(syn_infer_wait(jn, 5000), 0, "NORMAL wait failed");

	zassert_equal(done_count, 2, "expected 2 completions");
	zassert_equal(done_order[0], 2, "RT must complete first");
	zassert_equal(done_order[1], 1, "NORMAL must complete second");

	syn_tensor_t rn, rr;

	zassert_equal(syn_infer_get_result(jr, &rr), 0, "RT result failed");
	zassert_equal(syn_infer_get_result(jn, &rn), 0,
		      "NORMAL result failed");
	zassert_equal(rn.size, LP_OUTPUT_SIZE, "NORMAL output size");
	zassert_mem_equal(rn.data, baseline, LP_OUTPUT_SIZE,
			  "resumed job output must be bit-exact");

	syn_infer_get_stats(&st1);
	zassert_true(st1.preemptions > st0.preemptions,
		     "preemption counter must increase");
	zassert_true(st1.resumes > st0.resumes,
		     "resume counter must increase");
	zassert_true(st1.last_save_us < 1000U,
		     "context save took %u us (>= 1 ms)", st1.last_save_us);

	syn_pipeline_destroy(pn);
	syn_pipeline_destroy(pr);
	syn_mem_reset_ephemeral();
}

/** A non-preemptible NORMAL job finishes before a later RT job. */
ZTEST(syn_layer_preempt_suite, test_non_preemptible_runs_through)
{
	syn_infer_stats_t st0, st1;

	syn_infer_get_stats(&st0);

	syn_pipeline_t *pn = make_pipe("nopree_n");
	syn_pipeline_t *pr = make_pipe("nopree_r");

	syn_infer_params_t normal_params = {
		.priority = SYN_PRIORITY_NORMAL,
		.preemptible = false,
		.callback = order_cb,
		.user_data = (void *)1,
	};
	syn_job_id_t jn = syn_infer_submit(pn, &tensor_a, &normal_params);

	zassert_not_equal(jn, SYN_JOB_INVALID, "NORMAL submit failed");
	k_msleep(8);

	syn_infer_params_t rt_params = {
		.priority = SYN_PRIORITY_REALTIME,
		.callback = order_cb,
		.user_data = (void *)2,
	};
	syn_job_id_t jr = syn_infer_submit(pr, &tensor_b, &rt_params);

	zassert_not_equal(jr, SYN_JOB_INVALID, "RT submit failed");

	zassert_equal(syn_infer_wait(jn, 5000), 0, "NORMAL wait failed");
	zassert_equal(syn_infer_wait(jr, 5000), 0, "RT wait failed");

	zassert_equal(done_order[0], 1,
		      "non-preemptible NORMAL must finish first");
	zassert_equal(done_order[1], 2, "RT second");

	syn_infer_get_stats(&st1);
	zassert_equal(st1.preemptions, st0.preemptions,
		      "no preemption may occur");

	syn_tensor_t r;

	zassert_equal(syn_infer_get_result(jn, &r), 0, "result n failed");
	zassert_equal(syn_infer_get_result(jr, &r), 0, "result r failed");
	syn_pipeline_destroy(pn);
	syn_pipeline_destroy(pr);
	syn_mem_reset_ephemeral();
}

/** Equal-priority jobs dispatch earliest-deadline-first, no-deadline
 *  jobs last.
 */
ZTEST(syn_layer_preempt_suite, test_deadline_ordering)
{
	syn_pipeline_t *pipe = make_pipe("edf");

	/* All queued while this cooperative thread keeps the CPU */
	static const uint32_t deadlines[4] = { 300000, 100000, 200000, 0 };
	static const uint32_t tags[4] = { 300, 100, 200, 999 };
	syn_job_id_t ids[4];

	for (int i = 0; i < 4; i++) {
		syn_infer_params_t params = {
			.priority = SYN_PRIORITY_BEST_EFFORT,
			.deadline_us = deadlines[i],
			.callback = order_cb,
			.user_data = (void *)(uintptr_t)tags[i],
		};

		ids[i] = syn_infer_submit(pipe, &tensor_a, &params);
		zassert_not_equal(ids[i], SYN_JOB_INVALID,
				  "submit %d failed", i);
	}

	for (int i = 0; i < 4; i++) {
		zassert_equal(syn_infer_wait(ids[i], 10000), 0,
			      "wait %d failed", i);
	}

	zassert_equal(done_count, 4, "expected 4 completions");
	zassert_equal(done_order[0], 100, "earliest deadline first");
	zassert_equal(done_order[1], 200, "second deadline next");
	zassert_equal(done_order[2], 300, "third deadline next");
	zassert_equal(done_order[3], 999, "no-deadline job last");

	syn_tensor_t r;

	for (int i = 0; i < 4; i++) {
		zassert_equal(syn_infer_get_result(ids[i], &r), 0,
			      "result %d failed", i);
	}
	syn_pipeline_destroy(pipe);
	syn_mem_reset_ephemeral();
}

/** A completion later than deadline_us counts as a miss. */
ZTEST(syn_layer_preempt_suite, test_deadline_miss_counted)
{
	syn_infer_stats_t st0, st1;

	syn_infer_get_stats(&st0);

	syn_pipeline_t *pipe = make_pipe("miss");
	syn_infer_params_t params = {
		.priority = SYN_PRIORITY_NORMAL,
		.deadline_us = 1000, /* 1 ms; the job runs ~40 ms */
	};
	syn_job_id_t id = syn_infer_submit(pipe, &tensor_a, &params);

	zassert_not_equal(id, SYN_JOB_INVALID, "submit failed");
	zassert_equal(syn_infer_wait(id, 5000), 0, "wait failed");

	syn_tensor_t r;

	zassert_equal(syn_infer_get_result(id, &r), 0, "result failed");

	syn_infer_get_stats(&st1);
	zassert_true(st1.deadline_misses > st0.deadline_misses,
		     "deadline miss must be counted");

	syn_pipeline_destroy(pipe);
	syn_mem_reset_ephemeral();
}

/** A suspended job can be cancelled; its context slot is released. */
ZTEST(syn_layer_preempt_suite, test_cancel_suspended)
{
	syn_infer_stats_t st0, st1;

	syn_infer_get_stats(&st0);

	syn_pipeline_t *pn = make_pipe("cansusp_n");
	syn_pipeline_t *pr = make_pipe("cansusp_r");

	syn_infer_params_t normal_params = {
		.priority = SYN_PRIORITY_NORMAL,
		.preemptible = true,
	};
	syn_job_id_t jn = syn_infer_submit(pn, &tensor_a, &normal_params);

	zassert_not_equal(jn, SYN_JOB_INVALID, "NORMAL submit failed");
	k_msleep(8);

	/* RT job also spans ~40 ms: while it runs, NORMAL sits SUSPENDED */
	syn_infer_params_t rt_params = {
		.priority = SYN_PRIORITY_REALTIME,
	};
	syn_job_id_t jr = syn_infer_submit(pr, &tensor_b, &rt_params);

	zassert_not_equal(jr, SYN_JOB_INVALID, "RT submit failed");

	/* Give the scheduler time to suspend NORMAL and start RT, but
	 * wake well before RT finishes.
	 */
	k_msleep(10);

	zassert_equal(syn_infer_cancel(jn), 0,
		      "cancel of suspended job failed");
	zassert_equal(syn_infer_wait(jn, 100), -ECANCELED,
		      "cancelled job must report -ECANCELED");

	zassert_equal(syn_infer_wait(jr, 5000), 0, "RT wait failed");

	syn_tensor_t r;

	zassert_equal(syn_infer_get_result(jr, &r), 0, "RT result failed");
	zassert_equal(syn_infer_get_result(jn, &r), -ECANCELED,
		      "cancelled result must be -ECANCELED");

	syn_infer_get_stats(&st1);
	zassert_true(st1.preemptions > st0.preemptions,
		     "NORMAL must have been suspended before the cancel");
	zassert_true(st1.cancelled > st0.cancelled,
		     "cancel counter must increase");

	/* The freed slot must be reusable: preempt once more */
	done_count = 0;
	normal_params.callback = order_cb;
	normal_params.user_data = (void *)1;
	jn = syn_infer_submit(pn, &tensor_a, &normal_params);
	zassert_not_equal(jn, SYN_JOB_INVALID, "second NORMAL submit failed");
	k_msleep(8);
	rt_params.callback = order_cb;
	rt_params.user_data = (void *)2;
	jr = syn_infer_submit(pr, &tensor_b, &rt_params);
	zassert_not_equal(jr, SYN_JOB_INVALID, "second RT submit failed");

	zassert_equal(syn_infer_wait(jr, 5000), 0, "RT wait 2 failed");
	zassert_equal(syn_infer_wait(jn, 5000), 0, "NORMAL wait 2 failed");
	zassert_equal(done_order[0], 2, "RT first after slot reuse");

	zassert_equal(syn_infer_get_result(jr, &r), 0, "result failed");
	zassert_equal(syn_infer_get_result(jn, &r), 0, "result failed");

	syn_pipeline_destroy(pn);
	syn_pipeline_destroy(pr);
	syn_mem_reset_ephemeral();
}
