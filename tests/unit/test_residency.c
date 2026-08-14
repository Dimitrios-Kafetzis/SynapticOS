/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file test_residency.c
 * @brief Unit tests for the NPU residency contract (Phase 6, S8)
 *
 * S7 board findings on the single-residency Neutron HAL, verified
 * here on the stub analog through the engine-level behavior:
 *  - unloading the resident model must release the HAL residency
 *    (on Neutron: neutron_unprepare, or the prepared model's
 *    input-size gate fails every other invoke until reboot);
 *  - a job on a model that is not loaded is refused loudly
 *    (-ENOEXEC) instead of silently running whatever model happens
 *    to be NPU-resident (the S7 1353 us fake vs 6581 us real trap);
 *  - among loaded models the engine swaps the single-residency NPU
 *    on demand at dispatch, so mixed-model pipelines run correctly.
 *
 * The ztest thread is cooperative: submitted jobs stay QUEUED until
 * the test blocks, which makes the submit-then-unload window
 * deterministic. The suite keeps its registry footprint at two
 * models (and unregisters both in teardown): the registry caps at
 * CONFIG_SYNAPTIC_MAX_MODELS across all suites.
 */

#include <zephyr/ztest.h>
#include <synaptic/syn_infer.h>
#include <synaptic/syn_mem.h>
#include <synaptic/syn_model.h>
#include <synaptic/syn_hal_npu.h>
#include <string.h>

#include "syn_model_internal.h"
#include "syn_npu_layered.h"
#include "syn_hal_npu_internal.h"
#include "test_common.h"

#define res_arena test_shared_arena

#define RES_PLAIN_INPUT  48
#define RES_LAYER_INPUT  16
#define RES_LAYER_OUT    8

static const uint8_t plain_blob[32] = {0};
static uint8_t layer_blob[64];

static syn_model_handle_t plain_model;
static syn_model_handle_t layer_model;

/* Static inputs: must stay valid while jobs execute */
static uint8_t plain_in[RES_PLAIN_INPUT];
static uint8_t layer_in[RES_LAYER_INPUT];
static syn_tensor_t plain_tensor;
static syn_tensor_t layer_tensor;

static void make_inputs(void)
{
	uint32_t shape_p[1] = { RES_PLAIN_INPUT };
	uint32_t shape_l[1] = { RES_LAYER_INPUT };

	memset(plain_in, 0x11, sizeof(plain_in));
	syn_mem_tensor_init(&plain_tensor, shape_p, 1, SYN_NPU_DTYPE_INT8);
	plain_tensor.data = plain_in;
	plain_tensor.lifetime = SYN_MEM_SHARED;

	for (size_t i = 0; i < sizeof(layer_in); i++) {
		layer_in[i] = (uint8_t)(i * 3U + 1U);
	}
	syn_mem_tensor_init(&layer_tensor, shape_l, 1, SYN_NPU_DTYPE_INT8);
	layer_tensor.data = layer_in;
	layer_tensor.lifetime = SYN_MEM_SHARED;
}

static void register_one(const char *name, uint32_t in_size,
			 uint32_t out_size, const uint8_t *data,
			 size_t data_size, syn_model_handle_t *handle)
{
	if (syn_model_get_by_name(name, handle) != 0) {
		syn_model_info_t info = {0};

		strncpy(info.name, name, sizeof(info.name) - 1);
		strncpy(info.version, "1.0.0", sizeof(info.version) - 1);
		info.input_size = in_size;
		info.output_size = out_size;
		info.input_dtype = SYN_NPU_DTYPE_INT8;
		info.output_dtype = SYN_NPU_DTYPE_INT8;
		zassert_equal(syn_model_register(&info, handle), 0,
			      "register '%s' failed", name);
	}
	zassert_equal(syn_model_set_data(*handle, data, data_size), 0,
		      "set_data '%s' failed", name);
}

static void *res_suite_setup(void)
{
	/* The registry caps at CONFIG_SYNAPTIC_MAX_MODELS (4 in this
	 * app) shared across all suites: start from a clean registry,
	 * like test_model_registry does. Later suites re-register
	 * through the get_by_name guard in their setups.
	 */
	syn_model_reset_all();

	syn_hal_npu_deinit();
	zassert_equal(syn_hal_npu_init(), 0, "NPU init failed");

	uint16_t out_sizes[2] = { 16, RES_LAYER_OUT };
	uint16_t work[2] = { 0, 0 };
	int size = syn_npu_layered_make_model(layer_blob, sizeof(layer_blob),
					      RES_LAYER_INPUT, 2,
					      out_sizes, work);

	zassert_true(size > 0, "make_model failed: %d", size);

	register_one("res_plain", RES_PLAIN_INPUT, 10,
		     plain_blob, sizeof(plain_blob), &plain_model);
	register_one("res_layer", RES_LAYER_INPUT, RES_LAYER_OUT,
		     layer_blob, (size_t)size, &layer_model);

	zassert_equal(syn_model_load(plain_model), 0, "plain load failed");
	zassert_equal(syn_model_load(layer_model), 0, "layer load failed");

	make_inputs();
	return NULL;
}

static void res_suite_teardown(void *fixture)
{
	ARG_UNUSED(fixture);

	/* Leave no registry footprint for the other suites */
	syn_model_unregister(plain_model);
	syn_model_unregister(layer_model);
}

static void res_before(void *fixture)
{
	ARG_UNUSED(fixture);
	syn_mem_init(res_arena, sizeof(res_arena));

	/* Every test starts from "both models loaded" (tests that
	 * unload restore the state themselves, but a zassert abort
	 * inside one must not cascade into the next).
	 */
	if (!syn_model_is_loaded(plain_model)) {
		zassert_equal(syn_model_load(plain_model), 0,
			      "plain re-load failed");
	}
	if (!syn_model_is_loaded(layer_model)) {
		zassert_equal(syn_model_load(layer_model), 0,
			      "layer re-load failed");
	}
}

ZTEST_SUITE(syn_residency_suite, NULL, res_suite_setup, res_before,
	    NULL, res_suite_teardown);

/** run_sync on a registered-but-not-loaded model is refused loudly. */
ZTEST(syn_residency_suite, test_run_sync_refuses_not_loaded)
{
	zassert_equal(syn_model_unload(plain_model), 0, "unload failed");

	int8_t out_buf[16];
	syn_tensor_t out = { .data = out_buf, .size = sizeof(out_buf) };

	int ret = syn_infer_run_sync(plain_model, &plain_tensor, &out,
				     SYN_PRIORITY_NORMAL);

	zassert_equal(ret, -ENOEXEC,
		      "not-loaded run must be -ENOEXEC, got %d", ret);
	zassert_equal(syn_model_load(plain_model), 0, "re-load failed");
}

/** Submit stays permissive (the store hot-swap pattern queues jobs
 *  against a model being swapped in), but a job on a model that is
 *  still not loaded when it dispatches fails with -ENOEXEC.
 */
ZTEST(syn_residency_suite, test_submit_permissive_dispatch_refuses)
{
	syn_pipeline_t *pipe = syn_pipeline_create("res_pipe_r");

	zassert_not_null(pipe, "pipeline create failed");
	zassert_equal(syn_pipeline_add_model(pipe, plain_model), 0,
		      "add_model failed");
	zassert_equal(syn_pipeline_build(pipe), 0, "build failed");

	zassert_equal(syn_model_unload(plain_model), 0, "unload failed");

	syn_job_id_t job = syn_infer_submit(pipe, &plain_tensor, NULL);

	zassert_not_equal(job, SYN_JOB_INVALID,
			  "submit must stay permissive (hot-swap pattern)");

	int ret = syn_infer_wait(job, 5000);

	zassert_equal(ret, -ENOEXEC,
		      "dispatch on a not-loaded model must fail "
		      "-ENOEXEC, got %d", ret);

	syn_tensor_t discard;

	(void)syn_infer_get_result(job, &discard);
	syn_pipeline_destroy(pipe);
	zassert_equal(syn_model_load(plain_model), 0, "re-load failed");
}

/** A job queued on a model that is unloaded before dispatch fails
 *  with -ENOEXEC at dispatch (the authoritative check).
 */
ZTEST(syn_residency_suite, test_dispatch_refuses_unloaded_queued_job)
{
	syn_pipeline_t *pipe = syn_pipeline_create("res_pipe_q");

	zassert_not_null(pipe, "pipeline create failed");
	zassert_equal(syn_pipeline_add_model(pipe, plain_model), 0,
		      "add_model failed");
	zassert_equal(syn_pipeline_build(pipe), 0, "build failed");

	syn_job_id_t job = syn_infer_submit(pipe, &plain_tensor, NULL);

	zassert_not_equal(job, SYN_JOB_INVALID, "submit failed");

	/* Still queued (cooperative thread): unload wins the race */
	zassert_equal(syn_model_unload(plain_model), 0, "unload failed");

	int ret = syn_infer_wait(job, 5000);

	zassert_equal(ret, -ENOEXEC,
		      "queued job on an unloaded model must fail "
		      "-ENOEXEC, got %d", ret);

	syn_tensor_t discard;

	(void)syn_infer_get_result(job, &discard);
	syn_pipeline_destroy(pipe);
	zassert_equal(syn_model_load(plain_model), 0, "re-load failed");
}

/** Unloading the resident model releases the HAL residency (the S7
 *  Item C fix: on Neutron this is the neutron_unprepare call).
 */
ZTEST(syn_residency_suite, test_unload_releases_hal)
{
	int8_t out_buf[16];
	syn_tensor_t out = { .data = out_buf, .size = sizeof(out_buf) };

	/* Make plain_model the resident model, then unload it */
	zassert_equal(syn_infer_run_sync(plain_model, &plain_tensor, &out,
					 SYN_PRIORITY_NORMAL), 0,
		      "plain run failed");
	zassert_equal(syn_model_unload(plain_model), 0, "unload failed");

	/* Nothing resident: a direct HAL probe must be refused */
	uint8_t probe[8] = {0};

	zassert_equal(syn_hal_npu_set_input(0, probe, sizeof(probe)),
		      -EPERM, "HAL must hold no model after unload");

	/* Restore for the remaining tests */
	zassert_equal(syn_model_load(plain_model), 0, "re-load failed");
}

/** Loaded models swap the single-residency NPU on demand at
 *  dispatch: a plain-stub job and a layered job alternate correctly.
 */
ZTEST(syn_residency_suite, test_on_demand_swap)
{
	uint32_t swaps_before, swaps_after;

	syn_model_residency_stats(&swaps_before, NULL);

	int8_t out_buf[16];
	syn_tensor_t out = { .data = out_buf, .size = sizeof(out_buf) };

	/* Plain model: stub signature is 10 bytes, winner = sum % 10 */
	zassert_equal(syn_infer_run_sync(plain_model, &plain_tensor, &out,
					 SYN_PRIORITY_NORMAL), 0,
		      "plain run failed");
	zassert_equal(out.size, 10, "stub output must be 10 bytes");

	uint32_t sum = 0;

	for (size_t i = 0; i < sizeof(plain_in); i++) {
		sum += plain_in[i];
	}
	zassert_equal(out_buf[sum % 10U], 127, "stub winner mismatch");

	/* Layered model: only runs if its blob became resident again
	 * (before the swap fix this returned the 10-byte stub output).
	 */
	out.data = out_buf;
	out.size = sizeof(out_buf);
	zassert_equal(syn_infer_run_sync(layer_model, &layer_tensor, &out,
					 SYN_PRIORITY_NORMAL), 0,
		      "layered run failed");
	zassert_equal(out.size, RES_LAYER_OUT,
		      "layered output must be %u bytes, got %u",
		      RES_LAYER_OUT, (unsigned)out.size);

	/* And back again */
	out.data = out_buf;
	out.size = sizeof(out_buf);
	zassert_equal(syn_infer_run_sync(plain_model, &plain_tensor, &out,
					 SYN_PRIORITY_NORMAL), 0,
		      "plain re-run failed");
	zassert_equal(out.size, 10, "stub output must be 10 bytes");

	syn_model_residency_stats(&swaps_after, NULL);
	zassert_true(swaps_after >= swaps_before + 2U,
		     "expected at least 2 residency swaps, got %u",
		     swaps_after - swaps_before);
}
