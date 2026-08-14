/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file test_prof.c
 * @brief Unit tests for profiling and diagnostics (Phase 5.9)
 *
 * The profiler was the least-covered core module (17% lines at the
 * first Phase 5 coverage measurement); these tests exercise the
 * enable/disable gate, the stage marks driven through a real
 * inference, layer tracing, and the summary printer.
 */

#include <zephyr/ztest.h>
#include <synaptic/syn_prof.h>
#include <synaptic/syn_infer.h>
#include <synaptic/syn_mem.h>
#include <synaptic/syn_model.h>
#include <synaptic/syn_hal_npu.h>
#include <string.h>

#include "syn_prof_internal.h"
#include "test_common.h"

#define PROF_INPUT_SIZE 48

static syn_model_handle_t prof_model;
static const uint8_t prof_model_bin[32] = {0};
static uint8_t prof_input[PROF_INPUT_SIZE];

static void *prof_suite_setup(void)
{
	syn_hal_npu_deinit();
	zassert_equal(syn_hal_npu_init(), 0, "NPU init failed");
	zassert_equal(syn_hal_npu_load_model(prof_model_bin,
					     sizeof(prof_model_bin)), 0,
		      "NPU model load failed");

	if (syn_model_get_by_name("prof_test", &prof_model) != 0) {
		syn_model_info_t info = {0};

		strncpy(info.name, "prof_test", sizeof(info.name));
		strncpy(info.version, "1.0.0", sizeof(info.version));
		info.input_size = PROF_INPUT_SIZE;
		info.output_size = 10;
		info.input_dtype = SYN_NPU_DTYPE_INT8;
		info.output_dtype = SYN_NPU_DTYPE_INT8;
		zassert_equal(syn_model_register(&info, &prof_model), 0,
			      "model register failed");
	}

	/* S8 residency contract: registry load makes the model eligible
	 * to run (no data attached; the HAL blob above stays resident).
	 */
	int lret = syn_model_load(prof_model);

	zassert_true(lret == 0 || lret == -EALREADY,
		     "model load failed: %d", lret);
	return NULL;
}

static void prof_before(void *fixture)
{
	ARG_UNUSED(fixture);
	syn_mem_init(test_shared_arena, sizeof(test_shared_arena));
	syn_prof_enable();
}

ZTEST_SUITE(syn_prof_suite, NULL, prof_suite_setup, prof_before, NULL,
	    NULL);

static int run_one_inference(void)
{
	uint32_t shape[1] = { PROF_INPUT_SIZE };
	syn_tensor_t in;
	int8_t out_buf[16];
	syn_tensor_t out = { .data = out_buf, .size = sizeof(out_buf) };

	for (int i = 0; i < PROF_INPUT_SIZE; i++) {
		prof_input[i] = (uint8_t)(i * 5 + 3);
	}
	syn_mem_tensor_init(&in, shape, 1, SYN_NPU_DTYPE_INT8);
	in.data = prof_input;
	in.lifetime = SYN_MEM_SHARED;

	int ret = syn_infer_run_sync(prof_model, &in, &out,
				     SYN_PRIORITY_NORMAL);

	syn_mem_reset_ephemeral();
	return ret;
}

/** A profiled inference produces a consistent stage breakdown. */
ZTEST(syn_prof_suite, test_prof_marks_via_inference)
{
	zassert_equal(run_one_inference(), 0, "inference failed");

	syn_prof_result_t r;

	zassert_equal(syn_prof_get_last(&r), 0, "get_last failed");
	zassert_true(r.total_us > 0, "total time missing");
	zassert_true(r.npu_us <= r.total_us, "npu time exceeds total");
	zassert_true(r.preprocess_us + r.npu_us + r.postprocess_us <=
		     r.total_us + 1000,
		     "stage sum wildly exceeds total");
	zassert_true(r.npu_utilization_pct <= 100,
		     "utilization above 100%%");

	/* The summary printer must handle a valid record */
	syn_prof_print_summary();
}

/** Disabled profiling records nothing new. */
ZTEST(syn_prof_suite, test_prof_disable)
{
	zassert_equal(run_one_inference(), 0, "inference failed");

	syn_prof_result_t before;

	zassert_equal(syn_prof_get_last(&before), 0, "get_last failed");

	zassert_equal(syn_prof_disable(), 0, "disable failed");

	/* Marks while disabled must not touch the last record */
	syn_prof_mark_start();
	syn_prof_mark_preprocess_done();
	syn_prof_mark_npu_done();
	syn_prof_mark_end();

	syn_prof_result_t after;

	zassert_equal(syn_prof_get_last(&after), 0,
		      "last record must survive disable");
	zassert_equal(after.total_us, before.total_us,
		      "disabled profiling must not update the record");

	zassert_equal(syn_prof_enable(), 0, "re-enable failed");
	syn_prof_print_summary();
}

/** get_last validates its argument. */
ZTEST(syn_prof_suite, test_prof_get_last_null)
{
	zassert_equal(syn_prof_get_last(NULL), -EINVAL,
		      "NULL result must be rejected");
}

/** Layer tracing API responds sanely on the stub backend. */
ZTEST(syn_prof_suite, test_prof_layer_trace)
{
	int ret = syn_prof_enable_layer_trace();

	zassert_true(ret == 0 || ret == -ENOTSUP,
		     "unexpected layer-trace result: %d", ret);

	uint32_t us = 0;

	ret = syn_prof_get_layer_time(0, &us);
	zassert_true(ret == 0 || ret == -ENOTSUP || ret == -ENOENT ||
		     ret == -EINVAL,
		     "unexpected layer-time result: %d", ret);
	zassert_equal(syn_prof_get_layer_time(0, NULL), -EINVAL,
		      "NULL out must be rejected");
}
