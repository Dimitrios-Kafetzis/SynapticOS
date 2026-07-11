/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file test_scheduler.c
 * @brief Unit tests for the inference job scheduler (Phase 2.2)
 *
 * The ztest thread is cooperative, so jobs submitted here stay QUEUED
 * until the test blocks (e.g. in syn_infer_wait). This makes priority
 * ordering and queued-state assertions deterministic.
 */

#include <zephyr/ztest.h>
#include <synaptic/syn_infer.h>
#include <synaptic/syn_mem.h>
#include <synaptic/syn_model.h>
#include <synaptic/syn_hal_npu.h>
#include <string.h>

#define SCHED_INPUT_SIZE 48

static uint8_t sched_arena[4096] __aligned(16);
static const uint8_t sched_model_bin[32] = {0};
static syn_model_handle_t sched_model;

/* Static input buffers: must stay valid while jobs execute */
static uint8_t input_bufs[3][SCHED_INPUT_SIZE];
static syn_tensor_t input_tensors[3];

/* Execution-order recording via completion callbacks */
static syn_priority_t exec_order[3];
static volatile int exec_count;

static void record_cb(syn_job_id_t job, const syn_tensor_t *output,
		      void *user_data)
{
	ARG_UNUSED(job);
	ARG_UNUSED(output);

	if (exec_count < 3) {
		exec_order[exec_count] = (syn_priority_t)(uintptr_t)user_data;
	}
	exec_count++;
}

static void make_input(int idx, uint8_t fill)
{
	uint32_t shape[1] = { SCHED_INPUT_SIZE };

	memset(input_bufs[idx], fill, SCHED_INPUT_SIZE);
	syn_mem_tensor_init(&input_tensors[idx], shape, 1,
			    SYN_NPU_DTYPE_INT8);
	input_tensors[idx].data = input_bufs[idx];
	input_tensors[idx].lifetime = SYN_MEM_SHARED;
}

/** Expected stub NPU prediction: sum of input bytes mod 10 */
static uint32_t expected_class(const uint8_t *data, size_t len)
{
	uint32_t sum = 0;

	for (size_t i = 0; i < len; i++) {
		sum += data[i];
	}
	return sum % 10;
}

static void *sched_suite_setup(void)
{
	/* Fresh NPU stub with a loaded model */
	syn_hal_npu_deinit();
	zassert_equal(syn_hal_npu_init(), 0, "NPU init failed");
	zassert_equal(syn_hal_npu_load_model(sched_model_bin,
					     sizeof(sched_model_bin)),
		      0, "NPU model load failed");

	if (syn_model_get_by_name("sched_test", &sched_model) != 0) {
		syn_model_info_t info = {0};

		strncpy(info.name, "sched_test", sizeof(info.name));
		strncpy(info.version, "1.0.0", sizeof(info.version));
		info.input_size = SCHED_INPUT_SIZE;
		info.output_size = 10;
		info.sram_required = 256;
		info.input_dtype = SYN_NPU_DTYPE_INT8;
		info.output_dtype = SYN_NPU_DTYPE_INT8;
		zassert_equal(syn_model_register(&info, &sched_model), 0,
			      "Model registration failed");
	}
	return NULL;
}

static void sched_before(void *fixture)
{
	ARG_UNUSED(fixture);
	syn_mem_init(sched_arena, sizeof(sched_arena));
	exec_count = 0;
	memset(exec_order, 0xFF, sizeof(exec_order));
}

ZTEST_SUITE(syn_sched_suite, NULL, sched_suite_setup, sched_before,
	    NULL, NULL);

ZTEST(syn_sched_suite, test_run_sync)
{
	make_input(0, 0x11);

	int8_t out_buf[16];
	syn_tensor_t output = {
		.data = out_buf,
		.size = sizeof(out_buf),
	};

	int ret = syn_infer_run_sync(sched_model, &input_tensors[0],
				     &output, SYN_PRIORITY_NORMAL);

	zassert_equal(ret, 0, "run_sync failed: %d", ret);
	zassert_equal(output.size, 10, "Wrong output size: %u",
		      (unsigned)output.size);

	uint32_t expect = expected_class(input_bufs[0], SCHED_INPUT_SIZE);

	zassert_equal(out_buf[expect], 127,
		      "Expected class %u to have confidence 127", expect);
}

ZTEST(syn_sched_suite, test_priority_order)
{
	/* Submit in reverse priority order; verify execution order is
	 * REALTIME -> NORMAL -> BEST_EFFORT (acceptance criterion 2.2).
	 */
	syn_pipeline_t *pipe = syn_pipeline_create("prio");

	zassert_not_null(pipe, "Pipeline create failed");
	zassert_equal(syn_pipeline_add_model(pipe, sched_model), 0,
		      "add_model failed");
	zassert_equal(syn_pipeline_build(pipe), 0, "build failed");

	static const syn_priority_t submit_order[3] = {
		SYN_PRIORITY_BEST_EFFORT,
		SYN_PRIORITY_NORMAL,
		SYN_PRIORITY_REALTIME,
	};
	syn_job_id_t ids[3];

	for (int i = 0; i < 3; i++) {
		make_input(i, (uint8_t)(0x20 + i));

		syn_infer_params_t params = {
			.priority = submit_order[i],
			.callback = record_cb,
			.user_data = (void *)(uintptr_t)submit_order[i],
		};

		ids[i] = syn_infer_submit(pipe, &input_tensors[i], &params);
		zassert_not_equal(ids[i], SYN_JOB_INVALID,
				  "Submit %d failed", i);
	}

	/* Wait for every job, then check the recorded order */
	for (int i = 0; i < 3; i++) {
		zassert_equal(syn_infer_wait(ids[i], 2000), 0,
			      "Wait %d failed", i);
	}
	zassert_equal(exec_count, 3, "Expected 3 completions, got %d",
		      exec_count);
	zassert_equal(exec_order[0], SYN_PRIORITY_REALTIME,
		      "First executed should be REALTIME");
	zassert_equal(exec_order[1], SYN_PRIORITY_NORMAL,
		      "Second executed should be NORMAL");
	zassert_equal(exec_order[2], SYN_PRIORITY_BEST_EFFORT,
		      "Third executed should be BEST_EFFORT");

	for (int i = 0; i < 3; i++) {
		syn_tensor_t result;

		zassert_equal(syn_infer_get_result(ids[i], &result), 0,
			      "get_result %d failed", i);
	}

	syn_pipeline_destroy(pipe);
}

ZTEST(syn_sched_suite, test_wait_timeout)
{
	syn_pipeline_t *pipe = syn_pipeline_create("timeout");

	zassert_not_null(pipe, "Pipeline create failed");
	zassert_equal(syn_pipeline_add_model(pipe, sched_model), 0,
		      "add_model failed");
	zassert_equal(syn_pipeline_build(pipe), 0, "build failed");

	make_input(0, 0x33);

	syn_job_id_t id = syn_infer_submit(pipe, &input_tensors[0], NULL);

	zassert_not_equal(id, SYN_JOB_INVALID, "Submit failed");

	/* Zero timeout from a cooperative thread: the scheduler has not
	 * run yet, so the job is still queued -> -EAGAIN.
	 */
	zassert_equal(syn_infer_wait(id, 0), -EAGAIN,
		      "Zero-timeout wait on queued job should be -EAGAIN");

	/* get_result on an unfinished job is -EBUSY */
	syn_tensor_t result;

	zassert_equal(syn_infer_get_result(id, &result), -EBUSY,
		      "get_result on queued job should be -EBUSY");

	/* Now let it finish and consume the slot */
	zassert_equal(syn_infer_wait(id, 2000), 0, "Wait failed");
	zassert_equal(syn_infer_get_result(id, &result), 0,
		      "get_result failed");

	syn_pipeline_destroy(pipe);
}

ZTEST(syn_sched_suite, test_cancel)
{
	syn_pipeline_t *pipe = syn_pipeline_create("cancel");

	zassert_not_null(pipe, "Pipeline create failed");
	zassert_equal(syn_pipeline_add_model(pipe, sched_model), 0,
		      "add_model failed");
	zassert_equal(syn_pipeline_build(pipe), 0, "build failed");

	make_input(0, 0x44);
	make_input(1, 0x55);

	syn_job_id_t id1 = syn_infer_submit(pipe, &input_tensors[0], NULL);
	syn_job_id_t id2 = syn_infer_submit(pipe, &input_tensors[1], NULL);

	zassert_not_equal(id1, SYN_JOB_INVALID, "Submit 1 failed");
	zassert_not_equal(id2, SYN_JOB_INVALID, "Submit 2 failed");

	/* Both still queued (cooperative thread): cancel the second */
	zassert_equal(syn_infer_cancel(id2), 0, "Cancel queued job failed");
	zassert_equal(syn_infer_wait(id2, 100), -ECANCELED,
		      "Wait on cancelled job should be -ECANCELED");

	syn_tensor_t result;

	zassert_equal(syn_infer_get_result(id2, &result), -ECANCELED,
		      "get_result on cancelled job should be -ECANCELED");

	/* First job completes normally */
	zassert_equal(syn_infer_wait(id1, 2000), 0, "Wait 1 failed");

	/* Cancelling a finished job is -EALREADY */
	zassert_equal(syn_infer_cancel(id1), -EALREADY,
		      "Cancel of finished job should be -EALREADY");
	zassert_equal(syn_infer_get_result(id1, &result), 0,
		      "get_result 1 failed");

	syn_pipeline_destroy(pipe);
}

ZTEST(syn_sched_suite, test_invalid_job_id)
{
	syn_tensor_t result;

	zassert_equal(syn_infer_wait(0xDEAD, 10), -ENOENT,
		      "Wait on bogus id should be -ENOENT");
	zassert_equal(syn_infer_cancel(0xDEAD), -ENOENT,
		      "Cancel on bogus id should be -ENOENT");
	zassert_equal(syn_infer_get_result(0xDEAD, &result), -ENOENT,
		      "get_result on bogus id should be -ENOENT");
	zassert_equal(syn_infer_wait(SYN_JOB_INVALID, 10), -ENOENT,
		      "Wait on SYN_JOB_INVALID should be -ENOENT");
}

ZTEST(syn_sched_suite, test_submit_validation)
{
	/* Unbuilt pipeline is rejected */
	syn_pipeline_t *pipe = syn_pipeline_create("unbuilt");

	zassert_not_null(pipe, "Pipeline create failed");
	zassert_equal(syn_pipeline_add_model(pipe, sched_model), 0,
		      "add_model failed");

	make_input(0, 0x66);
	zassert_equal(syn_infer_submit(pipe, &input_tensors[0], NULL),
		      SYN_JOB_INVALID, "Submit to unbuilt pipe should fail");

	zassert_equal(syn_pipeline_build(pipe), 0, "build failed");

	/* NULL input is rejected */
	zassert_equal(syn_infer_submit(pipe, NULL, NULL), SYN_JOB_INVALID,
		      "Submit with NULL input should fail");

	syn_pipeline_destroy(pipe);
}

ZTEST(syn_sched_suite, test_max_concurrent)
{
	syn_pipeline_t *pipe = syn_pipeline_create("maxjobs");

	zassert_not_null(pipe, "Pipeline create failed");
	zassert_equal(syn_pipeline_add_model(pipe, sched_model), 0,
		      "add_model failed");
	zassert_equal(syn_pipeline_build(pipe), 0, "build failed");

	zassert_equal(syn_infer_set_max_concurrent(0), -EINVAL,
		      "Zero max should be rejected");
	zassert_equal(syn_infer_set_max_concurrent(1), 0,
		      "set_max_concurrent(1) failed");

	make_input(0, 0x77);
	make_input(1, 0x88);

	syn_job_id_t id1 = syn_infer_submit(pipe, &input_tensors[0], NULL);

	zassert_not_equal(id1, SYN_JOB_INVALID, "Submit 1 failed");

	/* Limit reached: second submit rejected */
	zassert_equal(syn_infer_submit(pipe, &input_tensors[1], NULL),
		      SYN_JOB_INVALID, "Submit above limit should fail");

	zassert_equal(syn_infer_wait(id1, 2000), 0, "Wait failed");

	syn_tensor_t result;

	zassert_equal(syn_infer_get_result(id1, &result), 0,
		      "get_result failed");

	/* Restore the configured limit */
	zassert_equal(syn_infer_set_max_concurrent(
			      CONFIG_SYNAPTIC_MAX_CONCURRENT_JOBS),
		      0, "Restore limit failed");

	syn_pipeline_destroy(pipe);
}
