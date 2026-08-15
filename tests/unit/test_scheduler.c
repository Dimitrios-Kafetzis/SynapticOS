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

#include "test_common.h"

#define sched_arena test_shared_arena
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

	/* S8 residency contract: jobs are refused unless the model is
	 * loaded. The suite drives the HAL blob directly above, so the
	 * registry load is eligibility-only (no data attached).
	 */
	int lret = syn_model_load(sched_model);

	zassert_true(lret == 0 || lret == -EALREADY,
		     "model load failed: %d", lret);
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

/* ------------------------------------------------------------------ */
/* Phase 6 S13: scheduler negative paths and quiesce-window edges     */
/* ------------------------------------------------------------------ */

#include "syn_infer_internal.h"

/* Slow passthrough stage: pins the scheduler thread mid-job long
 * enough for the cooperative test thread to observe RUNNING state
 * (volatile loop: k_busy_wait needs timer hardware QEMU lacks).
 */
static int slow_stage(const syn_tensor_t *in, syn_tensor_t *out,
                      const void *config)
{
    ARG_UNUSED(config);

    for (volatile int i = 0; i < 300000; i++) {
    }
    if (out->size < in->size) {
        return -ENOMEM;
    }
    memcpy(out->data, in->data, in->size);
    out->size = in->size;
    out->dtype = in->dtype;
    out->ndim = in->ndim;
    memcpy(out->shape, in->shape, sizeof(out->shape));
    return 0;
}

static syn_pipeline_t *sched_pipe(const char *name)
{
    syn_pipeline_t *pipe = syn_pipeline_create(name);

    zassert_not_null(pipe, "pipeline create failed");
    zassert_equal(syn_pipeline_add_model(pipe, sched_model), 0,
                  "add_model failed");
    zassert_equal(syn_pipeline_build(pipe), 0, "build failed");
    return pipe;
}

ZTEST(syn_sched_suite, test_stats_reset)
{
    syn_infer_get_stats(NULL); /* NULL-safe */

    make_input(0, 0x21);

    int8_t out_buf[16];
    syn_tensor_t out = { .data = out_buf, .size = sizeof(out_buf) };

    zassert_ok(syn_infer_run_sync(sched_model, &input_tensors[0], &out,
                                  SYN_PRIORITY_NORMAL), "run failed");

    syn_infer_stats_t st;

    syn_infer_get_stats(&st);
    zassert_true(st.completed > 0, "no completion recorded");

    syn_infer_reset_stats();
    syn_infer_get_stats(&st);
    zassert_equal(st.completed, 0, "completed not reset");
    zassert_equal(st.errors, 0, "errors not reset");
}

ZTEST(syn_sched_suite, test_quiesce_defers_then_replays)
{
    syn_pipeline_t *pipe = sched_pipe("defer");

    make_input(0, 0x31);
    make_input(1, 0x32);

    /* nothing running: quiesce returns immediately but parks dispatch */
    syn_infer_quiesce();

    syn_job_id_t id1 = syn_infer_submit(pipe, &input_tensors[0], NULL);
    syn_job_id_t id2 = syn_infer_submit(pipe, &input_tensors[1], NULL);

    zassert_not_equal(id1, SYN_JOB_INVALID, "submit 1 failed");
    zassert_not_equal(id2, SYN_JOB_INVALID, "submit 2 failed");

    /* the scheduler wakes, sees the pause, and defers both wakeups.
     * Equal priority, no deadlines: dispatch falls to the seq tie.
     */
    k_msleep(2);
    zassert_equal(syn_infer_wait(id1, 0), -EAGAIN,
                  "job dispatched through a paused gate");

    syn_infer_release();

    zassert_ok(syn_infer_wait(id1, 2000), "wait 1 failed");
    zassert_ok(syn_infer_wait(id2, 2000), "wait 2 failed");

    syn_tensor_t r;

    zassert_ok(syn_infer_get_result(id1, &r), "result 1 failed");
    zassert_ok(syn_infer_get_result(id2, &r), "result 2 failed");
    syn_pipeline_destroy(pipe);
}

ZTEST(syn_sched_suite, test_release_replay_finds_nothing)
{
    syn_pipeline_t *pipe = sched_pipe("replay");

    make_input(0, 0x33);
    syn_infer_quiesce();

    syn_job_id_t id = syn_infer_submit(pipe, &input_tensors[0], NULL);

    zassert_not_equal(id, SYN_JOB_INVALID, "submit failed");
    k_msleep(2); /* the wake is deferred */

    zassert_ok(syn_infer_cancel(id), "cancel failed");
    syn_infer_release();
    k_msleep(2); /* replayed wake dispatches into an empty queue */

    zassert_equal(syn_infer_wait(id, 100), -ECANCELED, "not cancelled");

    syn_tensor_t r;

    zassert_equal(syn_infer_get_result(id, &r), -ECANCELED,
                  "cancelled result");
    syn_pipeline_destroy(pipe);
}

ZTEST(syn_sched_suite, test_destroy_cancels_queued_job)
{
    syn_pipeline_t *pipe = sched_pipe("dcq");

    make_input(0, 0x34);
    syn_infer_quiesce();

    syn_job_id_t id = syn_infer_submit(pipe, &input_tensors[0], NULL);

    zassert_not_equal(id, SYN_JOB_INVALID, "submit failed");

    syn_pipeline_destroy(pipe);
    syn_infer_release();

    zassert_equal(syn_infer_wait(id, 100), -ECANCELED,
                  "destroy must cancel the queued job");

    syn_tensor_t r;

    zassert_equal(syn_infer_get_result(id, &r), -ECANCELED,
                  "cancelled result");
}

ZTEST(syn_sched_suite, test_cancel_running_and_quiesce_drain)
{
    syn_pipeline_t *pipe = syn_pipeline_create("slowrun");

    zassert_not_null(pipe, "create failed");
    zassert_equal(syn_pipeline_add_preprocess(pipe, slow_stage, NULL), 0,
                  "add slow stage failed");
    zassert_equal(syn_pipeline_add_model(pipe, sched_model), 0,
                  "add_model failed");
    zassert_equal(syn_pipeline_build(pipe), 0, "build failed");

    make_input(0, 0x35);

    syn_job_id_t id = syn_infer_submit(pipe, &input_tensors[0], NULL);

    zassert_not_equal(id, SYN_JOB_INVALID, "submit failed");

    /* land inside the slow stage */
    k_msleep(5);
    zassert_equal(syn_infer_cancel(id), -EBUSY,
                  "cancel of a RUNNING job must be refused");

    /* quiesce now has a live job to drain */
    syn_infer_quiesce();
    syn_infer_release();

    zassert_ok(syn_infer_wait(id, 5000), "wait failed");

    syn_tensor_t r;

    zassert_ok(syn_infer_get_result(id, &r), "result failed");
    syn_pipeline_destroy(pipe);
}

ZTEST(syn_sched_suite, test_wait_twice_on_error_job)
{
    syn_pipeline_t *pipe = sched_pipe("errwait");

    make_input(0, 0x36);
    zassert_ok(syn_model_unload(sched_model), "unload failed");

    syn_job_id_t id = syn_infer_submit(pipe, &input_tensors[0], NULL);

    zassert_not_equal(id, SYN_JOB_INVALID, "submit failed");
    zassert_equal(syn_infer_wait(id, 2000), -ENOEXEC, "first wait");
    zassert_equal(syn_infer_wait(id, 2000), -ENOEXEC,
                  "second wait must read the stored result");

    zassert_equal(syn_infer_get_result(id, NULL), -EINVAL,
                  "NULL output accepted");

    syn_tensor_t r;

    zassert_equal(syn_infer_get_result(id, &r), -ENOEXEC,
                  "error result");

    int lret = syn_model_load(sched_model);

    zassert_true(lret == 0 || lret == -EALREADY, "re-load failed");
    syn_pipeline_destroy(pipe);
}

/* Cancel fired from the system workqueue while the submitter blocks
 * in wait: the waiter must see -ECANCELED, not a result.
 */
static syn_job_id_t cancel_target;

static void cancel_work_fn(struct k_work *work)
{
    ARG_UNUSED(work);
    (void)syn_infer_cancel(cancel_target);
}

static K_WORK_DELAYABLE_DEFINE(cancel_work, cancel_work_fn);

ZTEST(syn_sched_suite, test_cancel_lands_during_wait)
{
    syn_pipeline_t *pipe = sched_pipe("cxwait");

    make_input(0, 0x37);
    syn_infer_quiesce(); /* keep the job queued while we wait on it */

    syn_job_id_t id = syn_infer_submit(pipe, &input_tensors[0], NULL);

    zassert_not_equal(id, SYN_JOB_INVALID, "submit failed");
    cancel_target = id;
    k_work_schedule(&cancel_work, K_MSEC(20));

    zassert_equal(syn_infer_wait(id, 2000), -ECANCELED,
                  "waiter must observe the cancel");

    syn_infer_release();

    syn_tensor_t r;

    zassert_equal(syn_infer_get_result(id, &r), -ECANCELED,
                  "cancelled result");
    syn_pipeline_destroy(pipe);
}

ZTEST(syn_sched_suite, test_done_slots_block_submission)
{
    syn_pipeline_t *pipe = sched_pipe("slots");

    make_input(0, 0x38);

    /* run the table full of DONE-but-unconsumed jobs */
    syn_job_id_t ids[CONFIG_SYNAPTIC_MAX_CONCURRENT_JOBS];

    for (int i = 0; i < CONFIG_SYNAPTIC_MAX_CONCURRENT_JOBS; i++) {
        ids[i] = syn_infer_submit(pipe, &input_tensors[0], NULL);
        zassert_not_equal(ids[i], SYN_JOB_INVALID, "submit %d failed", i);

        int wret = syn_infer_wait(ids[i], 2000);

        zassert_equal(wret, 0, "wait %d failed: %d", i, wret);
    }

    /* no active jobs, but every slot still holds a result */
    zassert_equal(syn_infer_submit(pipe, &input_tensors[0], NULL),
                  SYN_JOB_INVALID, "submit into a full table accepted");

    /* run_sync trips over the same wall and cleans up after itself */
    int8_t out_buf[16];
    syn_tensor_t out = { .data = out_buf, .size = sizeof(out_buf) };

    zassert_equal(syn_infer_run_sync(sched_model, &input_tensors[0],
                                     &out, SYN_PRIORITY_NORMAL), -EBUSY,
                  "run_sync into a full table accepted");

    syn_tensor_t r;

    for (int i = 0; i < CONFIG_SYNAPTIC_MAX_CONCURRENT_JOBS; i++) {
        zassert_ok(syn_infer_get_result(ids[i], &r),
                   "consume %d failed", i);
    }
    syn_pipeline_destroy(pipe);
}

ZTEST(syn_sched_suite, test_run_sync_argument_edges)
{
    make_input(0, 0x39);

    int8_t out_buf[16];
    syn_tensor_t out = { .data = out_buf, .size = sizeof(out_buf) };

    zassert_equal(syn_infer_run_sync(sched_model, NULL, &out,
                                     SYN_PRIORITY_NORMAL), -EINVAL,
                  "NULL input accepted");
    zassert_equal(syn_infer_run_sync(sched_model, &input_tensors[0],
                                     NULL, SYN_PRIORITY_NORMAL), -EINVAL,
                  "NULL output accepted");
    zassert_equal(syn_infer_run_sync(SYN_MODEL_INVALID,
                                     &input_tensors[0], &out,
                                     SYN_PRIORITY_NORMAL), -EINVAL,
                  "invalid model accepted");

    /* pipeline pool exhausted: run_sync cannot build its pipeline */
    syn_pipeline_t *pipes[4];

    for (int i = 0; i < 4; i++) {
        pipes[i] = syn_pipeline_create("hog");
        zassert_not_null(pipes[i], "hog create %d failed", i);
    }
    zassert_equal(syn_infer_run_sync(sched_model, &input_tensors[0],
                                     &out, SYN_PRIORITY_NORMAL), -ENOMEM,
                  "run_sync without a free pipeline accepted");
    for (int i = 0; i < 4; i++) {
        syn_pipeline_destroy(pipes[i]);
    }

    /* NULL data hands back the arena-backed descriptor */
    syn_tensor_t arena_out = { .data = NULL, .size = 0 };

    zassert_ok(syn_infer_run_sync(sched_model, &input_tensors[0],
                                  &arena_out, SYN_PRIORITY_NORMAL),
               "arena-descriptor run failed");
    zassert_not_null(arena_out.data, "no arena descriptor returned");
    zassert_equal(arena_out.size, 10, "wrong arena output size");

    zassert_false(syn_infer_model_suspended(SYN_MODEL_INVALID),
                  "invalid model reported as suspended");
}
