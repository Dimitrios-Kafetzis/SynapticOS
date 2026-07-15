/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file test_model_swap.c
 * @brief Unit tests for model hot-swap (Phase 4.3)
 *
 * Verifies the syn_model_swap() semantics: waits for the in-flight
 * inference, never corrupts its result, leaves queued jobs parked
 * until the new model is resident, and completes fast. A deliberately
 * slow preprocess stage stands in for a long-running inference.
 */

#include <zephyr/ztest.h>
#include <zephyr/sys/crc.h>
#include <string.h>
#include <synaptic/syn_api.h>
#include <synaptic/syn_model.h>
#include <synaptic/syn_infer.h>
#include <synaptic/syn_hal_npu.h>

#include "syn_model_internal.h"

#define SWAP_MODEL_BYTES 256U

static uint8_t model_a_data[SWAP_MODEL_BYTES];
static uint8_t model_b_data[SWAP_MODEL_BYTES];

static syn_model_handle_t h_a, h_b;

/* observability for the queued-job test */
static volatile bool slow_pre_done;
static volatile bool post_saw_b_loaded;

static void fill(uint8_t *buf, uint8_t seed)
{
    for (uint32_t i = 0; i < SWAP_MODEL_BYTES; i++) {
        buf[i] = (uint8_t)(seed + i * 7U);
    }
}

static syn_model_handle_t register_one(const char *name, uint8_t *data,
                                       uint8_t seed)
{
    syn_model_info_t info = {0};
    syn_model_handle_t h;

    fill(data, seed);
    strncpy(info.name, name, sizeof(info.name) - 1);
    info.input_size = 64;
    info.output_size = 16;
    info.crc32 = crc32_ieee(data, SWAP_MODEL_BYTES);

    zassert_ok(syn_model_register(&info, &h), "register failed");
    zassert_ok(syn_model_set_data(h, data, SWAP_MODEL_BYTES));
    return h;
}

static void swap_fresh(void)
{
    /* pipelines allocate stage buffers from the tensor arena */
    (void)syn_init(); /* -EALREADY after the first test is fine */
    syn_model_reset_all();
    syn_hal_npu_init();
    slow_pre_done = false;
    post_saw_b_loaded = false;
    h_a = register_one("swap_a", model_a_data, 3);
    h_b = register_one("swap_b", model_b_data, 91);
}

/* ~200 ms passthrough preprocess: the "in-flight inference" */
static int slow_passthrough(const syn_tensor_t *in, syn_tensor_t *out,
                            const void *config)
{
    ARG_UNUSED(config);

    k_sleep(K_MSEC(200));

    size_t n = MIN(in->size, out->size);

    memcpy(out->data, in->data, n);
    out->size = n;
    slow_pre_done = true;
    return 0;
}

/* records whether model B was NPU-resident while this job executed */
static int probe_post(const syn_tensor_t *in, syn_tensor_t *out,
                      const void *config)
{
    ARG_UNUSED(config);

    post_saw_b_loaded = syn_model_is_loaded(h_b);

    size_t n = MIN(in->size, out->size);

    memcpy(out->data, in->data, n);
    out->size = n;
    return 0;
}

static syn_pipeline_t *build_pipe(const char *name, syn_model_handle_t model,
                                  bool slow, bool probe)
{
    syn_pipeline_t *p = syn_pipeline_create(name);

    zassert_not_null(p, "pipeline create failed");
    if (slow) {
        zassert_ok(syn_pipeline_add_preprocess(p, slow_passthrough, NULL));
    }
    zassert_ok(syn_pipeline_add_model(p, model));
    if (probe) {
        zassert_ok(syn_pipeline_add_postprocess(p, probe_post, NULL));
    }
    zassert_ok(syn_pipeline_build(p), "pipeline build failed");
    return p;
}

static syn_tensor_t make_input(uint8_t *buf, size_t n)
{
    syn_tensor_t t = {0};

    memset(buf, 0x5A, n);
    t.data = buf;
    t.size = n;
    t.ndim = 1;
    t.shape[0] = n;
    return t;
}

ZTEST_SUITE(syn_swap_suite, NULL, NULL, NULL, NULL, NULL);

ZTEST(syn_swap_suite, test_swap_basic)
{
    swap_fresh();

    zassert_ok(syn_model_load(h_a), "load A failed");
    zassert_true(syn_model_is_loaded(h_a), "A not loaded");

    zassert_ok(syn_model_swap(h_a, h_b), "swap failed");
    zassert_false(syn_model_is_loaded(h_a), "A still loaded");
    zassert_true(syn_model_is_loaded(h_b), "B not loaded");

    /* acceptance: < 500 ms (idle NPU here; the in-flight case below
     * measures the waiting variant separately)
     */
    zassert_true(syn_model_last_swap_us() < 500000U,
                 "swap took %u us", syn_model_last_swap_us());
}

ZTEST(syn_swap_suite, test_swap_invalid_args)
{
    swap_fresh();

    zassert_equal(syn_model_swap(h_a, h_a), -EINVAL, "self-swap allowed");
    zassert_equal(syn_model_swap(SYN_MODEL_INVALID, h_b), -EINVAL,
                  "invalid old handle allowed");
    zassert_equal(syn_model_swap(h_a, 99), -EINVAL,
                  "invalid new handle allowed");
}

ZTEST(syn_swap_suite, test_swap_refuses_corrupt_new_model)
{
    swap_fresh();

    zassert_ok(syn_model_load(h_a));

    model_b_data[10] ^= 0x01U; /* CRC gate must catch this */

    zassert_equal(syn_model_swap(h_a, h_b), -EILSEQ,
                  "corrupt model swapped in");
    zassert_false(syn_model_is_loaded(h_a), "old still marked loaded");
    zassert_false(syn_model_is_loaded(h_b), "corrupt marked loaded");

    model_b_data[10] ^= 0x01U; /* restore for later suites */
}

ZTEST(syn_swap_suite, test_swap_waits_for_inflight)
{
    swap_fresh();

    zassert_ok(syn_model_load(h_a));

    syn_pipeline_t *pipe = build_pipe("slow_a", h_a, true, false);
    static uint8_t inbuf[64];
    syn_tensor_t in = make_input(inbuf, sizeof(inbuf));
    syn_infer_params_t params = { .priority = SYN_PRIORITY_NORMAL };

    syn_job_id_t job = syn_infer_submit(pipe, &in, &params);

    zassert_not_equal(job, SYN_JOB_INVALID, "submit failed");

    /* let the scheduler enter the slow stage, then swap: it must
     * block until the job has fully executed
     */
    k_sleep(K_MSEC(20));
    zassert_false(slow_pre_done, "job finished too early for the test");

    zassert_ok(syn_model_swap(h_a, h_b), "swap failed");
    zassert_true(slow_pre_done,
                 "swap returned while the job was still executing");

    zassert_ok(syn_infer_wait(job, 1000), "job lost");

    syn_tensor_t out;

    zassert_ok(syn_infer_get_result(job, &out),
               "in-flight result corrupted by swap");

    zassert_true(syn_model_is_loaded(h_b), "B not loaded after swap");
    zassert_true(syn_model_last_swap_us() < 500000U,
                 "swap incl. wait took %u us", syn_model_last_swap_us());

    syn_pipeline_destroy(pipe);
}

ZTEST(syn_swap_suite, test_queued_job_runs_after_swap_on_new_model)
{
    swap_fresh();

    zassert_ok(syn_model_load(h_a));

    syn_pipeline_t *slow_pipe = build_pipe("slow_a2", h_a, true, false);
    syn_pipeline_t *probe_pipe = build_pipe("probe_b", h_b, false, true);

    static uint8_t inbuf1[64], inbuf2[64];
    syn_tensor_t in1 = make_input(inbuf1, sizeof(inbuf1));
    syn_tensor_t in2 = make_input(inbuf2, sizeof(inbuf2));
    syn_infer_params_t params = { .priority = SYN_PRIORITY_NORMAL };

    syn_job_id_t job1 = syn_infer_submit(slow_pipe, &in1, &params);

    zassert_not_equal(job1, SYN_JOB_INVALID);

    k_sleep(K_MSEC(20)); /* job1 is now inside the slow stage */

    /* queued while job1 runs; must execute only after the swap */
    syn_job_id_t job2 = syn_infer_submit(probe_pipe, &in2, &params);

    zassert_not_equal(job2, SYN_JOB_INVALID);

    zassert_ok(syn_model_swap(h_a, h_b), "swap failed");

    zassert_ok(syn_infer_wait(job1, 1000), "job1 lost");
    zassert_ok(syn_infer_wait(job2, 1000), "queued job2 never ran");
    zassert_true(post_saw_b_loaded,
                 "queued job executed before the new model was resident");

    /* recycle the job slots */
    syn_tensor_t out;

    zassert_ok(syn_infer_get_result(job1, &out));
    zassert_ok(syn_infer_get_result(job2, &out));

    syn_pipeline_destroy(slow_pipe);
    syn_pipeline_destroy(probe_pipe);
}
