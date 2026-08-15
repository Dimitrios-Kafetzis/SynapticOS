/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file test_model_registry.c
 * @brief Unit tests for syn_model registry
 */

#include <zephyr/ztest.h>
#include <string.h>
#include <synaptic/syn_model.h>

static void cleanup_registry(void)
{
    /* Unregister all models to start fresh */
    for (uint32_t h = 1; h <= 8; h++) {
        syn_model_unregister(h);
    }
}

ZTEST_SUITE(syn_model_suite, NULL, NULL, NULL, NULL, NULL);

ZTEST(syn_model_suite, test_register_model)
{
    cleanup_registry();

    syn_model_info_t info = {0};
    strncpy(info.name, "test_model_v1", sizeof(info.name));
    strncpy(info.version, "1.0.0", sizeof(info.version));
    info.input_size = 27648;   /* 96*96*3 */
    info.output_size = 10;
    info.sram_required = 65536;

    syn_model_handle_t handle;
    int ret = syn_model_register(&info, &handle);
    zassert_equal(ret, 0, "Register failed: %d", ret);
    zassert_not_equal(handle, SYN_MODEL_INVALID, "Invalid handle");

    cleanup_registry();
}

ZTEST(syn_model_suite, test_get_by_name)
{
    cleanup_registry();

    syn_model_info_t info = {0};
    strncpy(info.name, "lookup_test", sizeof(info.name));
    syn_model_handle_t handle, found;

    syn_model_register(&info, &handle);
    int ret = syn_model_get_by_name("lookup_test", &found);
    zassert_equal(ret, 0, "Lookup failed");
    zassert_equal(handle, found, "Handle mismatch");

    cleanup_registry();
}

ZTEST(syn_model_suite, test_list_models)
{
    cleanup_registry();

    syn_model_info_t info1 = {0};
    strncpy(info1.name, "list_m1", sizeof(info1.name));
    syn_model_info_t info2 = {0};
    strncpy(info2.name, "list_m2", sizeof(info2.name));

    syn_model_handle_t h1, h2;
    syn_model_register(&info1, &h1);
    syn_model_register(&info2, &h2);

    syn_model_handle_t handles[8];
    uint8_t count = 0;
    int ret = syn_model_list(handles, &count, 8);
    zassert_equal(ret, 0, "List failed");
    zassert_equal(count, 2, "Expected 2 models, got %u", count);

    cleanup_registry();
}

ZTEST(syn_model_suite, test_duplicate_name)
{
    cleanup_registry();

    syn_model_info_t info = {0};
    strncpy(info.name, "dup_model", sizeof(info.name));

    syn_model_handle_t h1, h2;
    int ret = syn_model_register(&info, &h1);
    zassert_equal(ret, 0, "First register failed");

    ret = syn_model_register(&info, &h2);
    zassert_equal(ret, -EEXIST, "Duplicate name should return -EEXIST, got %d", ret);

    cleanup_registry();
}

ZTEST(syn_model_suite, test_max_models)
{
    cleanup_registry();

    syn_model_handle_t handles[CONFIG_SYNAPTIC_MAX_MODELS + 1];
    char name[32];

    for (int i = 0; i < CONFIG_SYNAPTIC_MAX_MODELS; i++) {
        syn_model_info_t info = {0};
        snprintf(name, sizeof(name), "model_%d", i);
        strncpy(info.name, name, sizeof(info.name));
        int ret = syn_model_register(&info, &handles[i]);
        zassert_equal(ret, 0, "Register model_%d failed: %d", i, ret);
    }

    /* One more should fail */
    syn_model_info_t extra = {0};
    strncpy(extra.name, "overflow", sizeof(extra.name));
    syn_model_handle_t hx;
    int ret = syn_model_register(&extra, &hx);
    zassert_equal(ret, -ENOMEM, "Should return -ENOMEM when full, got %d", ret);

    cleanup_registry();
}

ZTEST(syn_model_suite, test_unregister_and_reuse)
{
    cleanup_registry();

    syn_model_info_t info = {0};
    strncpy(info.name, "reuse_model", sizeof(info.name));

    syn_model_handle_t h1;
    int ret = syn_model_register(&info, &h1);
    zassert_equal(ret, 0, "First register failed");

    ret = syn_model_unregister(h1);
    zassert_equal(ret, 0, "Unregister failed");

    /* Should be able to register same name again */
    syn_model_handle_t h2;
    ret = syn_model_register(&info, &h2);
    zassert_equal(ret, 0, "Re-register failed: %d", ret);

    cleanup_registry();
}

ZTEST(syn_model_suite, test_load_unload)
{
    cleanup_registry();

    syn_model_info_t info = {0};
    strncpy(info.name, "loadable", sizeof(info.name));

    syn_model_handle_t handle;
    syn_model_register(&info, &handle);

    zassert_false(syn_model_is_loaded(handle), "Should not be loaded initially");

    int ret = syn_model_load(handle);
    zassert_equal(ret, 0, "Load failed: %d", ret);
    zassert_true(syn_model_is_loaded(handle), "Should be loaded");

    /* Double load should return -EALREADY */
    ret = syn_model_load(handle);
    zassert_equal(ret, -EALREADY, "Double load should return -EALREADY");

    ret = syn_model_unload(handle);
    zassert_equal(ret, 0, "Unload failed: %d", ret);
    zassert_false(syn_model_is_loaded(handle), "Should not be loaded after unload");

    /* Double unload should return -EALREADY */
    ret = syn_model_unload(handle);
    zassert_equal(ret, -EALREADY, "Double unload should return -EALREADY");

    cleanup_registry();
}

ZTEST(syn_model_suite, test_register_null)
{
    syn_model_handle_t handle;
    syn_model_info_t info = {0};

    zassert_equal(syn_model_register(NULL, &handle), -EINVAL, "");
    zassert_equal(syn_model_register(&info, NULL), -EINVAL, "");
}

ZTEST(syn_model_suite, test_get_info)
{
    cleanup_registry();

    syn_model_info_t info = {0};
    strncpy(info.name, "info_test", sizeof(info.name));
    strncpy(info.version, "2.1.0", sizeof(info.version));
    info.input_size = 768;
    info.output_size = 10;
    info.sram_required = 4096;
    info.input_dtype = SYN_NPU_DTYPE_INT8;

    syn_model_handle_t handle;
    syn_model_register(&info, &handle);

    syn_model_info_t retrieved = {0};
    int ret = syn_model_get_info(handle, &retrieved);
    zassert_equal(ret, 0, "get_info failed");
    zassert_true(strncmp(retrieved.name, "info_test", 32) == 0, "Name mismatch");
    zassert_true(strncmp(retrieved.version, "2.1.0", 16) == 0, "Version mismatch");
    zassert_equal(retrieved.input_size, 768, "input_size mismatch");
    zassert_equal(retrieved.output_size, 10, "output_size mismatch");
    zassert_equal(retrieved.input_dtype, SYN_NPU_DTYPE_INT8, "dtype mismatch");

    cleanup_registry();
}

/* ------------------------------------------------------------------ */
/* Phase 6 S13: registry negative paths, CRC gate and swap edges      */
/* ------------------------------------------------------------------ */

#include <zephyr/sys/crc.h>
#include <synaptic/syn_hal_npu.h>
#include "syn_model_internal.h"

static uint8_t reg_blob_a[32];
static uint8_t reg_blob_b[32];

static syn_model_handle_t reg_data_model(const char *name,
                                         const uint8_t *blob, size_t size,
                                         uint32_t crc)
{
    syn_model_info_t info = {0};
    syn_model_handle_t h;

    strncpy(info.name, name, sizeof(info.name) - 1);
    info.input_size = 16;
    info.output_size = 10;
    info.crc32 = crc;
    zassert_equal(syn_model_register(&info, &h), 0,
                  "register '%s' failed", name);
    zassert_equal(syn_model_set_data(h, blob, size), 0,
                  "set_data '%s' failed", name);
    return h;
}

ZTEST(syn_model_suite, test_lookup_null_args)
{
    syn_model_handle_t h;
    uint8_t count;

    zassert_equal(syn_model_get_by_name(NULL, &h), -EINVAL,
                  "NULL name accepted");
    zassert_equal(syn_model_get_by_name("x", NULL), -EINVAL,
                  "NULL handle accepted");
    zassert_equal(syn_model_list(NULL, &count, 4), -EINVAL,
                  "NULL list accepted");
}

ZTEST(syn_model_suite, test_invalid_handle_probes)
{
    zassert_equal(syn_model_load(SYN_MODEL_INVALID), -EINVAL, "load");
    zassert_equal(syn_model_unload(SYN_MODEL_INVALID), -EINVAL, "unload");
    zassert_equal(syn_model_ensure_resident(SYN_MODEL_INVALID), -EINVAL,
                  "ensure_resident");
    zassert_false(syn_model_is_loaded(SYN_MODEL_INVALID), "is_loaded");
    zassert_equal(syn_model_set_data(SYN_MODEL_INVALID, reg_blob_a, 1),
                  -EINVAL, "set_data");
}

ZTEST(syn_model_suite, test_set_data_rejects_empty)
{
    cleanup_registry();

    syn_model_info_t info = {0};
    syn_model_handle_t h;

    strncpy(info.name, "sd_probe", sizeof(info.name));
    zassert_ok(syn_model_register(&info, &h), "register failed");
    zassert_equal(syn_model_set_data(h, reg_blob_a, 0), -EINVAL,
                  "data with zero size accepted");
    zassert_ok(syn_model_set_data(h, NULL, 0), "detach refused");

    cleanup_registry();
}

ZTEST(syn_model_suite, test_load_crc_gate)
{
    cleanup_registry();
    syn_hal_npu_init();

    for (size_t i = 0; i < sizeof(reg_blob_a); i++) {
        reg_blob_a[i] = (uint8_t)(i * 7U + 1U);
    }

    /* registered CRC disagrees with the bytes: load is refused */
    syn_model_handle_t h = reg_data_model("crc_bad", reg_blob_a,
                                          sizeof(reg_blob_a), 0x1234U);

    zassert_equal(syn_model_load(h), -EILSEQ,
                  "corrupt model load accepted");
    zassert_false(syn_model_is_loaded(h), "loaded despite CRC gate");

    cleanup_registry();
}

ZTEST(syn_model_suite, test_load_hal_reject_propagates)
{
    cleanup_registry();
    syn_hal_npu_init();

    /* lie about the size: the stub HAL rejects oversized models */
    syn_model_handle_t h = reg_data_model("too_big", reg_blob_a,
                                          400U * 1024U, 0U);

    zassert_equal(syn_model_load(h), -ENOMEM,
                  "oversized model load accepted");

    cleanup_registry();
}

ZTEST(syn_model_suite, test_residency_stats_snapshot)
{
    uint32_t swaps = 0xAAAAAAAA, last = 0xAAAAAAAA;

    syn_model_residency_stats(&swaps, &last);
    zassert_not_equal(swaps, 0xAAAAAAAA, "swaps not written");
    zassert_not_equal(last, 0xAAAAAAAA, "last_us not written");
    syn_model_residency_stats(NULL, NULL); /* NULL-safe */
}

ZTEST(syn_model_suite, test_swap_and_edges)
{
    cleanup_registry();
    syn_hal_npu_deinit();
    zassert_ok(syn_hal_npu_init(), "NPU init failed");

    for (size_t i = 0; i < sizeof(reg_blob_a); i++) {
        reg_blob_a[i] = (uint8_t)(i + 3U);
        reg_blob_b[i] = (uint8_t)(0x80U - i);
    }

    syn_model_handle_t ha = reg_data_model(
        "swap_a", reg_blob_a, sizeof(reg_blob_a),
        crc32_ieee(reg_blob_a, sizeof(reg_blob_a)));
    syn_model_handle_t hb = reg_data_model(
        "swap_b", reg_blob_b, sizeof(reg_blob_b),
        crc32_ieee(reg_blob_b, sizeof(reg_blob_b)));

    /* argument edges */
    zassert_equal(syn_model_swap(SYN_MODEL_INVALID, hb), -EINVAL,
                  "invalid old accepted");
    zassert_equal(syn_model_swap(ha, ha), -EINVAL, "self-swap accepted");

    /* the real thing: A resident, swap to B */
    zassert_ok(syn_model_load(ha), "load A failed");
    zassert_ok(syn_model_swap(ha, hb), "swap failed");
    zassert_false(syn_model_is_loaded(ha), "old still loaded");
    zassert_true(syn_model_is_loaded(hb), "new not loaded");
    zassert_true(syn_model_last_swap_us() < 1000000U,
                 "swap duration implausible");

    /* swap onto a corrupt model fails and reports it */
    reg_blob_a[5] ^= 0x20U; /* break A against its registered CRC */
    zassert_equal(syn_model_swap(hb, ha), -EILSEQ,
                  "swap onto a corrupt model accepted");
    reg_blob_a[5] ^= 0x20U;

    /* on-demand residency swap refuses corrupt data the same way:
     * load both while intact (B loaded but A resident), then corrupt
     * B behind the registry's back
     */
    zassert_ok(syn_model_load(hb), "re-load B failed");
    zassert_ok(syn_model_load(ha), "re-load A failed");
    reg_blob_b[9] ^= 0x10U;
    zassert_equal(syn_model_ensure_resident(hb), -EILSEQ,
                  "ensure_resident of corrupt data accepted");
    reg_blob_b[9] ^= 0x10U;

    cleanup_registry();
}
