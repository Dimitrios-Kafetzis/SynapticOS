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
