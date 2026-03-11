/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file test_npu_hal.c
 * @brief Unit tests for NPU HAL stub
 */

#include <zephyr/ztest.h>
#include <synaptic/syn_hal_npu.h>
#include <string.h>

ZTEST_SUITE(syn_npu_suite, NULL, NULL, NULL, NULL, NULL);

ZTEST(syn_npu_suite, test_npu_init)
{
    /* Deinit first to ensure clean state */
    syn_hal_npu_deinit();

    int ret = syn_hal_npu_init();
    zassert_equal(ret, 0, "NPU init failed: %d", ret);

    /* Double init should return -EALREADY */
    ret = syn_hal_npu_init();
    zassert_equal(ret, -EALREADY, "Double init should return -EALREADY");

    /* State should be IDLE after init */
    zassert_equal(syn_hal_npu_get_state(), SYN_NPU_STATE_IDLE,
                  "State should be IDLE after init");

    syn_hal_npu_deinit();
}

ZTEST(syn_npu_suite, test_npu_caps)
{
    syn_hal_npu_deinit();
    syn_hal_npu_init();

    syn_npu_caps_t caps;
    int ret = syn_hal_npu_get_caps(&caps);
    zassert_equal(ret, 0, "get_caps failed");
    zassert_true(strcmp(caps.name, "stub") == 0, "Wrong name");
    zassert_equal(caps.supports_async, false, "Stub should not support async");

    /* NULL caps should fail */
    ret = syn_hal_npu_get_caps(NULL);
    zassert_equal(ret, -EINVAL, "NULL caps should return -EINVAL");

    syn_hal_npu_deinit();
}

ZTEST(syn_npu_suite, test_npu_load_model)
{
    syn_hal_npu_deinit();
    syn_hal_npu_init();

    uint8_t fake_model[64];
    memset(fake_model, 0xAA, sizeof(fake_model));

    int ret = syn_hal_npu_load_model(fake_model, sizeof(fake_model));
    zassert_equal(ret, 0, "load_model failed: %d", ret);

    /* NULL model should fail */
    ret = syn_hal_npu_load_model(NULL, 64);
    zassert_equal(ret, -EINVAL, "NULL model should return -EINVAL");

    /* Zero size should fail */
    ret = syn_hal_npu_load_model(fake_model, 0);
    zassert_equal(ret, -EINVAL, "Zero size should return -EINVAL");

    syn_hal_npu_deinit();
}

ZTEST(syn_npu_suite, test_npu_invoke)
{
    syn_hal_npu_deinit();
    syn_hal_npu_init();

    uint8_t fake_model[32];
    memset(fake_model, 0, sizeof(fake_model));
    syn_hal_npu_load_model(fake_model, sizeof(fake_model));

    /* Set input: all zeros */
    uint8_t input[16];
    memset(input, 0, sizeof(input));
    int ret = syn_hal_npu_set_input(0, input, sizeof(input));
    zassert_equal(ret, 0, "set_input failed: %d", ret);

    /* Invoke */
    ret = syn_hal_npu_invoke();
    zassert_equal(ret, 0, "invoke failed: %d", ret);

    /* Get output */
    uint8_t output[256];
    size_t out_size = 0;
    ret = syn_hal_npu_get_output(0, output, &out_size);
    zassert_equal(ret, 0, "get_output failed: %d", ret);
    zassert_equal(out_size, 10, "Expected 10-class output");

    /* With all-zero input, sum=0, winner=0%10=0, so output[0] should be 127 */
    zassert_equal(output[0], 127, "Class 0 should have max confidence");

    /* Verify determinism: same input produces same output */
    syn_hal_npu_set_input(0, input, sizeof(input));
    syn_hal_npu_invoke();

    uint8_t output2[256];
    size_t out_size2 = 0;
    syn_hal_npu_get_output(0, output2, &out_size2);
    zassert_mem_equal(output, output2, out_size, "Output should be deterministic");

    syn_hal_npu_deinit();
}

ZTEST(syn_npu_suite, test_npu_state_machine)
{
    syn_hal_npu_deinit();
    syn_hal_npu_init();

    /* After init: IDLE */
    zassert_equal(syn_hal_npu_get_state(), SYN_NPU_STATE_IDLE, "Should be IDLE");

    /* Suspend: IDLE -> SUSPENDED */
    int ret = syn_hal_npu_suspend();
    zassert_equal(ret, 0, "suspend failed");
    zassert_equal(syn_hal_npu_get_state(), SYN_NPU_STATE_SUSPENDED,
                  "Should be SUSPENDED");

    /* Resume: SUSPENDED -> IDLE */
    ret = syn_hal_npu_resume();
    zassert_equal(ret, 0, "resume failed");
    zassert_equal(syn_hal_npu_get_state(), SYN_NPU_STATE_IDLE,
                  "Should be IDLE after resume");

    /* Resume when not suspended should fail */
    ret = syn_hal_npu_resume();
    zassert_equal(ret, -EINVAL, "Resume when IDLE should return -EINVAL");

    syn_hal_npu_deinit();
}

ZTEST(syn_npu_suite, test_npu_error_handling)
{
    syn_hal_npu_deinit();
    syn_hal_npu_init();

    /* Invoke without model should fail */
    int ret = syn_hal_npu_invoke();
    zassert_equal(ret, -EPERM, "Invoke without model should return -EPERM");

    /* Set input without model should fail */
    uint8_t buf[8] = {0};
    ret = syn_hal_npu_set_input(0, buf, sizeof(buf));
    zassert_equal(ret, -EPERM, "set_input without model should return -EPERM");

    /* Operations when not initialized */
    syn_hal_npu_deinit();
    uint8_t model[16] = {0};
    ret = syn_hal_npu_load_model(model, sizeof(model));
    zassert_equal(ret, -EPERM, "load_model when not initialized should return -EPERM");

    ret = syn_hal_npu_suspend();
    zassert_equal(ret, -EPERM, "suspend when not initialized should return -EPERM");

    /* Async invoke not supported */
    syn_hal_npu_init();
    ret = syn_hal_npu_invoke_async(NULL, NULL);
    zassert_equal(ret, -ENOTSUP, "invoke_async should return -ENOTSUP");

    /* get_output with NULL args */
    ret = syn_hal_npu_get_output(0, NULL, NULL);
    zassert_equal(ret, -EINVAL, "NULL args should return -EINVAL");

    /* Invalid input index */
    ret = syn_hal_npu_set_input(1, buf, sizeof(buf));
    zassert_equal(ret, -EINVAL, "Index 1 should return -EINVAL");

    syn_hal_npu_deinit();
}
