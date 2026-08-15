/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file test_init.c
 * @brief Unit tests for syn_init / syn_shutdown runtime lifecycle
 */

#include <zephyr/ztest.h>
#include <synaptic/syn_api.h>

ZTEST_SUITE(syn_init_suite, NULL, NULL, NULL, NULL, NULL);

ZTEST(syn_init_suite, test_init_succeeds)
{
    /* Shutdown first in case a previous test left it initialized */
    syn_shutdown();

    int ret = syn_init();
    zassert_equal(ret, 0, "syn_init() failed: %d", ret);

    syn_shutdown();
}

ZTEST(syn_init_suite, test_double_init)
{
    syn_shutdown();

    int ret = syn_init();
    zassert_equal(ret, 0, "First init failed: %d", ret);

    ret = syn_init();
    zassert_equal(ret, -EALREADY, "Double init should return -EALREADY, got %d", ret);

    syn_shutdown();
}

ZTEST(syn_init_suite, test_shutdown_before_init)
{
    /* Make sure we're not initialized */
    syn_shutdown();
    syn_shutdown(); /* Double shutdown to ensure clean state */

    int ret = syn_shutdown();
    zassert_equal(ret, -EPERM, "Shutdown before init should return -EPERM, got %d", ret);
}

ZTEST(syn_init_suite, test_reinit_after_shutdown)
{
    syn_shutdown();

    int ret = syn_init();
    zassert_equal(ret, 0, "First init failed: %d", ret);

    ret = syn_shutdown();
    zassert_equal(ret, 0, "Shutdown failed: %d", ret);

    ret = syn_init();
    zassert_equal(ret, 0, "Re-init after shutdown failed: %d", ret);

    syn_shutdown();
}

/* ------------------------------------------------------------------ */
/* Phase 6 S13: version string and HAL-collision propagation          */
/* ------------------------------------------------------------------ */

#include <string.h>
#include <synaptic/syn_hal_npu.h>

ZTEST(syn_init_suite, test_version_string)
{
    const char *v = syn_version();

    zassert_not_null(v, "version NULL");
    zassert_true(strlen(v) >= 5, "version implausibly short: '%s'", v);
}

ZTEST(syn_init_suite, test_init_propagates_npu_collision)
{
    syn_shutdown();

    /* someone brought the NPU HAL up outside the runtime */
    syn_hal_npu_deinit();
    zassert_equal(syn_hal_npu_init(), 0, "direct NPU init failed");

    int ret = syn_init();

    zassert_equal(ret, -EALREADY,
                  "init over a live NPU HAL must propagate: %d", ret);

    /* clean retry succeeds */
    syn_hal_npu_deinit();
    zassert_equal(syn_init(), 0, "clean re-init failed");
    syn_shutdown();
}
