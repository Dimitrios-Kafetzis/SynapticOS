/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file test_dual_core.c
 * @brief Integration test: dual-core IPC and inference
 *
 * TODO: Implementation pending (Phase 3).
 */

#include <zephyr/ztest.h>
#include <synaptic/syn_api.h>

ZTEST_SUITE(syn_dual_core_suite, NULL, NULL, NULL, NULL, NULL);

ZTEST(syn_dual_core_suite, test_placeholder)
{
    zassert_true(true, "Placeholder — requires dual-core target");
}
