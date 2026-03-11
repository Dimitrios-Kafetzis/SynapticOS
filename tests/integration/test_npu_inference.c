/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file test_npu_inference.c
 * @brief Integration test: end-to-end NPU inference
 *
 * TODO: Implementation pending (Phase 2).
 * Requires on-target execution with FRDM-MCXN947.
 */

#include <zephyr/ztest.h>
#include <synaptic/syn_api.h>

ZTEST_SUITE(syn_npu_integration_suite, NULL, NULL, NULL, NULL, NULL);

ZTEST(syn_npu_integration_suite, test_placeholder)
{
    zassert_true(true, "Placeholder — requires on-target execution");
}
