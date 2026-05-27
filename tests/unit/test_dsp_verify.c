/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file test_dsp_verify.c
 * @brief DSP softmax and argmax verification with printable output
 *
 * Produces results R7 (softmax correctness) and R8 (argmax correctness)
 * for community content. Tests print detailed output for screenshots.
 */

#include <zephyr/ztest.h>
#include <synaptic/syn_hal_dsp.h>
#include <math.h>
#include <zephyr/logging/log.h>

LOG_MODULE_REGISTER(test_dsp_verify, LOG_LEVEL_INF);

ZTEST_SUITE(syn_dsp_verify_suite, NULL, NULL, NULL, NULL, NULL);

/**
 * R7: Softmax correctness  - verify output sums to 1.0 with known inputs.
 */
ZTEST(syn_dsp_verify_suite, test_softmax_known_input)
{
	float in[] = {1.0f, 2.0f, 3.0f, 4.0f};
	float out[4];

	LOG_INF("=== R7: Softmax Correctness (known input) ===");
	LOG_INF("  Input:  [1.0, 2.0, 3.0, 4.0]");

	int ret = syn_hal_dsp_softmax_f32(in, out, 4);
	zassert_equal(ret, 0, "softmax failed: %d", ret);

	float sum = 0.0f;

	for (int i = 0; i < 4; i++) {
		sum += out[i];
	}

	/* Print each value (LOG_INF doesn't support %f on all platforms,
	 * so we multiply by 10000 and print as integer for precision) */
	LOG_INF("  Output: [%u.%04u, %u.%04u, %u.%04u, %u.%04u]",
		(unsigned)(out[0]), (unsigned)(out[0] * 10000) % 10000,
		(unsigned)(out[1]), (unsigned)(out[1] * 10000) % 10000,
		(unsigned)(out[2]), (unsigned)(out[2] * 10000) % 10000,
		(unsigned)(out[3]), (unsigned)(out[3] * 10000) % 10000);
	LOG_INF("  Sum:    %u.%06u",
		(unsigned)(sum),
		(unsigned)(sum * 1000000) % 1000000);

	/* Verify sum is ~1.0 */
	zassert_true(fabsf(sum - 1.0f) < 0.001f,
		     "Softmax sum should be ~1.0, got sum with error %d/1000",
		     (int)(fabsf(sum - 1.0f) * 1000));

	/* Verify ordering: in[3]=4.0 should have highest probability */
	zassert_true(out[3] > out[2], "out[3] should be > out[2]");
	zassert_true(out[2] > out[1], "out[2] should be > out[1]");
	zassert_true(out[1] > out[0], "out[1] should be > out[0]");

	/* Verify approximate expected values:
	 * softmax([1,2,3,4]) ≈ [0.0321, 0.0871, 0.2369, 0.6439] */
	zassert_true(fabsf(out[0] - 0.0321f) < 0.01f,
		     "out[0] should be ~0.032");
	zassert_true(fabsf(out[3] - 0.6439f) < 0.01f,
		     "out[3] should be ~0.644");

	LOG_INF("  Ordering: out[3] > out[2] > out[1] > out[0] VERIFIED");
	LOG_INF("=== R7: PASS  - softmax sums to 1.0, correct distribution ===");
}

/**
 * R7 (continued): Softmax with all-zeros input  - should give uniform output.
 */
ZTEST(syn_dsp_verify_suite, test_softmax_uniform)
{
	float in[] = {0.0f, 0.0f, 0.0f, 0.0f};
	float out[4];

	LOG_INF("=== R7: Softmax Uniform Case (all zeros) ===");
	LOG_INF("  Input:  [0.0, 0.0, 0.0, 0.0]");

	int ret = syn_hal_dsp_softmax_f32(in, out, 4);
	zassert_equal(ret, 0, "softmax failed: %d", ret);

	float sum = 0.0f;

	for (int i = 0; i < 4; i++) {
		sum += out[i];
	}

	/* Each output should be ~0.25 (1/4) */
	for (int i = 0; i < 4; i++) {
		zassert_true(fabsf(out[i] - 0.25f) < 0.01f,
			     "out[%d] should be ~0.25", i);
	}

	zassert_true(fabsf(sum - 1.0f) < 0.001f,
		     "Sum should be ~1.0");

	LOG_INF("  Output: ~[0.25, 0.25, 0.25, 0.25] (uniform)");
	LOG_INF("  Sum:    ~1.0");
	LOG_INF("=== R7: PASS  - uniform input gives uniform softmax ===");
}

/**
 * R7 (continued): Softmax numerical stability with large values.
 */
ZTEST(syn_dsp_verify_suite, test_softmax_large_values)
{
	float in[] = {100.0f, 101.0f, 102.0f};
	float out[3];

	LOG_INF("=== R7: Softmax Numerical Stability (large values) ===");
	LOG_INF("  Input:  [100.0, 101.0, 102.0]");

	int ret = syn_hal_dsp_softmax_f32(in, out, 3);
	zassert_equal(ret, 0, "softmax failed: %d", ret);

	float sum = out[0] + out[1] + out[2];

	/* Should not produce NaN or Inf */
	for (int i = 0; i < 3; i++) {
		zassert_false(isnan(out[i]), "out[%d] is NaN", i);
		zassert_false(isinf(out[i]), "out[%d] is Inf", i);
	}

	zassert_true(fabsf(sum - 1.0f) < 0.001f,
		     "Sum should be ~1.0 even with large inputs");

	/* Ordering should be preserved */
	zassert_true(out[2] > out[1], "out[2] should be > out[1]");
	zassert_true(out[1] > out[0], "out[1] should be > out[0]");

	LOG_INF("  No NaN/Inf  - numerically stable (max-subtraction trick)");
	LOG_INF("  Sum: ~1.0, ordering preserved");
	LOG_INF("=== R7: PASS  - numerically stable with large values ===");
}

/**
 * R8: Argmax correctness  - verify on known data patterns.
 */
ZTEST(syn_dsp_verify_suite, test_argmax_known_patterns)
{
	uint32_t idx;

	LOG_INF("=== R8: Argmax Correctness ===");

	/* Pattern 1: clear maximum in the middle */
	int8_t data1[] = {-10, -5, 0, 127, 50, 10, -128};
	int ret = syn_hal_dsp_argmax(data1, 7, &idx);

	zassert_equal(ret, 0, "argmax failed");
	zassert_equal(idx, 3, "Expected index 3 (value 127), got %u", idx);
	LOG_INF("  Pattern 1: [-10,-5,0,127,50,10,-128] -> argmax=%u (val=%d) CORRECT",
		idx, data1[idx]);

	/* Pattern 2: maximum at the start */
	int8_t data2[] = {100, 50, 0, -50, -100};
	ret = syn_hal_dsp_argmax(data2, 5, &idx);
	zassert_equal(ret, 0, "argmax failed");
	zassert_equal(idx, 0, "Expected index 0, got %u", idx);
	LOG_INF("  Pattern 2: [100,50,0,-50,-100] -> argmax=%u (val=%d) CORRECT",
		idx, data2[idx]);

	/* Pattern 3: maximum at the end */
	int8_t data3[] = {-100, -50, 0, 50, 100};
	ret = syn_hal_dsp_argmax(data3, 5, &idx);
	zassert_equal(ret, 0, "argmax failed");
	zassert_equal(idx, 4, "Expected index 4, got %u", idx);
	LOG_INF("  Pattern 3: [-100,-50,0,50,100] -> argmax=%u (val=%d) CORRECT",
		idx, data3[idx]);

	/* Pattern 4: simulated model output (10 classes, one clear winner) */
	int8_t model_output[] = {-20, -15, -10, -5, 0, 5, 10, 80, -30, -40};
	ret = syn_hal_dsp_argmax(model_output, 10, &idx);
	zassert_equal(ret, 0, "argmax failed");
	zassert_equal(idx, 7, "Expected index 7 (value 80), got %u", idx);
	LOG_INF("  Pattern 4 (model output): 10 classes -> "
		"argmax=%u (confidence=%d) CORRECT",
		idx, model_output[idx]);

	LOG_INF("=== R8: PASS  - argmax correct on all patterns ===");
}

/**
 * R8 (continued): Argmax matches manual scan  - cross-validation.
 */
ZTEST(syn_dsp_verify_suite, test_argmax_vs_manual_scan)
{
	/* Use a realistic-looking inference output */
	int8_t output[] = {-50, -30, 15, -80, -10, 127, -128, 42, 90, -5};
	const size_t len = 10;
	uint32_t dsp_idx;
	uint32_t manual_idx = 0;
	int8_t manual_max = output[0];

	LOG_INF("=== R8: Argmax vs Manual Scan Cross-Validation ===");
	LOG_INF("  Output buffer: [-50,-30,15,-80,-10,127,-128,42,90,-5]");

	/* Manual scan */
	for (size_t i = 1; i < len; i++) {
		if (output[i] > manual_max) {
			manual_max = output[i];
			manual_idx = i;
		}
	}

	/* DSP argmax */
	int ret = syn_hal_dsp_argmax(output, len, &dsp_idx);
	zassert_equal(ret, 0, "argmax failed");

	LOG_INF("  Manual scan: index=%u, value=%d", manual_idx, manual_max);
	LOG_INF("  DSP argmax:  index=%u, value=%d", dsp_idx, output[dsp_idx]);

	zassert_equal(dsp_idx, manual_idx,
		      "DSP argmax (%u) != manual scan (%u)",
		      dsp_idx, manual_idx);
	zassert_equal(output[dsp_idx], manual_max,
		      "Values don't match");

	LOG_INF("=== R8: PASS  - DSP argmax matches manual scan ===");
}
