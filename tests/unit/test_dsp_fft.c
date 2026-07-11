/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file test_dsp_fft.c
 * @brief Unit tests for DSP FFT and Q15 matrix multiply (Phase 2.3)
 */

#include <zephyr/ztest.h>
#include <synaptic/syn_hal_dsp.h>
#include <math.h>
#include <string.h>

#define FFT_N 16

static float fft_in[FFT_N * 2];
static float fft_out[FFT_N * 2];

ZTEST_SUITE(syn_dsp_fft_suite, NULL, NULL, NULL, NULL, NULL);

ZTEST(syn_dsp_fft_suite, test_fft_impulse)
{
	/* delta[0] = 1: spectrum is flat, all bins (1, 0) */
	memset(fft_in, 0, sizeof(fft_in));
	fft_in[0] = 1.0f;

	zassert_equal(syn_hal_dsp_fft_f32(fft_in, fft_out, FFT_N), 0,
		      "FFT failed");

	for (int k = 0; k < FFT_N; k++) {
		zassert_within(fft_out[2 * k], 1.0f, 0.001f,
			       "Bin %d re should be 1.0", k);
		zassert_within(fft_out[2 * k + 1], 0.0f, 0.001f,
			       "Bin %d im should be 0.0", k);
	}
}

ZTEST(syn_dsp_fft_suite, test_fft_dc)
{
	/* Constant signal: all energy in bin 0 with value N */
	for (int i = 0; i < FFT_N; i++) {
		fft_in[2 * i] = 1.0f;
		fft_in[2 * i + 1] = 0.0f;
	}

	zassert_equal(syn_hal_dsp_fft_f32(fft_in, fft_out, FFT_N), 0,
		      "FFT failed");

	zassert_within(fft_out[0], (float)FFT_N, 0.001f,
		       "DC bin should be N");
	for (int k = 1; k < FFT_N; k++) {
		float mag = sqrtf(fft_out[2 * k] * fft_out[2 * k] +
				  fft_out[2 * k + 1] * fft_out[2 * k + 1]);

		zassert_within(mag, 0.0f, 0.001f,
			       "Bin %d should be empty", k);
	}
}

ZTEST(syn_dsp_fft_suite, test_fft_sine)
{
	/* Real sine at bin 3: peaks of magnitude N/2 at bins 3 and N-3 */
	const int bin = 3;

	for (int i = 0; i < FFT_N; i++) {
		fft_in[2 * i] = sinf(2.0f * 3.14159265f * bin * i / FFT_N);
		fft_in[2 * i + 1] = 0.0f;
	}

	zassert_equal(syn_hal_dsp_fft_f32(fft_in, fft_out, FFT_N), 0,
		      "FFT failed");

	for (int k = 0; k < FFT_N; k++) {
		float mag = sqrtf(fft_out[2 * k] * fft_out[2 * k] +
				  fft_out[2 * k + 1] * fft_out[2 * k + 1]);

		if (k == bin || k == FFT_N - bin) {
			zassert_within(mag, FFT_N / 2.0f, 0.01f,
				       "Peak expected at bin %d", k);
		} else {
			zassert_within(mag, 0.0f, 0.01f,
				       "Bin %d should be empty", k);
		}
	}
}

ZTEST(syn_dsp_fft_suite, test_fft_in_place)
{
	memset(fft_in, 0, sizeof(fft_in));
	fft_in[0] = 1.0f;

	zassert_equal(syn_hal_dsp_fft_f32(fft_in, fft_in, FFT_N), 0,
		      "In-place FFT failed");
	zassert_within(fft_in[2], 1.0f, 0.001f, "In-place result wrong");
}

ZTEST(syn_dsp_fft_suite, test_fft_invalid_args)
{
	zassert_equal(syn_hal_dsp_fft_f32(NULL, fft_out, FFT_N), -EINVAL,
		      "NULL in should fail");
	zassert_equal(syn_hal_dsp_fft_f32(fft_in, NULL, FFT_N), -EINVAL,
		      "NULL out should fail");
	zassert_equal(syn_hal_dsp_fft_f32(fft_in, fft_out, 12), -EINVAL,
		      "Non-power-of-2 length should fail");
	zassert_equal(syn_hal_dsp_fft_f32(fft_in, fft_out, 0), -EINVAL,
		      "Zero length should fail");
}

ZTEST(syn_dsp_fft_suite, test_matmul_identity)
{
	/* Q15 identity (0.99997 on the diagonal) times a vector */
	static const int16_t identity[9] = {
		32767, 0, 0,
		0, 32767, 0,
		0, 0, 32767,
	};
	static const int16_t vec[3] = { 1000, -2000, 3000 };
	int16_t out[3];

	zassert_equal(syn_hal_dsp_mat_mult_q15(identity, vec, out, 3, 3), 0,
		      "matmul failed");
	for (int i = 0; i < 3; i++) {
		zassert_within(out[i], vec[i], 1,
			       "Identity should preserve element %d", i);
	}
}

ZTEST(syn_dsp_fft_suite, test_matmul_known_values)
{
	/* 0.5 * 0.5 = 0.25 -> 8192 in Q15 */
	static const int16_t half[1] = { 16384 };
	static const int16_t vec[1] = { 16384 };
	int16_t out[1];

	zassert_equal(syn_hal_dsp_mat_mult_q15(half, vec, out, 1, 1), 0,
		      "matmul failed");
	zassert_equal(out[0], 8192, "0.5*0.5 should be 0.25 (8192), got %d",
		      out[0]);

	/* Row sum: [0.25 0.25] . [0.5 0.5] = 0.25 -> 8192 */
	static const int16_t row[2] = { 8192, 8192 };
	static const int16_t v2[2] = { 16384, 16384 };

	zassert_equal(syn_hal_dsp_mat_mult_q15(row, v2, out, 1, 2), 0,
		      "matmul failed");
	zassert_equal(out[0], 8192, "Row dot product wrong: %d", out[0]);
}

ZTEST(syn_dsp_fft_suite, test_matmul_saturation)
{
	/* -1.0 * -1.0 = +1.0 saturates to Q15 max (32767) */
	static const int16_t neg[1] = { -32768 };
	int16_t out[1];

	zassert_equal(syn_hal_dsp_mat_mult_q15(neg, neg, out, 1, 1), 0,
		      "matmul failed");
	zassert_equal(out[0], 32767, "Should saturate to 32767, got %d",
		      out[0]);
}

ZTEST(syn_dsp_fft_suite, test_matmul_invalid_args)
{
	static const int16_t a[1] = { 0 };
	int16_t out[1];

	zassert_equal(syn_hal_dsp_mat_mult_q15(NULL, a, out, 1, 1), -EINVAL,
		      "NULL a should fail");
	zassert_equal(syn_hal_dsp_mat_mult_q15(a, NULL, out, 1, 1), -EINVAL,
		      "NULL b should fail");
	zassert_equal(syn_hal_dsp_mat_mult_q15(a, a, NULL, 1, 1), -EINVAL,
		      "NULL out should fail");
	zassert_equal(syn_hal_dsp_mat_mult_q15(a, a, out, 0, 1), -EINVAL,
		      "Zero rows should fail");
}
