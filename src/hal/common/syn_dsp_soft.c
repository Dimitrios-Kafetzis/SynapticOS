/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file syn_dsp_soft.c
 * @brief SynapticOS — Shared software DSP kernels
 *
 * Iterative radix-2 Cooley-Tukey FFT and saturating Q15
 * matrix-vector multiply. These serve as the QEMU stub
 * implementation and as the reference/fallback for the
 * PowerQuad-accelerated driver.
 */

#include <zephyr/kernel.h>
#include <math.h>
#include <string.h>

#include "syn_dsp_soft.h"

#define FFT_MAX_LEN 1024

static bool is_pow2(size_t x)
{
	return x != 0 && (x & (x - 1)) == 0;
}

int syn_dsp_soft_fft_f32(const float *in, float *out, size_t len)
{
	if (in == NULL || out == NULL || len < 2 || len > FFT_MAX_LEN ||
	    !is_pow2(len)) {
		return -EINVAL;
	}

	if (out != in) {
		memcpy(out, in, len * 2 * sizeof(float));
	}

	/* In-place bit-reversal permutation */
	for (size_t i = 0, j = 0; i < len; i++) {
		if (i < j) {
			float tr = out[2 * i];
			float ti = out[2 * i + 1];

			out[2 * i] = out[2 * j];
			out[2 * i + 1] = out[2 * j + 1];
			out[2 * j] = tr;
			out[2 * j + 1] = ti;
		}
		size_t bit = len >> 1;

		while (j & bit) {
			j ^= bit;
			bit >>= 1;
		}
		j |= bit;
	}

	/* Iterative butterflies */
	for (size_t stage = 2; stage <= len; stage <<= 1) {
		float angle = -2.0f * 3.14159265358979f / (float)stage;
		float wr_step = cosf(angle);
		float wi_step = sinf(angle);

		for (size_t group = 0; group < len; group += stage) {
			float wr = 1.0f;
			float wi = 0.0f;

			for (size_t pair = 0; pair < stage / 2; pair++) {
				size_t top = group + pair;
				size_t bot = top + stage / 2;

				float br = out[2 * bot] * wr -
					   out[2 * bot + 1] * wi;
				float bi = out[2 * bot] * wi +
					   out[2 * bot + 1] * wr;

				out[2 * bot] = out[2 * top] - br;
				out[2 * bot + 1] = out[2 * top + 1] - bi;
				out[2 * top] += br;
				out[2 * top + 1] += bi;

				float wr_new = wr * wr_step - wi * wi_step;

				wi = wr * wi_step + wi * wr_step;
				wr = wr_new;
			}
		}
	}

	return 0;
}

int syn_dsp_soft_mat_mult_q15(const int16_t *a, const int16_t *b,
			      int16_t *out, uint16_t rows, uint16_t cols)
{
	if (a == NULL || b == NULL || out == NULL || rows == 0 || cols == 0) {
		return -EINVAL;
	}

	for (uint16_t r = 0; r < rows; r++) {
		int64_t acc = 0;

		for (uint16_t c = 0; c < cols; c++) {
			acc += (int32_t)a[(size_t)r * cols + c] *
			       (int32_t)b[c];
		}

		acc >>= 15;
		if (acc > INT16_MAX) {
			acc = INT16_MAX;
		} else if (acc < INT16_MIN) {
			acc = INT16_MIN;
		}
		out[r] = (int16_t)acc;
	}

	return 0;
}
