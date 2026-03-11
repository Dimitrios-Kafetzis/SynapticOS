/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file syn_hal_dsp_pq.c
 * @brief SynapticOS — PowerQuad DSP Driver (MCXN947)
 *
 * DSP operations using the MCXN947 PowerQuad hardware accelerator.
 *
 * Current implementation: Software fallback (identical to stub).
 * PowerQuad-accelerated versions will be added in Phase 2.
 */

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <synaptic/syn_hal_dsp.h>
#include <string.h>
#include <math.h>

LOG_MODULE_REGISTER(syn_hal_dsp_pq, CONFIG_SYNAPTIC_LOG_LEVEL);

#ifndef CLAMP
#define CLAMP(x, lo, hi) ((x) < (lo) ? (lo) : ((x) > (hi) ? (hi) : (x)))
#endif

int syn_hal_dsp_init(void)
{
	/* TODO: Initialize PowerQuad hardware:
	 *   PQ_Init(POWERQUAD);
	 */
	LOG_INF("DSP PowerQuad initialized (software fallback — PQ not yet integrated)");
	return 0;
}

int syn_hal_dsp_normalize_int8(const uint8_t *in, int8_t *out,
			       size_t len, float scale, int32_t zero_point)
{
	if (in == NULL || out == NULL || len == 0) {
		return -EINVAL;
	}

	/* TODO: Use PowerQuad for batch fixed-point multiply */
	for (size_t i = 0; i < len; i++) {
		int32_t val = (int32_t)((float)in[i] * scale) + zero_point;

		out[i] = (int8_t)CLAMP(val, -128, 127);
	}

	return 0;
}

int syn_hal_dsp_softmax_f32(const float *in, float *out, size_t len)
{
	if (in == NULL || out == NULL || len == 0) {
		return -EINVAL;
	}

	/* TODO: Use PowerQuad for exp and division */
	float max_val = in[0];

	for (size_t i = 1; i < len; i++) {
		if (in[i] > max_val) {
			max_val = in[i];
		}
	}

	float sum = 0.0f;

	for (size_t i = 0; i < len; i++) {
		out[i] = expf(in[i] - max_val);
		sum += out[i];
	}

	if (sum > 0.0f) {
		for (size_t i = 0; i < len; i++) {
			out[i] /= sum;
		}
	}

	return 0;
}

int syn_hal_dsp_argmax(const int8_t *data, size_t len, uint32_t *index)
{
	if (data == NULL || index == NULL || len == 0) {
		return -EINVAL;
	}

	int8_t max_val = data[0];
	uint32_t max_idx = 0;

	for (size_t i = 1; i < len; i++) {
		if (data[i] > max_val) {
			max_val = data[i];
			max_idx = (uint32_t)i;
		}
	}

	*index = max_idx;
	return 0;
}

int syn_hal_dsp_fft_f32(const float *in, float *out, size_t len)
{
	ARG_UNUSED(in);
	ARG_UNUSED(out);
	ARG_UNUSED(len);

	/* TODO: Implement using PowerQuad FFT engine */
	LOG_WRN("FFT not implemented in PowerQuad driver");
	return -ENOTSUP;
}

int syn_hal_dsp_mat_mult_q15(const int16_t *a, const int16_t *b,
			     int16_t *out, uint16_t rows, uint16_t cols)
{
	ARG_UNUSED(a);
	ARG_UNUSED(b);
	ARG_UNUSED(out);
	ARG_UNUSED(rows);
	ARG_UNUSED(cols);

	/* TODO: Implement using PowerQuad matrix engine */
	LOG_WRN("Matrix multiply not implemented in PowerQuad driver");
	return -ENOTSUP;
}
