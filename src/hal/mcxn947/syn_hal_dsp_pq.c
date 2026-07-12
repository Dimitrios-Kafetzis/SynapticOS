/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file syn_hal_dsp_pq.c
 * @brief SynapticOS — PowerQuad DSP Driver (MCXN947)
 *
 * FFT and Q15 matrix-vector multiply route to the PowerQuad
 * coprocessor; normalize/softmax/argmax remain software (they are
 * element-wise or data-dependent and gain little from PQ offload at
 * these sizes).
 *
 * The PowerQuad FFT engine is fixed-point internally and its output
 * gain is not architecturally guaranteed across lengths, so init runs
 * a self-calibration: impulse FFTs at two lengths determine the gain
 * model (constant vs 1/N), and known-answer matrix probes (including
 * a saturation case) validate the Q15 conversion path. If any probe
 * fails, the corresponding operation transparently falls back to the
 * shared software kernels and the failure is logged.
 */

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <synaptic/syn_hal_dsp.h>
#include <string.h>
#include <math.h>

#include "../common/syn_dsp_soft.h"

#ifdef CONFIG_SYNAPTIC_POWERQUAD
#include "fsl_powerquad.h"
#endif

LOG_MODULE_REGISTER(syn_hal_dsp_pq, CONFIG_SYNAPTIC_LOG_LEVEL);

#ifndef CLAMP
#define CLAMP(x, lo, hi) ((x) < (lo) ? (lo) : ((x) > (hi) ? (hi) : (x)))
#endif

#ifdef CONFIG_SYNAPTIC_POWERQUAD

#define PQ_FFT_MAX_LEN    512
#define PQ_FFT_MIN_LEN    8
#define PQ_MAT_MAX_DIM    16

/* Fixed-point staging buffers for the FFT engine (interleaved re/im) */
static int32_t pq_fft_buf_in[PQ_FFT_MAX_LEN * 2] __aligned(4);
static int32_t pq_fft_buf_out[PQ_FFT_MAX_LEN * 2] __aligned(4);

static K_MUTEX_DEFINE(pq_lock);

static bool pq_fft_ok;
static bool pq_mat_ok;
static bool pq_fft_gain_is_1_over_n;
static float pq_fft_gain;      /* Measured raw gain at calibration N  */
static uint32_t pq_fft_cal_n;  /* Calibration length                  */

static void pq_config_fixed32(void)
{
	pq_config_t cfg;

	PQ_GetDefaultConfig(&cfg);
	cfg.inputAFormat = kPQ_32Bit;
	cfg.inputBFormat = kPQ_32Bit;
	cfg.outputFormat = kPQ_32Bit;
	cfg.tmpFormat = kPQ_32Bit;
	cfg.machineFormat = kPQ_32Bit;
	PQ_SetConfig(POWERQUAD, &cfg);
}

static void pq_config_q15(void)
{
	pq_config_t cfg;

	PQ_GetDefaultConfig(&cfg);
	cfg.inputAFormat = kPQ_16Bit;
	cfg.inputBFormat = kPQ_16Bit;
	cfg.outputFormat = kPQ_16Bit;
	cfg.outputPrescale = -15;    /* out = (A x b) >> 15 */
	cfg.tmpFormat = kPQ_Float;
	cfg.machineFormat = kPQ_Float;
	PQ_SetConfig(POWERQUAD, &cfg);
}

/** Raw fixed-point CFFT of an impulse of amplitude `amp` at length
 *  `n`; returns the DC-bin output (the hardware gain times amp).
 */
static int32_t pq_fft_impulse_probe(uint32_t n, int32_t amp)
{
	memset(pq_fft_buf_in, 0, n * 2 * sizeof(int32_t));
	pq_fft_buf_in[0] = amp;

	pq_config_fixed32();
	PQ_TransformCFFT(POWERQUAD, n, pq_fft_buf_in, pq_fft_buf_out);
	PQ_WaitDone(POWERQUAD);

	return pq_fft_buf_out[0];
}

static void pq_calibrate_fft(void)
{
	const int32_t amp = 1 << 13;
	const uint32_t n1 = 16;
	const uint32_t n2 = 64;

	int32_t r1 = pq_fft_impulse_probe(n1, amp);
	int32_t r2 = pq_fft_impulse_probe(n2, amp);

	if (r1 == 0 || r2 == 0) {
		LOG_WRN("PQ FFT probe returned zero (r1=%d r2=%d), "
			"using software FFT", r1, r2);
		return;
	}

	float g1 = (float)r1 / (float)amp;
	float g2 = (float)r2 / (float)amp;
	float ratio = g1 / g2;

	if (fabsf(ratio - 1.0f) < 0.05f) {
		pq_fft_gain_is_1_over_n = false;
	} else if (fabsf(ratio - (float)n2 / (float)n1) < 0.2f *
		   ((float)n2 / (float)n1)) {
		pq_fft_gain_is_1_over_n = true;
	} else {
		LOG_WRN("PQ FFT gain model unclear (g1=%d/16384 g2=%d/16384),"
			" using software FFT", (int)(g1 * 16384.0f),
			(int)(g2 * 16384.0f));
		return;
	}

	pq_fft_gain = g1;
	pq_fft_cal_n = n1;
	pq_fft_ok = true;

	LOG_INF("PQ FFT calibrated: gain %d/16384 at N=%u, model %s",
		(int)(g1 * 16384.0f), n1,
		pq_fft_gain_is_1_over_n ? "1/N" : "constant");
}

/** Effective hardware gain for a given FFT length */
static float pq_fft_gain_at(uint32_t n)
{
	if (pq_fft_gain_is_1_over_n) {
		return pq_fft_gain * (float)pq_fft_cal_n / (float)n;
	}
	return pq_fft_gain;
}

static int pq_fft_f32(const float *in, float *out, size_t len)
{
	/* Power-of-2 scale so the largest input uses ~2^13: headroom
	 * for up to 512-point growth inside the 24-bit engine.
	 */
	float max_abs = 0.0f;

	for (size_t i = 0; i < len * 2; i++) {
		float a = fabsf(in[i]);

		if (a > max_abs) {
			max_abs = a;
		}
	}
	if (max_abs == 0.0f) {
		memset(out, 0, len * 2 * sizeof(float));
		return 0;
	}

	int exp;

	(void)frexpf(max_abs, &exp);         /* max_abs = m * 2^exp   */
	float scale = ldexpf(1.0f, 13 - exp); /* max*scale in [2^12,2^13) */

	k_mutex_lock(&pq_lock, K_FOREVER);

	for (size_t i = 0; i < len * 2; i++) {
		pq_fft_buf_in[i] = (int32_t)lrintf(in[i] * scale);
	}

	pq_config_fixed32();
	PQ_TransformCFFT(POWERQUAD, len, pq_fft_buf_in, pq_fft_buf_out);
	PQ_WaitDone(POWERQUAD);

	float inv = 1.0f / (scale * pq_fft_gain_at(len));

	for (size_t i = 0; i < len * 2; i++) {
		out[i] = (float)pq_fft_buf_out[i] * inv;
	}

	k_mutex_unlock(&pq_lock);
	return 0;
}

static int pq_mat_mult_q15(const int16_t *a, const int16_t *b,
			   int16_t *out, uint16_t rows, uint16_t cols)
{
	k_mutex_lock(&pq_lock, K_FOREVER);

	pq_config_q15();
	PQ_MatrixMultiplication(POWERQUAD,
				POWERQUAD_MAKE_MATRIX_LEN(rows, cols, 1),
				(void *)a, (void *)b, out);
	PQ_WaitDone(POWERQUAD);

	k_mutex_unlock(&pq_lock);
	return 0;
}

static void pq_calibrate_mat(void)
{
	/* Known-answer probes, 4-byte-aligned static data */
	static const int16_t a_kb[4] __aligned(4) = {
		16384, 8192, -16384, 4096,     /* 0.5 0.25 ; -0.5 0.125 */
	};
	static const int16_t b_kb[2] __aligned(4) = { 16384, -16384 };
	/* Expected: [0.5*0.5 + 0.25*-0.5, -0.5*0.5 + 0.125*-0.5]
	 *         = [0.125, -0.3125] = [4096, -10240]
	 */
	static const int16_t sat_a[2] __aligned(4) = { -32768, -32768 };
	static const int16_t sat_b[2] __aligned(4) = { -32768, -32768 };
	/* -1*-1 + -1*-1 = 2.0: must saturate to 32767 */
	int16_t r[2] __aligned(4) = {0};

	if (pq_mat_mult_q15(a_kb, b_kb, r, 2, 2) != 0) {
		return;
	}
	if (r[0] != 4096 || r[1] != -10240) {
		LOG_WRN("PQ matmul known-answer failed (%d, %d), "
			"using software matmul", r[0], r[1]);
		return;
	}

	if (pq_mat_mult_q15(sat_a, sat_b, r, 1, 2) != 0) {
		return;
	}
	if (r[0] != 32767) {
		LOG_WRN("PQ matmul does not saturate (got %d), "
			"using software matmul", r[0]);
		return;
	}

	/* Odd-dimension probe (16-bit operands pack two per word) */
	static const int16_t odd_a[3] __aligned(4) = { 16384, 16384, 16384 };
	static const int16_t odd_b[3] __aligned(4) = { 16384, 16384, 16384 };
	int16_t r_odd[1] __aligned(4) = {0};

	if (pq_mat_mult_q15(odd_a, odd_b, r_odd, 1, 3) != 0) {
		return;
	}
	if (r_odd[0] != 24576) {  /* 3 * 0.25 = 0.75 */
		LOG_WRN("PQ matmul odd-dim failed (%d), "
			"using software matmul", r_odd[0]);
		return;
	}

	pq_mat_ok = true;
	LOG_INF("PQ matmul calibrated: Q15 known-answer + saturation pass");
}

#endif /* CONFIG_SYNAPTIC_POWERQUAD */

int syn_hal_dsp_init(void)
{
#ifdef CONFIG_SYNAPTIC_POWERQUAD
	PQ_Init(POWERQUAD);
	pq_calibrate_fft();
	pq_calibrate_mat();
	LOG_INF("DSP PowerQuad initialized (FFT: %s, matmul: %s)",
		pq_fft_ok ? "hardware" : "software",
		pq_mat_ok ? "hardware" : "software");
#else
	LOG_INF("DSP PowerQuad disabled by config (software kernels)");
#endif
	return 0;
}

int syn_hal_dsp_normalize_int8(const uint8_t *in, int8_t *out,
			       size_t len, float scale, int32_t zero_point)
{
	if (in == NULL || out == NULL || len == 0) {
		return -EINVAL;
	}

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
#ifdef CONFIG_SYNAPTIC_POWERQUAD
	if (pq_fft_ok && len >= PQ_FFT_MIN_LEN && len <= PQ_FFT_MAX_LEN &&
	    (len & (len - 1)) == 0 && in != NULL && out != NULL) {
		return pq_fft_f32(in, out, len);
	}
#endif
	return syn_dsp_soft_fft_f32(in, out, len);
}

int syn_hal_dsp_mat_mult_q15(const int16_t *a, const int16_t *b,
			     int16_t *out, uint16_t rows, uint16_t cols)
{
#ifdef CONFIG_SYNAPTIC_POWERQUAD
	if (pq_mat_ok && a != NULL && b != NULL && out != NULL &&
	    rows > 0 && rows <= PQ_MAT_MAX_DIM &&
	    cols > 0 && cols <= PQ_MAT_MAX_DIM &&
	    ((uintptr_t)a & 3) == 0 && ((uintptr_t)b & 3) == 0 &&
	    ((uintptr_t)out & 3) == 0) {
		return pq_mat_mult_q15(a, b, out, rows, cols);
	}
#endif
	return syn_dsp_soft_mat_mult_q15(a, b, out, rows, cols);
}
