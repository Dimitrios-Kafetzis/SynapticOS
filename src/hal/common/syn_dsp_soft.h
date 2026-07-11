/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file syn_dsp_soft.h
 * @brief SynapticOS — Shared software DSP kernels (internal)
 *
 * Reference implementations used by the stub DSP HAL and as the
 * fallback path of the PowerQuad driver. Not part of the public API.
 */
#ifndef SYNAPTIC_SYN_DSP_SOFT_H_
#define SYNAPTIC_SYN_DSP_SOFT_H_

#include <stdint.h>
#include <stddef.h>

/**
 * In-order radix-2 complex FFT.
 *
 * @param in   Interleaved complex input: 2*len floats (re0, im0, re1, ...).
 * @param out  Interleaved complex output, 2*len floats. May equal @p in.
 * @param len  Number of complex points. Must be a power of two, 2..1024.
 * @return 0 on success, -EINVAL on bad arguments.
 */
int syn_dsp_soft_fft_f32(const float *in, float *out, size_t len);

/**
 * Q15 matrix-vector multiply: out[r] = sat(sum_c a[r*cols+c] * b[c] >> 15).
 *
 * @param a     Matrix, rows x cols, row-major Q15.
 * @param b     Vector, cols entries, Q15.
 * @param out   Result vector, rows entries, Q15 (saturated).
 * @param rows  Number of matrix rows (> 0).
 * @param cols  Number of matrix columns (> 0).
 * @return 0 on success, -EINVAL on bad arguments.
 */
int syn_dsp_soft_mat_mult_q15(const int16_t *a, const int16_t *b,
			      int16_t *out, uint16_t rows, uint16_t cols);

#endif /* SYNAPTIC_SYN_DSP_SOFT_H_ */
