/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file syn_hal_dsp.h
 * @brief SynapticOS — DSP Hardware Abstraction Layer (PowerQuad)
 *
 * Fixed-function DSP kernels used by the built-in pre/post-processors.
 * On FRDM-MCXN947 the FFT and matrix-multiply paths are accelerated by
 * the PowerQuad coprocessor when eligible (size/alignment permitting);
 * all other cases fall back to the portable software kernels.
 */
#ifndef SYNAPTIC_SYN_HAL_DSP_H_
#define SYNAPTIC_SYN_HAL_DSP_H_

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initialize the DSP HAL backend.
 *
 * @return 0 on success, negative errno on failure.
 */
int syn_hal_dsp_init(void);

/**
 * @brief Scale-and-offset quantization of a uint8 buffer to int8.
 *
 * Computes q = (int32)(x * scale) + zero_point per element, saturated
 * to [-128, 127].
 *
 * @param in         Input samples (uint8), @p len elements.
 * @param out        Output buffer (int8), @p len elements.
 * @param len        Number of elements (must be > 0).
 * @param scale      Multiplicative scale applied to each input.
 * @param zero_point Offset added after scaling.
 *
 * @return 0 on success, -EINVAL on bad arguments.
 */
int syn_hal_dsp_normalize_int8(const uint8_t *in, int8_t *out,
                               size_t len, float scale, int32_t zero_point);

/**
 * @brief Numerically-stable softmax over a float32 vector.
 *
 * @param in  Input logits, @p len elements.
 * @param out Output probabilities (sum to 1), @p len elements.
 * @param len Number of elements (must be > 0).
 *
 * @return 0 on success, -EINVAL on bad arguments.
 */
int syn_hal_dsp_softmax_f32(const float *in, float *out, size_t len);

/**
 * @brief Index of the maximum element in an int8 vector.
 *
 * @param data  Input vector, @p len elements.
 * @param len   Number of elements (must be > 0).
 * @param index Out: index of the maximum element (first on ties).
 *
 * @return 0 on success, -EINVAL on bad arguments.
 */
int syn_hal_dsp_argmax(const int8_t *data, size_t len, uint32_t *index);

/**
 * @brief Complex FFT, float32, interleaved re/im
 *        (PowerQuad-accelerated on MCXN947 when eligible).
 *
 * Both buffers hold @p len complex points as 2 * @p len interleaved
 * floats {re0, im0, re1, im1, ...}. In-place operation (out == in) is
 * supported.
 *
 * @param in  Input, @p len complex points (2 * @p len floats).
 * @param out Output transform, same layout and length as @p in.
 * @param len Number of complex points; must be a power of two
 *            within the backend's supported range.
 *
 * @return 0 on success, -EINVAL on bad arguments or unsupported
 *         length.
 */
int syn_hal_dsp_fft_f32(const float *in, float *out, size_t len);

/**
 * @brief Saturating Q15 matrix-vector multiply: out = a x b
 *        (PowerQuad-accelerated on MCXN947 when eligible).
 *
 * @param a    Matrix, @p rows x @p cols, row-major Q15.
 * @param b    Vector, @p cols elements, Q15.
 * @param out  Output vector, @p rows elements, Q15 (saturated).
 * @param rows Number of matrix rows (must be > 0).
 * @param cols Number of matrix columns / vector elements (must be > 0).
 *
 * @return 0 on success, -EINVAL on bad arguments.
 */
int syn_hal_dsp_mat_mult_q15(const int16_t *a, const int16_t *b,
                             int16_t *out, uint16_t rows, uint16_t cols);

#ifdef __cplusplus
}
#endif
#endif /* SYNAPTIC_SYN_HAL_DSP_H_ */
