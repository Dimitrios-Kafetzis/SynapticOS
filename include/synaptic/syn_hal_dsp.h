/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file syn_hal_dsp.h
 * @brief SynapticOS — DSP Hardware Abstraction Layer (PowerQuad)
 */
#ifndef SYNAPTIC_SYN_HAL_DSP_H_
#define SYNAPTIC_SYN_HAL_DSP_H_

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

int syn_hal_dsp_init(void);
int syn_hal_dsp_normalize_int8(const uint8_t *in, int8_t *out,
                               size_t len, float scale, int32_t zero_point);
int syn_hal_dsp_softmax_f32(const float *in, float *out, size_t len);
int syn_hal_dsp_argmax(const int8_t *data, size_t len, uint32_t *index);
int syn_hal_dsp_fft_f32(const float *in, float *out, size_t len);
int syn_hal_dsp_mat_mult_q15(const int16_t *a, const int16_t *b,
                             int16_t *out, uint16_t rows, uint16_t cols);

#ifdef __cplusplus
}
#endif
#endif /* SYNAPTIC_SYN_HAL_DSP_H_ */
