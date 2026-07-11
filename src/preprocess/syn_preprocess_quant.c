/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file syn_preprocess_quant.c
 * @brief SynapticOS — Quantization pre-processor (float32 -> int8)
 */

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <synaptic/syn_infer.h>
#include <synaptic/syn_process.h>
#include <string.h>
#include <math.h>

LOG_MODULE_REGISTER(syn_pre_quant, CONFIG_SYNAPTIC_LOG_LEVEL);

#ifndef CLAMP
#define CLAMP(x, lo, hi) ((x) < (lo) ? (lo) : ((x) > (hi) ? (hi) : (x)))
#endif

/** q = round(x / scale) + zero_point, saturated to int8 */
static int quantize_int8(const syn_tensor_t *in, syn_tensor_t *out,
			 const void *config)
{
	const syn_quantize_config_t *cfg = config;

	if (in == NULL || in->data == NULL || out == NULL ||
	    out->data == NULL || cfg == NULL || cfg->scale == 0.0f ||
	    in->dtype != SYN_NPU_DTYPE_FLOAT32) {
		return -EINVAL;
	}

	size_t count = in->size / sizeof(float);

	if (out->size < count) {
		LOG_ERR("Quantize output needs %u bytes, capacity %u",
			(unsigned)count, (unsigned)out->size);
		return -ENOMEM;
	}

	const float *src = in->data;
	int8_t *dst = out->data;

	for (size_t i = 0; i < count; i++) {
		int32_t q = (int32_t)lroundf(src[i] / cfg->scale) +
			    cfg->zero_point;

		dst[i] = (int8_t)CLAMP(q, -128, 127);
	}

	out->size = count;
	out->dtype = SYN_NPU_DTYPE_INT8;
	out->ndim = in->ndim;
	memcpy(out->shape, in->shape, sizeof(out->shape));
	return 0;
}

syn_preprocess_fn_t syn_preprocess_quantize_int8 = quantize_int8;
