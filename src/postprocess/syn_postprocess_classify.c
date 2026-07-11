/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file syn_postprocess_classify.c
 * @brief SynapticOS — Classification post-processors
 *
 * softmax, argmax, top-k and dequantize stages. Accept int8 or
 * float32 inputs; argmax/top-k emit syn_classification_t records.
 */

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <synaptic/syn_infer.h>
#include <synaptic/syn_process.h>
#include <synaptic/syn_hal_dsp.h>
#include <string.h>

LOG_MODULE_REGISTER(syn_post_classify, CONFIG_SYNAPTIC_LOG_LEVEL);

#define TOPK_MAX 16

static bool valid_pair(const syn_tensor_t *in, const syn_tensor_t *out)
{
	return in != NULL && in->data != NULL && in->size > 0 &&
	       out != NULL && out->data != NULL;
}

static size_t element_count(const syn_tensor_t *t)
{
	return (t->dtype == SYN_NPU_DTYPE_FLOAT32) ?
		t->size / sizeof(float) : t->size;
}

/** Fetch element i as float (int8 raw or dequantized via cfg) */
static float get_score(const syn_tensor_t *t, size_t i,
		       const syn_dequantize_config_t *cfg)
{
	if (t->dtype == SYN_NPU_DTYPE_FLOAT32) {
		return ((const float *)t->data)[i];
	}

	float q = (float)((const int8_t *)t->data)[i];

	if (cfg != NULL) {
		return (q - (float)cfg->zero_point) * cfg->scale;
	}
	return q;
}

/**
 * Softmax over int8 or float32 logits to float32 probabilities.
 * Optional config: syn_dequantize_config_t applied to int8 logits.
 */
static int post_softmax(const syn_tensor_t *in, syn_tensor_t *out,
			const void *config)
{
	if (!valid_pair(in, out)) {
		return -EINVAL;
	}
	if (in->dtype != SYN_NPU_DTYPE_INT8 &&
	    in->dtype != SYN_NPU_DTYPE_FLOAT32) {
		return -EINVAL;
	}

	size_t count = element_count(in);
	size_t needed = count * sizeof(float);

	if (out->size < needed) {
		return -ENOMEM;
	}

	float *dst = out->data;

	for (size_t i = 0; i < count; i++) {
		dst[i] = get_score(in, i, config);
	}

	int ret = syn_hal_dsp_softmax_f32(dst, dst, count);

	if (ret != 0) {
		return ret;
	}

	out->size = needed;
	out->dtype = SYN_NPU_DTYPE_FLOAT32;
	out->ndim = in->ndim;
	memcpy(out->shape, in->shape, sizeof(out->shape));
	return 0;
}

/** Emit the top classification as one syn_classification_t */
static int post_argmax(const syn_tensor_t *in, syn_tensor_t *out,
		       const void *config)
{
	ARG_UNUSED(config);

	if (!valid_pair(in, out)) {
		return -EINVAL;
	}
	if (in->dtype != SYN_NPU_DTYPE_INT8 &&
	    in->dtype != SYN_NPU_DTYPE_FLOAT32) {
		return -EINVAL;
	}
	if (out->size < sizeof(syn_classification_t)) {
		return -ENOMEM;
	}

	size_t count = element_count(in);
	uint32_t best = 0;

	if (in->dtype == SYN_NPU_DTYPE_INT8) {
		int ret = syn_hal_dsp_argmax(in->data, count, &best);

		if (ret != 0) {
			return ret;
		}
	} else {
		const float *v = in->data;

		for (size_t i = 1; i < count; i++) {
			if (v[i] > v[best]) {
				best = (uint32_t)i;
			}
		}
	}

	syn_classification_t *result = out->data;

	result->index = best;
	result->score = get_score(in, best, NULL);

	out->size = sizeof(syn_classification_t);
	out->dtype = SYN_NPU_DTYPE_FLOAT32;
	out->ndim = 1;
	memset(out->shape, 0, sizeof(out->shape));
	out->shape[0] = 1;
	return 0;
}

/** Emit the top-k classifications, ordered by descending score */
static int post_top_k(const syn_tensor_t *in, syn_tensor_t *out,
		      const void *config)
{
	const syn_topk_config_t *cfg = config;

	if (!valid_pair(in, out) || cfg == NULL || cfg->k == 0 ||
	    cfg->k > TOPK_MAX) {
		return -EINVAL;
	}
	if (in->dtype != SYN_NPU_DTYPE_INT8 &&
	    in->dtype != SYN_NPU_DTYPE_FLOAT32) {
		return -EINVAL;
	}

	size_t count = element_count(in);
	uint8_t k = (cfg->k < count) ? cfg->k : (uint8_t)count;

	if (out->size < k * sizeof(syn_classification_t)) {
		return -ENOMEM;
	}

	syn_classification_t *results = out->data;
	uint32_t chosen[TOPK_MAX];

	for (uint8_t round = 0; round < k; round++) {
		int32_t best = -1;
		float best_score = 0.0f;

		for (size_t i = 0; i < count; i++) {
			bool used = false;

			for (uint8_t j = 0; j < round; j++) {
				if (chosen[j] == (uint32_t)i) {
					used = true;
					break;
				}
			}
			if (used) {
				continue;
			}

			float score = get_score(in, i, NULL);

			if (best < 0 || score > best_score) {
				best = (int32_t)i;
				best_score = score;
			}
		}

		chosen[round] = (uint32_t)best;
		results[round].index = (uint32_t)best;
		results[round].score = best_score;
	}

	out->size = k * sizeof(syn_classification_t);
	out->dtype = SYN_NPU_DTYPE_FLOAT32;
	out->ndim = 1;
	memset(out->shape, 0, sizeof(out->shape));
	out->shape[0] = k;
	return 0;
}

/** Dequantize int8 to float32: x = (q - zero_point) * scale */
static int post_dequantize(const syn_tensor_t *in, syn_tensor_t *out,
			   const void *config)
{
	const syn_dequantize_config_t *cfg = config;

	if (!valid_pair(in, out) || cfg == NULL ||
	    in->dtype != SYN_NPU_DTYPE_INT8) {
		return -EINVAL;
	}

	size_t count = in->size;
	size_t needed = count * sizeof(float);

	if (out->size < needed) {
		return -ENOMEM;
	}

	const int8_t *src = in->data;
	float *dst = out->data;

	for (size_t i = 0; i < count; i++) {
		dst[i] = ((float)src[i] - (float)cfg->zero_point) *
			 cfg->scale;
	}

	out->size = needed;
	out->dtype = SYN_NPU_DTYPE_FLOAT32;
	out->ndim = in->ndim;
	memcpy(out->shape, in->shape, sizeof(out->shape));
	return 0;
}

syn_postprocess_fn_t syn_postprocess_softmax = post_softmax;
syn_postprocess_fn_t syn_postprocess_argmax = post_argmax;
syn_postprocess_fn_t syn_postprocess_top_k = post_top_k;
syn_postprocess_fn_t syn_postprocess_dequantize = post_dequantize;
