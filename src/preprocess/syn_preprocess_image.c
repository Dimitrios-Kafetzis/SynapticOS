/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file syn_preprocess_image.c
 * @brief SynapticOS — Image pre-processors (resize, normalize)
 *
 * Stage functions follow the pipeline buffer convention: the output
 * tensor arrives with its capacity in out->size and the stage sets
 * the final geometry before returning.
 */

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <synaptic/syn_infer.h>
#include <synaptic/syn_process.h>
#include <string.h>

LOG_MODULE_REGISTER(syn_pre_image, CONFIG_SYNAPTIC_LOG_LEVEL);

static bool is_byte_image(const syn_tensor_t *t)
{
	return t != NULL && t->data != NULL && t->ndim == 4 &&
	       t->shape[0] == 1 &&
	       (t->dtype == SYN_NPU_DTYPE_UINT8 ||
		t->dtype == SYN_NPU_DTYPE_INT8);
}

/**
 * Bilinear resize of a [1, H, W, C] byte image to the dimensions in
 * syn_resize_config_t. Output keeps the input dtype.
 */
static int image_resize(const syn_tensor_t *in, syn_tensor_t *out,
			const void *config)
{
	const syn_resize_config_t *cfg = config;

	if (!is_byte_image(in) || out == NULL || out->data == NULL ||
	    cfg == NULL || cfg->w == 0 || cfg->h == 0) {
		return -EINVAL;
	}

	uint32_t src_h = in->shape[1];
	uint32_t src_w = in->shape[2];
	uint32_t ch = in->shape[3];
	uint32_t dst_h = cfg->h;
	uint32_t dst_w = cfg->w;
	size_t needed = (size_t)dst_h * dst_w * ch;

	if (src_h == 0 || src_w == 0 || ch == 0 || ch > 4) {
		return -EINVAL;
	}
	if (out->size < needed) {
		LOG_ERR("Resize output needs %u bytes, capacity %u",
			(unsigned)needed, (unsigned)out->size);
		return -ENOMEM;
	}

	const uint8_t *src = in->data;
	uint8_t *dst = out->data;

	/* Edge-aligned bilinear: map pixel centers, clamp at borders */
	float x_ratio = (dst_w > 1) ?
		(float)(src_w - 1) / (float)(dst_w - 1) : 0.0f;
	float y_ratio = (dst_h > 1) ?
		(float)(src_h - 1) / (float)(dst_h - 1) : 0.0f;

	for (uint32_t y = 0; y < dst_h; y++) {
		float sy = y * y_ratio;
		uint32_t y0 = (uint32_t)sy;
		uint32_t y1 = (y0 + 1 < src_h) ? y0 + 1 : y0;
		float fy = sy - (float)y0;

		for (uint32_t x = 0; x < dst_w; x++) {
			float sx = x * x_ratio;
			uint32_t x0 = (uint32_t)sx;
			uint32_t x1 = (x0 + 1 < src_w) ? x0 + 1 : x0;
			float fx = sx - (float)x0;

			for (uint32_t c = 0; c < ch; c++) {
				float p00 = src[(y0 * src_w + x0) * ch + c];
				float p01 = src[(y0 * src_w + x1) * ch + c];
				float p10 = src[(y1 * src_w + x0) * ch + c];
				float p11 = src[(y1 * src_w + x1) * ch + c];

				float top = p00 + fx * (p01 - p00);
				float bot = p10 + fx * (p11 - p10);
				float val = top + fy * (bot - top);

				dst[(y * dst_w + x) * ch + c] =
					(uint8_t)(val + 0.5f);
			}
		}
	}

	out->size = needed;
	out->dtype = in->dtype;
	out->ndim = 4;
	out->shape[0] = 1;
	out->shape[1] = dst_h;
	out->shape[2] = dst_w;
	out->shape[3] = ch;
	return 0;
}

/**
 * Per-channel (x - mean) / std normalization of a [1, H, W, C] byte
 * image to float32.
 */
static int image_normalize(const syn_tensor_t *in, syn_tensor_t *out,
			   const void *config)
{
	const syn_normalize_config_t *cfg = config;

	if (!is_byte_image(in) || out == NULL || out->data == NULL ||
	    cfg == NULL) {
		return -EINVAL;
	}

	uint32_t ch = in->shape[3];

	if (ch == 0 || ch > 4 || cfg->channels != ch) {
		return -EINVAL;
	}
	for (uint32_t c = 0; c < ch; c++) {
		if (cfg->std[c] == 0.0f) {
			return -EINVAL;
		}
	}

	size_t num_px = in->size;
	size_t needed = num_px * sizeof(float);

	if (out->size < needed) {
		LOG_ERR("Normalize output needs %u bytes, capacity %u",
			(unsigned)needed, (unsigned)out->size);
		return -ENOMEM;
	}

	const uint8_t *src = in->data;
	float *dst = out->data;

	for (size_t i = 0; i < num_px; i++) {
		uint32_t c = i % ch;
		float val = (in->dtype == SYN_NPU_DTYPE_INT8) ?
			(float)((const int8_t *)src)[i] : (float)src[i];

		dst[i] = (val - cfg->mean[c]) / cfg->std[c];
	}

	out->size = needed;
	out->dtype = SYN_NPU_DTYPE_FLOAT32;
	out->ndim = in->ndim;
	memcpy(out->shape, in->shape, sizeof(out->shape));
	return 0;
}

syn_preprocess_fn_t syn_preprocess_image_resize = image_resize;
syn_preprocess_fn_t syn_preprocess_image_normalize = image_normalize;
