/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file syn_postprocess_detect.c
 * @brief SynapticOS — Detection post-processor (NMS)
 *
 * Greedy per-class non-maximum suppression over syn_bbox_t records.
 */

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <synaptic/syn_infer.h>
#include <synaptic/syn_process.h>
#include <string.h>

LOG_MODULE_REGISTER(syn_post_detect, CONFIG_SYNAPTIC_LOG_LEVEL);

#define NMS_MAX_BOXES 64

static float box_iou(const syn_bbox_t *a, const syn_bbox_t *b)
{
	float ix1 = (a->x1 > b->x1) ? a->x1 : b->x1;
	float iy1 = (a->y1 > b->y1) ? a->y1 : b->y1;
	float ix2 = (a->x2 < b->x2) ? a->x2 : b->x2;
	float iy2 = (a->y2 < b->y2) ? a->y2 : b->y2;

	float iw = ix2 - ix1;
	float ih = iy2 - iy1;

	if (iw <= 0.0f || ih <= 0.0f) {
		return 0.0f;
	}

	float inter = iw * ih;
	float area_a = (a->x2 - a->x1) * (a->y2 - a->y1);
	float area_b = (b->x2 - b->x1) * (b->y2 - b->y1);
	float uni = area_a + area_b - inter;

	return (uni > 0.0f) ? inter / uni : 0.0f;
}

static int post_nms(const syn_tensor_t *in, syn_tensor_t *out,
		    const void *config)
{
	const syn_nms_config_t *cfg = config;

	if (in == NULL || in->data == NULL || out == NULL ||
	    out->data == NULL || cfg == NULL || cfg->max_boxes == 0 ||
	    in->dtype != SYN_NPU_DTYPE_FLOAT32 ||
	    in->size % sizeof(syn_bbox_t) != 0) {
		return -EINVAL;
	}

	size_t count = in->size / sizeof(syn_bbox_t);

	if (count > NMS_MAX_BOXES) {
		LOG_ERR("NMS supports up to %d boxes, got %u",
			NMS_MAX_BOXES, (unsigned)count);
		return -EINVAL;
	}

	const syn_bbox_t *boxes = in->data;
	syn_bbox_t *kept = out->data;

	/* Candidates above the score threshold, sorted by descending
	 * score via index array (insertion sort, N <= 64).
	 */
	uint8_t order[NMS_MAX_BOXES];
	uint8_t num_cand = 0;

	for (size_t i = 0; i < count; i++) {
		if (boxes[i].score < cfg->score_threshold) {
			continue;
		}

		uint8_t pos = num_cand;

		while (pos > 0 &&
		       boxes[order[pos - 1]].score < boxes[i].score) {
			order[pos] = order[pos - 1];
			pos--;
		}
		order[pos] = (uint8_t)i;
		num_cand++;
	}

	bool suppressed[NMS_MAX_BOXES] = {false};
	uint8_t num_kept = 0;

	for (uint8_t i = 0; i < num_cand && num_kept < cfg->max_boxes;
	     i++) {
		if (suppressed[i]) {
			continue;
		}

		const syn_bbox_t *cand = &boxes[order[i]];

		if ((num_kept + 1) * sizeof(syn_bbox_t) > out->size) {
			LOG_WRN("NMS output capacity reached at %u boxes",
				num_kept);
			break;
		}
		kept[num_kept++] = *cand;

		/* Suppress lower-scored same-class boxes overlapping it */
		for (uint8_t j = i + 1; j < num_cand; j++) {
			if (suppressed[j]) {
				continue;
			}

			const syn_bbox_t *other = &boxes[order[j]];

			if (other->class_id == cand->class_id &&
			    box_iou(cand, other) > cfg->iou_threshold) {
				suppressed[j] = true;
			}
		}
	}

	out->size = num_kept * sizeof(syn_bbox_t);
	out->dtype = SYN_NPU_DTYPE_FLOAT32;
	out->ndim = 2;
	memset(out->shape, 0, sizeof(out->shape));
	out->shape[0] = num_kept;
	out->shape[1] = sizeof(syn_bbox_t) / sizeof(float);
	return 0;
}

syn_postprocess_fn_t syn_postprocess_nms = post_nms;
