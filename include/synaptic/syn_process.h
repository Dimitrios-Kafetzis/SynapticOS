/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file syn_process.h
 * @brief SynapticOS — Built-in pre/post-processor configuration
 *
 * Configuration structures for the built-in pipeline stages declared
 * in syn_infer.h. Pass a pointer to the matching structure as the
 * `config` argument of syn_pipeline_add_preprocess() /
 * syn_pipeline_add_postprocess().
 *
 * Added in Phase 2; supplements the frozen Phase 1 API headers.
 */
#ifndef SYNAPTIC_SYN_PROCESS_H_
#define SYNAPTIC_SYN_PROCESS_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** syn_preprocess_image_normalize: per-channel (x - mean) / std,
 *  uint8/int8 input, float32 output. `channels` must match the last
 *  input dimension (max 4).
 */
typedef struct {
	float   mean[4];
	float   std[4];
	uint8_t channels;
} syn_normalize_config_t;

/** syn_preprocess_quantize_int8: q = round(x / scale) + zero_point,
 *  float32 input, int8 output.
 */
typedef struct {
	float   scale;
	int32_t zero_point;
} syn_quantize_config_t;

/** syn_postprocess_dequantize: x = (q - zero_point) * scale,
 *  int8 input, float32 output. Also accepted (optionally) by
 *  syn_postprocess_softmax to dequantize int8 logits first.
 */
typedef struct {
	float   scale;
	int32_t zero_point;
} syn_dequantize_config_t;

/** syn_preprocess_audio_mfcc: MFCC feature extraction.
 *  Input: [1, N] float32 mono samples. Output:
 *  [num_frames, num_coeffs] float32 where num_frames = N / frame_len
 *  (non-overlapping frames). frame_len must be a power of two.
 */
typedef struct {
	uint32_t sample_rate_hz;
	uint16_t frame_len;    /**< Samples per frame (power of two)   */
	uint8_t  num_mel;      /**< Mel filterbank size (e.g. 20)      */
	uint8_t  num_coeffs;   /**< DCT coefficients kept (e.g. 10)    */
} syn_mfcc_config_t;

/** syn_postprocess_top_k: number of top classifications to emit. */
typedef struct {
	uint8_t k;
} syn_topk_config_t;

/** Result element written by syn_postprocess_argmax (one entry)
 *  and syn_postprocess_top_k (k entries).
 */
typedef struct {
	uint32_t index;
	float    score;
} syn_classification_t;

/** Detection box layout consumed and produced by
 *  syn_postprocess_nms: input tensor holds N consecutive
 *  syn_bbox_t entries (float32 tensor), output holds the kept boxes.
 */
typedef struct {
	float    x1;
	float    y1;
	float    x2;
	float    y2;
	float    score;
	float    class_id;
} syn_bbox_t;

/** syn_postprocess_nms: greedy per-class non-maximum suppression. */
typedef struct {
	float   iou_threshold;    /**< Suppress overlap above this     */
	float   score_threshold;  /**< Drop boxes below this first     */
	uint8_t max_boxes;        /**< Cap on emitted boxes            */
} syn_nms_config_t;

#ifdef __cplusplus
}
#endif
#endif /* SYNAPTIC_SYN_PROCESS_H_ */
