/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file main.c
 * @brief SynapticOS — face_detection sample (Phase 2)
 *
 * Continuous detection demo exercising the full Phase 2 stack:
 *
 *   synthetic frame -> resize -> normalize -> quantize -> model
 *                   -> decode -> NMS -> report
 *
 * A deterministic synthetic frame source stands in for the OV7670
 * camera, and detections are reported over the console instead of an
 * LCD overlay. Camera (DVP) and LCD (LCD-PAR-S035) integration is the
 * hardware bring-up step tracked for the FRDM-MCXN947 board; the
 * pipeline, scheduler, profiling, and post-processing below are the
 * final production path.
 */

#include <zephyr/kernel.h>
#include <synaptic/syn_api.h>
#include <synaptic/syn_process.h>
#include <string.h>

#include <zephyr/logging/log.h>
LOG_MODULE_REGISTER(face_detection, LOG_LEVEL_INF);

/* Synthetic "camera" frame and model geometry */
#define FRAME_W   24
#define FRAME_H   24
#define FRAME_C   3
#define MODEL_W   12
#define MODEL_H   12
#define NUM_CELLS 10   /* Stub detector emits one score per grid cell */

#define NUM_FRAMES 30

static const uint8_t dummy_model[64] = {0};
static uint8_t frame[FRAME_W * FRAME_H * FRAME_C];

/* Grid cells: 5 columns x 2 rows over the frame */
#define GRID_COLS 5
#define GRID_ROWS 2
#define CELL_W    (FRAME_W / GRID_COLS)
#define CELL_H    (FRAME_H / GRID_ROWS)

/** Deterministic frame: dark gradient plus a bright 8x8 blob that
 *  moves one grid cell per frame.
 */
static void generate_frame(uint32_t n)
{
	for (uint32_t y = 0; y < FRAME_H; y++) {
		for (uint32_t x = 0; x < FRAME_W; x++) {
			for (uint32_t c = 0; c < FRAME_C; c++) {
				frame[(y * FRAME_W + x) * FRAME_C + c] =
					(uint8_t)((x + y + c) & 0x3F);
			}
		}
	}

	uint32_t cell = n % NUM_CELLS;
	uint32_t bx = (cell % GRID_COLS) * CELL_W;
	uint32_t by = (cell / GRID_COLS) * CELL_H;

	for (uint32_t y = by; y < by + 8 && y < FRAME_H; y++) {
		for (uint32_t x = bx; x < bx + 8 && x < FRAME_W; x++) {
			for (uint32_t c = 0; c < FRAME_C; c++) {
				frame[(y * FRAME_W + x) * FRAME_C + c] = 100;
			}
		}
	}
}

/** Decode raw per-cell scores into candidate boxes in frame
 *  coordinates (the stand-in for anchor decoding of a real detector).
 */
static uint8_t decode_cells(const int8_t *scores, size_t count,
			    syn_bbox_t *boxes, uint8_t max)
{
	uint8_t n = 0;

	for (size_t i = 0; i < count && i < NUM_CELLS && n < max; i++) {
		if (scores[i] <= 0) {
			continue;
		}

		boxes[n].x1 = (float)((i % GRID_COLS) * CELL_W);
		boxes[n].y1 = (float)((i / GRID_COLS) * CELL_H);
		boxes[n].x2 = boxes[n].x1 + CELL_W;
		boxes[n].y2 = boxes[n].y1 + CELL_H;
		boxes[n].score = (float)scores[i] / 127.0f;
		boxes[n].class_id = 0.0f;
		n++;
	}
	return n;
}

int main(void)
{
	int ret;

	LOG_INF("=== SynapticOS %s: face_detection ===", syn_version());

	ret = syn_init();
	if (ret != 0) {
		LOG_ERR("syn_init() failed: %d", ret);
		return ret;
	}

	/* Register and load the detector model */
	syn_model_info_t model_info = {0};

	strncpy(model_info.name, "face_detect", sizeof(model_info.name));
	strncpy(model_info.version, "1.0.0", sizeof(model_info.version));
	model_info.input_size = MODEL_W * MODEL_H * FRAME_C;
	model_info.output_size = NUM_CELLS;
	model_info.flash_size = sizeof(dummy_model);
	model_info.sram_required = 4096;
	model_info.input_dtype = SYN_NPU_DTYPE_INT8;
	model_info.output_dtype = SYN_NPU_DTYPE_INT8;

	syn_model_handle_t model;

	ret = syn_model_register(&model_info, &model);
	if (ret != 0) {
		LOG_ERR("Model register failed: %d", ret);
		return ret;
	}

	ret = syn_hal_npu_load_model(dummy_model, sizeof(dummy_model));
	if (ret != 0) {
		LOG_ERR("NPU load failed: %d", ret);
		return ret;
	}
	syn_model_load(model);

	/* Build the vision pipeline once; it is reused every frame */
	static syn_resize_config_t resize_cfg = {
		.w = MODEL_W, .h = MODEL_H,
	};
	static syn_normalize_config_t norm_cfg = {
		.mean = { 127.5f, 127.5f, 127.5f },
		.std = { 127.5f, 127.5f, 127.5f },
		.channels = FRAME_C,
	};
	static syn_quantize_config_t quant_cfg = {
		.scale = 1.0f / 127.0f, .zero_point = 0,
	};

	syn_pipeline_t *pipe = syn_pipeline_create("face_detect");

	if (pipe == NULL) {
		return -ENOMEM;
	}

	ret = syn_pipeline_add_preprocess(pipe, syn_preprocess_image_resize,
					  &resize_cfg);
	ret |= syn_pipeline_add_preprocess(pipe,
					   syn_preprocess_image_normalize,
					   &norm_cfg);
	ret |= syn_pipeline_add_preprocess(pipe,
					   syn_preprocess_quantize_int8,
					   &quant_cfg);
	ret |= syn_pipeline_add_model(pipe, model);
	if (ret != 0) {
		LOG_ERR("Pipeline construction failed");
		return -EINVAL;
	}

	ret = syn_pipeline_build(pipe);
	if (ret != 0) {
		LOG_ERR("Pipeline build failed: %d", ret);
		return ret;
	}

	/* Static input descriptor over the frame buffer */
	syn_tensor_t input = {
		.data = frame,
		.size = sizeof(frame),
		.dtype = SYN_NPU_DTYPE_UINT8,
		.ndim = 4,
		.shape = { 1, FRAME_H, FRAME_W, FRAME_C },
		.lifetime = SYN_MEM_SHARED,
	};

	syn_nms_config_t nms_cfg = {
		.iou_threshold = 0.5f,
		.score_threshold = 0.3f,
		.max_boxes = 4,
	};

	uint32_t total_us = 0;
	uint32_t detections = 0;

	LOG_INF("Running %u frames through the detection pipeline",
		NUM_FRAMES);

	for (uint32_t n = 0; n < NUM_FRAMES; n++) {
		generate_frame(n);

		uint32_t start = k_cycle_get_32();

		syn_infer_params_t params = {
			.priority = SYN_PRIORITY_REALTIME,
		};
		syn_job_id_t job = syn_infer_submit(pipe, &input, &params);

		if (job == SYN_JOB_INVALID) {
			LOG_ERR("Frame %u: submit failed", n);
			break;
		}

		ret = syn_infer_wait(job, 1000);
		if (ret != 0) {
			LOG_ERR("Frame %u: wait failed: %d", n, ret);
			break;
		}

		syn_tensor_t scores;

		ret = syn_infer_get_result(job, &scores);
		if (ret != 0) {
			LOG_ERR("Frame %u: get_result failed: %d", n, ret);
			break;
		}

		/* Decode grid scores to boxes, then suppress overlaps */
		syn_bbox_t candidates[NUM_CELLS];
		syn_bbox_t kept[4];
		uint8_t num_cand = decode_cells(scores.data, scores.size,
						candidates,
						ARRAY_SIZE(candidates));

		syn_tensor_t cand_tensor = {
			.data = candidates,
			.size = (size_t)num_cand * sizeof(syn_bbox_t),
			.dtype = SYN_NPU_DTYPE_FLOAT32,
			.ndim = 2,
			.shape = { num_cand, 6 },
		};
		syn_tensor_t kept_tensor = {
			.data = kept,
			.size = sizeof(kept),
		};

		uint8_t num_kept = 0;

		if (num_cand > 0 &&
		    syn_postprocess_nms(&cand_tensor, &kept_tensor,
					&nms_cfg) == 0) {
			num_kept = (uint8_t)kept_tensor.shape[0];
		}

		uint32_t frame_us = k_cyc_to_us_ceil32(k_cycle_get_32() -
						       start);

		total_us += frame_us;
		detections += num_kept;

		if (num_kept > 0) {
			LOG_INF("Frame %02u: %u face(s), best [%d,%d %dx%d] "
				"score %d%%, %u us",
				n, num_kept,
				(int)kept[0].x1, (int)kept[0].y1,
				(int)(kept[0].x2 - kept[0].x1),
				(int)(kept[0].y2 - kept[0].y1),
				(int)(kept[0].score * 100.0f), frame_us);
		} else {
			LOG_INF("Frame %02u: no detections, %u us",
				n, frame_us);
		}

		/* Zero-fragmentation policy: drop all per-frame tensors */
		syn_mem_reset_ephemeral();
	}

	uint32_t avg_us = total_us / NUM_FRAMES;
	uint32_t fps_x10 = (avg_us > 0) ? 10000000U / avg_us : 0;

	LOG_INF("=== face_detection summary ===");
	LOG_INF("  Frames:     %u", NUM_FRAMES);
	LOG_INF("  Detections: %u", detections);
	LOG_INF("  Avg frame:  %u us (%u.%u FPS)",
		avg_us, fps_x10 / 10, fps_x10 % 10);
	syn_prof_print_summary();
	syn_mem_print_stats();
	LOG_INF("=== face_detection complete ===");

	while (1) {
		k_sleep(K_SECONDS(1));
	}

	return 0;
}
