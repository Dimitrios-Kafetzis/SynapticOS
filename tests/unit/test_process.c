/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file test_process.c
 * @brief Unit tests for built-in pre/post-processors (Phase 2.4)
 */

#include <zephyr/ztest.h>
#include <synaptic/syn_infer.h>
#include <synaptic/syn_process.h>
#include <synaptic/syn_mem.h>
#include <synaptic/syn_model.h>
#include <synaptic/syn_hal_npu.h>
#include <string.h>
#include <math.h>

static uint8_t proc_arena[4096] __aligned(16);
static const uint8_t proc_model_bin[32] = {0};
static syn_model_handle_t proc_model;

/* Reusable output buffer for standalone stage tests */
static uint8_t out_buf[512] __aligned(4);

static syn_tensor_t make_out(size_t capacity)
{
	syn_tensor_t t = {
		.data = out_buf,
		.size = (capacity <= sizeof(out_buf)) ?
			capacity : sizeof(out_buf),
	};

	return t;
}

static void *proc_suite_setup(void)
{
	syn_hal_npu_deinit();
	zassert_equal(syn_hal_npu_init(), 0, "NPU init failed");
	zassert_equal(syn_hal_npu_load_model(proc_model_bin,
					     sizeof(proc_model_bin)),
		      0, "NPU model load failed");

	if (syn_model_get_by_name("proc_test", &proc_model) != 0) {
		syn_model_info_t info = {0};

		strncpy(info.name, "proc_test", sizeof(info.name));
		strncpy(info.version, "1.0.0", sizeof(info.version));
		info.input_size = 48;
		info.output_size = 10;
		info.sram_required = 256;
		info.input_dtype = SYN_NPU_DTYPE_INT8;
		info.output_dtype = SYN_NPU_DTYPE_INT8;
		zassert_equal(syn_model_register(&info, &proc_model), 0,
			      "Model registration failed");
	}
	return NULL;
}

static void proc_before(void *fixture)
{
	ARG_UNUSED(fixture);
	syn_mem_init(proc_arena, sizeof(proc_arena));
}

ZTEST_SUITE(syn_process_suite, NULL, proc_suite_setup, proc_before,
	    NULL, NULL);

/* ---------------- image resize ---------------- */

ZTEST(syn_process_suite, test_resize_upscale)
{
	static uint8_t src[4] = { 10, 30, 50, 70 }; /* 2x2, 1 channel */
	syn_tensor_t in = {
		.data = src, .size = 4, .dtype = SYN_NPU_DTYPE_UINT8,
		.ndim = 4, .shape = { 1, 2, 2, 1 },
	};
	syn_tensor_t out = make_out(16);
	syn_resize_config_t cfg = { .w = 4, .h = 4 };

	zassert_equal(syn_preprocess_image_resize(&in, &out, &cfg), 0,
		      "Resize failed");
	zassert_equal(out.shape[1], 4, "Wrong height");
	zassert_equal(out.shape[2], 4, "Wrong width");
	zassert_equal(out.size, 16, "Wrong size");

	uint8_t *dst = out.data;

	/* Edge-aligned bilinear preserves the corners */
	zassert_equal(dst[0], 10, "Top-left corner");
	zassert_equal(dst[3], 30, "Top-right corner");
	zassert_equal(dst[12], 50, "Bottom-left corner");
	zassert_equal(dst[15], 70, "Bottom-right corner");

	/* Interior is interpolated: dst(1,1) = 30 +- 1 */
	zassert_within(dst[5], 30, 1, "Center interpolation");
}

ZTEST(syn_process_suite, test_resize_downscale)
{
	static uint8_t src[16];
	syn_tensor_t in = {
		.data = src, .size = 16, .dtype = SYN_NPU_DTYPE_UINT8,
		.ndim = 4, .shape = { 1, 4, 4, 1 },
	};

	for (int i = 0; i < 16; i++) {
		src[i] = (uint8_t)(i * 16);
	}

	syn_tensor_t out = make_out(4);
	syn_resize_config_t cfg = { .w = 2, .h = 2 };

	zassert_equal(syn_preprocess_image_resize(&in, &out, &cfg), 0,
		      "Resize failed");

	uint8_t *dst = out.data;

	zassert_equal(dst[0], src[0], "Top-left corner");
	zassert_equal(dst[1], src[3], "Top-right corner");
	zassert_equal(dst[2], src[12], "Bottom-left corner");
	zassert_equal(dst[3], src[15], "Bottom-right corner");
}

ZTEST(syn_process_suite, test_resize_invalid)
{
	static uint8_t src[4];
	syn_tensor_t in = {
		.data = src, .size = 4, .dtype = SYN_NPU_DTYPE_UINT8,
		.ndim = 4, .shape = { 1, 2, 2, 1 },
	};
	syn_tensor_t out = make_out(16);
	syn_resize_config_t cfg = { .w = 4, .h = 4 };

	zassert_equal(syn_preprocess_image_resize(&in, &out, NULL),
		      -EINVAL, "NULL config should fail");

	syn_tensor_t small = make_out(8);

	zassert_equal(syn_preprocess_image_resize(&in, &small, &cfg),
		      -ENOMEM, "Small output should fail");

	in.ndim = 2;
	zassert_equal(syn_preprocess_image_resize(&in, &out, &cfg),
		      -EINVAL, "Non-image tensor should fail");
}

/* ---------------- normalize ---------------- */

ZTEST(syn_process_suite, test_normalize)
{
	static uint8_t src[3] = { 0, 128, 255 };
	syn_tensor_t in = {
		.data = src, .size = 3, .dtype = SYN_NPU_DTYPE_UINT8,
		.ndim = 4, .shape = { 1, 1, 3, 1 },
	};
	syn_tensor_t out = make_out(12);
	syn_normalize_config_t cfg = {
		.mean = { 127.5f }, .std = { 127.5f }, .channels = 1,
	};

	zassert_equal(syn_preprocess_image_normalize(&in, &out, &cfg), 0,
		      "Normalize failed");
	zassert_equal(out.dtype, SYN_NPU_DTYPE_FLOAT32, "Wrong dtype");

	float *v = out.data;

	zassert_within(v[0], -1.0f, 0.001f, "Min pixel");
	zassert_within(v[1], 0.0f, 0.01f, "Mid pixel");
	zassert_within(v[2], 1.0f, 0.001f, "Max pixel");
}

ZTEST(syn_process_suite, test_normalize_invalid)
{
	static uint8_t src[3];
	syn_tensor_t in = {
		.data = src, .size = 3, .dtype = SYN_NPU_DTYPE_UINT8,
		.ndim = 4, .shape = { 1, 1, 3, 1 },
	};
	syn_tensor_t out = make_out(12);

	syn_normalize_config_t zero_std = {
		.mean = { 0.0f }, .std = { 0.0f }, .channels = 1,
	};

	zassert_equal(syn_preprocess_image_normalize(&in, &out, &zero_std),
		      -EINVAL, "Zero std should fail");

	syn_normalize_config_t wrong_ch = {
		.mean = { 0.0f }, .std = { 1.0f }, .channels = 3,
	};

	zassert_equal(syn_preprocess_image_normalize(&in, &out, &wrong_ch),
		      -EINVAL, "Channel mismatch should fail");
}

/* ---------------- quantize / dequantize ---------------- */

ZTEST(syn_process_suite, test_quantize_int8)
{
	static float src[4] = { -1.0f, 0.0f, 0.5f, 2.0f };
	syn_tensor_t in = {
		.data = src, .size = sizeof(src),
		.dtype = SYN_NPU_DTYPE_FLOAT32,
		.ndim = 1, .shape = { 4 },
	};
	syn_tensor_t out = make_out(4);
	syn_quantize_config_t cfg = {
		.scale = 1.0f / 127.0f, .zero_point = 0,
	};

	zassert_equal(syn_preprocess_quantize_int8(&in, &out, &cfg), 0,
		      "Quantize failed");

	int8_t *q = out.data;

	zassert_equal(q[0], -127, "Quantize -1.0");
	zassert_equal(q[1], 0, "Quantize 0.0");
	zassert_within(q[2], 64, 1, "Quantize 0.5");
	zassert_equal(q[3], 127, "Quantize 2.0 saturates");
}

ZTEST(syn_process_suite, test_dequantize_roundtrip)
{
	static int8_t src[4] = { -127, 0, 64, 127 };
	syn_tensor_t in = {
		.data = src, .size = 4, .dtype = SYN_NPU_DTYPE_INT8,
		.ndim = 1, .shape = { 4 },
	};
	syn_tensor_t out = make_out(16);
	syn_dequantize_config_t cfg = {
		.scale = 1.0f / 127.0f, .zero_point = 0,
	};

	zassert_equal(syn_postprocess_dequantize(&in, &out, &cfg), 0,
		      "Dequantize failed");

	float *v = out.data;

	zassert_within(v[0], -1.0f, 0.001f, "Dequantize -127");
	zassert_within(v[1], 0.0f, 0.001f, "Dequantize 0");
	zassert_within(v[2], 0.504f, 0.001f, "Dequantize 64");
	zassert_within(v[3], 1.0f, 0.001f, "Dequantize 127");
}

/* ---------------- softmax / argmax / top-k ---------------- */

ZTEST(syn_process_suite, test_softmax_int8)
{
	static int8_t src[10] = { 0, 0, 0, 0, 0, 127, 0, 0, 0, 0 };
	syn_tensor_t in = {
		.data = src, .size = 10, .dtype = SYN_NPU_DTYPE_INT8,
		.ndim = 1, .shape = { 10 },
	};
	syn_tensor_t out = make_out(40);

	zassert_equal(syn_postprocess_softmax(&in, &out, NULL), 0,
		      "Softmax failed");
	zassert_equal(out.dtype, SYN_NPU_DTYPE_FLOAT32, "Wrong dtype");

	float *p = out.data;
	float sum = 0.0f;

	for (int i = 0; i < 10; i++) {
		sum += p[i];
	}
	zassert_within(sum, 1.0f, 0.001f, "Probabilities must sum to 1");
	zassert_true(p[5] > 0.99f, "Winner should dominate");
}

ZTEST(syn_process_suite, test_argmax)
{
	static int8_t src[10] = { -5, 3, 88, 1, 0, -100, 42, 7, 9, 11 };
	syn_tensor_t in = {
		.data = src, .size = 10, .dtype = SYN_NPU_DTYPE_INT8,
		.ndim = 1, .shape = { 10 },
	};
	syn_tensor_t out = make_out(sizeof(syn_classification_t));

	zassert_equal(syn_postprocess_argmax(&in, &out, NULL), 0,
		      "Argmax failed");

	syn_classification_t *r = out.data;

	zassert_equal(r->index, 2, "Wrong argmax index");
	zassert_within(r->score, 88.0f, 0.001f, "Wrong argmax score");
}

ZTEST(syn_process_suite, test_top_k)
{
	static float src[5] = { 0.1f, 0.5f, 0.3f, 0.9f, 0.2f };
	syn_tensor_t in = {
		.data = src, .size = sizeof(src),
		.dtype = SYN_NPU_DTYPE_FLOAT32,
		.ndim = 1, .shape = { 5 },
	};
	syn_tensor_t out = make_out(3 * sizeof(syn_classification_t));
	syn_topk_config_t cfg = { .k = 3 };

	zassert_equal(syn_postprocess_top_k(&in, &out, &cfg), 0,
		      "Top-k failed");
	zassert_equal(out.shape[0], 3, "Should emit 3 results");

	syn_classification_t *r = out.data;

	zassert_equal(r[0].index, 3, "First should be index 3");
	zassert_equal(r[1].index, 1, "Second should be index 1");
	zassert_equal(r[2].index, 2, "Third should be index 2");
	zassert_true(r[0].score >= r[1].score && r[1].score >= r[2].score,
		     "Scores must be descending");
}

/* ---------------- NMS ---------------- */

ZTEST(syn_process_suite, test_nms)
{
	static syn_bbox_t boxes[3] = {
		{ 0.0f, 0.0f, 10.0f, 10.0f, 0.9f, 0.0f },
		{ 1.0f, 1.0f, 11.0f, 11.0f, 0.8f, 0.0f }, /* IoU 0.68 w/ 1st */
		{ 0.0f, 0.0f, 10.0f, 10.0f, 0.7f, 1.0f }, /* other class */
	};
	syn_tensor_t in = {
		.data = boxes, .size = sizeof(boxes),
		.dtype = SYN_NPU_DTYPE_FLOAT32,
		.ndim = 2, .shape = { 3, 6 },
	};
	syn_tensor_t out = make_out(sizeof(boxes));
	syn_nms_config_t cfg = {
		.iou_threshold = 0.5f, .score_threshold = 0.1f,
		.max_boxes = 8,
	};

	zassert_equal(syn_postprocess_nms(&in, &out, &cfg), 0, "NMS failed");
	zassert_equal(out.shape[0], 2, "Should keep 2 boxes, got %u",
		      out.shape[0]);

	syn_bbox_t *kept = out.data;

	zassert_within(kept[0].score, 0.9f, 0.001f,
		       "Highest score kept first");
	zassert_within(kept[1].score, 0.7f, 0.001f,
		       "Other class survives suppression");
}

ZTEST(syn_process_suite, test_nms_score_threshold)
{
	static syn_bbox_t boxes[2] = {
		{ 0.0f, 0.0f, 10.0f, 10.0f, 0.9f, 0.0f },
		{ 20.0f, 20.0f, 30.0f, 30.0f, 0.05f, 0.0f },
	};
	syn_tensor_t in = {
		.data = boxes, .size = sizeof(boxes),
		.dtype = SYN_NPU_DTYPE_FLOAT32,
		.ndim = 2, .shape = { 2, 6 },
	};
	syn_tensor_t out = make_out(sizeof(boxes));
	syn_nms_config_t cfg = {
		.iou_threshold = 0.5f, .score_threshold = 0.5f,
		.max_boxes = 8,
	};

	zassert_equal(syn_postprocess_nms(&in, &out, &cfg), 0, "NMS failed");
	zassert_equal(out.shape[0], 1,
		      "Low-score box should be dropped");
}

/* ---------------- MFCC ---------------- */

ZTEST(syn_process_suite, test_mfcc)
{
	/* Two 64-sample frames: 1 kHz sine, then silence */
	static float samples[128];

	for (int i = 0; i < 64; i++) {
		samples[i] = sinf(2.0f * 3.14159265f * 1000.0f * i /
				  8000.0f);
	}
	memset(&samples[64], 0, 64 * sizeof(float));

	syn_tensor_t in = {
		.data = samples, .size = sizeof(samples),
		.dtype = SYN_NPU_DTYPE_FLOAT32,
		.ndim = 2, .shape = { 1, 128 },
	};
	syn_tensor_t out = make_out(2 * 6 * sizeof(float));
	syn_mfcc_config_t cfg = {
		.sample_rate_hz = 8000,
		.frame_len = 64,
		.num_mel = 12,
		.num_coeffs = 6,
	};

	zassert_equal(syn_preprocess_audio_mfcc(&in, &out, &cfg), 0,
		      "MFCC failed");
	zassert_equal(out.shape[0], 2, "Expected 2 frames");
	zassert_equal(out.shape[1], 6, "Expected 6 coefficients");

	float *c = out.data;

	for (int i = 0; i < 12; i++) {
		zassert_true(isfinite(c[i]), "Coefficient %d not finite", i);
	}

	/* Energy coefficient of the sine frame exceeds the silent frame */
	zassert_true(c[0] > c[6],
		     "Sine frame c0 should exceed silence c0");
}

ZTEST(syn_process_suite, test_mfcc_invalid)
{
	static float samples[64];
	syn_tensor_t in = {
		.data = samples, .size = sizeof(samples),
		.dtype = SYN_NPU_DTYPE_FLOAT32,
		.ndim = 2, .shape = { 1, 64 },
	};
	syn_tensor_t out = make_out(256);

	syn_mfcc_config_t bad_frame = {
		.sample_rate_hz = 8000, .frame_len = 60, /* not pow2 */
		.num_mel = 12, .num_coeffs = 6,
	};

	zassert_equal(syn_preprocess_audio_mfcc(&in, &out, &bad_frame),
		      -EINVAL, "Non-pow2 frame_len should fail");

	syn_mfcc_config_t bad_coeffs = {
		.sample_rate_hz = 8000, .frame_len = 64,
		.num_mel = 6, .num_coeffs = 12, /* coeffs > mel */
	};

	zassert_equal(syn_preprocess_audio_mfcc(&in, &out, &bad_coeffs),
		      -EINVAL, "coeffs > mel should fail");
}

/* ---------------- pipeline integration ---------------- */

ZTEST(syn_process_suite, test_pipeline_with_builtins)
{
	/* Acceptance 2.4: built-ins run inside the pipeline system.
	 * normalize -> quantize -> model -> softmax -> argmax
	 */
	static uint8_t frame[48];
	static syn_normalize_config_t norm_cfg = {
		.mean = { 0.0f, 0.0f, 0.0f }, .std = { 1.0f, 1.0f, 1.0f },
		.channels = 3,
	};
	static syn_quantize_config_t quant_cfg = {
		.scale = 1.0f, .zero_point = 0,
	};

	/* Values 0..47 survive int8 quantization unchanged, so the
	 * stub NPU sees the same bytes: winner = sum % 10.
	 */
	uint32_t sum = 0;

	for (int i = 0; i < 48; i++) {
		frame[i] = (uint8_t)i;
		sum += (uint8_t)i;
	}

	syn_tensor_t input = {
		.data = frame, .size = sizeof(frame),
		.dtype = SYN_NPU_DTYPE_UINT8,
		.ndim = 4, .shape = { 1, 4, 4, 3 },
	};

	syn_pipeline_t *pipe = syn_pipeline_create("builtin_e2e");

	zassert_not_null(pipe, "Pipeline create failed");
	zassert_equal(syn_pipeline_add_preprocess(
			      pipe, syn_preprocess_image_normalize,
			      &norm_cfg), 0, "add normalize failed");
	zassert_equal(syn_pipeline_add_preprocess(
			      pipe, syn_preprocess_quantize_int8,
			      &quant_cfg), 0, "add quantize failed");
	zassert_equal(syn_pipeline_add_model(pipe, proc_model), 0,
		      "add model failed");
	zassert_equal(syn_pipeline_add_postprocess(
			      pipe, syn_postprocess_softmax, NULL), 0,
		      "add softmax failed");
	zassert_equal(syn_pipeline_add_postprocess(
			      pipe, syn_postprocess_argmax, NULL), 0,
		      "add argmax failed");
	zassert_equal(syn_pipeline_build(pipe), 0, "build failed");

	syn_job_id_t job = syn_infer_submit(pipe, &input, NULL);

	zassert_not_equal(job, SYN_JOB_INVALID, "Submit failed");
	zassert_equal(syn_infer_wait(job, 2000), 0, "Wait failed");

	syn_tensor_t result;

	zassert_equal(syn_infer_get_result(job, &result), 0,
		      "get_result failed");
	zassert_equal(result.size, sizeof(syn_classification_t),
		      "Result should be one classification record");

	syn_classification_t *top = result.data;

	zassert_equal(top->index, sum % 10,
		      "Expected class %u, got %u", sum % 10, top->index);
	zassert_true(top->score > 0.99f,
		     "Winner probability should dominate");

	syn_pipeline_destroy(pipe);
}
