/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file test_infer_pipeline.c
 * @brief Unit tests for pipeline construction (Phase 2.1)
 */

#include <zephyr/ztest.h>
#include <synaptic/syn_infer.h>
#include <synaptic/syn_mem.h>
#include <synaptic/syn_model.h>
#include <string.h>

static uint8_t pipe_arena[4096] __aligned(16);
static syn_model_handle_t test_model;

/* Passthrough stage: copy input to output, keep geometry */
static int passthrough(const syn_tensor_t *in, syn_tensor_t *out,
		       const void *config)
{
	ARG_UNUSED(config);

	if (out->size < in->size) {
		return -ENOMEM;
	}
	memcpy(out->data, in->data, in->size);
	out->size = in->size;
	out->dtype = in->dtype;
	out->ndim = in->ndim;
	memcpy(out->shape, in->shape, sizeof(out->shape));
	return 0;
}

static void *pipeline_suite_setup(void)
{
	syn_mem_init(pipe_arena, sizeof(pipe_arena));

	if (syn_model_get_by_name("pipe_test", &test_model) != 0) {
		syn_model_info_t info = {0};

		strncpy(info.name, "pipe_test", sizeof(info.name));
		strncpy(info.version, "1.0.0", sizeof(info.version));
		info.input_size = 48;
		info.output_size = 10;
		info.sram_required = 256;
		info.input_dtype = SYN_NPU_DTYPE_INT8;
		info.output_dtype = SYN_NPU_DTYPE_INT8;
		zassert_equal(syn_model_register(&info, &test_model), 0,
			      "Model registration failed");
	}
	return NULL;
}

ZTEST_SUITE(syn_pipeline_suite, NULL, pipeline_suite_setup, NULL, NULL, NULL);

ZTEST(syn_pipeline_suite, test_create_destroy)
{
	syn_pipeline_t *p = syn_pipeline_create("basic");

	zassert_not_null(p, "Create failed");
	syn_pipeline_destroy(p);
}

ZTEST(syn_pipeline_suite, test_pool_exhaustion)
{
	syn_pipeline_t *pipes[4];

	for (int i = 0; i < 4; i++) {
		pipes[i] = syn_pipeline_create("pool");
		zassert_not_null(pipes[i], "Create %d failed", i);
	}

	/* Pool is full: 5th create must fail */
	zassert_is_null(syn_pipeline_create("overflow"),
			"5th pipeline should fail");

	/* Destroying one frees a slot */
	syn_pipeline_destroy(pipes[0]);
	pipes[0] = syn_pipeline_create("reuse");
	zassert_not_null(pipes[0], "Reuse after destroy failed");

	for (int i = 0; i < 4; i++) {
		syn_pipeline_destroy(pipes[i]);
	}
}

ZTEST(syn_pipeline_suite, test_build_requires_model)
{
	syn_pipeline_t *p = syn_pipeline_create("nomodel");

	zassert_not_null(p, "Create failed");
	zassert_equal(syn_pipeline_add_preprocess(p, passthrough, NULL), 0,
		      "add_preprocess failed");
	zassert_equal(syn_pipeline_build(p), -EINVAL,
		      "Build without model should fail");
	syn_pipeline_destroy(p);
}

ZTEST(syn_pipeline_suite, test_stage_ordering)
{
	syn_pipeline_t *p = syn_pipeline_create("order");

	zassert_not_null(p, "Create failed");

	/* Postprocess before model is rejected */
	zassert_equal(syn_pipeline_add_postprocess(p, passthrough, NULL),
		      -EINVAL, "Postprocess before model should fail");

	zassert_equal(syn_pipeline_add_model(p, test_model), 0,
		      "add_model failed");

	/* Preprocess after model is rejected */
	zassert_equal(syn_pipeline_add_preprocess(p, passthrough, NULL),
		      -EINVAL, "Preprocess after model should fail");

	/* Second model is rejected */
	zassert_equal(syn_pipeline_add_model(p, test_model), -EALREADY,
		      "Second model should fail");

	syn_pipeline_destroy(p);
}

ZTEST(syn_pipeline_suite, test_unregistered_model)
{
	syn_pipeline_t *p = syn_pipeline_create("badmodel");

	zassert_not_null(p, "Create failed");
	zassert_equal(syn_pipeline_add_model(p, 99), -ENOENT,
		      "Unregistered model should fail");
	zassert_equal(syn_pipeline_add_model(p, SYN_MODEL_INVALID), -EINVAL,
		      "Invalid handle should fail");
	syn_pipeline_destroy(p);
}

ZTEST(syn_pipeline_suite, test_full_pipeline_build)
{
	/* Acceptance: 2 preprocess + 1 model + 1 postprocess */
	syn_pipeline_t *p = syn_pipeline_create("full");

	zassert_not_null(p, "Create failed");
	zassert_equal(syn_pipeline_add_preprocess(p, passthrough, NULL), 0,
		      "pre 1 failed");
	zassert_equal(syn_pipeline_add_preprocess(p, passthrough, NULL), 0,
		      "pre 2 failed");
	zassert_equal(syn_pipeline_add_model(p, test_model), 0,
		      "model failed");
	zassert_equal(syn_pipeline_add_postprocess(p, passthrough, NULL), 0,
		      "post failed");
	zassert_equal(syn_pipeline_build(p), 0, "Build failed");

	/* Build twice is rejected, and so is adding after build */
	zassert_equal(syn_pipeline_build(p), -EALREADY,
		      "Double build should fail");
	zassert_equal(syn_pipeline_add_postprocess(p, passthrough, NULL),
		      -EPERM, "Add after build should fail");

	syn_pipeline_destroy(p);
}

ZTEST(syn_pipeline_suite, test_stage_overflow)
{
	syn_pipeline_t *p = syn_pipeline_create("overflow");

	zassert_not_null(p, "Create failed");

	/* Fill all but one slot with preprocess stages, then the model */
	for (int i = 0; i < CONFIG_SYNAPTIC_MAX_PIPELINE_STAGES - 1; i++) {
		zassert_equal(syn_pipeline_add_preprocess(p, passthrough,
							  NULL),
			      0, "pre %d failed", i);
	}
	zassert_equal(syn_pipeline_add_model(p, test_model), 0,
		      "model failed");

	/* Table is full now */
	zassert_equal(syn_pipeline_add_postprocess(p, passthrough, NULL),
		      -ENOMEM, "Stage overflow should fail");

	syn_pipeline_destroy(p);
}

ZTEST(syn_pipeline_suite, test_invalid_args)
{
	syn_pipeline_t *p = syn_pipeline_create("args");

	zassert_not_null(p, "Create failed");
	zassert_equal(syn_pipeline_add_preprocess(p, NULL, NULL), -EINVAL,
		      "NULL fn should fail");
	zassert_equal(syn_pipeline_add_preprocess(NULL, passthrough, NULL),
		      -EINVAL, "NULL pipe should fail");
	zassert_equal(syn_pipeline_build(NULL), -EINVAL,
		      "NULL build should fail");
	syn_pipeline_destroy(p);
	syn_pipeline_destroy(NULL); /* Must not crash */
}
