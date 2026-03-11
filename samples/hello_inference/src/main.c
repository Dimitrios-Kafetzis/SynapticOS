/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file main.c
 * @brief SynapticOS — Hello Inference Sample
 *
 * End-to-end demo: initializes the runtime, registers and loads a model,
 * runs a single inference through the NPU HAL, finds the top prediction
 * via DSP argmax, and prints profiling/memory statistics.
 */

#include <zephyr/kernel.h>
#include <synaptic/syn_api.h>
#include <string.h>

#include <zephyr/logging/log.h>
LOG_MODULE_REGISTER(hello_inference, LOG_LEVEL_INF);

/* Dummy model binary for Phase 1 demo */
static const uint8_t dummy_model[64] = {0};

/* Input size that fits both QEMU stub (1 KB max) and FRDM hardware */
#define DEMO_INPUT_H  16
#define DEMO_INPUT_W  16
#define DEMO_INPUT_C  3
#define DEMO_INPUT_SIZE (DEMO_INPUT_H * DEMO_INPUT_W * DEMO_INPUT_C)

int main(void)
{
	int ret;

	LOG_INF("=== SynapticOS %s — Hello Inference ===", syn_version());

	/* 1. Initialize runtime */
	ret = syn_init();
	if (ret != 0) {
		LOG_ERR("syn_init() failed: %d", ret);
		return ret;
	}

	/* 2. Register a model */
	syn_model_info_t model_info = {0};

	strncpy(model_info.name, "test_classify", sizeof(model_info.name));
	strncpy(model_info.version, "1.0.0", sizeof(model_info.version));
	model_info.input_size = DEMO_INPUT_SIZE;
	model_info.output_size = 10;
	model_info.flash_size = sizeof(dummy_model);
	model_info.sram_required = 4096;
	model_info.input_dtype = SYN_NPU_DTYPE_INT8;
	model_info.output_dtype = SYN_NPU_DTYPE_INT8;

	syn_model_handle_t handle;

	ret = syn_model_register(&model_info, &handle);
	if (ret != 0) {
		LOG_ERR("Model register failed: %d", ret);
		return ret;
	}
	LOG_INF("Registered model '%s' (handle=%u)", model_info.name, handle);

	/* 3. Load model to NPU */
	ret = syn_hal_npu_load_model(dummy_model, sizeof(dummy_model));
	if (ret != 0) {
		LOG_ERR("NPU load failed: %d", ret);
		return ret;
	}
	LOG_INF("Model loaded to NPU");

	/* Mark model as loaded in registry */
	syn_model_load(handle);

	/* 4. Allocate input tensor */
	uint32_t input_shape[] = {1, DEMO_INPUT_H, DEMO_INPUT_W, DEMO_INPUT_C};
	syn_tensor_t *input = syn_mem_tensor_alloc(input_shape, 4,
						   SYN_NPU_DTYPE_INT8,
						   SYN_MEM_EPHEMERAL);
	if (input == NULL) {
		LOG_ERR("Failed to allocate input tensor");
		return -ENOMEM;
	}
	LOG_INF("Input tensor: %ux%ux%ux%u (%u bytes)",
		input->shape[0], input->shape[1],
		input->shape[2], input->shape[3],
		(unsigned)input->size);

	/* 5. Fill with test pattern (gradient) */
	uint8_t *data = (uint8_t *)input->data;

	for (size_t i = 0; i < input->size; i++) {
		data[i] = (uint8_t)(i & 0xFF);
	}

	/* 6. Run inference */
	ret = syn_hal_npu_set_input(0, input->data, input->size);
	if (ret != 0) {
		LOG_ERR("Set input failed: %d", ret);
		return ret;
	}

	uint32_t start = k_cycle_get_32();

	ret = syn_hal_npu_invoke();

	uint32_t elapsed_us = k_cyc_to_us_ceil32(k_cycle_get_32() - start);

	if (ret != 0) {
		LOG_ERR("Invoke failed: %d", ret);
		return ret;
	}
	LOG_INF("Inference completed in %u us", elapsed_us);

	/* 7. Get output */
	int8_t output_buf[256];
	size_t output_size = sizeof(output_buf);

	ret = syn_hal_npu_get_output(0, output_buf, &output_size);
	if (ret != 0) {
		LOG_ERR("Get output failed: %d", ret);
		return ret;
	}

	/* 8. Find top prediction via DSP argmax */
	uint32_t top_class = 0;

	ret = syn_hal_dsp_argmax(output_buf, output_size, &top_class);
	if (ret != 0) {
		LOG_WRN("Argmax failed: %d, using manual scan", ret);
		int8_t max_val = output_buf[0];

		for (size_t i = 1; i < output_size; i++) {
			if (output_buf[i] > max_val) {
				max_val = output_buf[i];
				top_class = i;
			}
		}
	}
	LOG_INF("Prediction: class %u (confidence %d)", top_class,
		output_buf[top_class]);

	/* 9. Print stats */
	syn_mem_print_stats();

	/* 10. Print profiling */
	syn_prof_print_summary();

	LOG_INF("=== Hello Inference complete ===");
	LOG_INF("Use 'syn' shell commands to inspect runtime.");

	/* Keep shell alive */
	while (1) {
		k_sleep(K_SECONDS(1));
	}

	return 0;
}
