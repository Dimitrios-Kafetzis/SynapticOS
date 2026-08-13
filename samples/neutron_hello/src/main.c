/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file main.c
 * @brief SynapticOS — Neutron NPU Hello Sample (FRDM-MCXN947 CPU0)
 *
 * First-light demo for the real eIQ Neutron invoke path: loads a
 * neutron-converter-compiled INT8 classifier (embedded as a "SYNN"
 * blob at build time), runs repeated inferences through
 * syn_infer_run_sync, and prints the raw logits, the top-1 class and
 * latency statistics for comparison against the host reference
 * (community/phase6/s3-artifacts/host-reference.md).
 */

#include <zephyr/kernel.h>
#include <synaptic/syn_api.h>
#include <string.h>

#include <zephyr/logging/log.h>
LOG_MODULE_REGISTER(neutron_hello, LOG_LEVEL_INF);

#include "model_blob.inc"

#define INPUT_H 32
#define INPUT_W 32
#define INPUT_C 3
#define INPUT_SIZE (INPUT_H * INPUT_W * INPUT_C)
#define OUTPUT_CLASSES 10
/* The Neutron-converted graph pads the NPU data output to 12 lanes;
 * the on-CPU SLICE that trims it back to 10 is not part of the NPU
 * subgraph, so the raw NPU output is what arrives here.
 */
#define NPU_OUTPUT_SIZE 12
#define NUM_RUNS 20

int main(void)
{
	int ret;

	LOG_INF("=== SynapticOS %s, Neutron NPU hello ===", syn_version());

	ret = syn_init();
	if (ret != 0) {
		LOG_ERR("syn_init() failed: %d", ret);
		return ret;
	}

	syn_model_info_t model_info = {0};

	strncpy(model_info.name, "resnet_cifar10", sizeof(model_info.name));
	strncpy(model_info.version, "1.0.0", sizeof(model_info.version));
	model_info.input_size = INPUT_SIZE;
	model_info.output_size = NPU_OUTPUT_SIZE;
	model_info.flash_size = sizeof(model_blob);
	model_info.sram_required = 4096;
	model_info.input_dtype = SYN_NPU_DTYPE_INT8;
	model_info.output_dtype = SYN_NPU_DTYPE_INT8;

	syn_model_handle_t handle;

	ret = syn_model_register(&model_info, &handle);
	if (ret != 0) {
		LOG_ERR("Model register failed: %d", ret);
		return ret;
	}

	ret = syn_hal_npu_load_model(model_blob, sizeof(model_blob));
	if (ret != 0) {
		LOG_ERR("NPU load failed: %d", ret);
		return ret;
	}
	syn_model_load(handle);
	LOG_INF("SYNN blob loaded: %u bytes", (unsigned)sizeof(model_blob));

	uint32_t input_shape[] = {1, INPUT_H, INPUT_W, INPUT_C};
	syn_tensor_t *input = syn_mem_tensor_alloc(input_shape, 4,
						   SYN_NPU_DTYPE_INT8,
						   SYN_MEM_EPHEMERAL);
	if (input == NULL) {
		LOG_ERR("Failed to allocate input tensor");
		return -ENOMEM;
	}

	/* Deterministic pattern shared with the host reference run:
	 * int8 value ((i * 7 + 13) & 0xFF) - 128 at flat index i.
	 */
	int8_t *data = (int8_t *)input->data;

	for (size_t i = 0; i < input->size; i++) {
		data[i] = (int8_t)((int)((i * 7U + 13U) & 0xFF) - 128);
	}

	int8_t output_buf[NPU_OUTPUT_SIZE];
	uint32_t lat_min = UINT32_MAX, lat_max = 0, lat_sum = 0;
	size_t output_size = 0;

	for (int run = 0; run < NUM_RUNS; run++) {
		syn_tensor_t output = {
			.data = output_buf,
			.size = sizeof(output_buf),
		};

		uint32_t start = k_cycle_get_32();

		ret = syn_infer_run_sync(handle, input, &output,
					 SYN_PRIORITY_NORMAL);

		uint32_t us = k_cyc_to_us_ceil32(k_cycle_get_32() - start);

		if (ret != 0) {
			LOG_ERR("Inference %d failed: %d", run, ret);
			return ret;
		}
		output_size = output.size;
		lat_sum += us;
		lat_min = MIN(lat_min, us);
		lat_max = MAX(lat_max, us);
	}

	LOG_INF("%d inferences: latency min %u / avg %u / max %u us",
		NUM_RUNS, lat_min, lat_sum / NUM_RUNS, lat_max);

	LOG_INF("Output (%u B logits, pre-softmax):", (unsigned)output_size);
	for (size_t i = 0; i < output_size; i++) {
		LOG_INF("  class %u: %d", (unsigned)i, output_buf[i]);
	}

	/* Rank only the real classes; lanes 10..11 are converter pad. */
	size_t rank_size = MIN(output_size, (size_t)OUTPUT_CLASSES);
	uint32_t top_class = 0;

	ret = syn_hal_dsp_argmax(output_buf, rank_size, &top_class);
	if (ret != 0) {
		int8_t max_val = output_buf[0];

		top_class = 0;
		for (size_t i = 1; i < rank_size; i++) {
			if (output_buf[i] > max_val) {
				max_val = output_buf[i];
				top_class = i;
			}
		}
	}
	LOG_INF("Top-1: class %u (host reference for this pattern: 3)",
		top_class);

	syn_mem_print_stats();
	syn_prof_print_summary();

	LOG_INF("=== Neutron hello complete ===");

	while (1) {
		k_sleep(K_SECONDS(1));
	}

	return 0;
}
