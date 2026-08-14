/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file main.c
 * @brief SynapticOS application template
 *
 * Minimal out-of-tree consumer of the SynapticOS runtime, using the
 * public API only (include/synaptic/): initialize, register and load
 * a model, run one inference through the scheduler, report the top
 * class. The 64-byte zero blob exercises the software (stub) NPU
 * path on every target; replace it with a packed .synm payload (see
 * tools/syn_model_pack.py in the synaptic-os repo) to target the
 * eIQ Neutron NPU on FRDM-MCXN947 builds with the `neutron` module.
 */

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <synaptic/syn_api.h>
#include <string.h>

LOG_MODULE_REGISTER(synaptic_app, LOG_LEVEL_INF);

static const uint8_t demo_model[64] = {0};

/* Fits the stub NPU input bound (1 KB) on QEMU */
#define DEMO_INPUT_SIZE 768

int main(void)
{
	int ret;

	LOG_INF("SynapticOS %s application template", syn_version());

	ret = syn_init();
	if (ret != 0) {
		LOG_ERR("syn_init failed: %d", ret);
		return ret;
	}

	/* Describe and register the model */
	syn_model_info_t info = {0};

	strncpy(info.name, "template_model", sizeof(info.name) - 1);
	strncpy(info.version, "1.0.0", sizeof(info.version) - 1);
	info.input_size = DEMO_INPUT_SIZE;
	info.output_size = 10;
	info.flash_size = sizeof(demo_model);
	info.sram_required = 4096;
	info.input_dtype = SYN_NPU_DTYPE_INT8;
	info.output_dtype = SYN_NPU_DTYPE_INT8;

	syn_model_handle_t model;

	ret = syn_model_register(&info, &model);
	if (ret != 0) {
		LOG_ERR("register failed: %d", ret);
		return ret;
	}

	/* Make the blob NPU-resident, then mark the model runnable.
	 * Jobs on models that are not loaded are refused by contract.
	 */
	ret = syn_hal_npu_load_model(demo_model, sizeof(demo_model));
	if (ret != 0) {
		LOG_ERR("NPU load failed: %d", ret);
		return ret;
	}
	ret = syn_model_load(model);
	if (ret != 0) {
		LOG_ERR("model load failed: %d", ret);
		return ret;
	}

	/* Ephemeral input tensor with a test pattern */
	uint32_t shape[1] = { DEMO_INPUT_SIZE };
	syn_tensor_t *input = syn_mem_tensor_alloc(shape, 1,
						   SYN_NPU_DTYPE_INT8,
						   SYN_MEM_EPHEMERAL);

	if (input == NULL) {
		LOG_ERR("arena too small for a %u-byte input",
			DEMO_INPUT_SIZE);
		return -ENOMEM;
	}

	uint8_t *data = input->data;

	for (size_t i = 0; i < input->size; i++) {
		data[i] = (uint8_t)(i & 0xFF);
	}

	/* One synchronous inference through the scheduler */
	int8_t out_buf[16];
	syn_tensor_t output = {
		.data = out_buf,
		.size = sizeof(out_buf),
	};

	ret = syn_infer_run_sync(model, input, &output, SYN_PRIORITY_NORMAL);
	if (ret != 0) {
		LOG_ERR("inference failed: %d", ret);
		return ret;
	}

	uint32_t top = 0;

	syn_hal_dsp_argmax(output.data, output.size, &top);
	LOG_INF("Inference OK: %u output bytes, top class %u",
		(unsigned)output.size, top);

	/* Reclaim the ephemeral region between inferences */
	syn_mem_reset_ephemeral();

	syn_mem_print_stats();
	LOG_INF("Template run complete");
	return 0;
}
