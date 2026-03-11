/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file syn_hal_npu_stub.c
 * @brief SynapticOS — NPU Stub (CPU-only software fallback)
 *
 * Software-emulated NPU for QEMU testing and integration development.
 * Simulates the full NPU lifecycle with deterministic fake inference.
 */

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <synaptic/syn_hal_npu.h>
#include <string.h>

LOG_MODULE_REGISTER(syn_hal_npu_stub, CONFIG_SYNAPTIC_LOG_LEVEL);

#define STUB_MAX_MODEL_SIZE    (256 * 1024)
#define STUB_MAX_INPUT_SIZE    1024
#define STUB_MAX_OUTPUT_SIZE   256

static struct {
	syn_npu_state_t state;
	bool            model_loaded;
	size_t          model_size;
	uint8_t         input_buf[STUB_MAX_INPUT_SIZE];
	size_t          input_size;
	uint8_t         output_buf[STUB_MAX_OUTPUT_SIZE];
	size_t          output_size;
	bool            initialized;
} stub;

int syn_hal_npu_init(void)
{
	if (stub.initialized) {
		return -EALREADY;
	}

	memset(&stub, 0, sizeof(stub));
	stub.state = SYN_NPU_STATE_IDLE;
	stub.initialized = true;

	LOG_INF("NPU stub initialized (software fallback)");
	return 0;
}

void syn_hal_npu_deinit(void)
{
	stub.initialized = false;
	stub.state = SYN_NPU_STATE_IDLE;
	stub.model_loaded = false;
	LOG_INF("NPU stub deinitialized");
}

int syn_hal_npu_get_caps(syn_npu_caps_t *caps)
{
	if (caps == NULL) {
		return -EINVAL;
	}

	caps->name = "stub";
	caps->max_ops_per_sec = 0;
	caps->scratch_size = 0;
	caps->supported_dtypes = 0x1F; /* All types */
	caps->supports_async = false;

	return 0;
}

syn_npu_state_t syn_hal_npu_get_state(void)
{
	return stub.state;
}

int syn_hal_npu_load_model(const uint8_t *model_data, size_t model_size)
{
	if (model_data == NULL || model_size == 0) {
		return -EINVAL;
	}
	if (!stub.initialized) {
		return -EPERM;
	}
	if (stub.state == SYN_NPU_STATE_BUSY) {
		return -EBUSY;
	}
	if (model_size > STUB_MAX_MODEL_SIZE) {
		return -ENOMEM;
	}

	stub.model_size = model_size;
	stub.model_loaded = true;

	LOG_INF("Model loaded: %zu bytes", model_size);
	return 0;
}

int syn_hal_npu_set_input(uint8_t index, const void *data, size_t size)
{
	if (data == NULL || size == 0 || index != 0) {
		return -EINVAL;
	}
	if (!stub.initialized || !stub.model_loaded) {
		return -EPERM;
	}
	if (stub.state == SYN_NPU_STATE_BUSY) {
		return -EBUSY;
	}
	if (size > STUB_MAX_INPUT_SIZE) {
		return -ENOMEM;
	}

	memcpy(stub.input_buf, data, size);
	stub.input_size = size;

	return 0;
}

int syn_hal_npu_invoke(void)
{
	if (!stub.initialized || !stub.model_loaded) {
		return -EPERM;
	}
	if (stub.state == SYN_NPU_STATE_BUSY) {
		return -EBUSY;
	}

	stub.state = SYN_NPU_STATE_BUSY;

	/* Generate deterministic fake output: 10-class classification */
	stub.output_size = 10;
	memset(stub.output_buf, 0, stub.output_size);

	/* Simple hash: sum input bytes to pick "winner" class */
	uint32_t sum = 0;

	for (size_t i = 0; i < stub.input_size; i++) {
		sum += stub.input_buf[i];
	}

	uint8_t winner = sum % stub.output_size;

	stub.output_buf[winner] = 127; /* Max INT8 confidence */

	/* Simulate inference latency with volatile loop (no timer HW needed) */
	for (volatile int delay = 0; delay < 1000; delay++) {
	}

	stub.state = SYN_NPU_STATE_IDLE;

	LOG_DBG("Stub inference complete: class %u (sum=%u)", winner, sum);
	return 0;
}

int syn_hal_npu_invoke_async(syn_npu_done_cb_t cb, void *user_data)
{
	ARG_UNUSED(cb);
	ARG_UNUSED(user_data);

	return -ENOTSUP;
}

int syn_hal_npu_get_output(uint8_t index, void *data, size_t *size)
{
	if (data == NULL || size == NULL || index != 0) {
		return -EINVAL;
	}
	if (!stub.initialized) {
		return -EPERM;
	}
	if (stub.state == SYN_NPU_STATE_BUSY) {
		return -EBUSY;
	}

	memcpy(data, stub.output_buf, stub.output_size);
	*size = stub.output_size;

	return 0;
}

int syn_hal_npu_suspend(void)
{
	if (!stub.initialized) {
		return -EPERM;
	}

	stub.state = SYN_NPU_STATE_SUSPENDED;
	LOG_DBG("NPU stub suspended");
	return 0;
}

int syn_hal_npu_resume(void)
{
	if (!stub.initialized) {
		return -EPERM;
	}
	if (stub.state != SYN_NPU_STATE_SUSPENDED) {
		return -EINVAL;
	}

	stub.state = SYN_NPU_STATE_IDLE;
	LOG_DBG("NPU stub resumed");
	return 0;
}
