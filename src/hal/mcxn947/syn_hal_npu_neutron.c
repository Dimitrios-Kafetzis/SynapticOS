/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file syn_hal_npu_neutron.c
 * @brief SynapticOS — eIQ Neutron NPU Driver (MCXN947)
 *
 * Hardware NPU driver for the NXP FRDM-MCXN947 eIQ Neutron NPU.
 *
 * Current implementation: Software-emulated stub with hardware-specific
 * initialization. Real Neutron SDK integration will be added when the
 * SDK becomes available in hal_nxp.
 */

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <synaptic/syn_hal_npu.h>
#include <string.h>

LOG_MODULE_REGISTER(syn_hal_npu_neutron, CONFIG_SYNAPTIC_LOG_LEVEL);

/* TODO: Replace with real Neutron SDK includes when available:
 * #include <neutron.h>
 */

/* Stub-inference bound: the HAL only keeps an XIP pointer, so accept
 * anything a flash model slot can hold (440 KB minus the .synm
 * header). Board-found: the old arbitrary 256 KB cap rejected a
 * valid slot-max OTA model AFTER it was stored and activated. The
 * real bound comes with the Neutron SDK invoke path.
 */
#define NEUTRON_MAX_MODEL_SIZE    (440 * 1024 - 64)
#define NEUTRON_MAX_INPUT_SIZE    (96 * 96 * 3)
#define NEUTRON_MAX_OUTPUT_SIZE   256

static struct {
	syn_npu_state_t state;
	bool            model_loaded;
	size_t          model_size;
	const uint8_t  *model_data;
	uint8_t         input_buf[NEUTRON_MAX_INPUT_SIZE];
	size_t          input_size;
	uint8_t         output_buf[NEUTRON_MAX_OUTPUT_SIZE];
	size_t          output_size;
	bool            initialized;
} npu;

int syn_hal_npu_init(void)
{
	if (npu.initialized) {
		return -EALREADY;
	}

	memset(&npu, 0, sizeof(npu));

	/* TODO: Enable NPU clock gate via Zephyr clock control or SYSCON:
	 *   clock_control_on(...)
	 * or:
	 *   SYSCON->AHBCLKCTRL2 |= SYSCON_AHBCLKCTRL2_NPU_MASK;
	 */

	/* TODO: Configure NPU power domain if needed */

	/* TODO: Initialize Neutron SDK:
	 *   neutron_init(&npu_config);
	 */

	npu.state = SYN_NPU_STATE_IDLE;
	npu.initialized = true;

	LOG_INF("Neutron NPU initialized (stub — SDK not yet integrated)");
	return 0;
}

void syn_hal_npu_deinit(void)
{
	/* TODO: Deinitialize Neutron SDK and disable clock gate */

	npu.initialized = false;
	npu.state = SYN_NPU_STATE_IDLE;
	npu.model_loaded = false;
	LOG_INF("Neutron NPU deinitialized");
}

int syn_hal_npu_get_caps(syn_npu_caps_t *caps)
{
	if (caps == NULL) {
		return -EINVAL;
	}

	caps->name = "neutron";
	/* MCXN947 Neutron NPU: ~100 GOPS at INT8 */
	caps->max_ops_per_sec = 100000000;
	caps->scratch_size = CONFIG_SYNAPTIC_SCRATCH_POOL_SIZE;
	caps->supported_dtypes = 0x03; /* INT8 and UINT8 for Neutron */
	caps->supports_async = true;

	return 0;
}

syn_npu_state_t syn_hal_npu_get_state(void)
{
	return npu.state;
}

int syn_hal_npu_load_model(const uint8_t *model_data, size_t model_size)
{
	if (model_data == NULL || model_size == 0) {
		return -EINVAL;
	}
	if (!npu.initialized) {
		return -EPERM;
	}
	if (npu.state == SYN_NPU_STATE_BUSY) {
		return -EBUSY;
	}
	if (model_size > NEUTRON_MAX_MODEL_SIZE) {
		return -ENOMEM;
	}

	/* TODO: Load model via Neutron SDK:
	 *   neutron_load_model(model_data, model_size);
	 */
	npu.model_data = model_data;
	npu.model_size = model_size;
	npu.model_loaded = true;

	LOG_INF("Model loaded: %zu bytes (stub inference)", model_size);
	return 0;
}

int syn_hal_npu_set_input(uint8_t index, const void *data, size_t size)
{
	if (data == NULL || size == 0 || index != 0) {
		return -EINVAL;
	}
	if (!npu.initialized || !npu.model_loaded) {
		return -EPERM;
	}
	if (npu.state == SYN_NPU_STATE_BUSY) {
		return -EBUSY;
	}
	if (size > NEUTRON_MAX_INPUT_SIZE) {
		return -ENOMEM;
	}

	/* TODO: Set input via Neutron SDK:
	 *   neutron_set_input(index, data, size);
	 */
	memcpy(npu.input_buf, data, size);
	npu.input_size = size;

	return 0;
}

int syn_hal_npu_invoke(void)
{
	if (!npu.initialized || !npu.model_loaded) {
		return -EPERM;
	}
	if (npu.state == SYN_NPU_STATE_BUSY) {
		return -EBUSY;
	}

	npu.state = SYN_NPU_STATE_BUSY;

	/* TODO: Replace with real Neutron SDK invoke:
	 *   int ret = neutron_invoke();
	 *   if (ret != 0) {
	 *       npu.state = SYN_NPU_STATE_ERROR;
	 *       return -EIO;
	 *   }
	 */

	/* Stub inference: deterministic 10-class classification */
	npu.output_size = 10;
	memset(npu.output_buf, 0, npu.output_size);

	uint32_t sum = 0;

	for (size_t i = 0; i < npu.input_size; i++) {
		sum += npu.input_buf[i];
	}

	uint8_t winner = sum % npu.output_size;

	npu.output_buf[winner] = 127;

	/* Simulate NPU inference latency */
	k_busy_wait(1000);

	npu.state = SYN_NPU_STATE_IDLE;

	LOG_DBG("Neutron inference complete (stub): class %u", winner);
	return 0;
}

int syn_hal_npu_invoke_async(syn_npu_done_cb_t cb, void *user_data)
{
	/* TODO: Implement async invoke using Neutron NPU interrupt */
	ARG_UNUSED(cb);
	ARG_UNUSED(user_data);

	return -ENOTSUP;
}

int syn_hal_npu_get_output(uint8_t index, void *data, size_t *size)
{
	if (data == NULL || size == NULL || index != 0) {
		return -EINVAL;
	}
	if (!npu.initialized) {
		return -EPERM;
	}
	if (npu.state == SYN_NPU_STATE_BUSY) {
		return -EBUSY;
	}

	/* TODO: Get output via Neutron SDK:
	 *   neutron_get_output(index, data, size);
	 */
	memcpy(data, npu.output_buf, npu.output_size);
	*size = npu.output_size;

	return 0;
}

int syn_hal_npu_suspend(void)
{
	if (!npu.initialized) {
		return -EPERM;
	}

	/* TODO: Gate NPU clock for power savings */
	npu.state = SYN_NPU_STATE_SUSPENDED;
	LOG_DBG("Neutron NPU suspended");
	return 0;
}

int syn_hal_npu_resume(void)
{
	if (!npu.initialized) {
		return -EPERM;
	}
	if (npu.state != SYN_NPU_STATE_SUSPENDED) {
		return -EINVAL;
	}

	/* TODO: Ungate NPU clock */
	npu.state = SYN_NPU_STATE_IDLE;
	LOG_DBG("Neutron NPU resumed");
	return 0;
}
