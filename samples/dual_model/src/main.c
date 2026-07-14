/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file main.c
 * @brief SynapticOS - dual_model sample, CPU0 (AI runtime core)
 *
 * The AI-runtime side of the asymmetric dual-core demo:
 *  1. boots the SynapticOS runtime
 *  2. registers two models: face_detect (96x96x3 vision) and
 *     keyword_spot (49x10 MFCC audio)
 *  3. initializes IPC over the shared region and starts serving
 *     cross-core inference requests
 *  4. releases CPU1 and completes the STATUS_REQ/RESP handshake
 *
 * CPU1 (samples/dual_model/remote, flashed to flash bank 1) then
 * alternates inference requests between the two models over IPC.
 * If CPU1 does not respond, this core continues single-core with
 * the full shell available.
 */

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <synaptic/syn_api.h>
#include <string.h>

LOG_MODULE_REGISTER(dual_model, LOG_LEVEL_INF);

#ifdef CONFIG_SYNAPTIC_DUAL_CORE

#include <zephyr/devicetree.h>
#include "syn_boot_internal.h"
#include "syn_infer_remote.h"

#define SYN_SHARED_NODE DT_NODELABEL(syn_shared)

/* Dummy model binary: the Neutron NPU HAL runs its stub path until
 * real compiled models arrive in a later phase (honest labeling:
 * latencies measured with this blob exercise the full runtime path
 * but not real NPU math).
 */
static const uint8_t dummy_model[64] = {0};

static int register_model(const char *name, uint32_t input_size,
			  uint32_t output_size)
{
	syn_model_info_t info = {0};
	syn_model_handle_t handle;
	int ret;

	strncpy(info.name, name, sizeof(info.name) - 1);
	strncpy(info.version, "1.0.0", sizeof(info.version) - 1);
	info.input_size = input_size;
	info.output_size = output_size;
	info.flash_size = sizeof(dummy_model);
	info.sram_required = 4096;
	info.input_dtype = SYN_NPU_DTYPE_INT8;
	info.output_dtype = SYN_NPU_DTYPE_INT8;

	ret = syn_model_register(&info, &handle);
	if (ret != 0) {
		LOG_ERR("Register '%s' failed: %d", name, ret);
		return ret;
	}

	syn_model_load(handle);
	LOG_INF("Model '%s' ready (handle %u, in %u out %u)",
		name, handle, input_size, output_size);
	return 0;
}

int main(void)
{
	int ret;
	void *shm_base = (void *)DT_REG_ADDR(SYN_SHARED_NODE);
	size_t shm_size = DT_REG_SIZE(SYN_SHARED_NODE);

	LOG_INF("SynapticOS %s dual_model (CPU0, AI runtime)",
		syn_version());

	ret = syn_init();
	if (ret != 0) {
		LOG_ERR("Runtime init failed: %d", ret);
		return ret;
	}

	/* Two models for CPU1 to alternate between */
	ret = register_model("face_detect", 96 * 96 * 3, 144);
	if (ret != 0) {
		return ret;
	}
	ret = register_model("keyword_spot", 49 * 10, 12);
	if (ret != 0) {
		return ret;
	}

	ret = syn_hal_npu_load_model(dummy_model, sizeof(dummy_model));
	if (ret != 0) {
		LOG_ERR("NPU load failed: %d", ret);
		return ret;
	}

	ret = syn_ipc_init(shm_base, shm_size);
	if (ret != 0) {
		LOG_ERR("IPC init failed: %d", ret);
		return ret;
	}

	ret = syn_remote_serve_init();
	if (ret != 0) {
		LOG_ERR("Inference serving init failed: %d", ret);
		return ret;
	}

	ret = syn_boot_secondary(shm_base, 200);
	if (ret != 0) {
		LOG_WRN("CPU1 not responding (%d): single-core mode", ret);
	} else {
		LOG_INF("Dual-core up: CPU1 boot %u us, handshake %u us",
			syn_boot_cpu1_boot_us(), syn_boot_handshake_us());
	}

	/* Shell owns the console from here. CPU1 traffic is served by
	 * the IPC dispatch thread; check with: syn ipc status
	 */
	return 0;
}

#else /* !CONFIG_SYNAPTIC_DUAL_CORE */

int main(void)
{
	LOG_INF("dual_model requires the FRDM-MCXN947 dual-core build");
	return 0;
}

#endif /* CONFIG_SYNAPTIC_DUAL_CORE */
