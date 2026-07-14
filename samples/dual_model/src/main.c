/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file main.c
 * @brief SynapticOS - dual_model sample, CPU0 (AI runtime core)
 *
 * Boots the SynapticOS runtime, initializes IPC over the shared
 * region, releases CPU1 and completes the STATUS_REQ/RESP handshake.
 * If CPU1 does not respond the demo continues single-core.
 *
 * CPU1 runs the companion application in samples/dual_model/remote,
 * flashed separately to flash bank 1.
 */

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <synaptic/syn_api.h>

LOG_MODULE_REGISTER(dual_model, LOG_LEVEL_INF);

#ifdef CONFIG_SYNAPTIC_DUAL_CORE

#include <zephyr/devicetree.h>
#include "syn_boot_internal.h"

#define SYN_SHARED_NODE DT_NODELABEL(syn_shared)

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

	ret = syn_ipc_init(shm_base, shm_size);
	if (ret != 0) {
		LOG_ERR("IPC init failed: %d", ret);
		return ret;
	}

	ret = syn_boot_secondary(shm_base, 200);
	if (ret != 0) {
		LOG_WRN("CPU1 not responding (%d): single-core mode", ret);
	} else {
		LOG_INF("Dual-core up: CPU1 boot %u us, handshake %u us",
			syn_boot_cpu1_boot_us(), syn_boot_handshake_us());
	}

	/* Shell owns the console from here; CPU1 heartbeats keep
	 * arriving via the STATUS_REQ auto-responder. Check with:
	 * syn ipc status
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
