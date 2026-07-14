/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file syn_boot.c
 * @brief SynapticOS - Dual-core boot sequence (CPU0 primary)
 *
 * CPU0 owns CPU1 startup:
 *  1. CPU0 boots (Zephyr default), runs syn_init() + syn_ipc_init(),
 *     which publishes the shared control block (magic + cpu0_ready).
 *  2. syn_boot_secondary() points SYSCON CPBOOT at the CPU1 image in
 *     flash bank 1 and toggles CPU1 clock/reset in SYSCON CPUCTRL
 *     (keyed writes, per NXP's boot_multicore_slave reference).
 *  3. CPU1 boots its own Zephyr kernel, attaches to the shared region
 *     (validating magic and layout), sets cpu1_ready, and sends
 *     SYN_IPC_STATUS_REQ.
 *  4. CPU0's auto-responder answers with SYN_IPC_STATUS_RESP; the
 *     link is up.
 *
 * If CPU1 never responds (no image flashed, corrupted image), CPU0
 * logs the failure and continues single-core: nothing in the runtime
 * depends on CPU1 being alive.
 *
 * Compiled only for CPU0 dual-core builds.
 */

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>

#include <synaptic/syn_ipc.h>

#include "syn_shared_layout.h"
#include "syn_boot_internal.h"

#if !defined(CONFIG_SOC_MCXN947_CPU1)

#include <fsl_device_registers.h>

LOG_MODULE_REGISTER(syn_boot, CONFIG_SYNAPTIC_LOG_LEVEL);

/* SYSCON->CPUCTRL writes must carry this key in bits 31:16 */
#define SYN_CPUCTRL_KEY 0xC0C40000UL

static volatile syn_shm_ctrl_t *shm_ctrl;
static atomic_t status_req_count;
static atomic_t link_up;
static uint32_t cpu1_boot_us;
static uint32_t handshake_us;

static void status_req_handler(const syn_ipc_msg_t *msg, void *ctx)
{
	ARG_UNUSED(ctx);

	syn_ipc_msg_t resp = {
		.msg_id = 0, /* auto-assigned */
		.type = SYN_IPC_STATUS_RESP,
		.priority = msg->priority,
		.payload_len = 0,
		.payload_offset = msg->msg_id, /* echo for correlation */
		.status = 0,
	};

	atomic_inc(&status_req_count);
	atomic_set(&link_up, 1);

	if (syn_ipc_send(&resp) != 0) {
		LOG_WRN("STATUS_RESP send failed (ring full)");
	}
}

int syn_boot_secondary(void *shared_base, uint32_t timeout_ms)
{
	if (shared_base == NULL || timeout_ms == 0U) {
		return -EINVAL;
	}

	shm_ctrl = &((syn_shm_region_t *)shared_base)->ctrl;

	if (shm_ctrl->magic != SYN_SHM_MAGIC || shm_ctrl->cpu0_ready != 1U) {
		LOG_ERR("IPC not initialized; call syn_ipc_init() first");
		return -EINVAL;
	}

	int ret = syn_ipc_register_handler(SYN_IPC_STATUS_REQ,
					   status_req_handler, NULL);
	if (ret != 0) {
		return ret;
	}

	LOG_INF("Releasing CPU1 (vector table at 0x%08lx)",
		(unsigned long)SYN_BOOT_CPU1_VECTOR);

	uint32_t t_release = k_cycle_get_32();

	/* NXP boot_multicore_slave sequence: boot address, then keyed
	 * clock-enable + reset-assert, then keyed reset-release.
	 */
	SYSCON0->CPBOOT = SYN_BOOT_CPU1_VECTOR;

	uint32_t ctrl = SYSCON0->CPUCTRL | SYN_CPUCTRL_KEY;

	SYSCON0->CPUCTRL = ctrl | SYSCON_CPUCTRL_CPU1CLKEN_MASK |
			   SYSCON_CPUCTRL_CPU1RSTEN_MASK;
	SYSCON0->CPUCTRL = (ctrl | SYSCON_CPUCTRL_CPU1CLKEN_MASK) &
			   ~SYSCON_CPUCTRL_CPU1RSTEN_MASK;

	/* Phase 1: CPU1 kernel up = ready flag in the control block */
	uint32_t waited_us = 0;
	uint32_t budget_us = timeout_ms * 1000U;

	while (shm_ctrl->cpu1_ready != 1U && waited_us < budget_us) {
		k_busy_wait(100);
		waited_us += 100;
	}

	if (shm_ctrl->cpu1_ready != 1U) {
		LOG_ERR("CPU1 did not boot within %u ms; continuing single-core",
			timeout_ms);
		return -ETIMEDOUT;
	}

	cpu1_boot_us = k_cyc_to_us_floor32(k_cycle_get_32() - t_release);
	LOG_INF("CPU1 ready flag after %u us", cpu1_boot_us);

	/* Phase 2: first STATUS_REQ answered = handshake complete */
	while (atomic_get(&link_up) == 0 && waited_us < budget_us) {
		k_sleep(K_MSEC(1));
		waited_us += 1000;
	}

	if (atomic_get(&link_up) == 0) {
		LOG_ERR("CPU1 booted but no STATUS_REQ within %u ms; "
			"continuing single-core", timeout_ms);
		return -ETIMEDOUT;
	}

	handshake_us = k_cyc_to_us_floor32(k_cycle_get_32() - t_release);
	LOG_INF("IPC handshake complete %u us after release", handshake_us);
	return 0;
}

bool syn_boot_secondary_linked(void)
{
	return atomic_get(&link_up) == 1;
}

uint32_t syn_boot_cpu1_boot_us(void)
{
	return cpu1_boot_us;
}

uint32_t syn_boot_handshake_us(void)
{
	return handshake_us;
}

uint32_t syn_boot_status_req_count(void)
{
	return (uint32_t)atomic_get(&status_req_count);
}

#endif /* !CONFIG_SOC_MCXN947_CPU1 */
