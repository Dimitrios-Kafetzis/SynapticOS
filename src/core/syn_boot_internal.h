/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file syn_boot_internal.h
 * @brief SynapticOS - Dual-core boot sequence, CPU0 side (private)
 */
#ifndef SYNAPTIC_SYN_BOOT_INTERNAL_H_
#define SYNAPTIC_SYN_BOOT_INTERNAL_H_

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Release CPU1 from reset and wait for the IPC handshake.
 *
 * Preconditions: syn_ipc_init() has completed on this core, so the
 * shared control block is published. Registers the STATUS_REQ
 * auto-responder, points SYSCON CPBOOT at the CPU1 image in flash
 * bank 1, releases the core, then waits for (a) CPU1's ready flag
 * and (b) the first STATUS_REQ message.
 *
 * On timeout the system stays fully functional in single-core mode;
 * the caller just carries on without CPU1.
 *
 * @param shared_base Base of the shared region (same as syn_ipc_init).
 * @param timeout_ms  Total budget for flag + handshake.
 * @return 0 on success, -ETIMEDOUT if CPU1 did not respond,
 *         -EINVAL on bad arguments.
 */
int syn_boot_secondary(void *shared_base, uint32_t timeout_ms);

/**
 * @brief Hold CPU1 in reset (clock gated).
 *
 * Used by the OTA engine before erasing or programming flash bank 1:
 * CPU1 executes in place from that bank, and flash operations there
 * disturb its instruction fetches. The shared-region ready flag and
 * link state are cleared so a later resume re-runs the handshake.
 * Only call with the cross-core offload quiescent (no in-flight
 * remote jobs); messages still queued in the rings are handled late,
 * after resume.
 *
 * @return 0 on success, -ENODEV if CPU1 was never booted.
 */
int syn_boot_secondary_stop(void);

/**
 * @brief Re-release CPU1 after syn_boot_secondary_stop().
 *
 * Re-runs the release sequence and waits for the ready flag and the
 * STATUS_REQ handshake, like syn_boot_secondary(), but skips the
 * blank check (the CPU1 image reserve is never touched by OTA) and
 * does not re-register the auto-responder.
 *
 * @return 0 on success, -ENODEV if never booted, -ETIMEDOUT on
 *         handshake timeout.
 */
int syn_boot_secondary_resume(uint32_t timeout_ms);

/** True once the STATUS_REQ/RESP handshake has completed. */
bool syn_boot_secondary_linked(void);

/** Microseconds from reset release to CPU1's ready flag (0 if never). */
uint32_t syn_boot_cpu1_boot_us(void);

/** Microseconds from reset release to the first STATUS_REQ (0 if never). */
uint32_t syn_boot_handshake_us(void);

/** Number of STATUS_REQ messages answered so far. */
uint32_t syn_boot_status_req_count(void);

#ifdef __cplusplus
}
#endif
#endif /* SYNAPTIC_SYN_BOOT_INTERNAL_H_ */
