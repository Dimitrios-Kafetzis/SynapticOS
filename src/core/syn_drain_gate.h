/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file syn_drain_gate.h
 * @brief SynapticOS - Single-server drain gate (private header)
 *
 * Small synchronization helper for a server loop that must be
 * drainable: a controller can pause admission and wait until the
 * one in-flight unit of work (if any) has finished. Used by the
 * cross-core inference serve path so syn_ota_begin() never parks
 * CPU1 mid-request, and unit-testable on single-core QEMU.
 *
 * Contract: at most ONE worker is inside the gate at a time (the
 * serve side runs on the single IPC dispatch thread).
 */
#ifndef SYNAPTIC_SYN_DRAIN_GATE_H_
#define SYNAPTIC_SYN_DRAIN_GATE_H_

#include <zephyr/kernel.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
	struct k_mutex lock;
	struct k_sem idle;
	bool paused;
	bool busy;
} syn_drain_gate_t;

/** @brief Initialize the gate (open, idle). */
void syn_drain_gate_init(syn_drain_gate_t *gate);

/**
 * @brief Worker: try to enter the gate before starting work.
 * @return true to proceed (exit() must follow), false when the gate
 *         is paused (reject the work, do NOT call exit()).
 */
bool syn_drain_gate_enter(syn_drain_gate_t *gate);

/** @brief Worker: leave the gate after finishing work. */
void syn_drain_gate_exit(syn_drain_gate_t *gate);

/**
 * @brief Controller: pause admission and wait for the in-flight
 *        work to finish.
 * @return 0 once idle (admission stays paused until resume()),
 *         -ETIMEDOUT when the in-flight work outlived @p timeout_ms
 *         (admission stays paused; the worker may still be running).
 */
int syn_drain_gate_drain(syn_drain_gate_t *gate, uint32_t timeout_ms);

/** @brief Controller: reopen the gate. */
void syn_drain_gate_resume(syn_drain_gate_t *gate);

#ifdef __cplusplus
}
#endif
#endif /* SYNAPTIC_SYN_DRAIN_GATE_H_ */
