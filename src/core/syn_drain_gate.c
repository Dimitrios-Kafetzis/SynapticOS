/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file syn_drain_gate.c
 * @brief SynapticOS - Single-server drain gate
 *
 * See syn_drain_gate.h for the contract.
 */

#include <errno.h>

#include "syn_drain_gate.h"

void syn_drain_gate_init(syn_drain_gate_t *gate)
{
	k_mutex_init(&gate->lock);
	k_sem_init(&gate->idle, 0, 1);
	gate->paused = false;
	gate->busy = false;
}

bool syn_drain_gate_enter(syn_drain_gate_t *gate)
{
	bool admitted;

	k_mutex_lock(&gate->lock, K_FOREVER);
	admitted = !gate->paused;
	if (admitted) {
		gate->busy = true;
	}
	k_mutex_unlock(&gate->lock);
	return admitted;
}

void syn_drain_gate_exit(syn_drain_gate_t *gate)
{
	bool notify;

	k_mutex_lock(&gate->lock, K_FOREVER);
	gate->busy = false;
	notify = gate->paused;
	k_mutex_unlock(&gate->lock);

	if (notify) {
		k_sem_give(&gate->idle);
	}
}

int syn_drain_gate_drain(syn_drain_gate_t *gate, uint32_t timeout_ms)
{
	bool busy;

	k_mutex_lock(&gate->lock, K_FOREVER);
	gate->paused = true;
	k_sem_reset(&gate->idle);
	busy = gate->busy;
	k_mutex_unlock(&gate->lock);

	if (!busy) {
		return 0;
	}
	if (k_sem_take(&gate->idle, K_MSEC(timeout_ms)) != 0) {
		return -ETIMEDOUT;
	}
	return 0;
}

void syn_drain_gate_resume(syn_drain_gate_t *gate)
{
	k_mutex_lock(&gate->lock, K_FOREVER);
	gate->paused = false;
	k_mutex_unlock(&gate->lock);
}
