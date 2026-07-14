/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file syn_ipc_ring.c
 * @brief SynapticOS - Lock-free SPSC ring over shared SRAM
 *
 * Indices are free-running 32-bit counters: slot = index % entries,
 * count = head - tail (well-defined across wraparound with unsigned
 * arithmetic). Only the producer writes head, only the consumer
 * writes tail, so neither index needs atomic read-modify-write.
 *
 * Memory ordering (barrier_dmem_fence_full() emits DMB on ARM):
 *  - producer: fill slot, DMB, publish head. The consumer can only
 *    observe the new head after the slot contents are visible.
 *  - consumer: read head, DMB, read slot, DMB, publish tail. The
 *    slot is read only after head confirmed it, and the producer
 *    can only observe the freed slot after the copy-out completed.
 */

#include <errno.h>
#include <zephyr/sys/barrier.h>

#include "syn_ipc_ring.h"

void syn_ipc_ring_reset(syn_ipc_ring_t *ring)
{
	ring->head = 0;
	ring->tail = 0;
	barrier_dmem_fence_full();
}

int syn_ipc_ring_push(syn_ipc_ring_t *ring, const syn_ipc_msg_t *msg)
{
	uint32_t head = ring->head;

	if (head - ring->tail >= SYN_IPC_RING_ENTRIES) {
		return -EAGAIN;
	}

	ring->slots[head % SYN_IPC_RING_ENTRIES] = *msg;
	barrier_dmem_fence_full();

	ring->head = head + 1;
	barrier_dmem_fence_full();

	return 0;
}

int syn_ipc_ring_pop(syn_ipc_ring_t *ring, syn_ipc_msg_t *msg)
{
	uint32_t tail = ring->tail;

	if (ring->head == tail) {
		return -EAGAIN;
	}
	barrier_dmem_fence_full();

	*msg = ring->slots[tail % SYN_IPC_RING_ENTRIES];
	barrier_dmem_fence_full();

	ring->tail = tail + 1;
	barrier_dmem_fence_full();

	return 0;
}

bool syn_ipc_ring_is_empty(const syn_ipc_ring_t *ring)
{
	return ring->head == ring->tail;
}

bool syn_ipc_ring_is_full(const syn_ipc_ring_t *ring)
{
	return (ring->head - ring->tail) >= SYN_IPC_RING_ENTRIES;
}

uint32_t syn_ipc_ring_count(const syn_ipc_ring_t *ring)
{
	return ring->head - ring->tail;
}
