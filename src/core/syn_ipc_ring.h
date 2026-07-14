/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file syn_ipc_ring.h
 * @brief SynapticOS - Lock-free SPSC ring operations (private header)
 *
 * Pure single-producer single-consumer ring logic over the shared
 * syn_ipc_ring_t structure. No locks: correctness comes from each
 * index having exactly one writer plus data memory barriers (DMB)
 * ordering slot contents against index publication.
 *
 * This file has no dual-core dependency on purpose: the same code
 * runs cross-core on the MCXN947 (producer and consumer on different
 * CPUs) and single-core on qemu_cortex_m3 where the unit tests drive
 * both roles.
 */
#ifndef SYNAPTIC_SYN_IPC_RING_H_
#define SYNAPTIC_SYN_IPC_RING_H_

#include <stdbool.h>
#include "syn_shared_layout.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Reset a ring to empty. Call from the initializing core only. */
void syn_ipc_ring_reset(syn_ipc_ring_t *ring);

/**
 * @brief Push a message. Producer side only.
 * @return 0 on success, -EAGAIN if the ring is full.
 */
int syn_ipc_ring_push(syn_ipc_ring_t *ring, const syn_ipc_msg_t *msg);

/**
 * @brief Pop the oldest message. Consumer side only.
 * @return 0 on success, -EAGAIN if the ring is empty.
 */
int syn_ipc_ring_pop(syn_ipc_ring_t *ring, syn_ipc_msg_t *msg);

/** True when the ring holds no messages. */
bool syn_ipc_ring_is_empty(const syn_ipc_ring_t *ring);

/** True when a push would return -EAGAIN. */
bool syn_ipc_ring_is_full(const syn_ipc_ring_t *ring);

/** Number of messages currently queued. */
uint32_t syn_ipc_ring_count(const syn_ipc_ring_t *ring);

#ifdef __cplusplus
}
#endif
#endif /* SYNAPTIC_SYN_IPC_RING_H_ */
