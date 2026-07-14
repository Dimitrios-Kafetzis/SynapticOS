/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file syn_ipc_internal.h
 * @brief SynapticOS - IPC internals shared between core modules
 */
#ifndef SYNAPTIC_SYN_IPC_INTERNAL_H_
#define SYNAPTIC_SYN_IPC_INTERNAL_H_

#include "syn_shared_layout.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Shared region this core attached to via syn_ipc_init().
 * @return Region pointer, or NULL before init.
 */
syn_shm_region_t *syn_ipc_region(void);

#ifdef __cplusplus
}
#endif
#endif /* SYNAPTIC_SYN_IPC_INTERNAL_H_ */
