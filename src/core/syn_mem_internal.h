/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file syn_mem_internal.h
 * @brief SynapticOS - Arena layout introspection (private header)
 *
 * Read-only view of the arena bookkeeping for diagnostic consumers
 * (shell `syn mem dump`). Not part of the public API.
 */
#ifndef SYNAPTIC_SYN_MEM_INTERNAL_H_
#define SYNAPTIC_SYN_MEM_INTERNAL_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Snapshot of the arena region boundaries. */
typedef struct {
	const uint8_t *base;      /**< Arena base address                  */
	size_t total;             /**< Full arena size incl. scratch       */
	size_t usable;            /**< Tensor-usable size (total - scratch)*/
	size_t persistent_used;   /**< [base, base+persistent_used)        */
	size_t ephemeral_used;    /**< Follows the persistent region       */
	size_t scratch_total;     /**< Scratch pool at top of arena        */
	size_t scratch_used;      /**< Current scratch offset              */
} syn_mem_layout_t;

/**
 * @brief Fill @p layout from the live arena bookkeeping.
 * @return 0 on success, -ENODEV if the arena is not initialized.
 */
int syn_mem_get_layout(syn_mem_layout_t *layout);

#ifdef __cplusplus
}
#endif
#endif /* SYNAPTIC_SYN_MEM_INTERNAL_H_ */
