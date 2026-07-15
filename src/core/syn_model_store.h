/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file syn_model_store.h
 * @brief SynapticOS - flash-backed model store (internal API)
 *
 * Persistence layer under the RAM model registry. Owns the two
 * registry sectors (ping-pong commit: alternating copies, monotonic
 * generation, CRC32, newest-valid-wins) and the two A/B model slots.
 *
 * The store manages one UPDATE PAIR, matching the A/B flash layout:
 * at most one active model and one non-active occupant (the rollback
 * candidate or a staged-but-not-activated update). Applications may
 * still register any number of RAM-resident models through the
 * public syn_model API; only slot-backed models persist across
 * reboot.
 *
 * Commit is the atomicity point: a power loss anywhere before a
 * complete, CRC-valid copy hits flash leaves the previous generation
 * authoritative.
 */
#ifndef SYNAPTIC_SYN_MODEL_STORE_H_
#define SYNAPTIC_SYN_MODEL_STORE_H_

#include <synaptic/syn_model.h>
#include "syn_flash_port.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SYN_STORE_SLOT_NONE 0xFFU

typedef struct {
	/* window-relative offsets */
	uint32_t registry_off[2];
	uint32_t registry_size;   /* per copy; multiple of sector size */
	uint32_t slot_off[2];
	uint32_t slot_size;       /* per slot; multiple of sector size */
} syn_store_layout_t;

/* Scan both registry copies, adopt the newest valid one, and register
 * the active model (if any) into the RAM registry with its payload
 * mapped from flash. Idempotent via syn_store_deinit().
 */
int syn_store_init(const syn_flash_port_t *port,
		   const syn_store_layout_t *layout);
void syn_store_deinit(void);
bool syn_store_ready(void);

/* One-call install: write a .synm image (synthesized header + payload)
 * into the staging slot, activate it, and register it. The previous
 * active model, if any, becomes the rollback candidate.
 */
int syn_store_install(const syn_model_info_t *info,
		      const void *data, size_t size,
		      syn_model_handle_t *handle);

/* OTA engine primitives */
int syn_store_staging_slot(uint8_t *slot);
int syn_store_slot_bounds(uint8_t slot, uint32_t *off, uint32_t *size);
int syn_store_mark_staged(uint8_t slot, const syn_model_info_t *info);
int syn_store_activate(uint8_t slot);
int syn_store_rollback(void);
int syn_store_clear_staged(void);

/* Introspection */
uint8_t syn_store_active_slot(void);
uint8_t syn_store_staged_slot(void);
int syn_store_slot_info(uint8_t slot, syn_model_info_t *info);
uint32_t syn_store_generation(void);
/* cumulative erase count of registry copy 0/1 (wear tracking) */
uint32_t syn_store_wear(uint8_t copy);
/* duration of the most recent commit / the boot scan, microseconds */
uint32_t syn_store_last_commit_us(void);
uint32_t syn_store_scan_us(void);

const syn_flash_port_t *syn_store_port(void);
const syn_store_layout_t *syn_store_layout_get(void);

#ifdef __cplusplus
}
#endif
#endif /* SYNAPTIC_SYN_MODEL_STORE_H_ */
