/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file syn_model_ota_internal.h
 * @brief SynapticOS - OTA engine internals (not public API)
 *
 * include/synaptic/syn_model_ota.h is frozen; extra introspection for
 * the shell and samples lives here.
 */
#ifndef SYNAPTIC_SYN_MODEL_OTA_INTERNAL_H_
#define SYNAPTIC_SYN_MODEL_OTA_INTERNAL_H_

#include <stdint.h>
#include <synaptic/syn_model_ota.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
	syn_ota_state_t state;
	uint8_t slot;          /* staging slot of the session (0xFF none) */
	uint32_t total_size;   /* expected .synm stream bytes */
	uint32_t received;     /* bytes accepted so far */
	uint32_t session_us;   /* begin -> latest state change */
	int last_error;        /* errno of the transition into ERROR */
} syn_ota_status_t;

void syn_ota_get_status(syn_ota_status_t *status);

/* Human-readable state name ("IDLE", "DOWNLOADING", ...) */
const char *syn_ota_state_str(syn_ota_state_t state);

/* Drop all session state back to IDLE without touching flash.
 * Test fixtures use it to emulate the RAM loss of a reboot.
 */
void syn_ota_reset(void);

#ifdef __cplusplus
}
#endif
#endif /* SYNAPTIC_SYN_MODEL_OTA_INTERNAL_H_ */
