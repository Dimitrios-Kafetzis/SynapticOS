/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file syn_model_internal.h
 * @brief SynapticOS - model registry internals (not public API)
 *
 * Hooks the persistent model store needs into the RAM registry.
 * include/synaptic/syn_model.h is frozen; these stay internal.
 */
#ifndef SYNAPTIC_SYN_MODEL_INTERNAL_H_
#define SYNAPTIC_SYN_MODEL_INTERNAL_H_

#include <synaptic/syn_model.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Attach the raw model payload to a registered model. The pointer
 * must stay valid for the lifetime of the registration (flash XIP
 * alias or static buffer). Passing NULL detaches.
 */
int syn_model_set_data(syn_model_handle_t handle,
		       const uint8_t *data, size_t size);

/* Drop every registration (test fixtures and store re-init). */
void syn_model_reset_all(void);

/* Duration of the most recent syn_model_swap(), microseconds
 * (includes waiting out the in-flight inference).
 */
uint32_t syn_model_last_swap_us(void);

#ifdef __cplusplus
}
#endif
#endif /* SYNAPTIC_SYN_MODEL_INTERNAL_H_ */
