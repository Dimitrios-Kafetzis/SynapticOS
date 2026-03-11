/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file syn_model_ota.h
 * @brief SynapticOS — Over-The-Air Model Updates
 */
#ifndef SYNAPTIC_SYN_MODEL_OTA_H_
#define SYNAPTIC_SYN_MODEL_OTA_H_

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    SYN_OTA_STATE_IDLE,
    SYN_OTA_STATE_DOWNLOADING,
    SYN_OTA_STATE_VALIDATING,
    SYN_OTA_STATE_STAGING,
    SYN_OTA_STATE_READY,
    SYN_OTA_STATE_ERROR,
} syn_ota_state_t;

int  syn_ota_begin(const char *model_name, size_t total_size);
int  syn_ota_write_chunk(const uint8_t *data, size_t len);
int  syn_ota_finish(void);
int  syn_ota_activate(void);
int  syn_ota_rollback(void);
syn_ota_state_t syn_ota_get_state(void);

#ifdef __cplusplus
}
#endif
#endif /* SYNAPTIC_SYN_MODEL_OTA_H_ */
