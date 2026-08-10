/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file syn_model_ota.h
 * @brief SynapticOS — Over-The-Air Model Updates
 *
 * Power-loss-safe staged model updates into the flash model store:
 * begin() -> write_chunk()... -> finish() stages and CRC-verifies the
 * image; activate() commits it via the ping-pong registry; rollback()
 * returns to the previous model. A staged image survives reboot and
 * an interrupted transfer never corrupts the active model.
 */
#ifndef SYNAPTIC_SYN_MODEL_OTA_H_
#define SYNAPTIC_SYN_MODEL_OTA_H_

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/** OTA session state. */
typedef enum {
    SYN_OTA_STATE_IDLE,          /**< No session in progress          */
    SYN_OTA_STATE_DOWNLOADING,   /**< Accepting chunks                */
    SYN_OTA_STATE_VALIDATING,    /**< Verifying the received image    */
    SYN_OTA_STATE_STAGING,       /**< Writing staging metadata        */
    SYN_OTA_STATE_READY,         /**< Staged; activate() may commit   */
    SYN_OTA_STATE_ERROR,         /**< Failed; begin() to start over   */
} syn_ota_state_t;

/**
 * @brief Start an OTA session for a model.
 *
 * Drains in-flight remote inference work before touching session
 * state, erases the staging slot and enters DOWNLOADING.
 *
 * @param model_name Name the staged model will carry.
 * @param total_size Exact image size that will be streamed, in bytes.
 *
 * @return 0 on success, -EBUSY if a session is active or the drain
 *         timed out, -EFBIG if the image exceeds the slot, negative
 *         errno otherwise.
 */
int  syn_ota_begin(const char *model_name, size_t total_size);

/**
 * @brief Stream the next chunk of the model image.
 *
 * Chunk size is arbitrary; chunks are written to the staging slot in
 * order.
 *
 * @param data Chunk bytes.
 * @param len  Chunk length in bytes.
 *
 * @return 0 on success, -EPERM outside DOWNLOADING, negative errno on
 *         write failure (session enters ERROR).
 */
int  syn_ota_write_chunk(const uint8_t *data, size_t len);

/**
 * @brief Finish the transfer: verify size and CRC, stage the image.
 *
 * On success the session is READY and the staged image survives
 * reboot until activated or replaced.
 *
 * @return 0 on success, -EPERM outside DOWNLOADING, -EILSEQ on CRC
 *         mismatch, negative errno otherwise.
 */
int  syn_ota_finish(void);

/**
 * @brief Commit the staged model as the active one.
 *
 * Registry commit is atomic (ping-pong, generation-numbered). Also
 * accepts a staged slot left by a previous boot (power-loss
 * recovery from IDLE).
 *
 * @return 0 on success, negative errno if nothing is staged or the
 *         commit fails.
 */
int  syn_ota_activate(void);

/**
 * @brief Return to the previous model (the other registry slot).
 *
 * @return 0 on success, -EBUSY while a transfer is in progress,
 *         negative errno if there is no rollback candidate.
 */
int  syn_ota_rollback(void);

/**
 * @brief Current OTA session state.
 *
 * @return The session state (see syn_ota_state_t).
 */
syn_ota_state_t syn_ota_get_state(void);

#ifdef __cplusplus
}
#endif
#endif /* SYNAPTIC_SYN_MODEL_OTA_H_ */
