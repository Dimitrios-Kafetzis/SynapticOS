/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file syn_model.h
 * @brief SynapticOS — Model Lifecycle Management
 *
 * Registry of models known to the runtime (metadata + flash location),
 * plus load/unload onto the NPU and CRC-gated flash reads. Handles are
 * small opaque integers; SYN_MODEL_INVALID is never a valid handle.
 */
#ifndef SYNAPTIC_SYN_MODEL_H_
#define SYNAPTIC_SYN_MODEL_H_

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <synaptic/syn_hal_npu.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Opaque model handle (0 == SYN_MODEL_INVALID). */
typedef uint32_t syn_model_handle_t;

#define SYN_MODEL_INVALID  ((syn_model_handle_t)0)

/** Model metadata as registered with the runtime. */
typedef struct {
    char             name[32];        /**< Unique model name            */
    char             version[16];     /**< Version string               */
    uint32_t         input_size;      /**< Input size in bytes          */
    uint32_t         output_size;     /**< Output size in bytes         */
    uint32_t         flash_offset;    /**< Model image offset in flash  */
    uint32_t         flash_size;      /**< Model image size in bytes    */
    uint32_t         sram_required;   /**< Arena bytes needed to load   */
    uint32_t         crc32;           /**< CRC-32 of the flash image    */
    syn_npu_dtype_t  input_dtype;     /**< Input element type           */
    syn_npu_dtype_t  output_dtype;    /**< Output element type          */
    uint8_t          input_shape[4];  /**< Input shape [b, h, w, c]     */
    uint8_t          output_shape[4]; /**< Output shape [b, h, w, c]    */
} syn_model_info_t;

/* Registry operations */

/**
 * @brief Register a model with the runtime.
 *
 * @param info   Model metadata; copied by the call.
 * @param handle Out: handle for subsequent operations.
 *
 * @return 0 on success, -EINVAL on bad arguments, -EEXIST if the name
 *         is already registered, -ENOMEM if the registry is full.
 */
int  syn_model_register(const syn_model_info_t *info,
                        syn_model_handle_t *handle);

/**
 * @brief Remove a model from the registry.
 *
 * The model must not be loaded.
 *
 * @param handle Handle from syn_model_register().
 *
 * @return 0 on success, -EINVAL on bad handle, negative errno
 *         otherwise.
 */
int  syn_model_unregister(syn_model_handle_t handle);

/**
 * @brief Read back a registered model's metadata.
 *
 * @param handle Model handle.
 * @param info   Out: filled with the registered metadata.
 *
 * @return 0 on success, -EINVAL on bad arguments.
 */
int  syn_model_get_info(syn_model_handle_t handle, syn_model_info_t *info);

/**
 * @brief Look up a model handle by name.
 *
 * @param name   Model name (as registered).
 * @param handle Out: matching handle.
 *
 * @return 0 on success, -EINVAL on bad arguments, -ENOENT if no
 *         model has that name.
 */
int  syn_model_get_by_name(const char *name, syn_model_handle_t *handle);

/**
 * @brief List registered model handles.
 *
 * @param handles Out: array receiving up to @p max handles.
 * @param count   Out: number of handles written.
 * @param max     Capacity of @p handles.
 *
 * @return 0 on success, -EINVAL on bad arguments.
 */
int  syn_model_list(syn_model_handle_t *handles, uint8_t *count,
                    uint8_t max);

/* Loading */

/**
 * @brief Load a registered model onto the NPU.
 *
 * Reads the image from flash, verifies its CRC-32, and hands it to
 * the NPU backend. Fails with -EILSEQ on CRC mismatch.
 *
 * @param handle Model handle.
 *
 * @return 0 on success, -EALREADY if already loaded, -EILSEQ on CRC
 *         mismatch, negative errno otherwise.
 */
int  syn_model_load(syn_model_handle_t handle);

/**
 * @brief Unload a loaded model from the NPU.
 *
 * @param handle Model handle.
 *
 * @return 0 on success, -EALREADY if not loaded, negative errno
 *         otherwise.
 */
int  syn_model_unload(syn_model_handle_t handle);

/**
 * @brief Whether a model is currently loaded on the NPU.
 *
 * @param handle Model handle.
 *
 * @return true if loaded, false otherwise (including bad handles).
 */
bool syn_model_is_loaded(syn_model_handle_t handle);

/* Hot-swap */

/**
 * @brief Atomically replace one loaded model with another.
 *
 * Quiesces in-flight work on @p old_handle, unloads it and loads
 * @p new_handle in its place.
 *
 * @param old_handle Currently loaded model.
 * @param new_handle Replacement model.
 *
 * @return 0 on success, negative errno on failure (the old model
 *         remains authoritative on error).
 */
int  syn_model_swap(syn_model_handle_t old_handle,
                    syn_model_handle_t new_handle);

#ifdef __cplusplus
}
#endif
#endif /* SYNAPTIC_SYN_MODEL_H_ */
