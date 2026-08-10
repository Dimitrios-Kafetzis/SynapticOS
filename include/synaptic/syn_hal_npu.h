/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file syn_hal_npu.h
 * @brief SynapticOS — NPU Hardware Abstraction Layer
 *
 * Backend-neutral NPU session API. On FRDM-MCXN947 the backend targets
 * the eIQ Neutron NPU (placeholder invoke until the Neutron SDK module
 * lands); on QEMU a stub backend provides deterministic synthetic
 * execution. One model is resident at a time.
 */
#ifndef SYNAPTIC_SYN_HAL_NPU_H_
#define SYNAPTIC_SYN_HAL_NPU_H_

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/** NPU execution state. */
typedef enum {
    SYN_NPU_STATE_IDLE,        /**< Ready, no invoke in flight        */
    SYN_NPU_STATE_BUSY,        /**< Invoke in progress                */
    SYN_NPU_STATE_ERROR,       /**< Faulted; re-init required         */
    SYN_NPU_STATE_SUSPENDED,   /**< Power-suspended via suspend()     */
} syn_npu_state_t;

/** Tensor element data types supported by NPU backends. */
typedef enum {
    SYN_NPU_DTYPE_INT8,
    SYN_NPU_DTYPE_UINT8,
    SYN_NPU_DTYPE_INT16,
    SYN_NPU_DTYPE_FLOAT16,
    SYN_NPU_DTYPE_FLOAT32,
} syn_npu_dtype_t;

/** Static capabilities reported by the active NPU backend. */
typedef struct {
    const char *name;             /**< Backend name (static string)     */
    uint32_t    max_ops_per_sec;  /**< Peak throughput estimate         */
    uint32_t    scratch_size;     /**< Backend scratch requirement, B   */
    uint8_t     supported_dtypes; /**< Bitmask of syn_npu_dtype_t bits  */
    bool        supports_async;   /**< invoke_async() available         */
} syn_npu_caps_t;

/**
 * @brief Asynchronous-invoke completion callback.
 *
 * @param status    0 on success, negative errno on failure.
 * @param user_data Pointer registered with syn_hal_npu_invoke_async().
 */
typedef void (*syn_npu_done_cb_t)(int status, void *user_data);

/* Lifecycle */

/**
 * @brief Initialize the NPU backend.
 *
 * @return 0 on success, negative errno on failure.
 */
int  syn_hal_npu_init(void);

/**
 * @brief Release the NPU backend and any resident model.
 */
void syn_hal_npu_deinit(void);

/**
 * @brief Query backend capabilities.
 *
 * @param caps Out: filled with the active backend's capabilities.
 *
 * @return 0 on success, negative errno on failure.
 */
int  syn_hal_npu_get_caps(syn_npu_caps_t *caps);

/**
 * @brief Current NPU execution state.
 *
 * @return The backend state (see syn_npu_state_t).
 */
syn_npu_state_t syn_hal_npu_get_state(void);

/* Execution */

/**
 * @brief Load a model blob into the NPU, replacing any resident model.
 *
 * @param model_data Model image (backend-specific format).
 * @param model_size Image size in bytes.
 *
 * @return 0 on success, negative errno on bad image or size.
 */
int  syn_hal_npu_load_model(const uint8_t *model_data, size_t model_size);

/**
 * @brief Stage input data for the resident model.
 *
 * @param index Input tensor index (0-based).
 * @param data  Input bytes; copied/staged by the backend.
 * @param size  Input size in bytes.
 *
 * @return 0 on success, negative errno on bad index/size or no model.
 */
int  syn_hal_npu_set_input(uint8_t index, const void *data, size_t size);

/**
 * @brief Run the resident model to completion (blocking).
 *
 * @return 0 on success, negative errno on failure.
 */
int  syn_hal_npu_invoke(void);

/**
 * @brief Start an asynchronous invoke of the resident model.
 *
 * Requires caps.supports_async. @p cb fires on completion and may run
 * in interrupt or worker context.
 *
 * @param cb        Completion callback.
 * @param user_data Opaque pointer passed to @p cb.
 *
 * @return 0 if started, negative errno on failure or unsupported.
 */
int  syn_hal_npu_invoke_async(syn_npu_done_cb_t cb, void *user_data);

/**
 * @brief Read back an output tensor after a completed invoke.
 *
 * @param index Output tensor index (0-based).
 * @param data  Out: buffer receiving the output bytes.
 * @param size  In: capacity of @p data; out: bytes written.
 *
 * @return 0 on success, negative errno on bad index or short buffer.
 */
int  syn_hal_npu_get_output(uint8_t index, void *data, size_t *size);

/* Power management */

/**
 * @brief Suspend the NPU (state becomes SYN_NPU_STATE_SUSPENDED).
 *
 * @return 0 on success, negative errno on failure.
 */
int  syn_hal_npu_suspend(void);

/**
 * @brief Resume a suspended NPU.
 *
 * @return 0 on success, negative errno on failure.
 */
int  syn_hal_npu_resume(void);

#ifdef __cplusplus
}
#endif
#endif /* SYNAPTIC_SYN_HAL_NPU_H_ */
