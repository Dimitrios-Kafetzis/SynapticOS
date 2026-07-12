/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file syn_api.h
 * @brief SynapticOS — Top-level Application API
 *
 * This is the primary header for application developers. Include this
 * single header to access all SynapticOS functionality.
 */

#ifndef SYNAPTIC_SYN_API_H_
#define SYNAPTIC_SYN_API_H_

#include <synaptic/syn_mem.h>
#include <synaptic/syn_model.h>
#include <synaptic/syn_infer.h>
#include <synaptic/syn_prof.h>
#include <synaptic/syn_hal_npu.h>
#include <synaptic/syn_hal_dsp.h>

#ifdef CONFIG_SYNAPTIC_DUAL_CORE
#include <synaptic/syn_ipc.h>
#endif

#ifdef CONFIG_SYNAPTIC_OTA
#include <synaptic/syn_model_ota.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/** SynapticOS version */
#define SYNAPTIC_VERSION_MAJOR  0
#define SYNAPTIC_VERSION_MINOR  2
#define SYNAPTIC_VERSION_PATCH  0
#define SYNAPTIC_VERSION_STRING "0.2.0"

/**
 * @brief Initialize the SynapticOS runtime.
 *
 * Must be called once at startup before any other syn_* function.
 * Initializes the tensor arena, HAL drivers, model registry,
 * and inference scheduler.
 *
 * @return 0 on success, negative errno on failure.
 */
int syn_init(void);

/**
 * @brief Shut down the SynapticOS runtime.
 *
 * Cancels all pending jobs, unloads models, and releases resources.
 *
 * @return 0 on success, negative errno on failure.
 */
int syn_shutdown(void);

/**
 * @brief Get SynapticOS runtime version string.
 *
 * @return Null-terminated version string (e.g., "0.1.0").
 */
const char *syn_version(void);

#ifdef __cplusplus
}
#endif

#endif /* SYNAPTIC_SYN_API_H_ */
