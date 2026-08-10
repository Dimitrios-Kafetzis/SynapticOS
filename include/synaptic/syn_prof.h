/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file syn_prof.h
 * @brief SynapticOS — Profiling and Diagnostics
 *
 * Per-inference stage timing (pre-process / NPU / post-process / IPC)
 * captured by the pipeline engine when profiling is enabled.
 */
#ifndef SYNAPTIC_SYN_PROF_H_
#define SYNAPTIC_SYN_PROF_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Timing breakdown of the most recent profiled inference. */
typedef struct {
    uint32_t total_us;            /**< End-to-end wall time            */
    uint32_t preprocess_us;       /**< Pre-processing stages           */
    uint32_t npu_us;              /**< Model execution                 */
    uint32_t postprocess_us;      /**< Post-processing stages          */
    uint32_t ipc_overhead_us;     /**< Cross-core transport share      */
    uint32_t mem_peak_bytes;      /**< Arena peak during the job       */
    uint32_t npu_utilization_pct; /**< npu_us as a share of total      */
} syn_prof_result_t;

/**
 * @brief Enable per-inference profiling.
 *
 * @return 0 on success, negative errno on failure.
 */
int  syn_prof_enable(void);

/**
 * @brief Disable profiling (captured results remain readable).
 *
 * @return 0 on success, negative errno on failure.
 */
int  syn_prof_disable(void);

/**
 * @brief Fetch the timing of the last profiled inference.
 *
 * @param result Out: filled with the last capture.
 *
 * @return 0 on success, -EINVAL on NULL argument, negative errno if
 *         nothing has been captured yet.
 */
int  syn_prof_get_last(syn_prof_result_t *result);

/**
 * @brief Print a human-readable summary of the last capture.
 */
void syn_prof_print_summary(void);

/* Per-layer tracing */

/**
 * @brief Enable per-layer timing capture (layered models).
 *
 * @return 0 on success, negative errno on failure.
 */
int  syn_prof_enable_layer_trace(void);

/**
 * @brief Read the captured execution time of one layer.
 *
 * @param layer_index Layer position in execution order (0-based).
 * @param us          Out: layer time in microseconds.
 *
 * @return 0 on success, negative errno on bad index or if tracing is
 *         disabled.
 */
int  syn_prof_get_layer_time(uint32_t layer_index, uint32_t *us);

#ifdef __cplusplus
}
#endif
#endif /* SYNAPTIC_SYN_PROF_H_ */
