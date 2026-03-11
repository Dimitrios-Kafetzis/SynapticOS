/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file syn_prof.h
 * @brief SynapticOS — Profiling and Diagnostics
 */
#ifndef SYNAPTIC_SYN_PROF_H_
#define SYNAPTIC_SYN_PROF_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t total_us;
    uint32_t preprocess_us;
    uint32_t npu_us;
    uint32_t postprocess_us;
    uint32_t ipc_overhead_us;
    uint32_t mem_peak_bytes;
    uint32_t npu_utilization_pct;
} syn_prof_result_t;

int  syn_prof_enable(void);
int  syn_prof_disable(void);
int  syn_prof_get_last(syn_prof_result_t *result);
void syn_prof_print_summary(void);

/* Per-layer tracing */
int  syn_prof_enable_layer_trace(void);
int  syn_prof_get_layer_time(uint32_t layer_index, uint32_t *us);

#ifdef __cplusplus
}
#endif
#endif /* SYNAPTIC_SYN_PROF_H_ */
