/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file syn_infer_internal.h
 * @brief SynapticOS - inference engine internals (not public API)
 *
 * Quiesce gate for NPU residency changes (model load / hot-swap):
 * quiesce() pauses job dispatch and returns once the in-flight job,
 * if any, has finished executing; jobs submitted meanwhile stay
 * queued. release() resumes dispatch and replays deferred wakeups.
 * Not reentrant; callers serialize (syn_model.c holds its gate).
 */
#ifndef SYNAPTIC_SYN_INFER_INTERNAL_H_
#define SYNAPTIC_SYN_INFER_INTERNAL_H_

#include <stdint.h>
#include <stdbool.h>
#include <synaptic/syn_model.h>

#ifdef __cplusplus
extern "C" {
#endif

void syn_infer_quiesce(void);
void syn_infer_release(void);

/** true while any SUSPENDED (layer-preempted) job references the
 * model - its pipeline input and payload must stay valid, so
 * unload/unregister/eviction is refused until it resumes or is
 * cancelled (quiesce-gap closure, Phase 6.4).
 */
bool syn_infer_model_suspended(syn_model_handle_t model);

/** Scheduler counters (Phase 5.1: deadline dispatch + preemption). */
typedef struct {
	uint32_t completed;        /**< Jobs finished successfully       */
	uint32_t errors;           /**< Jobs finished with an error      */
	uint32_t cancelled;        /**< Jobs cancelled before finishing  */
	uint32_t deadline_misses;  /**< Completions after deadline_us    */
	uint32_t preemptions;      /**< Layer-boundary suspensions       */
	uint32_t resumes;          /**< Suspended jobs resumed           */
	uint32_t last_save_us;     /**< Last context-save duration       */
	uint32_t max_save_us;      /**< Worst context-save duration      */
} syn_infer_stats_t;

void syn_infer_get_stats(syn_infer_stats_t *stats);
void syn_infer_reset_stats(void);

#ifdef __cplusplus
}
#endif
#endif /* SYNAPTIC_SYN_INFER_INTERNAL_H_ */
