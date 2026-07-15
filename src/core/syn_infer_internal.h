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

#ifdef __cplusplus
extern "C" {
#endif

void syn_infer_quiesce(void);
void syn_infer_release(void);

#ifdef __cplusplus
}
#endif
#endif /* SYNAPTIC_SYN_INFER_INTERNAL_H_ */
