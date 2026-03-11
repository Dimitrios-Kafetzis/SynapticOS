/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file syn_prof_internal.h
 * @brief SynapticOS — Internal profiling helpers
 *
 * Used by the inference path to record stage timestamps.
 * Not part of the public API.
 */

#ifndef SYNAPTIC_SYN_PROF_INTERNAL_H_
#define SYNAPTIC_SYN_PROF_INTERNAL_H_

void syn_prof_mark_start(void);
void syn_prof_mark_preprocess_done(void);
void syn_prof_mark_npu_done(void);
void syn_prof_mark_end(void);

#endif /* SYNAPTIC_SYN_PROF_INTERNAL_H_ */
