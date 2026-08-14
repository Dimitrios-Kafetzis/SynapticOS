/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file syn_hal_npu_internal.h
 * @brief SynapticOS - NPU HAL internals (not public API)
 *
 * Residency hooks the model registry needs beyond the public session
 * API. include/synaptic/syn_hal_npu.h is frozen at 0.5.0; additions
 * stay internal until the v1.0.0 header revision.
 */
#ifndef SYNAPTIC_SYN_HAL_NPU_INTERNAL_H_
#define SYNAPTIC_SYN_HAL_NPU_INTERNAL_H_

#ifdef __cplusplus
extern "C" {
#endif

/* Drop the resident model from the NPU without releasing the backend.
 * On the Neutron backend this unprepares the model, releasing the
 * firmware's input-size gate (S7 board finding: a prepared model
 * gates EVERY invoke, and unload without unprepare left the gate
 * armed until reboot). Idempotent: returns 0 when nothing is
 * resident. -EPERM before init, -EBUSY during an invoke.
 */
int syn_hal_npu_unload_model(void);

#ifdef __cplusplus
}
#endif
#endif /* SYNAPTIC_SYN_HAL_NPU_INTERNAL_H_ */
