/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file syn_mpu_internal.h
 * @brief SynapticOS - Cross-core MPU protection (private header)
 */
#ifndef SYNAPTIC_SYN_MPU_INTERNAL_H_
#define SYNAPTIC_SYN_MPU_INTERNAL_H_

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Verify cross-core memory protection.
 *
 * Checks that the shared region is writable, then spawns a sacrificial
 * thread that writes into the other core's private SRAM. With the MPU
 * guard region in place the write raises a MemManage fault; the fault
 * policy aborts only that thread and the system keeps running.
 *
 * @return 0 if the guarded write faulted as expected,
 *         -EPERM if the write went through (MPU not enforcing),
 *         other negative errno on setup failure.
 */
int syn_mpu_selftest(void);

#ifdef __cplusplus
}
#endif
#endif /* SYNAPTIC_SYN_MPU_INTERNAL_H_ */
