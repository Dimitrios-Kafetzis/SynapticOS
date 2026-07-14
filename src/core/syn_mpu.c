/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file syn_mpu.c
 * @brief SynapticOS - Cross-core MPU protection and fault policy
 *
 * The guard regions themselves are declared in the device tree: each
 * core's build carries a read-only MPU region (ATTR_MPU_FLASH) over the
 * other core's private SRAM, so a stray cross-core write raises a
 * MemManage fault on the offending core.
 *
 * ARMv8-M limitation, stated plainly: the MPU access-permission field
 * has no privileged-no-access encoding, and Zephyr keeps the default
 * background map enabled for privileged code (required for ROM API and
 * peripheral access). Cross-core WRITES fault; cross-core reads remain
 * permitted. Writes are the corruption hazard in an AMP system, so this
 * is the protection that matters.
 *
 * This file adds the runtime pieces:
 *  - a fatal-error policy that turns an MPU violation into "log it,
 *    abort the offending thread, keep the core running" instead of a
 *    system halt, so a fault on one core never takes down the other;
 *  - a self-test that provokes a guarded write from a sacrificial
 *    thread to prove enforcement on real hardware.
 */

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <zephyr/logging/log_ctrl.h>
#include <zephyr/fatal.h>

#include "syn_shared_layout.h"
#include "syn_mpu_internal.h"

LOG_MODULE_REGISTER(syn_mpu, CONFIG_SYNAPTIC_LOG_LEVEL);

#if defined(CONFIG_SOC_MCXN947_CPU1)
#define SYN_MPU_GUARDED_BASE  SYN_SHM_CPU0_RAM_BASE
#define SYN_MPU_GUARDED_SIZE  SYN_SHM_CPU0_RAM_SIZE
#else
#define SYN_MPU_GUARDED_BASE  SYN_SHM_CPU1_RAM_BASE
#define SYN_MPU_GUARDED_SIZE  SYN_SHM_CPU1_RAM_SIZE
#endif

/*
 * Fatal-error policy: a CPU exception raised in thread context (which
 * is what a guarded cross-core write produces) is logged and only the
 * offending thread dies; z_fatal_error() aborts it when this handler
 * returns. Anything else (kernel panic, stack check, OOPS) halts the
 * core as usual.
 */
void k_sys_fatal_error_handler(unsigned int reason,
			       const struct arch_esf *esf)
{
	ARG_UNUSED(esf);

	/* MemManage data-access faults are what the cross-core guard
	 * regions raise (Zephyr's ARM fault decoder reports them as
	 * extended K_ERR_ARM_MEM_* reasons).
	 */
	if (reason == K_ERR_CPU_EXCEPTION ||
	    reason == K_ERR_ARM_MEM_GENERIC ||
	    reason == K_ERR_ARM_MEM_DATA_ACCESS) {
		LOG_ERR("MPU violation (reason %u): aborting offending thread, core continues",
			reason);
		return;
	}

	LOG_PANIC();
	LOG_ERR("Unrecoverable fatal error %u: halting this core", reason);
	for (;;) {
		/* Halt this core only; the other core is unaffected. */
	}
}

static K_THREAD_STACK_DEFINE(mpu_test_stack, 1024);
static struct k_thread mpu_test_thread;
static volatile uint32_t mpu_test_survived;

static void mpu_test_entry(void *p1, void *p2, void *p3)
{
	ARG_UNUSED(p1);
	ARG_UNUSED(p2);
	ARG_UNUSED(p3);

	/* Poke the middle of the other core's private SRAM. With the
	 * guard region active this write never lands; the MemManage
	 * fault aborts this thread before the flag below is set.
	 */
	volatile uint32_t *guarded =
		(volatile uint32_t *)(SYN_MPU_GUARDED_BASE +
				      SYN_MPU_GUARDED_SIZE / 2);

	*guarded = 0xDEADBEEFUL;

	mpu_test_survived = 1;
}

int syn_mpu_selftest(void)
{
	/* Step 1: shared region must be writable from this core. Use
	 * the last word of the shared region, which no IPC structure
	 * occupies.
	 */
	volatile uint32_t *shared_probe =
		(volatile uint32_t *)(SYN_SHM_SHARED_BASE +
				      SYN_SHM_SHARED_SIZE - 4);
	uint32_t before = *shared_probe;

	*shared_probe = 0xA5A5A5A5UL;
	if (*shared_probe != 0xA5A5A5A5UL) {
		LOG_ERR("Shared region write/readback failed");
		return -EIO;
	}
	*shared_probe = before;

	LOG_INF("Shared region write/readback OK");

	/* Step 2: a write to the other core's private SRAM must fault. */
	mpu_test_survived = 0;

	k_thread_create(&mpu_test_thread, mpu_test_stack,
			K_THREAD_STACK_SIZEOF(mpu_test_stack),
			mpu_test_entry, NULL, NULL, NULL,
			K_PRIO_PREEMPT(10), 0, K_NO_WAIT);
	k_thread_name_set(&mpu_test_thread, "syn_mpu_test");

	if (k_thread_join(&mpu_test_thread, K_MSEC(500)) != 0) {
		k_thread_abort(&mpu_test_thread);
		LOG_ERR("MPU test thread did not terminate");
		return -ETIMEDOUT;
	}

	if (mpu_test_survived) {
		LOG_ERR("Cross-core write went through: MPU NOT enforcing");
		return -EPERM;
	}

	LOG_INF("Cross-core write to 0x%08lx faulted as expected",
		(unsigned long)(SYN_MPU_GUARDED_BASE +
				SYN_MPU_GUARDED_SIZE / 2));
	return 0;
}
