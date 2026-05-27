/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file syn_init.c
 * @brief SynapticOS — Runtime Initialization
 *
 * Full initialization sequence: memory arena, NPU HAL, DSP HAL,
 * profiling. Enforces ordering and prevents double-initialization.
 */

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>

LOG_MODULE_REGISTER(syn_init, CONFIG_SYNAPTIC_LOG_LEVEL);

#include <synaptic/syn_api.h>
#include <synaptic/syn_mem.h>
#include <synaptic/syn_hal_npu.h>
#include <synaptic/syn_hal_dsp.h>
#ifdef CONFIG_SYNAPTIC_PROFILING
#include <synaptic/syn_prof.h>
#endif

static uint8_t tensor_arena[CONFIG_SYNAPTIC_TENSOR_ARENA_SIZE] __aligned(16);
static bool runtime_initialized;

int syn_init(void)
{
	int ret;

	if (runtime_initialized) {
		LOG_WRN("Runtime already initialized");
		return -EALREADY;
	}

	LOG_INF("Initializing SynapticOS runtime v%s", syn_version());

	/* Step 1: Memory arena */
	ret = syn_mem_init(tensor_arena, sizeof(tensor_arena));
	if (ret != 0) {
		LOG_ERR("Memory init failed: %d", ret);
		return ret;
	}

	/* Step 2: NPU HAL */
	ret = syn_hal_npu_init();
	if (ret != 0) {
		LOG_ERR("NPU HAL init failed: %d", ret);
		return ret;
	}

	/* Step 3: DSP HAL (non-critical) */
	ret = syn_hal_dsp_init();
	if (ret != 0) {
		LOG_WRN("DSP init failed: %d - continuing without DSP acceleration", ret);
	}

	/* Step 4: Profiling */
#ifdef CONFIG_SYNAPTIC_PROFILING
	ret = syn_prof_enable();
	if (ret != 0) {
		LOG_WRN("Profiling init failed: %d", ret);
	}
#endif

	runtime_initialized = true;

	LOG_INF("Runtime ready (arena=%u bytes, NPU=initialized)",
		(unsigned)sizeof(tensor_arena));
	return 0;
}

int syn_shutdown(void)
{
	if (!runtime_initialized) {
		return -EPERM;
	}

	LOG_INF("Shutting down SynapticOS runtime");

#ifdef CONFIG_SYNAPTIC_PROFILING
	syn_prof_disable();
#endif

	syn_hal_npu_deinit();

	runtime_initialized = false;
	LOG_INF("Runtime shut down");
	return 0;
}

const char *syn_version(void)
{
	return SYNAPTIC_VERSION_STRING;
}
