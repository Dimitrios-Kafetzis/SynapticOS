/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file syn_prof.c
 * @brief SynapticOS — Profiling and Diagnostics
 *
 * Tracks timing for inference stages using Zephyr cycle counter.
 */

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <synaptic/syn_prof.h>
#include <synaptic/syn_mem.h>
#include <string.h>

LOG_MODULE_REGISTER(syn_prof, CONFIG_SYNAPTIC_LOG_LEVEL);

#include "syn_prof_internal.h"

static struct {
	bool enabled;
	syn_prof_result_t last_result;
	uint32_t start_cycle;
	uint32_t preprocess_cycle;
	uint32_t npu_cycle;
	bool has_result;
} profiler;

int syn_prof_enable(void)
{
	profiler.enabled = true;
	profiler.has_result = false;
	LOG_INF("Profiling enabled");
	return 0;
}

int syn_prof_disable(void)
{
	profiler.enabled = false;
	LOG_INF("Profiling disabled");
	return 0;
}

int syn_prof_get_last(syn_prof_result_t *result)
{
	if (result == NULL) {
		return -EINVAL;
	}
	if (!profiler.has_result) {
		return -ENOENT;
	}

	*result = profiler.last_result;
	return 0;
}

void syn_prof_print_summary(void)
{
	if (!profiler.has_result) {
		LOG_INF("No profiling data available");
		return;
	}

	syn_prof_result_t *r = &profiler.last_result;

	LOG_INF("=== Inference Profile ===");
	LOG_INF("  Total:       %u us", r->total_us);
	LOG_INF("  Preprocess:  %u us", r->preprocess_us);
	LOG_INF("  NPU:         %u us", r->npu_us);
	LOG_INF("  Postprocess: %u us", r->postprocess_us);
	LOG_INF("  Memory peak: %u bytes", r->mem_peak_bytes);
	LOG_INF("  NPU util:    %u%%", r->npu_utilization_pct);
}

int syn_prof_enable_layer_trace(void)
{
	return -ENOTSUP;
}

int syn_prof_get_layer_time(uint32_t layer_index, uint32_t *us)
{
	ARG_UNUSED(layer_index);

	if (us == NULL) {
		return -EINVAL;
	}

	*us = 0;
	return -ENOTSUP;
}

/* Internal helpers called by inference path */

void syn_prof_mark_start(void)
{
	if (!profiler.enabled) {
		return;
	}

	memset(&profiler.last_result, 0, sizeof(profiler.last_result));
	profiler.start_cycle = k_cycle_get_32();
}

void syn_prof_mark_preprocess_done(void)
{
	if (!profiler.enabled) {
		return;
	}

	profiler.preprocess_cycle = k_cycle_get_32();
	profiler.last_result.preprocess_us =
		k_cyc_to_us_ceil32(profiler.preprocess_cycle - profiler.start_cycle);
}

void syn_prof_mark_npu_done(void)
{
	if (!profiler.enabled) {
		return;
	}

	profiler.npu_cycle = k_cycle_get_32();
	profiler.last_result.npu_us =
		k_cyc_to_us_ceil32(profiler.npu_cycle - profiler.preprocess_cycle);
}

void syn_prof_mark_end(void)
{
	if (!profiler.enabled) {
		return;
	}

	uint32_t end_cycle = k_cycle_get_32();

	profiler.last_result.postprocess_us =
		k_cyc_to_us_ceil32(end_cycle - profiler.npu_cycle);
	profiler.last_result.total_us =
		k_cyc_to_us_ceil32(end_cycle - profiler.start_cycle);

	/* Capture memory peak */
	syn_mem_stats_t mem_stats;

	if (syn_mem_get_stats(&mem_stats) == 0) {
		profiler.last_result.mem_peak_bytes = (uint32_t)mem_stats.arena_peak;
	}

	/* NPU utilization: ratio of NPU time to total */
	if (profiler.last_result.total_us > 0) {
		profiler.last_result.npu_utilization_pct =
			(profiler.last_result.npu_us * 100) /
			profiler.last_result.total_us;
	}

	profiler.has_result = true;
}
