/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file test_mem_bench.c
 * @brief Memory allocator benchmarks for performance characterization
 *
 * Produces results R1 (allocation throughput) for community content.
 * Allocates tensors of varying sizes, measures cycle counts, and prints
 * throughput statistics.
 */

#include <zephyr/ztest.h>
#include <zephyr/kernel.h>
#include <synaptic/syn_mem.h>
#include <zephyr/logging/log.h>

LOG_MODULE_REGISTER(test_mem_bench, LOG_LEVEL_INF);

#define BENCH_ARENA_SIZE (8 * 1024) /* 8 KB test arena */
static uint8_t __aligned(16) bench_arena[BENCH_ARENA_SIZE];

ZTEST_SUITE(syn_mem_bench_suite, NULL, NULL, NULL, NULL, NULL);

/**
 * R1: Allocation throughput — allocate many small tensors, measure cycles.
 */
ZTEST(syn_mem_bench_suite, test_alloc_throughput_small)
{
	syn_mem_init(bench_arena, BENCH_ARENA_SIZE);

	uint32_t shape[] = {1, 4, 4, 1}; /* 16 bytes per tensor */
	const int num_allocs = 20;
	uint32_t start, end;

	start = k_cycle_get_32();
	for (int i = 0; i < num_allocs; i++) {
		syn_tensor_t *t = syn_mem_tensor_alloc(shape, 4,
						       SYN_NPU_DTYPE_INT8,
						       SYN_MEM_EPHEMERAL);
		zassert_not_null(t, "Alloc %d failed", i);
	}
	end = k_cycle_get_32();

	uint32_t total_cycles = end - start;
	uint32_t avg_cycles = total_cycles / num_allocs;
	uint32_t total_us = k_cyc_to_us_ceil32(total_cycles);

	LOG_INF("=== R1: Allocation Throughput (16-byte tensors) ===");
	LOG_INF("  Allocations:   %d", num_allocs);
	LOG_INF("  Total cycles:  %u", total_cycles);
	LOG_INF("  Avg cycles:    %u per alloc", avg_cycles);
	LOG_INF("  Total time:    %u us", total_us);
	if (total_us > 0) {
		LOG_INF("  Throughput:    %u allocs/sec",
			(num_allocs * 1000000U) / total_us);
	}

	syn_mem_stats_t stats;
	syn_mem_get_stats(&stats);
	LOG_INF("  Arena used:    %zu / %zu bytes", stats.arena_used,
		stats.arena_total);
	LOG_INF("  Alloc count:   %u", stats.alloc_count);

	/* Sanity: all allocations should have been tracked */
	zassert_equal(stats.alloc_count, num_allocs, "Alloc count mismatch");
}

/**
 * R1 (continued): Allocation throughput with varying sizes.
 */
ZTEST(syn_mem_bench_suite, test_alloc_throughput_varying)
{
	syn_mem_init(bench_arena, BENCH_ARENA_SIZE);

	/* Varying tensor sizes: 4, 16, 32, 64 bytes */
	struct {
		uint32_t shape[4];
		uint8_t ndim;
	} configs[] = {
		{{1, 2, 2, 1}, 4},  /*  4 bytes */
		{{1, 4, 4, 1}, 4},  /* 16 bytes */
		{{1, 4, 4, 2}, 4},  /* 32 bytes */
		{{1, 8, 8, 1}, 4},  /* 64 bytes */
	};
	const int num_configs = 4;
	const int iters_per_config = 5;

	LOG_INF("=== R1: Allocation Throughput (varying sizes) ===");

	uint32_t total_start = k_cycle_get_32();
	int total_allocs = 0;

	for (int c = 0; c < num_configs; c++) {
		uint32_t start = k_cycle_get_32();

		for (int i = 0; i < iters_per_config; i++) {
			syn_tensor_t *t = syn_mem_tensor_alloc(
				configs[c].shape, configs[c].ndim,
				SYN_NPU_DTYPE_INT8, SYN_MEM_EPHEMERAL);
			zassert_not_null(t, "Alloc failed: config %d iter %d",
					 c, i);
			total_allocs++;
		}

		uint32_t elapsed = k_cycle_get_32() - start;
		uint32_t size = 1;

		for (int d = 0; d < configs[c].ndim; d++) {
			size *= configs[c].shape[d];
		}
		LOG_INF("  %3u-byte tensor: %u cycles/alloc (%d allocs)",
			size, elapsed / iters_per_config, iters_per_config);
	}

	uint32_t total_elapsed = k_cycle_get_32() - total_start;

	LOG_INF("  Total: %d allocs in %u cycles (%u us)", total_allocs,
		total_elapsed, k_cyc_to_us_ceil32(total_elapsed));
}

/**
 * R1 (continued): Measure reset speed.
 */
ZTEST(syn_mem_bench_suite, test_reset_speed)
{
	syn_mem_init(bench_arena, BENCH_ARENA_SIZE);

	uint32_t shape[] = {1, 8, 8, 1};
	const int num_rounds = 10;

	LOG_INF("=== R1: Ephemeral Reset Speed ===");

	uint32_t total_reset_cycles = 0;

	for (int r = 0; r < num_rounds; r++) {
		/* Fill with 5 tensors */
		for (int i = 0; i < 5; i++) {
			syn_tensor_t *t = syn_mem_tensor_alloc(
				shape, 4, SYN_NPU_DTYPE_INT8,
				SYN_MEM_EPHEMERAL);
			zassert_not_null(t, "Alloc failed round %d", r);
		}

		uint32_t start = k_cycle_get_32();
		syn_mem_reset_ephemeral();
		uint32_t elapsed = k_cycle_get_32() - start;

		total_reset_cycles += elapsed;
	}

	LOG_INF("  Resets:       %d", num_rounds);
	LOG_INF("  Avg cycles:   %u per reset", total_reset_cycles / num_rounds);
	LOG_INF("  Avg time:     %u us per reset",
		k_cyc_to_us_ceil32(total_reset_cycles / num_rounds));

	syn_mem_stats_t stats;
	syn_mem_get_stats(&stats);
	zassert_equal(stats.reset_count, num_rounds, "Reset count mismatch");
}
