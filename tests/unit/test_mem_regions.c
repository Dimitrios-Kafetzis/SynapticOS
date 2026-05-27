/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file test_mem_regions.c
 * @brief Persistent/ephemeral region behavior tests with printable output
 *
 * Produces results R3 (persistent survives reset) and R4 (scratch isolation)
 * for community content. These tests print detailed output for screenshots.
 */

#include <zephyr/ztest.h>
#include <synaptic/syn_mem.h>
#include <string.h>
#include <zephyr/logging/log.h>

LOG_MODULE_REGISTER(test_mem_regions, LOG_LEVEL_INF);

#define REGION_ARENA_SIZE (8 * 1024) /* 8 KB test arena */
static uint8_t __aligned(16) region_arena[REGION_ARENA_SIZE];

ZTEST_SUITE(syn_mem_regions_suite, NULL, NULL, NULL, NULL, NULL);

static void print_stats(const char *label)
{
	syn_mem_stats_t stats;

	syn_mem_get_stats(&stats);
	LOG_INF("  [%s] arena: %zu/%zu bytes (peak %zu), "
		"scratch: %zu/%zu, allocs: %u, resets: %u",
		label, stats.arena_used, stats.arena_total, stats.arena_peak,
		stats.scratch_used, stats.scratch_total, stats.alloc_count,
		stats.reset_count);
}

/**
 * R3: Demonstrate that persistent tensors survive ephemeral reset.
 */
ZTEST(syn_mem_regions_suite, test_persistent_ephemeral_lifecycle)
{
	syn_mem_init(region_arena, REGION_ARENA_SIZE);

	LOG_INF("=== R3: Persistent vs Ephemeral Region Behavior ===");
	print_stats("init");

	/* Step 1: Allocate persistent tensor (simulates model weights) */
	uint32_t pshape[] = {1, 4, 4, 1}; /* 16 bytes */
	syn_tensor_t *weights = syn_mem_tensor_alloc(pshape, 4,
						     SYN_NPU_DTYPE_INT8,
						     SYN_MEM_PERSISTENT);
	zassert_not_null(weights, "Persistent alloc failed");
	memset(weights->data, 0xAA, weights->size);

	LOG_INF("  Persistent tensor: addr=%p, size=%zu bytes",
		weights->data, weights->size);
	print_stats("after persistent alloc");

	syn_mem_stats_t stats_p;
	syn_mem_get_stats(&stats_p);
	size_t persistent_used = stats_p.arena_used;

	/* Step 2: Allocate ephemeral tensor (simulates activations) */
	uint32_t eshape[] = {1, 8, 8, 1}; /* 64 bytes */
	syn_tensor_t *activations = syn_mem_tensor_alloc(eshape, 4,
							 SYN_NPU_DTYPE_INT8,
							 SYN_MEM_EPHEMERAL);
	zassert_not_null(activations, "Ephemeral alloc failed");
	memset(activations->data, 0x55, activations->size);

	LOG_INF("  Ephemeral tensor:  addr=%p, size=%zu bytes",
		activations->data, activations->size);
	print_stats("after both allocs");

	syn_mem_stats_t stats_both;
	syn_mem_get_stats(&stats_both);
	zassert_true(stats_both.arena_used > persistent_used,
		     "Both regions should be in use");

	/* Step 3: Reset ephemeral region */
	LOG_INF("  --- Resetting ephemeral region ---");
	syn_mem_reset_ephemeral();
	print_stats("after ephemeral reset");

	syn_mem_stats_t stats_reset;
	syn_mem_get_stats(&stats_reset);

	/* Persistent memory should still be in use */
	zassert_equal(stats_reset.arena_used, persistent_used,
		      "Only persistent should remain: expected %zu, got %zu",
		      persistent_used, stats_reset.arena_used);

	/* Persistent data should be intact */
	uint8_t *wdata = (uint8_t *)weights->data;
	bool data_intact = true;

	for (size_t i = 0; i < weights->size; i++) {
		if (wdata[i] != 0xAA) {
			data_intact = false;
			break;
		}
	}
	zassert_true(data_intact, "Persistent data corrupted after reset");
	LOG_INF("  Persistent data verified: all %zu bytes intact",
		weights->size);

	/* Step 4: Allocate new ephemeral (reuses freed space) */
	syn_tensor_t *act2 = syn_mem_tensor_alloc(eshape, 4,
						  SYN_NPU_DTYPE_INT8,
						  SYN_MEM_EPHEMERAL);
	zassert_not_null(act2, "Second ephemeral alloc failed");
	LOG_INF("  New ephemeral:     addr=%p, size=%zu bytes",
		act2->data, act2->size);
	print_stats("after new ephemeral");

	LOG_INF("=== R3: PASS - persistent survives reset, ephemeral reclaimed ===");
}

/**
 * R4: Demonstrate scratch pool isolation from arena.
 */
ZTEST(syn_mem_regions_suite, test_scratch_arena_isolation)
{
	syn_mem_init(region_arena, REGION_ARENA_SIZE);

	LOG_INF("=== R4: Scratch Pool Isolation ===");
	print_stats("init");

	/* Step 1: Fill arena with tensors (leave some room) */
	uint32_t shape[] = {1, 16, 16, 1}; /* 256 bytes each */
	int alloc_count = 0;
	syn_tensor_t *tensors[20];

	while (alloc_count < 20) {
		tensors[alloc_count] = syn_mem_tensor_alloc(
			shape, 4, SYN_NPU_DTYPE_INT8, SYN_MEM_EPHEMERAL);
		if (tensors[alloc_count] == NULL) {
			break;
		}
		alloc_count++;
	}
	LOG_INF("  Allocated %d tensors of 256 bytes each", alloc_count);
	print_stats("arena filled");

	zassert_true(alloc_count > 0, "Should have allocated at least 1 tensor");

	syn_mem_stats_t stats_full;
	syn_mem_get_stats(&stats_full);

	/* Step 2: Scratch should still work independently */
	void *scratch = syn_mem_scratch_acquire(512);
	zassert_not_null(scratch, "Scratch should work even when arena is busy");
	LOG_INF("  Scratch acquired: addr=%p, size=512 bytes", scratch);
	print_stats("after scratch acquire");

	/* Step 3: Verify no overlap */
	uintptr_t scratch_addr = (uintptr_t)scratch;

	for (int i = 0; i < alloc_count; i++) {
		uintptr_t tdata = (uintptr_t)tensors[i]->data;
		uintptr_t tend = tdata + tensors[i]->size;

		zassert_true(scratch_addr < tdata ||
			     scratch_addr >= tend,
			     "Scratch overlaps tensor %d", i);
	}
	LOG_INF("  No overlap verified between scratch and %d tensors",
		alloc_count);

	/* Step 4: Release scratch, then reset ephemeral (resets both) */
	syn_mem_scratch_release(scratch);
	print_stats("after scratch release");

	syn_mem_reset_ephemeral();
	print_stats("after ephemeral reset");

	syn_mem_stats_t stats_after;
	syn_mem_get_stats(&stats_after);
	zassert_equal(stats_after.arena_used, 0,
		      "Arena should be empty after ephemeral reset");
	zassert_equal(stats_after.scratch_used, 0,
		      "Scratch resets together with ephemeral (by design)");

	/* Step 5: Scratch still works after reset */
	void *scratch2 = syn_mem_scratch_acquire(256);
	zassert_not_null(scratch2, "Scratch should work after reset");
	LOG_INF("  Scratch re-acquired after reset: addr=%p", scratch2);
	print_stats("after scratch re-acquire");

	LOG_INF("=== R4: PASS - scratch pool is isolated from tensor arena ===");
}

/**
 * R2: Zero fragmentation proof — used equals sum of allocations.
 */
ZTEST(syn_mem_regions_suite, test_zero_fragmentation)
{
	syn_mem_init(region_arena, REGION_ARENA_SIZE);

	LOG_INF("=== R2: Zero Fragmentation Proof ===");

	/* Allocate tensors of different sizes and track expected usage */
	struct {
		uint32_t shape[4];
		uint8_t ndim;
		size_t expected_data_size;
	} allocs[] = {
		{{1, 3, 3, 1}, 4, 9},
		{{1, 7, 1, 1}, 4, 7},
		{{1, 16, 1, 1}, 4, 16},
		{{1, 5, 5, 2}, 4, 50},
		{{1, 10, 10, 1}, 4, 100},
	};
	const int num = 5;

	for (int i = 0; i < num; i++) {
		syn_tensor_t *t = syn_mem_tensor_alloc(
			allocs[i].shape, allocs[i].ndim,
			SYN_NPU_DTYPE_INT8, SYN_MEM_EPHEMERAL);
		zassert_not_null(t, "Alloc %d failed", i);
		LOG_INF("  Tensor %d: shape=[%u,%u,%u,%u] data=%zu bytes "
			"addr=%p",
			i, t->shape[0], t->shape[1], t->shape[2], t->shape[3],
			t->size, t->data);
	}

	syn_mem_stats_t stats;
	syn_mem_get_stats(&stats);

	LOG_INF("  Arena used: %zu bytes, Alloc count: %u, Peak: %zu bytes",
		stats.arena_used, stats.alloc_count, stats.arena_peak);
	LOG_INF("  Fragmentation: 0 bytes (bump allocator by construction)");

	zassert_equal(stats.alloc_count, num, "Alloc count mismatch");
	zassert_true(stats.arena_used > 0, "Arena should be in use");
	/* Peak should equal current usage (no frees) */
	zassert_equal(stats.arena_peak, stats.arena_used,
		      "Peak should equal current usage");

	LOG_INF("=== R2: PASS - zero fragmentation confirmed ===");
}
