/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file test_mem.c
 * @brief Unit tests for syn_mem tensor arena allocator
 */

#include <zephyr/ztest.h>
#include <synaptic/syn_mem.h>

#define TEST_ARENA_SIZE  (8 * 1024)  /* 8 KB test arena */
static uint8_t __aligned(16) test_arena[TEST_ARENA_SIZE];

ZTEST_SUITE(syn_mem_suite, NULL, NULL, NULL, NULL, NULL);

ZTEST(syn_mem_suite, test_init)
{
    int ret = syn_mem_init(test_arena, TEST_ARENA_SIZE);
    zassert_equal(ret, 0, "syn_mem_init failed: %d", ret);
}

ZTEST(syn_mem_suite, test_init_null)
{
    int ret = syn_mem_init(NULL, TEST_ARENA_SIZE);
    zassert_equal(ret, -EINVAL, "Should reject NULL base");

    ret = syn_mem_init(test_arena, 0);
    zassert_equal(ret, -EINVAL, "Should reject zero size");
}

ZTEST(syn_mem_suite, test_tensor_alloc_basic)
{
    syn_mem_init(test_arena, TEST_ARENA_SIZE);

    uint32_t shape[] = {1, 10, 10, 3};
    syn_tensor_t *t = syn_mem_tensor_alloc(shape, 4,
                                           SYN_NPU_DTYPE_INT8,
                                           SYN_MEM_EPHEMERAL);
    zassert_not_null(t, "Tensor alloc returned NULL");
    zassert_equal(t->ndim, 4, "Wrong ndim");
    zassert_equal(t->shape[1], 10, "Wrong shape[1]");
    zassert_equal(t->size, 1 * 10 * 10 * 3, "Wrong total size");
}

ZTEST(syn_mem_suite, test_ephemeral_reset)
{
    syn_mem_init(test_arena, TEST_ARENA_SIZE);

    uint32_t shape[] = {1, 8, 8, 1};
    syn_tensor_t *t1 = syn_mem_tensor_alloc(shape, 4,
                                            SYN_NPU_DTYPE_INT8,
                                            SYN_MEM_EPHEMERAL);
    zassert_not_null(t1, "First alloc failed");

    syn_mem_stats_t stats;
    syn_mem_get_stats(&stats);
    zassert_true(stats.arena_used > 0, "Arena should be in use");

    syn_mem_reset_ephemeral();

    syn_mem_get_stats(&stats);
    zassert_equal(stats.arena_used, 0, "Arena should be empty after reset");
    zassert_equal(stats.reset_count, 1, "Reset count should be 1");
}

ZTEST(syn_mem_suite, test_arena_exhaustion)
{
    syn_mem_init(test_arena, TEST_ARENA_SIZE);

    /* Try to allocate more than usable arena size */
    uint32_t shape[] = {1, 256, 256, 3};  /* 196608 bytes > usable arena */
    syn_tensor_t *t = syn_mem_tensor_alloc(shape, 4,
                                           SYN_NPU_DTYPE_INT8,
                                           SYN_MEM_EPHEMERAL);
    zassert_is_null(t, "Should return NULL on exhaustion");
}

ZTEST(syn_mem_suite, test_scratch_pool)
{
    syn_mem_init(test_arena, TEST_ARENA_SIZE);

    void *buf = syn_mem_scratch_acquire(1024);
    zassert_not_null(buf, "Scratch acquire failed");

    syn_mem_stats_t stats;
    syn_mem_get_stats(&stats);
    zassert_true(stats.scratch_used > 0, "Scratch used should be > 0");
    zassert_equal(stats.scratch_total, CONFIG_SYNAPTIC_SCRATCH_POOL_SIZE,
                  "Scratch total mismatch");

    syn_mem_scratch_release(buf);
}

ZTEST(syn_mem_suite, test_stats)
{
    syn_mem_init(test_arena, TEST_ARENA_SIZE);

    syn_mem_stats_t stats;
    int ret = syn_mem_get_stats(&stats);
    zassert_equal(ret, 0, "get_stats failed");
    zassert_equal(stats.arena_total,
                  TEST_ARENA_SIZE - CONFIG_SYNAPTIC_SCRATCH_POOL_SIZE,
                  "Wrong arena total");
    zassert_equal(stats.arena_used, 0, "Should be empty");
    zassert_equal(stats.alloc_count, 0, "Should have zero allocs");
}

/* --- New tests for Deliverable 1.1 --- */

ZTEST(syn_mem_suite, test_alignment_int8)
{
    syn_mem_init(test_arena, TEST_ARENA_SIZE);

    uint32_t shape[] = {1, 7, 7, 3};  /* Odd size to stress alignment */
    syn_tensor_t *t = syn_mem_tensor_alloc(shape, 4,
                                           SYN_NPU_DTYPE_INT8,
                                           SYN_MEM_EPHEMERAL);
    zassert_not_null(t, "Alloc failed");
    zassert_equal((uintptr_t)t->data % 16, 0,
                  "INT8 tensor data not 16-byte aligned");
}

ZTEST(syn_mem_suite, test_alignment_int16)
{
    syn_mem_init(test_arena, TEST_ARENA_SIZE);

    uint32_t shape[] = {1, 5, 5, 1};
    syn_tensor_t *t = syn_mem_tensor_alloc(shape, 4,
                                           SYN_NPU_DTYPE_INT16,
                                           SYN_MEM_EPHEMERAL);
    zassert_not_null(t, "Alloc failed");
    zassert_equal((uintptr_t)t->data % 16, 0,
                  "INT16 tensor data not 16-byte aligned");
}

ZTEST(syn_mem_suite, test_alignment_float32)
{
    syn_mem_init(test_arena, TEST_ARENA_SIZE);

    uint32_t shape[] = {1, 3, 3, 1};
    syn_tensor_t *t = syn_mem_tensor_alloc(shape, 4,
                                           SYN_NPU_DTYPE_FLOAT32,
                                           SYN_MEM_EPHEMERAL);
    zassert_not_null(t, "Alloc failed");
    zassert_equal((uintptr_t)t->data % 16, 0,
                  "FLOAT32 tensor data not 16-byte aligned");
}

ZTEST(syn_mem_suite, test_alignment_multiple_allocs)
{
    syn_mem_init(test_arena, TEST_ARENA_SIZE);

    uint32_t shape1[] = {1, 7};
    uint32_t shape2[] = {1, 13};
    uint32_t shape3[] = {1, 3, 3, 1};

    syn_tensor_t *t1 = syn_mem_tensor_alloc(shape1, 2,
                                            SYN_NPU_DTYPE_INT8,
                                            SYN_MEM_EPHEMERAL);
    syn_tensor_t *t2 = syn_mem_tensor_alloc(shape2, 2,
                                            SYN_NPU_DTYPE_INT16,
                                            SYN_MEM_EPHEMERAL);
    syn_tensor_t *t3 = syn_mem_tensor_alloc(shape3, 4,
                                            SYN_NPU_DTYPE_FLOAT32,
                                            SYN_MEM_EPHEMERAL);

    zassert_not_null(t1, "Alloc 1 failed");
    zassert_not_null(t2, "Alloc 2 failed");
    zassert_not_null(t3, "Alloc 3 failed");

    zassert_equal((uintptr_t)t1->data % 16, 0, "t1 not aligned");
    zassert_equal((uintptr_t)t2->data % 16, 0, "t2 not aligned");
    zassert_equal((uintptr_t)t3->data % 16, 0, "t3 not aligned");
}

ZTEST(syn_mem_suite, test_persistent_survives_reset)
{
    syn_mem_init(test_arena, TEST_ARENA_SIZE);

    /* Allocate a persistent tensor */
    uint32_t pshape[] = {1, 4, 4, 1};
    syn_tensor_t *persistent = syn_mem_tensor_alloc(pshape, 4,
                                                    SYN_NPU_DTYPE_INT8,
                                                    SYN_MEM_PERSISTENT);
    zassert_not_null(persistent, "Persistent alloc failed");

    /* Write a known pattern into persistent data */
    memset(persistent->data, 0xAB, persistent->size);

    syn_mem_stats_t stats_before;
    syn_mem_get_stats(&stats_before);
    size_t persistent_bytes = stats_before.arena_used;

    /* Allocate an ephemeral tensor */
    uint32_t eshape[] = {1, 8, 8, 1};
    syn_tensor_t *ephemeral = syn_mem_tensor_alloc(eshape, 4,
                                                   SYN_NPU_DTYPE_INT8,
                                                   SYN_MEM_EPHEMERAL);
    zassert_not_null(ephemeral, "Ephemeral alloc failed");

    syn_mem_stats_t stats_after_both;
    syn_mem_get_stats(&stats_after_both);
    zassert_true(stats_after_both.arena_used > persistent_bytes,
                 "Arena should have grown");

    /* Reset ephemeral — persistent must survive */
    syn_mem_reset_ephemeral();

    syn_mem_stats_t stats_after_reset;
    syn_mem_get_stats(&stats_after_reset);
    zassert_equal(stats_after_reset.arena_used, persistent_bytes,
                  "After reset, only persistent should remain");

    /* Verify persistent data is intact */
    uint8_t *pdata = (uint8_t *)persistent->data;
    for (size_t i = 0; i < persistent->size; i++) {
        zassert_equal(pdata[i], 0xAB,
                      "Persistent data corrupted at offset %zu", i);
    }
}

ZTEST(syn_mem_suite, test_scratch_pool_separate)
{
    syn_mem_init(test_arena, TEST_ARENA_SIZE);

    /* Allocate a tensor from the main arena */
    uint32_t shape[] = {1, 4, 4, 1};
    syn_tensor_t *t = syn_mem_tensor_alloc(shape, 4,
                                           SYN_NPU_DTYPE_INT8,
                                           SYN_MEM_EPHEMERAL);
    zassert_not_null(t, "Tensor alloc failed");

    /* Acquire scratch */
    void *scratch = syn_mem_scratch_acquire(512);
    zassert_not_null(scratch, "Scratch acquire failed");

    /* Scratch and tensor data should not overlap */
    uintptr_t tensor_start = (uintptr_t)t;
    uintptr_t tensor_end = (uintptr_t)t->data + t->size;
    uintptr_t scratch_addr = (uintptr_t)scratch;

    zassert_true(scratch_addr < tensor_start || scratch_addr >= tensor_end,
                 "Scratch overlaps tensor region");

    /* Verify stats reflect both */
    syn_mem_stats_t stats;
    syn_mem_get_stats(&stats);
    zassert_true(stats.arena_used > 0, "Tensor arena should be in use");
    zassert_true(stats.scratch_used > 0, "Scratch should be in use");
}

ZTEST(syn_mem_suite, test_zero_size_tensor)
{
    syn_mem_init(test_arena, TEST_ARENA_SIZE);

    /* ndim = 0 should fail */
    uint32_t shape[] = {1, 4};
    syn_tensor_t *t = syn_mem_tensor_alloc(shape, 0,
                                           SYN_NPU_DTYPE_INT8,
                                           SYN_MEM_EPHEMERAL);
    zassert_is_null(t, "Should reject ndim=0");

    /* NULL shape should fail */
    t = syn_mem_tensor_alloc(NULL, 2, SYN_NPU_DTYPE_INT8, SYN_MEM_EPHEMERAL);
    zassert_is_null(t, "Should reject NULL shape");
}

ZTEST(syn_mem_suite, test_ndim_boundary)
{
    syn_mem_init(test_arena, TEST_ARENA_SIZE);

    /* ndim = 4 is the max — should succeed */
    uint32_t shape4[] = {1, 2, 3, 4};
    syn_tensor_t *t = syn_mem_tensor_alloc(shape4, 4,
                                           SYN_NPU_DTYPE_INT8,
                                           SYN_MEM_EPHEMERAL);
    zassert_not_null(t, "ndim=4 should succeed");
    zassert_equal(t->ndim, 4, "Wrong ndim");

    syn_mem_reset_ephemeral();

    /* ndim = 5 should fail */
    uint32_t shape5[] = {1, 2, 3, 4};
    t = syn_mem_tensor_alloc(shape5, 5, SYN_NPU_DTYPE_INT8, SYN_MEM_EPHEMERAL);
    zassert_is_null(t, "ndim=5 should fail");

    /* ndim = 1 should succeed */
    uint32_t shape1[] = {16};
    t = syn_mem_tensor_alloc(shape1, 1, SYN_NPU_DTYPE_INT8, SYN_MEM_EPHEMERAL);
    zassert_not_null(t, "ndim=1 should succeed");
    zassert_equal(t->shape[0], 16, "Wrong shape[0]");
}

ZTEST(syn_mem_suite, test_scratch_exhaustion)
{
    syn_mem_init(test_arena, TEST_ARENA_SIZE);

    /* Scratch pool is CONFIG_SYNAPTIC_SCRATCH_POOL_SIZE (4 KB in test) */
    void *buf = syn_mem_scratch_acquire(CONFIG_SYNAPTIC_SCRATCH_POOL_SIZE + 1);
    zassert_is_null(buf, "Should fail when exceeding scratch pool");
}

ZTEST(syn_mem_suite, test_multiple_alloc_and_reset)
{
    syn_mem_init(test_arena, TEST_ARENA_SIZE);

    /* Allocate 5 ephemeral tensors */
    uint32_t shape[] = {1, 4};
    for (int i = 0; i < 5; i++) {
        syn_tensor_t *t = syn_mem_tensor_alloc(shape, 2,
                                               SYN_NPU_DTYPE_INT8,
                                               SYN_MEM_EPHEMERAL);
        zassert_not_null(t, "Alloc %d failed", i);
    }

    syn_mem_stats_t stats1;
    syn_mem_get_stats(&stats1);
    zassert_true(stats1.arena_used > 0, "Should have used memory");

    /* Reset and allocate again — memory should be reused */
    syn_mem_reset_ephemeral();

    syn_mem_stats_t stats2;
    syn_mem_get_stats(&stats2);
    zassert_equal(stats2.arena_used, 0, "Should be empty after reset");

    for (int i = 0; i < 5; i++) {
        syn_tensor_t *t = syn_mem_tensor_alloc(shape, 2,
                                               SYN_NPU_DTYPE_INT8,
                                               SYN_MEM_EPHEMERAL);
        zassert_not_null(t, "Re-alloc %d failed", i);
    }
}

ZTEST(syn_mem_suite, test_zero_size_shape)
{
    syn_mem_init(test_arena, TEST_ARENA_SIZE);

    /* Shape with zeros — total elements = 0, but ndim > 0 */
    uint32_t shape[] = {0, 0, 0, 0};
    syn_tensor_t *t = syn_mem_tensor_alloc(shape, 4,
                                           SYN_NPU_DTYPE_INT8,
                                           SYN_MEM_EPHEMERAL);
    /* Should succeed but with size = 0 (0 elements) */
    /* The allocator calculates num_elements = 0, data_size = 0 */
    /* Still allocates descriptor space */
    if (t != NULL) {
        zassert_equal(t->size, 0, "Size should be 0 for zero shape");
    }
    /* Either behavior (NULL or 0-size) is acceptable */
}
