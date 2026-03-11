/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file syn_mem.c
 * @brief SynapticOS — Tensor Arena Allocator
 *
 * Bump-pointer arena with:
 *   - 16-byte alignment for NPU DMA compatibility
 *   - Persistent / ephemeral region split
 *   - Separate scratch pool at top of arena
 *
 * Arena layout:
 *   [base ... persistent ... ephemeral ... | scratch_pool ... base+total]
 */

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>

LOG_MODULE_REGISTER(syn_mem, CONFIG_SYNAPTIC_LOG_LEVEL);

#include <synaptic/syn_mem.h>
#include <string.h>

#define ALIGN_UP(x, align) (((x) + (align) - 1) & ~((size_t)(align) - 1))
#define TENSOR_ALIGNMENT 16

static struct {
	uint8_t *base;
	size_t   total;           /* Full arena size including scratch */
	size_t   usable;          /* Tensor-usable size (total - scratch) */
	size_t   persistent_used; /* Persistent region: [base, base + persistent_used) */
	size_t   ephemeral_used;  /* Ephemeral region: [base + persistent_used, ...) */
	size_t   peak;            /* High-water mark of persistent_used + ephemeral_used */
	size_t   scratch_total;   /* Scratch pool size */
	size_t   scratch_used;    /* Current scratch allocation offset */
	uint32_t alloc_count;
	uint32_t reset_count;
	bool     initialized;
} arena;

int syn_mem_init(void *arena_base, size_t arena_size)
{
	if (arena_base == NULL || arena_size == 0) {
		return -EINVAL;
	}

	size_t scratch_size = CONFIG_SYNAPTIC_SCRATCH_POOL_SIZE;

	if (arena_size <= scratch_size) {
		return -EINVAL;
	}

	arena.base = arena_base;
	arena.total = arena_size;
	arena.usable = arena_size - scratch_size;
	arena.persistent_used = 0;
	arena.ephemeral_used = 0;
	arena.peak = 0;
	arena.scratch_total = scratch_size;
	arena.scratch_used = 0;
	arena.alloc_count = 0;
	arena.reset_count = 0;
	arena.initialized = true;

	LOG_INF("Arena: %u KB total, %u KB tensor, %u KB scratch",
		(unsigned)(arena_size / 1024),
		(unsigned)(arena.usable / 1024),
		(unsigned)(scratch_size / 1024));
	return 0;
}

void syn_mem_reset_ephemeral(void)
{
	if (!arena.initialized) {
		return;
	}
	arena.ephemeral_used = 0;
	arena.scratch_used = 0;
	arena.reset_count++;
}

static size_t dtype_size(syn_npu_dtype_t dtype)
{
	switch (dtype) {
	case SYN_NPU_DTYPE_INT8:
	case SYN_NPU_DTYPE_UINT8:
		return 1;
	case SYN_NPU_DTYPE_INT16:
	case SYN_NPU_DTYPE_FLOAT16:
		return 2;
	case SYN_NPU_DTYPE_FLOAT32:
		return 4;
	default:
		return 1;
	}
}

/**
 * Allocate from the arena bump pointer.
 * Returns aligned pointer or NULL if insufficient space.
 */
static void *arena_alloc(syn_mem_lifetime_t lifetime, size_t size)
{
	size_t offset;
	size_t aligned_offset;

	if (lifetime == SYN_MEM_PERSISTENT) {
		offset = arena.persistent_used;
	} else {
		offset = arena.persistent_used + arena.ephemeral_used;
	}

	aligned_offset = ALIGN_UP(offset, TENSOR_ALIGNMENT);
	size_t end = aligned_offset + size;

	if (end > arena.usable) {
		return NULL;
	}

	void *ptr = arena.base + aligned_offset;

	if (lifetime == SYN_MEM_PERSISTENT) {
		arena.persistent_used = end;
	} else {
		arena.ephemeral_used = end - arena.persistent_used;
	}

	/* Update peak */
	size_t current = arena.persistent_used + arena.ephemeral_used;

	if (current > arena.peak) {
		arena.peak = current;
	}

	arena.alloc_count++;
	return ptr;
}

syn_tensor_t *syn_mem_tensor_alloc(const uint32_t *shape, uint8_t ndim,
				   syn_npu_dtype_t dtype,
				   syn_mem_lifetime_t lifetime)
{
	if (!arena.initialized || shape == NULL || ndim == 0 || ndim > 4) {
		return NULL;
	}

	/* Calculate total element count */
	size_t num_elements = 1;

	for (uint8_t i = 0; i < ndim; i++) {
		num_elements *= shape[i];
	}

	size_t data_size = num_elements * dtype_size(dtype);

	/*
	 * We need: sizeof(syn_tensor_t) + padding + data_size
	 * The descriptor is placed at the aligned offset. Then data follows
	 * at the next 16-byte aligned boundary after the descriptor.
	 */
	size_t desc_size = ALIGN_UP(sizeof(syn_tensor_t), TENSOR_ALIGNMENT);
	size_t total = desc_size + data_size;

	void *block = arena_alloc(lifetime, total);

	if (block == NULL) {
		return NULL;
	}

	syn_tensor_t *t = (syn_tensor_t *)block;

	/* Data starts at next aligned boundary after descriptor */
	t->data = (uint8_t *)block + desc_size;
	t->size = data_size;
	t->dtype = dtype;
	t->ndim = ndim;
	t->lifetime = lifetime;
	memset(t->shape, 0, sizeof(t->shape));
	for (uint8_t i = 0; i < ndim; i++) {
		t->shape[i] = shape[i];
	}

	return t;
}

void syn_mem_tensor_free(syn_tensor_t *tensor)
{
	ARG_UNUSED(tensor);
	/* Bump allocator — free is a no-op; use reset_ephemeral */
}

int syn_mem_tensor_init(syn_tensor_t *tensor, const uint32_t *shape,
			uint8_t ndim, syn_npu_dtype_t dtype)
{
	if (tensor == NULL || shape == NULL || ndim == 0 || ndim > 4) {
		return -EINVAL;
	}

	size_t num_elements = 1;

	for (uint8_t i = 0; i < ndim; i++) {
		num_elements *= shape[i];
	}

	tensor->size = num_elements * dtype_size(dtype);
	tensor->dtype = dtype;
	tensor->ndim = ndim;
	memset(tensor->shape, 0, sizeof(tensor->shape));
	for (uint8_t i = 0; i < ndim; i++) {
		tensor->shape[i] = shape[i];
	}

	return 0;
}

void *syn_mem_scratch_acquire(size_t size)
{
	if (!arena.initialized || size == 0) {
		return NULL;
	}

	size_t aligned_offset = ALIGN_UP(arena.scratch_used, TENSOR_ALIGNMENT);
	size_t end = aligned_offset + size;

	if (end > arena.scratch_total) {
		return NULL;
	}

	/* Scratch pool lives at the top of the arena */
	uint8_t *scratch_base = arena.base + arena.usable;
	void *ptr = scratch_base + aligned_offset;

	arena.scratch_used = end;
	arena.alloc_count++;

	return ptr;
}

void syn_mem_scratch_release(void *ptr)
{
	ARG_UNUSED(ptr);
	/* Scratch is reset when ephemeral is reset */
}

int syn_mem_get_stats(syn_mem_stats_t *stats)
{
	if (stats == NULL || !arena.initialized) {
		return -EINVAL;
	}

	stats->arena_total = arena.usable;
	stats->arena_used = arena.persistent_used + arena.ephemeral_used;
	stats->arena_peak = arena.peak;
	stats->scratch_total = arena.scratch_total;
	stats->scratch_used = arena.scratch_used;
	stats->alloc_count = arena.alloc_count;
	stats->reset_count = arena.reset_count;
	return 0;
}

void syn_mem_print_stats(void)
{
	if (!arena.initialized) {
		LOG_WRN("Arena not initialized");
		return;
	}

	LOG_INF("Arena: %u KB usable, %u KB used (persistent: %u, ephemeral: %u), "
		"peak %u KB, scratch: %u/%u KB",
		(unsigned)(arena.usable / 1024),
		(unsigned)((arena.persistent_used + arena.ephemeral_used) / 1024),
		(unsigned)(arena.persistent_used / 1024),
		(unsigned)(arena.ephemeral_used / 1024),
		(unsigned)(arena.peak / 1024),
		(unsigned)(arena.scratch_used / 1024),
		(unsigned)(arena.scratch_total / 1024));
}
