/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file syn_mem.h
 * @brief SynapticOS — Tensor Memory Management
 *
 * Static tensor arena with lifetime-segregated allocation (persistent
 * grows up, ephemeral grows down) plus a fixed-block scratch pool.
 * Ephemeral allocations are reclaimed in bulk via
 * syn_mem_reset_ephemeral() between inferences — there is no
 * per-tensor free for ephemeral memory.
 */

#ifndef SYNAPTIC_SYN_MEM_H_
#define SYNAPTIC_SYN_MEM_H_

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <synaptic/syn_hal_npu.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Memory lifetime classification */
typedef enum {
    SYN_MEM_PERSISTENT,   /**< Lives across inference calls (weights, biases) */
    SYN_MEM_EPHEMERAL,    /**< Freed after each inference (activations)       */
    SYN_MEM_SHARED,       /**< In shared IPC region (input/output tensors)    */
} syn_mem_lifetime_t;

/** Tensor descriptor */
typedef struct {
    void               *data;       /**< Pointer to tensor data             */
    size_t              size;       /**< Total size in bytes                */
    syn_npu_dtype_t     dtype;      /**< Data type                          */
    uint8_t             ndim;       /**< Number of dimensions (max 4)       */
    uint32_t            shape[4];   /**< Shape: [batch, height, width, ch]  */
    syn_mem_lifetime_t  lifetime;   /**< Memory lifetime class              */
} syn_tensor_t;

/** Memory statistics */
typedef struct {
    size_t   arena_total;     /**< Total arena size                   */
    size_t   arena_used;      /**< Currently allocated                */
    size_t   arena_peak;      /**< High-water mark                    */
    size_t   scratch_total;   /**< Total scratch pool size            */
    size_t   scratch_used;    /**< Currently used scratch             */
    uint32_t alloc_count;     /**< Total allocations since init       */
    uint32_t reset_count;     /**< Total ephemeral resets since init  */
} syn_mem_stats_t;

/* Arena management */

/**
 * @brief Initialize the tensor arena over a caller-provided region.
 *
 * Called once at startup (syn_init() does this for the configured
 * arena). Re-initializing resets all bookkeeping.
 *
 * @param arena_base Base address of the arena region.
 * @param arena_size Region size in bytes.
 *
 * @return 0 on success, -EINVAL on bad base/size.
 */
int  syn_mem_init(void *arena_base, size_t arena_size);

/**
 * @brief Release all ephemeral allocations at once.
 *
 * Must be called between inferences; persistent allocations are
 * untouched.
 */
void syn_mem_reset_ephemeral(void);

/* Tensor allocation */

/**
 * @brief Allocate a tensor (descriptor + data) from the arena.
 *
 * @param shape    Dimension sizes, @p ndim entries.
 * @param ndim     Number of dimensions (1..4).
 * @param dtype    Element data type.
 * @param lifetime Lifetime class (see syn_mem_lifetime_t).
 *
 * @return Tensor pointer, or NULL on bad arguments or arena
 *         exhaustion.
 */
syn_tensor_t *syn_mem_tensor_alloc(const uint32_t *shape, uint8_t ndim,
                                   syn_npu_dtype_t dtype,
                                   syn_mem_lifetime_t lifetime);

/**
 * @brief Free a persistent tensor allocated by syn_mem_tensor_alloc().
 *
 * Ephemeral tensors are reclaimed by syn_mem_reset_ephemeral()
 * instead.
 *
 * @param tensor Tensor to free (NULL is ignored).
 */
void syn_mem_tensor_free(syn_tensor_t *tensor);

/**
 * @brief Initialize a caller-owned tensor descriptor (no allocation).
 *
 * Fills size/dtype/shape bookkeeping; the caller supplies `data`
 * separately.
 *
 * @param tensor Descriptor to fill.
 * @param shape  Dimension sizes, @p ndim entries.
 * @param ndim   Number of dimensions (1..4).
 * @param dtype  Element data type.
 *
 * @return 0 on success, -EINVAL on bad arguments.
 */
int syn_mem_tensor_init(syn_tensor_t *tensor, const uint32_t *shape,
                        uint8_t ndim, syn_npu_dtype_t dtype);

/* Scratch pool */

/**
 * @brief Acquire a block from the fixed scratch pool.
 *
 * For short-lived working buffers (DSP kernels, staging). Blocks are
 * limited in size and count by Kconfig.
 *
 * @param size Bytes needed.
 *
 * @return Block pointer, or NULL if the pool is exhausted or @p size
 *         exceeds the block size.
 */
void *syn_mem_scratch_acquire(size_t size);

/**
 * @brief Return a scratch block to the pool.
 *
 * @param ptr Pointer from syn_mem_scratch_acquire() (NULL is ignored).
 */
void  syn_mem_scratch_release(void *ptr);

/* Statistics */

/**
 * @brief Snapshot allocator statistics.
 *
 * @param stats Out: filled with current counters.
 *
 * @return 0 on success, -EINVAL on NULL argument.
 */
int  syn_mem_get_stats(syn_mem_stats_t *stats);

/**
 * @brief Print allocator statistics to the console/log.
 */
void syn_mem_print_stats(void);

#ifdef __cplusplus
}
#endif

#endif /* SYNAPTIC_SYN_MEM_H_ */
