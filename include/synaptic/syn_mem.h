/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file syn_mem.h
 * @brief SynapticOS — Tensor Memory Management
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
int  syn_mem_init(void *arena_base, size_t arena_size);
void syn_mem_reset_ephemeral(void);

/* Tensor allocation */
syn_tensor_t *syn_mem_tensor_alloc(const uint32_t *shape, uint8_t ndim,
                                   syn_npu_dtype_t dtype,
                                   syn_mem_lifetime_t lifetime);
void syn_mem_tensor_free(syn_tensor_t *tensor);

/* Helper to initialize a tensor descriptor (no allocation) */
int syn_mem_tensor_init(syn_tensor_t *tensor, const uint32_t *shape,
                        uint8_t ndim, syn_npu_dtype_t dtype);

/* Scratch pool */
void *syn_mem_scratch_acquire(size_t size);
void  syn_mem_scratch_release(void *ptr);

/* Statistics */
int  syn_mem_get_stats(syn_mem_stats_t *stats);
void syn_mem_print_stats(void);

#ifdef __cplusplus
}
#endif

#endif /* SYNAPTIC_SYN_MEM_H_ */
