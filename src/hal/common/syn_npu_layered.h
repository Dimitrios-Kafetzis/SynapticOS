/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file syn_npu_layered.h
 * @brief SynapticOS - Layer-granular synthetic model execution (private)
 *
 * Executes models in the synthetic layered format one layer at a
 * time so the scheduler can preempt at layer boundaries. The format
 * and the per-layer transform are deterministic and CPU-evaluated on
 * every backend: results carry the stub-NPU label until the Neutron
 * SDK invoke path lands.
 *
 * Blob layout (little-endian, 4-byte header alignment not required):
 *   offset 0: uint32  magic "SYNL" (0x4C4E5953)
 *   offset 4: uint16  layer_count  (>= 1)
 *   offset 6: uint16  input_size   (exact input bytes, <= ACT_MAX)
 *   offset 8: layer_count x { uint16 out_size; uint16 work; }
 *
 * Every layer transform is order-sensitive and chains on the full
 * previous activation, so a resumed job is bit-exact only if resumed
 * from the correct saved context. `work` adds work*100 iterations of
 * busy looping to emulate compute time (volatile loop: QEMU has no
 * k_busy_wait).
 *
 * Concurrency: one live session at a time, owned by the scheduler
 * thread. save() parks the live session in a suspension slot;
 * restore() resumes it. Suspended sessions keep their model blob
 * POINTER - unloading or overwriting that model while a job is
 * suspended is undefined (the quiesce gate only drains the running
 * job).
 */
#ifndef SYNAPTIC_SYN_NPU_LAYERED_H_
#define SYNAPTIC_SYN_NPU_LAYERED_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SYN_LAYERED_MAGIC      0x4C4E5953UL /* "SYNL" */
#define SYN_LAYERED_HDR_SIZE   8U
#define SYN_LAYERED_DESC_SIZE  4U
#define SYN_LAYERED_CTX_SLOTS  2U

#ifdef CONFIG_SYNAPTIC_LAYER_EXEC

/**
 * @brief Backend hook: record the resident model after a successful
 *        syn_hal_npu_load_model(). Clears the record when the blob
 *        is not in the layered format.
 */
void syn_npu_layered_on_load(const uint8_t *data, size_t size);

/**
 * @brief Start a layered session on the resident model.
 * @return Layer count (> 0) when a layered model is resident,
 *         0 when the resident model is not layered (caller falls
 *         back to the monolithic invoke path), -EINVAL when
 *         @p in_size does not match the model's declared input size.
 */
int syn_npu_layered_begin(const void *input, size_t in_size);

/**
 * @brief Execute the next layer of the live session.
 * @return Layers remaining after this one (0 = model complete),
 *         -EPERM without a live session.
 */
int syn_npu_layered_step(void);

/** @return Layers remaining in the live session, -EPERM without one. */
int syn_npu_layered_remaining(void);

/**
 * @brief Copy the final activation out of a completed live session
 *        and end it.
 * @return 0 on success, -EPERM without a live session, -EBUSY when
 *         layers remain, -ENOMEM when @p *size is too small.
 */
int syn_npu_layered_output(void *out, size_t *size);

/**
 * @brief Park the live session in a free suspension slot.
 * @return Slot index (>= 0), -EPERM without a live session,
 *         -ENOSPC when all slots are taken.
 */
int syn_npu_layered_save(void);

/**
 * @brief Resume a parked session into the live context and free the
 *        slot.
 * @return 0 on success, -EINVAL on a bad or empty slot.
 */
int syn_npu_layered_restore(int slot);

/** @brief Discard a parked session (cancelled suspended job). */
void syn_npu_layered_slot_free(int slot);

/**
 * @brief Build a layered model blob (test/demo helper).
 * @return Total blob size in bytes, or -ENOMEM when @p buf_size is
 *         too small, -EINVAL on bad geometry (any size 0 or above
 *         SYNAPTIC_LAYER_ACT_MAX).
 */
int syn_npu_layered_make_model(uint8_t *buf, size_t buf_size,
			       uint16_t input_size, uint16_t layer_count,
			       const uint16_t *out_sizes,
			       const uint16_t *work);

#else /* !CONFIG_SYNAPTIC_LAYER_EXEC */

static inline void syn_npu_layered_on_load(const uint8_t *data, size_t size)
{
	(void)data;
	(void)size;
}

#endif /* CONFIG_SYNAPTIC_LAYER_EXEC */

#ifdef __cplusplus
}
#endif
#endif /* SYNAPTIC_SYN_NPU_LAYERED_H_ */
