/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file syn_npu_layered.h
 * @brief SynapticOS - Layer-granular synthetic model execution (private)
 *
 * Executes models in the synthetic layered formats one layer at a
 * time so the scheduler can preempt at layer boundaries, with
 * memory-optimal activation placement (Phase 5.2). The formats and
 * the per-layer transform are deterministic and CPU-evaluated on
 * every backend: results carry the stub-NPU label until the Neutron
 * SDK invoke path lands.
 *
 * Chain format "SYNL" (little-endian):
 *   offset 0: uint32  magic 0x4C4E5953
 *   offset 4: uint16  layer_count  (1..SYN_LAYERED_MAX_LAYERS)
 *   offset 6: uint16  input_size   (exact input bytes, <= ACT_MAX)
 *   offset 8: layer_count x { uint16 out_size; uint16 work; }
 *   Every layer consumes the previous activation.
 *
 * DAG format "SYND" (little-endian):
 *   offset 0: uint32  magic 0x444E5953
 *   offset 4: uint16  layer_count
 *   offset 6: uint16  input_size
 *   offset 8: layer_count x { uint16 out_size; uint16 work;
 *                             int16 src_a; int16 src_b; }
 *   src encoding: -1 = previous layer output (model input for layer
 *   0), -2 = model input, j >= 0 = output of layer j (j < i).
 *   src_b: -3 = none (single-input layer), else as src_a - the layer
 *   then mixes a second (skip) activation into its transform.
 *
 * Activation placement: all live activations (model input + one
 * buffer per layer output) share ONE static plan area. At session
 * begin a greedy first-fit planner assigns offsets from buffer
 * lifetimes (definition layer to last consuming layer), so storage
 * is reused the moment a buffer's last consumer ran. The all-live
 * sum (the naive baseline) and the planned peak are exposed via
 * syn_npu_layered_plan_info(). Planning is deterministic: a resumed
 * session replans identically.
 *
 * Every layer transform is order-sensitive and chains on the full
 * source activation, so a resumed job is bit-exact only if resumed
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

#define SYN_LAYERED_MAGIC       0x4C4E5953UL /* "SYNL" chain */
#define SYN_LAYERED_DAG_MAGIC   0x444E5953UL /* "SYND" DAG   */
#define SYN_LAYERED_HDR_SIZE    8U
#define SYN_LAYERED_DESC_SIZE   4U  /* chain descriptor  */
#define SYN_LAYERED_DDESC_SIZE  8U  /* DAG descriptor    */
#define SYN_LAYERED_CTX_SLOTS   2U
#define SYN_LAYERED_MAX_LAYERS  16U

#define SYN_LAYERED_SRC_PREV    (-1)
#define SYN_LAYERED_SRC_INPUT   (-2)
#define SYN_LAYERED_SRC_NONE    (-3)

/** One DAG layer for syn_npu_layered_make_dag(). */
typedef struct {
	uint16_t out_size;
	uint16_t work;
	int16_t  src_a;  /* SYN_LAYERED_SRC_* or a layer index */
	int16_t  src_b;  /* SYN_LAYERED_SRC_NONE for single input */
} syn_layered_dag_layer_t;

#ifdef CONFIG_SYNAPTIC_LAYER_EXEC

/**
 * @brief Backend hook: record the resident model after a successful
 *        syn_hal_npu_load_model(). Clears the record when the blob
 *        is not in a layered format.
 */
void syn_npu_layered_on_load(const uint8_t *data, size_t size);

/**
 * @brief Start a layered session on the resident model: plan the
 *        activation placement and stage the input.
 * @return Layer count (> 0) when a layered model is resident,
 *         0 when the resident model is not layered (caller falls
 *         back to the monolithic invoke path), -EINVAL when
 *         @p in_size does not match the model's declared input size,
 *         -ENOMEM when the planned peak exceeds the plan area.
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
 * @brief Placement metrics of the most recent begin()/restore().
 * @param planned_peak  Planned live-activation peak in bytes.
 * @param naive_peak    All-live (sum of every activation) baseline.
 * @param plan_us       Planning duration in microseconds.
 */
void syn_npu_layered_plan_info(uint32_t *planned_peak,
			       uint32_t *naive_peak, uint32_t *plan_us);

/**
 * @brief Park the live session in a free suspension slot.
 * @return Slot index (>= 0), -EPERM without a live session,
 *         -ENOSPC when all slots are taken.
 */
int syn_npu_layered_save(void);

/**
 * @brief Resume a parked session into the live context (replanning
 *        deterministically) and free the slot.
 * @return 0 on success, -EINVAL on a bad or empty slot.
 */
int syn_npu_layered_restore(int slot);

/** @brief Discard a parked session (cancelled suspended job). */
void syn_npu_layered_slot_free(int slot);

/**
 * @brief Build a chain-format model blob (test/demo helper).
 * @return Total blob size in bytes, or -ENOMEM when @p buf_size is
 *         too small, -EINVAL on bad geometry.
 */
int syn_npu_layered_make_model(uint8_t *buf, size_t buf_size,
			       uint16_t input_size, uint16_t layer_count,
			       const uint16_t *out_sizes,
			       const uint16_t *work);

/**
 * @brief Build a DAG-format model blob (test/demo helper).
 * @return Total blob size in bytes, or -ENOMEM when @p buf_size is
 *         too small, -EINVAL on bad geometry or source references.
 */
int syn_npu_layered_make_dag(uint8_t *buf, size_t buf_size,
			     uint16_t input_size, uint16_t layer_count,
			     const syn_layered_dag_layer_t *layers);

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
