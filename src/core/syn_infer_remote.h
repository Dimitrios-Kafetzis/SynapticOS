/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file syn_infer_remote.h
 * @brief SynapticOS - Cross-core inference protocol (private header)
 *
 * CPU1 (application core) requests inference; CPU0 (AI runtime)
 * executes it via syn_infer_run_sync() and returns the output.
 * Tensors travel through the fixed exchange slot in the shared
 * region (syn_shm_infer_slot_t); control messages use the frozen
 * syn_ipc.h message types:
 *
 *   SYN_IPC_MODEL_LOAD  CPU1->CPU0 lookup-by-name, handle comes back
 *   SYN_IPC_INFER_REQ   CPU1->CPU0 run the model in the slot
 *   SYN_IPC_INFER_RESP  CPU0->CPU1 status + output in the slot
 *
 * One request is outstanding at a time; concurrent CPU1 callers
 * serialize on an internal mutex.
 */
#ifndef SYNAPTIC_SYN_INFER_REMOTE_H_
#define SYNAPTIC_SYN_INFER_REMOTE_H_

#include <stddef.h>
#include <stdint.h>
#include <synaptic/syn_model.h>
#include <synaptic/syn_infer.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief CPU0: start serving cross-core inference requests.
 *
 * Registers the MODEL_LOAD and INFER_REQ handlers. Call after
 * syn_init() and syn_ipc_init(). Handlers run on the IPC dispatch
 * thread; a long inference delays other message dispatch, which is
 * acceptable for this runtime's single-NPU reality.
 *
 * @return 0 on success, negative errno otherwise.
 */
int syn_remote_serve_init(void);

/**
 * @brief CPU1: resolve a model name to a handle on CPU0.
 *
 * @return 0 on success (handle written), -ENOENT if CPU0 does not
 *         know the model, -ETIMEDOUT, or other negative errno.
 */
int syn_remote_model_lookup(const char *name, uint32_t timeout_ms,
			    syn_model_handle_t *handle);

/**
 * @brief CPU1: borrow the shared input buffer for zero-copy staging.
 *
 * Writing the input tensor directly into the shared slot avoids a
 * private staging copy (CPU1 has only 64 KB of RAM). Pass the
 * returned pointer as @p input to syn_remote_infer(), which then
 * skips its memcpy. Staging and the matching syn_remote_infer()
 * call must not interleave with other threads' requests.
 *
 * @param cap Written with the buffer capacity (may be NULL).
 * @return Slot input buffer, or NULL before syn_ipc_init().
 */
void *syn_remote_infer_input(size_t *cap);

/**
 * @brief CPU1: run one inference on CPU0 and fetch the output.
 *
 * Copies the input into the shared slot (skipped when @p input is
 * the zero-copy staging buffer), sends INFER_REQ with the given
 * priority, blocks for INFER_RESP, then copies the output out.
 *
 * @param model      Handle from syn_remote_model_lookup().
 * @param input      Input tensor bytes.
 * @param input_len  Must fit SYN_SHM_INFER_INPUT_SIZE.
 * @param output     Buffer for the output tensor.
 * @param output_cap Capacity of @p output.
 * @param output_len Actual output size (may be NULL).
 * @param priority   Scheduling priority on CPU0.
 * @param timeout_ms End-to-end budget.
 * @return 0 on success; inference errors from CPU0 propagate as
 *         their negative errno; -ETIMEDOUT if CPU0 never answered;
 *         -EMSGSIZE if input or output exceed the slot.
 */
int syn_remote_infer(syn_model_handle_t model,
		     const void *input, size_t input_len,
		     void *output, size_t output_cap, size_t *output_len,
		     syn_priority_t priority, uint32_t timeout_ms);

#ifdef __cplusplus
}
#endif
#endif /* SYNAPTIC_SYN_INFER_REMOTE_H_ */
