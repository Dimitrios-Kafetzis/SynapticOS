/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file syn_ipc.h
 * @brief SynapticOS — Inter-Core Communication (CPU0 <-> CPU1)
 *
 * Lock-free SPSC message rings over shared SRAM between the two
 * Cortex-M33 cores, with MCX inter-core interrupts for doorbells.
 * Payloads travel in a shared payload area referenced by offset.
 */
#ifndef SYNAPTIC_SYN_IPC_H_
#define SYNAPTIC_SYN_IPC_H_

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/** IPC message types */
typedef enum {
    SYN_IPC_INFER_REQ,     /**< Inference request (CPU1 -> CPU0)      */
    SYN_IPC_INFER_RESP,    /**< Inference response (CPU0 -> CPU1)     */
    SYN_IPC_MODEL_LOAD,    /**< Model load request                    */
    SYN_IPC_MODEL_UNLOAD,  /**< Model unload request                  */
    SYN_IPC_STATUS_REQ,    /**< Liveness/status request               */
    SYN_IPC_STATUS_RESP,   /**< Liveness/status response              */
} syn_ipc_type_t;

/** IPC message structure (shared SRAM) */
typedef struct {
    uint32_t       msg_id;         /**< Monotonic id, set by send()    */
    uint8_t        type;           /**< syn_ipc_type_t                 */
    uint8_t        priority;       /**< syn_priority_t of the request  */
    uint16_t       payload_len;    /**< Payload bytes at the offset    */
    uint32_t       payload_offset; /**< Offset into the shared payload area */
    uint32_t       timestamp_us;   /**< Sender timestamp               */
    int32_t        status;         /**< Result code on responses       */
} syn_ipc_msg_t;

/**
 * @brief Per-type message handler.
 *
 * Runs on the receiver's IPC RX thread.
 *
 * @param msg Received message (valid for the duration of the call).
 * @param ctx Pointer registered with syn_ipc_register_handler().
 */
typedef void (*syn_ipc_handler_t)(const syn_ipc_msg_t *msg, void *ctx);

/**
 * @brief Initialize IPC over a shared SRAM region.
 *
 * Both cores must use the same region and matching shared-layout
 * versions; mismatched layouts refuse to pair.
 *
 * @param shared_base Base of the shared SRAM region (core-local view).
 * @param shared_size Region size; must cover the control block, rings
 *                    and payload area.
 *
 * @return 0 on success, -EINVAL on bad region, -EALREADY if already
 *         initialized, negative errno otherwise.
 */
int  syn_ipc_init(void *shared_base, size_t shared_size);

/**
 * @brief Enqueue a message to the other core.
 *
 * Non-blocking; the message is copied into the TX ring and the other
 * core is signalled.
 *
 * @param msg Message to send (msg_id is assigned by the call).
 *
 * @return 0 on success, -ENODEV before init, -EINVAL on bad message,
 *         negative errno if the ring is full.
 */
int  syn_ipc_send(const syn_ipc_msg_t *msg);

/**
 * @brief Dequeue the next message from the other core.
 *
 * @param msg        Out: received message.
 * @param timeout_ms 0 to poll (no wait), otherwise wait up to this
 *                   many milliseconds.
 *
 * @return 0 on success, -ENODEV before init, -EINVAL on bad argument,
 *         negative errno on timeout with no message.
 */
int  syn_ipc_receive(syn_ipc_msg_t *msg, uint32_t timeout_ms);

/**
 * @brief Register a handler for one message type.
 *
 * The handler is dispatched from the IPC RX thread for every received
 * message of @p type. One handler per type; registering again
 * replaces the previous handler.
 *
 * @param type    Message type to handle.
 * @param handler Handler function (NULL to unregister).
 * @param ctx     Opaque pointer passed to the handler.
 *
 * @return 0 on success, -EINVAL on bad type.
 */
int  syn_ipc_register_handler(syn_ipc_type_t type,
                              syn_ipc_handler_t handler, void *ctx);

#ifdef __cplusplus
}
#endif
#endif /* SYNAPTIC_SYN_IPC_H_ */
