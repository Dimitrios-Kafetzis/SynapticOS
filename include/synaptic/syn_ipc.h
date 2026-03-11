/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file syn_ipc.h
 * @brief SynapticOS — Inter-Core Communication (CPU0 <-> CPU1)
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
    SYN_IPC_INFER_REQ,
    SYN_IPC_INFER_RESP,
    SYN_IPC_MODEL_LOAD,
    SYN_IPC_MODEL_UNLOAD,
    SYN_IPC_STATUS_REQ,
    SYN_IPC_STATUS_RESP,
} syn_ipc_type_t;

/** IPC message structure (shared SRAM) */
typedef struct {
    uint32_t       msg_id;
    uint8_t        type;
    uint8_t        priority;
    uint16_t       payload_len;
    uint32_t       payload_offset;
    uint32_t       timestamp_us;
    int32_t        status;
} syn_ipc_msg_t;

typedef void (*syn_ipc_handler_t)(const syn_ipc_msg_t *msg, void *ctx);

int  syn_ipc_init(void *shared_base, size_t shared_size);
int  syn_ipc_send(const syn_ipc_msg_t *msg);
int  syn_ipc_receive(syn_ipc_msg_t *msg, uint32_t timeout_ms);
int  syn_ipc_register_handler(syn_ipc_type_t type,
                              syn_ipc_handler_t handler, void *ctx);

#ifdef __cplusplus
}
#endif
#endif /* SYNAPTIC_SYN_IPC_H_ */
