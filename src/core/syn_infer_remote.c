/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file syn_infer_remote.c
 * @brief SynapticOS - Cross-core inference protocol
 *
 * See syn_infer_remote.h for the protocol. This file compiles on
 * both cores; the serve side is CPU0-only, the request side is
 * CPU1-only (guarded below), and both share the slot access code.
 *
 * Ownership of the exchange slot alternates strictly: CPU1 owns it
 * from mutex-take to INFER_REQ send, CPU0 owns it from INFER_REQ
 * receipt to INFER_RESP send. The message itself is the ownership
 * transfer, and the ring's DMB barriers order the slot contents
 * against it.
 */

#include <string.h>

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>

#include <synaptic/syn_ipc.h>
#include <synaptic/syn_model.h>
#include <synaptic/syn_infer.h>
#include <synaptic/syn_mem.h>

#include "syn_shared_layout.h"
#include "syn_ipc_internal.h"
#include "syn_infer_remote.h"

LOG_MODULE_REGISTER(syn_remote, CONFIG_SYNAPTIC_LOG_LEVEL);

static syn_shm_infer_slot_t *infer_slot(void)
{
	syn_shm_region_t *shm = syn_ipc_region();

	return (shm != NULL) ? (syn_shm_infer_slot_t *)shm->payload : NULL;
}

/* Send with a short retry window: the ring is only ever full under
 * heartbeat pileup, which drains in microseconds.
 */
static int send_with_retry(const syn_ipc_msg_t *msg)
{
	int ret = -EAGAIN;

	for (int i = 0; i < 10 && ret == -EAGAIN; i++) {
		ret = syn_ipc_send(msg);
		if (ret == -EAGAIN) {
			k_sleep(K_MSEC(1));
		}
	}
	return ret;
}

#if defined(CONFIG_SOC_MCXN947_CPU1)

/* ------------------------------------------------------------------
 * CPU1: request side
 */

static K_MUTEX_DEFINE(remote_lock);
static K_SEM_DEFINE(resp_sem, 0, 1);
static syn_ipc_msg_t resp_msg;
static bool handlers_ready;

void *syn_remote_infer_input(size_t *cap)
{
	syn_shm_infer_slot_t *slot = infer_slot();

	if (cap != NULL) {
		*cap = SYN_SHM_INFER_INPUT_SIZE;
	}
	return (slot != NULL) ? slot->input : NULL;
}

static void resp_handler(const syn_ipc_msg_t *msg, void *ctx)
{
	ARG_UNUSED(ctx);

	resp_msg = *msg;
	k_sem_give(&resp_sem);
}

static int ensure_handlers(void)
{
	if (handlers_ready) {
		return 0;
	}

	int ret = syn_ipc_register_handler(SYN_IPC_INFER_RESP,
					   resp_handler, NULL);

	if (ret == 0) {
		ret = syn_ipc_register_handler(SYN_IPC_MODEL_LOAD,
					       resp_handler, NULL);
	}
	if (ret == 0) {
		handlers_ready = true;
	}
	return ret;
}

/* Issue one request/response transaction against the shared slot.
 * Caller holds remote_lock.
 */
static int transact(syn_ipc_msg_t *req, uint32_t timeout_ms)
{
	k_sem_reset(&resp_sem);

	int ret = send_with_retry(req);

	if (ret != 0) {
		return ret;
	}

	if (k_sem_take(&resp_sem, K_MSEC(timeout_ms)) != 0) {
		return -ETIMEDOUT;
	}
	return 0;
}

int syn_remote_model_lookup(const char *name, uint32_t timeout_ms,
			    syn_model_handle_t *handle)
{
	syn_shm_infer_slot_t *slot = infer_slot();

	if (name == NULL || handle == NULL) {
		return -EINVAL;
	}
	if (slot == NULL) {
		return -ENODEV;
	}
	if (strlen(name) >= sizeof(slot->model_name)) {
		return -EMSGSIZE;
	}

	k_mutex_lock(&remote_lock, K_FOREVER);

	int ret = ensure_handlers();

	if (ret == 0) {
		strcpy(slot->model_name, name);

		syn_ipc_msg_t req = {
			.type = SYN_IPC_MODEL_LOAD,
			.payload_offset = SYN_SHM_PAYLOAD_OFFSET,
		};

		ret = transact(&req, timeout_ms);
		if (ret == 0) {
			ret = resp_msg.status;
			if (ret == 0) {
				*handle = slot->model;
			}
		}
	}

	k_mutex_unlock(&remote_lock);
	return ret;
}

int syn_remote_infer(syn_model_handle_t model,
		     const void *input, size_t input_len,
		     void *output, size_t output_cap, size_t *output_len,
		     syn_priority_t priority, uint32_t timeout_ms)
{
	syn_shm_infer_slot_t *slot = infer_slot();

	if (input == NULL || output == NULL || input_len == 0U) {
		return -EINVAL;
	}
	if (slot == NULL) {
		return -ENODEV;
	}
	if (input_len > SYN_SHM_INFER_INPUT_SIZE) {
		return -EMSGSIZE;
	}

	k_mutex_lock(&remote_lock, K_FOREVER);

	int ret = ensure_handlers();

	if (ret == 0) {
		slot->model = model;
		slot->input_len = input_len;
		if (input != slot->input) {
			memcpy(slot->input, input, input_len);
		}

		syn_ipc_msg_t req = {
			.type = SYN_IPC_INFER_REQ,
			.priority = (uint8_t)priority,
			.payload_len = (uint16_t)input_len,
			.payload_offset = SYN_SHM_PAYLOAD_OFFSET,
		};

		ret = transact(&req, timeout_ms);
	}

	if (ret == 0) {
		ret = resp_msg.status;
	}
	if (ret == 0) {
		if (slot->output_len > output_cap) {
			ret = -EMSGSIZE;
		} else {
			memcpy(output, slot->output, slot->output_len);
			if (output_len != NULL) {
				*output_len = slot->output_len;
			}
		}
	}

	k_mutex_unlock(&remote_lock);
	return ret;
}

#else /* CPU0: serve side */

static void model_load_handler(const syn_ipc_msg_t *msg, void *ctx)
{
	ARG_UNUSED(ctx);

	syn_shm_infer_slot_t *slot = infer_slot();
	syn_model_handle_t handle = SYN_MODEL_INVALID;

	slot->model_name[sizeof(slot->model_name) - 1] = '\0';

	int ret = syn_model_get_by_name(slot->model_name, &handle);

	slot->model = handle;

	syn_ipc_msg_t resp = {
		.type = SYN_IPC_MODEL_LOAD,
		.payload_offset = msg->msg_id,
		.status = ret,
	};

	LOG_INF("Model lookup '%s': %s (handle %u)", slot->model_name,
		(ret == 0) ? "found" : "not found", handle);

	if (send_with_retry(&resp) != 0) {
		LOG_ERR("Model lookup response send failed");
	}
}

static void infer_req_handler(const syn_ipc_msg_t *msg, void *ctx)
{
	ARG_UNUSED(ctx);

	syn_shm_infer_slot_t *slot = infer_slot();
	syn_model_info_t info;
	syn_tensor_t output = {0};
	int ret;

	syn_priority_t prio = (msg->priority <= SYN_PRIORITY_REALTIME)
			      ? (syn_priority_t)msg->priority
			      : SYN_PRIORITY_NORMAL;

	slot->output_len = 0;

	ret = syn_model_get_info(slot->model, &info);
	if (ret != 0) {
		goto respond;
	}
	if (slot->input_len > SYN_SHM_INFER_INPUT_SIZE ||
	    slot->input_len == 0U) {
		ret = -EMSGSIZE;
		goto respond;
	}

	/* Copy input from the shared slot into the local tensor arena */
	uint32_t shape[1] = { slot->input_len };
	syn_tensor_t *input = syn_mem_tensor_alloc(shape, 1,
						   info.input_dtype,
						   SYN_MEM_EPHEMERAL);

	if (input == NULL) {
		ret = -ENOMEM;
		goto respond;
	}
	memcpy(input->data, slot->input, slot->input_len);

	ret = syn_infer_run_sync(slot->model, input, &output, prio);

	if (ret == 0) {
		if (output.size > SYN_SHM_INFER_OUTPUT_SIZE) {
			ret = -EMSGSIZE;
		} else {
			memcpy(slot->output, output.data, output.size);
			slot->output_len = output.size;
		}
	}

	syn_mem_reset_ephemeral();

respond:
	slot->infer_status = ret;

	syn_ipc_msg_t resp = {
		.type = SYN_IPC_INFER_RESP,
		.priority = msg->priority,
		.payload_len = (uint16_t)MIN(slot->output_len, UINT16_MAX),
		.payload_offset = msg->msg_id,
		.status = ret,
	};

	if (ret != 0) {
		LOG_WRN("Cross-core inference failed: %d (model %u)",
			ret, slot->model);
	}

	if (send_with_retry(&resp) != 0) {
		LOG_ERR("INFER_RESP send failed; CPU1 will time out");
	}
}

int syn_remote_serve_init(void)
{
	if (syn_ipc_region() == NULL) {
		return -ENODEV;
	}

	int ret = syn_ipc_register_handler(SYN_IPC_MODEL_LOAD,
					   model_load_handler, NULL);

	if (ret == 0) {
		ret = syn_ipc_register_handler(SYN_IPC_INFER_REQ,
					       infer_req_handler, NULL);
	}

	if (ret == 0) {
		LOG_INF("Cross-core inference serving enabled");
	}
	return ret;
}

#endif /* CONFIG_SOC_MCXN947_CPU1 */
