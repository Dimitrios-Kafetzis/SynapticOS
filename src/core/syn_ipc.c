/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file syn_ipc.c
 * @brief SynapticOS - Inter-Core Communication (CPU0 <-> CPU1)
 *
 * Implements the frozen include/synaptic/syn_ipc.h API over two
 * lock-free SPSC rings in shared SRAM (see syn_shared_layout.h) with
 * MAILBOX interrupt notification. Each core produces into its own TX
 * ring and consumes the other, so both directions stay strictly
 * single-producer single-consumer.
 *
 * Receive path: the MAILBOX ISR just wakes the RX dispatch thread,
 * which is the only ring consumer. Messages with a registered handler
 * are dispatched from that thread; everything else lands in an
 * internal queue that syn_ipc_receive() blocks on. This keeps SPSC
 * ownership clean while offering both callback and blocking APIs.
 *
 * Compiled only under CONFIG_SYNAPTIC_DUAL_CORE (MCXN947 dual-core
 * builds). The ring logic itself lives in syn_ipc_ring.c and is unit
 * tested single-core on QEMU.
 */

#include <string.h>

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <zephyr/sys/barrier.h>

#include <synaptic/syn_ipc.h>

#include "syn_shared_layout.h"
#include "syn_ipc_ring.h"

#include <fsl_mailbox.h>

LOG_MODULE_REGISTER(syn_ipc, CONFIG_SYNAPTIC_LOG_LEVEL);

#if defined(CONFIG_SOC_MCXN947_CPU1)
#define SYN_IPC_MY_CORE    kMAILBOX_CM33_Core1
#define SYN_IPC_OTHER_CORE kMAILBOX_CM33_Core0
#else
#define SYN_IPC_MY_CORE    kMAILBOX_CM33_Core0
#define SYN_IPC_OTHER_CORE kMAILBOX_CM33_Core1
#endif

#define SYN_IPC_TYPE_COUNT (SYN_IPC_STATUS_RESP + 1)

static syn_shm_region_t *shm;
static syn_ipc_ring_t *tx_ring;
static syn_ipc_ring_t *rx_ring;
static bool ipc_ready;

static struct k_sem rx_event;
static struct k_msgq rx_queue;
static char rx_queue_buf[SYN_IPC_RING_ENTRIES * sizeof(syn_ipc_msg_t)];

static struct {
	syn_ipc_handler_t fn;
	void *ctx;
} handlers[SYN_IPC_TYPE_COUNT];

static atomic_t msg_id_next = ATOMIC_INIT(0);

static K_THREAD_STACK_DEFINE(rx_thread_stack, 2048);
static struct k_thread rx_thread;

static void syn_ipc_mailbox_isr(const void *arg)
{
	ARG_UNUSED(arg);

	uint32_t value = MAILBOX_GetValue(MAILBOX, SYN_IPC_MY_CORE);

	if (value != 0U) {
		MAILBOX_ClearValueBits(MAILBOX, SYN_IPC_MY_CORE, value);
	}

	k_sem_give(&rx_event);
}

static void syn_ipc_rx_thread(void *p1, void *p2, void *p3)
{
	ARG_UNUSED(p1);
	ARG_UNUSED(p2);
	ARG_UNUSED(p3);

	syn_ipc_msg_t msg;

	for (;;) {
		while (syn_ipc_ring_pop(rx_ring, &msg) == 0) {
			syn_ipc_handler_t fn = NULL;
			void *ctx = NULL;

			if (msg.type < SYN_IPC_TYPE_COUNT) {
				fn = handlers[msg.type].fn;
				ctx = handlers[msg.type].ctx;
			}

			if (fn != NULL) {
				fn(&msg, ctx);
			} else if (k_msgq_put(&rx_queue, &msg,
					      K_NO_WAIT) != 0) {
				LOG_WRN("RX queue full, message %u dropped",
					msg.msg_id);
			}
		}

		k_sem_take(&rx_event, K_FOREVER);
	}
}

int syn_ipc_init(void *shared_base, size_t shared_size)
{
	if (shared_base == NULL) {
		return -EINVAL;
	}
	if (shared_size < SYN_SHM_PAYLOAD_OFFSET +
			  SYN_SHM_INFER_INPUT_SIZE +
			  SYN_SHM_INFER_OUTPUT_SIZE) {
		return -EINVAL;
	}
	if (ipc_ready) {
		return -EALREADY;
	}

	shm = shared_base;

#if defined(CONFIG_SOC_MCXN947_CPU1)
	/* Secondary core: CPU0 must have initialized the region first. */
	if (shm->ctrl.magic != SYN_SHM_MAGIC) {
		LOG_ERR("Shared region not initialized by CPU0");
		return -ENODEV;
	}
	if (shm->ctrl.layout_version != SYN_SHM_LAYOUT_VERSION ||
	    shm->ctrl.ring_entries != SYN_IPC_RING_ENTRIES) {
		LOG_ERR("Layout mismatch: version %u ring %u (expect %u/%u)",
			shm->ctrl.layout_version, shm->ctrl.ring_entries,
			(unsigned)SYN_SHM_LAYOUT_VERSION,
			(unsigned)SYN_IPC_RING_ENTRIES);
		return -ENODEV;
	}

	tx_ring = &shm->ring_c1_to_c0;
	rx_ring = &shm->ring_c0_to_c1;
#else
	/* Primary core: own the region and publish the layout. */
	memset(shm, 0, shared_size);

	shm->ctrl.layout_version = SYN_SHM_LAYOUT_VERSION;
	shm->ctrl.ring_entries = SYN_IPC_RING_ENTRIES;
	shm->ctrl.shared_size = shared_size;
	syn_ipc_ring_reset(&shm->ring_c0_to_c1);
	syn_ipc_ring_reset(&shm->ring_c1_to_c0);

	barrier_dmem_fence_full();
	shm->ctrl.magic = SYN_SHM_MAGIC;
	barrier_dmem_fence_full();

	tx_ring = &shm->ring_c0_to_c1;
	rx_ring = &shm->ring_c1_to_c0;
#endif

	k_sem_init(&rx_event, 0, 1);
	k_msgq_init(&rx_queue, rx_queue_buf, sizeof(syn_ipc_msg_t),
		    SYN_IPC_RING_ENTRIES);

	MAILBOX_Init(MAILBOX);

	IRQ_CONNECT(MAILBOX_IRQn, 2, syn_ipc_mailbox_isr, NULL, 0);
	irq_enable(MAILBOX_IRQn);

	k_thread_create(&rx_thread, rx_thread_stack,
			K_THREAD_STACK_SIZEOF(rx_thread_stack),
			syn_ipc_rx_thread, NULL, NULL, NULL,
			K_PRIO_PREEMPT(2), 0, K_NO_WAIT);
	k_thread_name_set(&rx_thread, "syn_ipc_rx");

	ipc_ready = true;

	/* Announce readiness for the boot handshake (3.3). */
#if defined(CONFIG_SOC_MCXN947_CPU1)
	shm->ctrl.cpu1_ready = 1;
#else
	shm->ctrl.cpu0_ready = 1;
#endif
	barrier_dmem_fence_full();

	LOG_INF("IPC ready (%s, ring %u entries, region %u bytes)",
		IS_ENABLED(CONFIG_SOC_MCXN947_CPU1) ? "CPU1" : "CPU0",
		(unsigned)SYN_IPC_RING_ENTRIES, (unsigned)shared_size);
	return 0;
}

int syn_ipc_send(const syn_ipc_msg_t *msg)
{
	if (!ipc_ready) {
		return -ENODEV;
	}
	if (msg == NULL || msg->type >= SYN_IPC_TYPE_COUNT) {
		return -EINVAL;
	}

	syn_ipc_msg_t out = *msg;

	if (out.msg_id == 0U) {
		out.msg_id = (uint32_t)atomic_inc(&msg_id_next) + 1U;
	}
	out.timestamp_us = k_cyc_to_us_floor32(k_cycle_get_32());

	int ret = syn_ipc_ring_push(tx_ring, &out);

	if (ret != 0) {
		return ret; /* -EAGAIN: ring full */
	}

	/* Kick the other core's mailbox IRQ. */
	MAILBOX_SetValueBits(MAILBOX, SYN_IPC_OTHER_CORE, 1U);

	return 0;
}

int syn_ipc_receive(syn_ipc_msg_t *msg, uint32_t timeout_ms)
{
	if (!ipc_ready) {
		return -ENODEV;
	}
	if (msg == NULL) {
		return -EINVAL;
	}

	k_timeout_t timeout = (timeout_ms == 0U) ? K_NO_WAIT
						 : K_MSEC(timeout_ms);

	if (k_msgq_get(&rx_queue, msg, timeout) != 0) {
		return -EAGAIN;
	}
	return 0;
}

int syn_ipc_register_handler(syn_ipc_type_t type,
			     syn_ipc_handler_t handler, void *ctx)
{
	if ((int)type < 0 || type >= SYN_IPC_TYPE_COUNT) {
		return -EINVAL;
	}

	/* ctx must be visible before the handler pointer enables
	 * dispatch from the RX thread.
	 */
	handlers[type].ctx = ctx;
	barrier_dmem_fence_full();
	handlers[type].fn = handler;

	return 0;
}
