/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file test_ipc.c
 * @brief Unit tests for syn_ipc inter-core communication
 *
 * Runs single-core on qemu_cortex_m3: validates the shared memory
 * layout ABI that both MCXN947 cores must agree on. The SPSC ring
 * logic tests drive producer and consumer from the same core (the
 * ring code has no core affinity; only interrupt notification does).
 */

#include <zephyr/ztest.h>
#include <synaptic/syn_ipc.h>

#include "syn_shared_layout.h"
#include "syn_ipc_ring.h"

static syn_ipc_ring_t test_ring;

static void ipc_suite_before(void *fixture)
{
	ARG_UNUSED(fixture);
	syn_ipc_ring_reset(&test_ring);
}

ZTEST_SUITE(syn_ipc_suite, NULL, NULL, ipc_suite_before, NULL, NULL);

/* Build a message whose every field derives from the sequence number,
 * so any slot corruption or reordering is detectable on pop.
 */
static syn_ipc_msg_t make_msg(uint32_t seq)
{
	syn_ipc_msg_t m = {
		.msg_id = seq,
		.type = (uint8_t)(seq % 6U),
		.priority = (uint8_t)(seq % 3U),
		.payload_len = (uint16_t)(seq & 0xFFFFU),
		.payload_offset = seq * 7U,
		.timestamp_us = seq * 13U,
		.status = -(int32_t)(seq % 100U),
	};
	return m;
}

static void check_msg(const syn_ipc_msg_t *m, uint32_t seq)
{
	zassert_equal(m->msg_id, seq, "msg_id corrupted at seq %u", seq);
	zassert_equal(m->type, (uint8_t)(seq % 6U), "type corrupted");
	zassert_equal(m->priority, (uint8_t)(seq % 3U), "priority corrupted");
	zassert_equal(m->payload_len, (uint16_t)(seq & 0xFFFFU),
		      "payload_len corrupted");
	zassert_equal(m->payload_offset, seq * 7U, "payload_offset corrupted");
	zassert_equal(m->timestamp_us, seq * 13U, "timestamp corrupted");
	zassert_equal(m->status, -(int32_t)(seq % 100U), "status corrupted");
}

/* The IPC message is a wire format shared between two separately
 * compiled images; its layout is load-bearing ABI.
 */
ZTEST(syn_ipc_suite, test_msg_abi)
{
	zassert_equal(sizeof(syn_ipc_msg_t), 20, "message size changed");
	zassert_equal(offsetof(syn_ipc_msg_t, msg_id), 0, "msg_id moved");
	zassert_equal(offsetof(syn_ipc_msg_t, type), 4, "type moved");
	zassert_equal(offsetof(syn_ipc_msg_t, priority), 5, "priority moved");
	zassert_equal(offsetof(syn_ipc_msg_t, payload_len), 6,
		      "payload_len moved");
	zassert_equal(offsetof(syn_ipc_msg_t, payload_offset), 8,
		      "payload_offset moved");
	zassert_equal(offsetof(syn_ipc_msg_t, timestamp_us), 12,
		      "timestamp_us moved");
	zassert_equal(offsetof(syn_ipc_msg_t, status), 16, "status moved");
}

ZTEST(syn_ipc_suite, test_shared_region_layout)
{
	/* Control block first, then the two rings, then payload */
	zassert_equal(offsetof(syn_shm_region_t, ctrl), 0,
		      "control block must lead the region");
	zassert_equal(offsetof(syn_shm_region_t, ring_c0_to_c1),
		      sizeof(syn_shm_ctrl_t), "ring0 placement changed");
	zassert_equal(offsetof(syn_shm_region_t, ring_c1_to_c0),
		      sizeof(syn_shm_ctrl_t) + sizeof(syn_ipc_ring_t),
		      "ring1 placement changed");

	/* Producer and consumer indices in separate 64-byte blocks */
	zassert_equal(offsetof(syn_ipc_ring_t, head), 0, "head moved");
	zassert_equal(offsetof(syn_ipc_ring_t, tail), 64, "tail moved");
	zassert_equal(offsetof(syn_ipc_ring_t, slots), 128, "slots moved");

	/* Everything incl. the inference slot fits the 96 KB region */
	zassert_true(SYN_SHM_PAYLOAD_OFFSET + sizeof(syn_shm_infer_slot_t) <=
		     SYN_SHM_SHARED_SIZE,
		     "layout exceeds shared region");
	zassert_equal(offsetof(syn_shm_infer_slot_t, input), 64,
		      "infer slot header changed");
	zassert_equal(offsetof(syn_shm_infer_slot_t, output),
		      64 + SYN_SHM_INFER_INPUT_SIZE,
		      "infer slot output placement changed");
}

ZTEST(syn_ipc_suite, test_memory_map_tiles)
{
	/* CPU0 | shared | CPU1 regions must tile without gaps/overlap */
	zassert_equal(SYN_SHM_CPU0_RAM_BASE + SYN_SHM_CPU0_RAM_SIZE,
		      SYN_SHM_SHARED_BASE, "gap between CPU0 and shared");
	zassert_equal(SYN_SHM_SHARED_BASE + SYN_SHM_SHARED_SIZE,
		      SYN_SHM_CPU1_RAM_BASE, "gap between shared and CPU1");

	/* Fits MCXN947 main SRAM: 416 KB at secure alias 0x30000000 */
	zassert_true(SYN_SHM_CPU1_RAM_BASE + SYN_SHM_CPU1_RAM_SIZE <=
		     0x30000000UL + 416 * 1024UL,
		     "map exceeds physical SRAM");
}

ZTEST(syn_ipc_suite, test_ring_empty_after_reset)
{
	syn_ipc_msg_t msg;

	zassert_true(syn_ipc_ring_is_empty(&test_ring), "not empty");
	zassert_false(syn_ipc_ring_is_full(&test_ring), "full when empty");
	zassert_equal(syn_ipc_ring_count(&test_ring), 0, "count not 0");
	zassert_equal(syn_ipc_ring_pop(&test_ring, &msg), -EAGAIN,
		      "pop on empty must return -EAGAIN");
}

ZTEST(syn_ipc_suite, test_ring_push_pop_roundtrip)
{
	syn_ipc_msg_t in = make_msg(42);
	syn_ipc_msg_t out;

	zassert_equal(syn_ipc_ring_push(&test_ring, &in), 0, "push failed");
	zassert_equal(syn_ipc_ring_count(&test_ring), 1, "count not 1");
	zassert_equal(syn_ipc_ring_pop(&test_ring, &out), 0, "pop failed");
	check_msg(&out, 42);
	zassert_true(syn_ipc_ring_is_empty(&test_ring), "not drained");
}

ZTEST(syn_ipc_suite, test_ring_full_returns_eagain)
{
	syn_ipc_msg_t msg;

	for (uint32_t i = 0; i < SYN_IPC_RING_ENTRIES; i++) {
		msg = make_msg(i);
		zassert_equal(syn_ipc_ring_push(&test_ring, &msg), 0,
			      "push %u failed before capacity", i);
	}

	zassert_true(syn_ipc_ring_is_full(&test_ring), "not full");
	zassert_equal(syn_ipc_ring_count(&test_ring), SYN_IPC_RING_ENTRIES,
		      "count wrong at capacity");

	msg = make_msg(999);
	zassert_equal(syn_ipc_ring_push(&test_ring, &msg), -EAGAIN,
		      "push on full ring must return -EAGAIN");

	/* Draining one slot makes room for exactly one more */
	zassert_equal(syn_ipc_ring_pop(&test_ring, &msg), 0, "pop failed");
	check_msg(&msg, 0);
	zassert_equal(syn_ipc_ring_push(&test_ring, &msg), 0,
		      "push after drain failed");
}

ZTEST(syn_ipc_suite, test_ring_fifo_order)
{
	syn_ipc_msg_t msg;

	for (uint32_t i = 0; i < SYN_IPC_RING_ENTRIES; i++) {
		msg = make_msg(i);
		zassert_equal(syn_ipc_ring_push(&test_ring, &msg), 0,
			      "push failed");
	}
	for (uint32_t i = 0; i < SYN_IPC_RING_ENTRIES; i++) {
		zassert_equal(syn_ipc_ring_pop(&test_ring, &msg), 0,
			      "pop failed");
		check_msg(&msg, i);
	}
}

/* Acceptance criterion 3.2: 10,000 messages without loss or
 * corruption, including thousands of index wraparounds. Producer
 * and consumer are driven alternately from this core; the ring code
 * is identical to the cross-core configuration.
 */
ZTEST(syn_ipc_suite, test_ring_10k_messages_no_loss)
{
	syn_ipc_msg_t msg;
	uint32_t pushed = 0, popped = 0;

	while (popped < 10000U) {
		/* Bursts of varying size exercise every fill level */
		uint32_t burst = (pushed % SYN_IPC_RING_ENTRIES) + 1U;

		for (uint32_t i = 0; i < burst && pushed < 10000U; i++) {
			msg = make_msg(pushed);
			if (syn_ipc_ring_push(&test_ring, &msg) != 0) {
				break; /* full: drain below */
			}
			pushed++;
		}
		while (syn_ipc_ring_pop(&test_ring, &msg) == 0) {
			check_msg(&msg, popped);
			popped++;
		}
	}

	zassert_equal(pushed, 10000U, "pushed %u of 10000", pushed);
	zassert_equal(popped, 10000U, "popped %u of 10000", popped);
	zassert_true(syn_ipc_ring_is_empty(&test_ring), "residue in ring");
}

/* Free-running index wraparound: force head/tail near UINT32_MAX and
 * verify count arithmetic and slot addressing survive the wrap.
 */
ZTEST(syn_ipc_suite, test_ring_index_wraparound)
{
	syn_ipc_msg_t msg;

	syn_ipc_ring_reset(&test_ring);
	test_ring.head = UINT32_MAX - 2U;
	test_ring.tail = UINT32_MAX - 2U;

	for (uint32_t i = 0; i < SYN_IPC_RING_ENTRIES; i++) {
		msg = make_msg(i);
		zassert_equal(syn_ipc_ring_push(&test_ring, &msg), 0,
			      "push across wrap failed at %u", i);
	}
	zassert_true(syn_ipc_ring_is_full(&test_ring),
		     "full check broken across wrap");

	for (uint32_t i = 0; i < SYN_IPC_RING_ENTRIES; i++) {
		zassert_equal(syn_ipc_ring_pop(&test_ring, &msg), 0,
			      "pop across wrap failed at %u", i);
		check_msg(&msg, i);
	}
	zassert_true(syn_ipc_ring_is_empty(&test_ring),
		     "empty check broken across wrap");
	zassert_true(test_ring.head < SYN_IPC_RING_ENTRIES,
		     "head did not wrap");
}
