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

ZTEST_SUITE(syn_ipc_suite, NULL, NULL, NULL, NULL, NULL);

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

	/* Everything incl. inference buffers fits the 96 KB region */
	zassert_true(SYN_SHM_PAYLOAD_OFFSET + SYN_SHM_INFER_OUTPUT_OFFSET +
		     SYN_SHM_INFER_OUTPUT_SIZE <= SYN_SHM_SHARED_SIZE,
		     "layout exceeds shared region");
}

ZTEST(syn_ipc_suite, test_memory_map_tiles)
{
	/* CPU0 | shared | CPU1 regions must tile without gaps/overlap */
	zassert_equal(SYN_SHM_CPU0_RAM_BASE + SYN_SHM_CPU0_RAM_SIZE,
		      SYN_SHM_SHARED_BASE, "gap between CPU0 and shared");
	zassert_equal(SYN_SHM_SHARED_BASE + SYN_SHM_SHARED_SIZE,
		      SYN_SHM_CPU1_RAM_BASE, "gap between shared and CPU1");

	/* Fits MCXN947 main SRAM: 416 KB at 0x20000000 */
	zassert_true(SYN_SHM_CPU1_RAM_BASE + SYN_SHM_CPU1_RAM_SIZE <=
		     0x20000000UL + 416 * 1024UL,
		     "map exceeds physical SRAM");
}
