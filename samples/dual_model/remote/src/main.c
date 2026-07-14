/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file main.c
 * @brief SynapticOS - dual_model sample, CPU1 (application core)
 *
 * Runs on the FRDM-MCXN947 second core (board frdm_mcxn947_cpu1,
 * flashed to flash bank 1). CPU0 releases this core after publishing
 * the shared IPC region.
 *
 * Boot handshake: attach to the shared region (retrying briefly in
 * case of reset races), send SYN_IPC_STATUS_REQ, then keep a 1 Hz
 * heartbeat going. CPU1 has no console; its liveness is visible on
 * CPU0 via 'syn ipc status'.
 */

#include <zephyr/kernel.h>
#include <zephyr/devicetree.h>
#include <synaptic/syn_ipc.h>

#include "syn_infer_remote.h"

#define SYN_SHARED_NODE DT_NODELABEL(syn_shared)

/* Smoke-test the cross-core inference protocol once at boot: resolve
 * the face_detect model and run a single request. The result is
 * reported to CPU0 implicitly (its console logs the served request);
 * a local failure just skips ahead to the heartbeat loop.
 */
static void infer_smoke_test(void)
{
	syn_model_handle_t model;
	static uint8_t output[144];
	size_t in_cap = 0, out_len = 0;
	const size_t in_len = 96 * 96 * 3;

	if (syn_remote_model_lookup("face_detect", 500, &model) != 0) {
		return;
	}

	/* CPU1 has 64 KB of RAM: stage the frame zero-copy in the
	 * shared slot instead of a private 27 KB buffer.
	 */
	uint8_t *input = syn_remote_infer_input(&in_cap);

	if (input == NULL || in_cap < in_len) {
		return;
	}
	for (size_t i = 0; i < in_len; i++) {
		input[i] = (uint8_t)(i & 0xFF);
	}

	(void)syn_remote_infer(model, input, in_len,
			       output, sizeof(output), &out_len,
			       SYN_PRIORITY_NORMAL, 1000);
}

int main(void)
{
	void *shm_base = (void *)DT_REG_ADDR(SYN_SHARED_NODE);
	size_t shm_size = DT_REG_SIZE(SYN_SHARED_NODE);
	int ret = -ENODEV;

	/* CPU0 published the region before releasing this core, but
	 * tolerate a slow CPU0 after independent resets.
	 */
	for (int i = 0; i < 100 && ret != 0; i++) {
		ret = syn_ipc_init(shm_base, shm_size);
		if (ret != 0) {
			k_sleep(K_MSEC(10));
		}
	}
	if (ret != 0) {
		/* No shared region: nothing useful to do without CPU0 */
		for (;;) {
			k_sleep(K_FOREVER);
		}
	}

	infer_smoke_test();

	/* Handshake, then 1 Hz heartbeat */
	for (;;) {
		syn_ipc_msg_t req = {
			.msg_id = 0, /* auto-assigned */
			.type = SYN_IPC_STATUS_REQ,
			.priority = 0,
			.payload_len = 0,
			.payload_offset = 0,
			.status = 0,
		};

		(void)syn_ipc_send(&req);

		/* Drain any responses; correctness is judged on CPU0 */
		syn_ipc_msg_t msg;

		while (syn_ipc_receive(&msg, 100) == 0) {
			/* STATUS_RESP consumed; nothing else expected yet */
		}

		k_sleep(K_MSEC(900));
	}

	return 0;
}
