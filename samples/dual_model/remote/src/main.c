/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file main.c
 * @brief SynapticOS - dual_model sample, CPU1 (application core)
 *
 * Runs on the FRDM-MCXN947 second core (board frdm_mcxn947_cpu1,
 * flashed to flash bank 1). CPU0 releases this core after publishing
 * the shared IPC region.
 *
 * Application logic: after the boot handshake, alternate cross-core
 * inference between the two models CPU0 serves:
 *  - face_detect: 96x96x3 frame (synthetic camera pattern),
 *    REALTIME priority
 *  - keyword_spot: 49x10 MFCC window (synthetic audio pattern),
 *    NORMAL priority
 *
 * CPU1 has no console; progress is visible on CPU0 ('syn ipc
 * status' shows served counts and every 100th serve is logged).
 * A 1 Hz STATUS_REQ heartbeat rides alongside the inference
 * traffic so the link state stays fresh.
 */

#include <zephyr/kernel.h>
#include <zephyr/devicetree.h>
#include <synaptic/syn_ipc.h>

#include "syn_infer_remote.h"

#define SYN_SHARED_NODE DT_NODELABEL(syn_shared)

#define FACE_INPUT_LEN  (96 * 96 * 3)
#define FACE_OUTPUT_LEN 144
#define KWS_INPUT_LEN   (49 * 10)
#define KWS_OUTPUT_LEN  12

static uint8_t output_buf[FACE_OUTPUT_LEN]; /* largest of the two */

static void heartbeat(void)
{
	syn_ipc_msg_t req = {
		.msg_id = 0, /* auto-assigned */
		.type = SYN_IPC_STATUS_REQ,
	};

	(void)syn_ipc_send(&req);

	/* Drain responses; correctness is judged on CPU0 */
	syn_ipc_msg_t msg;

	while (syn_ipc_receive(&msg, 10) == 0) {
	}
}

static int run_one(syn_model_handle_t model, size_t in_len,
		   size_t out_len, syn_priority_t prio, uint32_t seed)
{
	size_t in_cap = 0;
	size_t got = 0;

	/* Zero-copy staging: CPU1's 64 KB RAM cannot hold a private
	 * 27 KB frame buffer, so patterns render straight into the
	 * shared slot.
	 */
	uint8_t *input = syn_remote_infer_input(&in_cap);

	if (input == NULL || in_cap < in_len) {
		return -ENOMEM;
	}
	for (size_t i = 0; i < in_len; i++) {
		input[i] = (uint8_t)((i + seed) & 0xFF);
	}

	return syn_remote_infer(model, input, in_len,
				output_buf, out_len, &got, prio, 1000);
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
		/* No shared region: nothing useful without CPU0 */
		for (;;) {
			k_sleep(K_FOREVER);
		}
	}

	/* Announce ourselves before the inference traffic starts */
	heartbeat();

	/* Resolve both models on CPU0 (retry until it has registered
	 * them; its runtime may still be starting up).
	 */
	syn_model_handle_t face = SYN_MODEL_INVALID;
	syn_model_handle_t kws = SYN_MODEL_INVALID;

	for (int i = 0; i < 50 && face == SYN_MODEL_INVALID; i++) {
		if (syn_remote_model_lookup("face_detect", 200, &face) != 0) {
			face = SYN_MODEL_INVALID;
			k_sleep(K_MSEC(100));
		}
	}
	(void)syn_remote_model_lookup("keyword_spot", 200, &kws);

	if (face == SYN_MODEL_INVALID || kws == SYN_MODEL_INVALID) {
		/* Keep the heartbeat alive so CPU0 can still see us */
		for (;;) {
			heartbeat();
			k_sleep(K_MSEC(1000));
		}
	}

	/* Alternate the two models with different priorities. The
	 * "camera" gets REALTIME, the "microphone" NORMAL, matching
	 * the phase plan's concurrent-priorities demo.
	 */
	uint32_t cycle = 0;

	for (;;) {
		if ((cycle & 1U) == 0U) {
			(void)run_one(face, FACE_INPUT_LEN, FACE_OUTPUT_LEN,
				      SYN_PRIORITY_REALTIME, cycle);
		} else {
			(void)run_one(kws, KWS_INPUT_LEN, KWS_OUTPUT_LEN,
				      SYN_PRIORITY_NORMAL, cycle);
		}

		cycle++;

		/* 1 Hz heartbeat woven into the inference cadence */
		if ((cycle % 20U) == 0U) {
			heartbeat();
		}

		k_sleep(K_MSEC(50));
	}

	return 0;
}
