/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file main.c
 * @brief SynapticOS - ota_update sample (Phase 4.5)
 *
 * End-to-end model OTA demo:
 *
 *  1. Boots with a built-in factory model installed into the store
 *     (first boot only; on later boots the persistent registry
 *     restores whatever was active, surviving updates and rollbacks).
 *  2. Runs a periodic inference tick that prints which model and
 *     which flash slot served it.
 *  3. A new model arrives over the shell UART, driven by the host:
 *
 *       python3 tools/syn_model_pack.py --input blob.bin --name demo \
 *           --input-shape 1,8,8 --output-shape 1,16 --output demo.synm
 *       python3 tools/syn_ota_send.py --port /dev/ttyACM0 demo.synm
 *
 *     which streams `syn ota begin/data/done/activate`. The tick
 *     switches models without dropping a beat (hot-swap keeps the
 *     in-flight inference intact; the store hot-loads the update).
 *  4. `syn ota rollback` on the shell restores the previous model.
 *
 * On the FRDM-MCXN947 the store is auto-initialized over the bank 1
 * partition map (SYNAPTIC_STORE_AUTO_INIT). On QEMU there is no
 * backing flash: a RAM-emulated port stands in, so the whole flow
 * (minus true reboot persistence) can be demonstrated in emulation.
 *
 * Dual-core note: this sample is single-core. On a board whose bank 1
 * still carries a CPU1 image (dual_model flashing), the partition map
 * keeps every OTA erase/write above the CPU1 reserve; the full
 * dual-core interaction (CPU1 parked during the session and resumed
 * after) is exercised by running the same `syn ota` commands inside
 * the dual_model sample, which also enables the store.
 */

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <string.h>

#include <synaptic/syn_api.h>
#include <synaptic/syn_model.h>
#include <synaptic/syn_infer.h>
#include <synaptic/syn_model_ota.h>

#include "syn_model_store.h"
#include "syn_model_internal.h"
#include "syn_flash_port.h"

LOG_MODULE_REGISTER(ota_update, LOG_LEVEL_INF);

#define TICK_SECONDS 3

#ifndef CONFIG_SOC_FLASH_MCUX
/* QEMU: RAM-emulated flash, board-like geometry, scaled down */
#define QSECTOR 1024U
#define QSLOT   6144U
static uint8_t qemu_flash[2U * QSECTOR + 2U * QSLOT];
static syn_flash_port_t qemu_port;
static syn_flash_ram_ctx_t qemu_ctx;

static const syn_store_layout_t qemu_layout = {
	.registry_off = { 0U, QSECTOR },
	.registry_size = QSECTOR,
	.slot_off = { 2U * QSECTOR, 2U * QSECTOR + QSLOT },
	.slot_size = QSLOT,
};
#endif

/* factory model: an opaque stub blob (the NPU HAL is a stub until
 * the Neutron invoke path lands; .synm packaging works the same for
 * real .tflite files)
 */
static uint8_t factory_blob[1024];

static int store_bringup(void)
{
#ifdef CONFIG_SOC_FLASH_MCUX
	/* SYNAPTIC_STORE_AUTO_INIT already ran before main() */
	if (!syn_store_ready()) {
		LOG_ERR("Model store unavailable (see boot log)");
		return -ENODEV;
	}
	return 0;
#else
	int ret = syn_flash_port_ram_init(&qemu_port, &qemu_ctx, qemu_flash,
					  sizeof(qemu_flash), QSECTOR, 128U);

	if (ret == 0) {
		ret = syn_store_init(&qemu_port, &qemu_layout);
	}
	if (ret != 0) {
		LOG_ERR("RAM store init failed: %d", ret);
	}
	return ret;
#endif
}

static int factory_install(void)
{
	if (syn_store_active_slot() != SYN_STORE_SLOT_NONE) {
		LOG_INF("Persistent model found; skipping factory install");
		return 0;
	}

	for (size_t i = 0; i < sizeof(factory_blob); i++) {
		factory_blob[i] = (uint8_t)(i * 31U + 5U);
	}

	syn_model_info_t info = {0};

	strncpy(info.name, "demo_model", sizeof(info.name) - 1);
	strncpy(info.version, "factory", sizeof(info.version) - 1);
	info.input_size = 64;
	info.output_size = 16;
	info.input_shape[0] = 1;
	info.input_shape[1] = 8;
	info.input_shape[2] = 8;
	info.output_shape[0] = 1;
	info.output_shape[1] = 16;

	syn_model_handle_t h;
	int ret = syn_store_install(&info, factory_blob,
				    sizeof(factory_blob), &h);

	if (ret != 0) {
		LOG_ERR("Factory install failed: %d", ret);
		return ret;
	}
	LOG_INF("Factory model installed into slot %u (stub blob, %u bytes)",
		syn_store_active_slot(), (unsigned)sizeof(factory_blob));
	return 0;
}

/* Resolve and, if needed, load whichever model is currently active. */
static int active_model(syn_model_handle_t *h, syn_model_info_t *info)
{
	uint8_t slot = syn_store_active_slot();

	if (slot == SYN_STORE_SLOT_NONE) {
		return -ENOENT;
	}
	if (syn_store_slot_info(slot, info) != 0 ||
	    syn_model_get_by_name(info->name, h) != 0) {
		return -ENOENT;
	}
	if (!syn_model_is_loaded(*h)) {
		int ret = syn_model_load(*h);

		if (ret != 0 && ret != -EALREADY) {
			return ret;
		}
	}
	return 0;
}

int main(void)
{
	LOG_INF("SynapticOS %s - ota_update", syn_version());

	int ret = syn_init();

	if (ret != 0 && ret != -EALREADY) {
		LOG_ERR("syn_init failed: %d", ret);
		return ret;
	}

	if (store_bringup() != 0 || factory_install() != 0) {
		return -ENODEV;
	}

	static uint8_t in_buf[64];
	static uint8_t out_buf[64];
	uint32_t tick = 0;

	memset(in_buf, 0x42, sizeof(in_buf));

	LOG_INF("Inference tick every %u s; update via "
		"tools/syn_ota_send.py, revert via 'syn ota rollback'",
		(unsigned)TICK_SECONDS);

	while (1) {
		syn_model_handle_t h;
		syn_model_info_t info;

		ret = active_model(&h, &info);
		if (ret != 0) {
			LOG_WRN("tick %u: no usable active model (%d)",
				tick, ret);
			k_sleep(K_SECONDS(TICK_SECONDS));
			tick++;
			continue;
		}

		syn_tensor_t in = {
			.data = in_buf,
			.size = MIN(info.input_size, sizeof(in_buf)),
			.ndim = 1,
		};
		in.shape[0] = in.size;

		syn_tensor_t out = {
			.data = out_buf,
			.size = sizeof(out_buf),
		};

		ret = syn_infer_run_sync(h, &in, &out, SYN_PRIORITY_NORMAL);
		if (ret == 0) {
			LOG_INF("tick %u: served by '%s' (%s) slot %u, "
				"%u output bytes",
				tick, info.name, info.version,
				syn_store_active_slot(), (unsigned)out.size);
		} else {
			LOG_ERR("tick %u: inference failed: %d", tick, ret);
		}

		/* stage buffers are ephemeral arena allocations; the
		 * result was copied into out_buf above
		 */
		syn_mem_reset_ephemeral();

		k_sleep(K_SECONDS(TICK_SECONDS));
		tick++;
	}
	return 0;
}
