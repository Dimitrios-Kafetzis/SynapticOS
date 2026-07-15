/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file syn_model_ota.c
 * @brief SynapticOS - A/B model updates (Phase 4.2)
 *
 * Streams a .synm image (64-byte header + model payload) into the
 * staging slot and hands the result to the model store:
 *
 *   begin()        erase staging slot, IDLE/ERROR/READY -> DOWNLOADING
 *   write_chunk()  page-buffered programming (any chunk size; the
 *                  reference transfer uses 4 KB chunks)
 *   finish()       -> VALIDATING (CRC32 of the payload as it actually
 *                  landed in flash vs the header) -> STAGING (registry
 *                  commit of the staged record) -> READY
 *   activate()     staged record becomes active, the previous active
 *                  model becomes the rollback candidate; -> IDLE
 *   rollback()     previous model becomes active again
 *
 * Power-loss safety: nothing before the STAGING registry commit is
 * referenced anywhere, so a reset mid-transfer leaves the old model
 * untouched; the commit itself is the store's ping-pong commit. A
 * staged-but-not-activated update survives reboot and can still be
 * activated (activate() from IDLE with a staged slot present).
 *
 * Dual-core (board only): CPU1 executes in place from bank 1, where
 * the staging slot lives, so begin() holds CPU1 in reset for the
 * whole session and it is re-released when the session ends
 * (activate, rollback, or the transition into ERROR). Cross-core
 * offload is paused during OTA; callers must be quiescent.
 */

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <zephyr/sys/crc.h>
#include <string.h>

LOG_MODULE_REGISTER(syn_model_ota, CONFIG_SYNAPTIC_LOG_LEVEL);

#include <synaptic/syn_model_ota.h>
#include <synaptic/syn_model.h>

#include "syn_model_ota_internal.h"
#include "syn_model_store.h"
#include "syn_flash_port.h"
#include "syn_synm.h"

#if defined(CONFIG_SYNAPTIC_DUAL_CORE) && !defined(CONFIG_SOC_MCXN947_CPU1)
#include "syn_boot_internal.h"
#define OTA_CPU1_RESUME_TIMEOUT_MS 500U
#endif

static struct {
	syn_ota_state_t state;
	uint8_t slot;
	bool cpu1_parked;      /* we stopped CPU1; resume at session end */
	char name[32];
	uint32_t total_size;   /* expected .synm bytes (header incl.) */
	uint32_t received;
	uint32_t t_begin;      /* cycle counter at begin() */
	uint32_t session_us;
	int last_error;

	struct syn_synm_hdr hdr;   /* filled from the first 64 bytes */

	/* page-write buffering */
	uint8_t page_buf[256];
	uint32_t buffered;     /* valid bytes in page_buf */
	uint32_t write_off;    /* slot-relative offset of next page write */
} ota;

static K_MUTEX_DEFINE(ota_lock);

static void session_clock(void)
{
	ota.session_us = k_cyc_to_us_floor32(k_cycle_get_32() - ota.t_begin);
}

static void cpu1_park(void)
{
#if defined(CONFIG_SYNAPTIC_DUAL_CORE) && !defined(CONFIG_SOC_MCXN947_CPU1)
	if (syn_boot_secondary_linked()) {
		if (syn_boot_secondary_stop() == 0) {
			ota.cpu1_parked = true;
			LOG_WRN("OTA session: CPU1 parked, offload paused");
		}
	}
#endif
}

static void cpu1_unpark(void)
{
#if defined(CONFIG_SYNAPTIC_DUAL_CORE) && !defined(CONFIG_SOC_MCXN947_CPU1)
	if (ota.cpu1_parked) {
		ota.cpu1_parked = false;
		int ret = syn_boot_secondary_resume(OTA_CPU1_RESUME_TIMEOUT_MS);

		if (ret != 0) {
			LOG_ERR("CPU1 resume after OTA failed: %d "
				"(continuing single-core)", ret);
		} else {
			LOG_INF("OTA session over: CPU1 resumed");
		}
	}
#endif
}

/* Session over, unsuccessfully. Always returns err for tail calls. */
static int enter_error(int err)
{
	ota.state = SYN_OTA_STATE_ERROR;
	ota.last_error = err;
	session_clock();
	cpu1_unpark();
	LOG_ERR("OTA error: %d", err);
	return err;
}

int syn_ota_begin(const char *model_name, size_t total_size)
{
	if (model_name == NULL || model_name[0] == '\0' ||
	    total_size <= SYN_SYNM_HDR_SIZE) {
		return -EINVAL;
	}
	if (!syn_store_ready()) {
		return -ENODEV;
	}

	k_mutex_lock(&ota_lock, K_FOREVER);

	if (ota.state == SYN_OTA_STATE_DOWNLOADING ||
	    ota.state == SYN_OTA_STATE_VALIDATING ||
	    ota.state == SYN_OTA_STATE_STAGING) {
		LOG_WRN("Abandoning OTA session in state %s",
			syn_ota_state_str(ota.state));
	}

	uint8_t slot;
	uint32_t slot_off, slot_size;

	(void)syn_store_staging_slot(&slot);
	(void)syn_store_slot_bounds(slot, &slot_off, &slot_size);

	if (total_size > slot_size) {
		k_mutex_unlock(&ota_lock);
		return -EFBIG;
	}

	/* a staged-but-unactivated predecessor is superseded */
	if (syn_store_staged_slot() != SYN_STORE_SLOT_NONE) {
		(void)syn_store_clear_staged();
	}

	ota.t_begin = k_cycle_get_32();
	ota.slot = slot;
	ota.total_size = (uint32_t)total_size;
	ota.received = 0;
	ota.buffered = 0;
	ota.write_off = 0;
	ota.last_error = 0;
	memset(ota.name, 0, sizeof(ota.name));
	strncpy(ota.name, model_name, sizeof(ota.name) - 1);
	memset(&ota.hdr, 0, sizeof(ota.hdr));

	/* bank 1 flash work starts here: quiesce CPU1 first */
	cpu1_park();

	const syn_flash_port_t *port = syn_store_port();
	uint32_t sector = port->sector_size;
	uint32_t erase_len = (ota.total_size + sector - 1U) / sector * sector;

	LOG_INF("Erasing staging slot %u: %u sectors",
		slot, erase_len / sector);

	int ret = syn_flash_erase_sectors(port, slot_off, erase_len);

	if (ret != 0) {
		k_mutex_unlock(&ota_lock);
		return enter_error(ret);
	}

	ota.state = SYN_OTA_STATE_DOWNLOADING;
	session_clock();
	k_mutex_unlock(&ota_lock);

	LOG_INF("OTA begin: '%s' %u bytes into slot %u",
		ota.name, ota.total_size, slot);
	return 0;
}

/* Flush buffered bytes in page multiples; final pads the tail 0xFF. */
static int flush_pages(bool final)
{
	const syn_flash_port_t *port = syn_store_port();
	uint32_t page = port->page_size;
	uint32_t slot_off;

	(void)syn_store_slot_bounds(ota.slot, &slot_off, NULL);

	uint32_t writable = final ? ((ota.buffered + page - 1U) / page * page)
				  : (ota.buffered / page * page);

	if (writable == 0U) {
		return 0;
	}
	if (final && writable > ota.buffered) {
		memset(&ota.page_buf[ota.buffered], 0xFF,
		       writable - ota.buffered);
	}

	int ret = syn_flash_write(port, slot_off + ota.write_off,
				  ota.page_buf, writable);
	if (ret != 0) {
		return ret;
	}

	ota.write_off += writable;

	uint32_t left = ota.buffered - MIN(ota.buffered, writable);

	if (left > 0U) {
		memmove(ota.page_buf, &ota.page_buf[writable], left);
	}
	ota.buffered = left;
	return 0;
}

int syn_ota_write_chunk(const uint8_t *data, size_t len)
{
	if (data == NULL || len == 0U) {
		return -EINVAL;
	}

	k_mutex_lock(&ota_lock, K_FOREVER);

	if (ota.state != SYN_OTA_STATE_DOWNLOADING) {
		k_mutex_unlock(&ota_lock);
		return -EPERM;
	}
	if (ota.received + len > ota.total_size) {
		k_mutex_unlock(&ota_lock);
		return enter_error(-EFBIG);
	}

	/* capture the header as it streams past */
	if (ota.received < SYN_SYNM_HDR_SIZE) {
		uint32_t need = SYN_SYNM_HDR_SIZE - ota.received;
		uint32_t take = MIN(need, (uint32_t)len);

		memcpy((uint8_t *)&ota.hdr + ota.received, data, take);

		if (ota.received + take == SYN_SYNM_HDR_SIZE) {
			if (ota.hdr.magic != SYN_SYNM_MAGIC ||
			    ota.hdr.version != SYN_SYNM_VERSION) {
				k_mutex_unlock(&ota_lock);
				return enter_error(-EILSEQ);
			}
			if (ota.hdr.model_size !=
			    ota.total_size - SYN_SYNM_HDR_SIZE) {
				k_mutex_unlock(&ota_lock);
				return enter_error(-EMSGSIZE);
			}
			if (strncmp(ota.hdr.name, ota.name,
				    sizeof(ota.hdr.name)) != 0) {
				LOG_ERR("OTA name mismatch: header '%.32s' "
					"vs begin '%s'", ota.hdr.name,
					ota.name);
				k_mutex_unlock(&ota_lock);
				return enter_error(-EINVAL);
			}
		}
	}

	/* page-buffered programming */
	const uint8_t *src = data;
	size_t remaining = len;

	while (remaining > 0U) {
		uint32_t space = sizeof(ota.page_buf) - ota.buffered;
		uint32_t take = MIN(space, (uint32_t)remaining);

		memcpy(&ota.page_buf[ota.buffered], src, take);
		ota.buffered += take;
		src += take;
		remaining -= take;

		if (ota.buffered == sizeof(ota.page_buf)) {
			int ret = flush_pages(false);

			if (ret != 0) {
				k_mutex_unlock(&ota_lock);
				return enter_error(ret);
			}
		}
	}

	ota.received += (uint32_t)len;
	session_clock();
	k_mutex_unlock(&ota_lock);
	return 0;
}

int syn_ota_finish(void)
{
	k_mutex_lock(&ota_lock, K_FOREVER);

	if (ota.state != SYN_OTA_STATE_DOWNLOADING) {
		k_mutex_unlock(&ota_lock);
		return -EPERM;
	}
	if (ota.received != ota.total_size) {
		k_mutex_unlock(&ota_lock);
		return enter_error(-ENODATA);
	}

	int ret = flush_pages(true);

	if (ret != 0) {
		k_mutex_unlock(&ota_lock);
		return enter_error(ret);
	}

	/* integrity: CRC32 of the payload as it landed in flash */
	ota.state = SYN_OTA_STATE_VALIDATING;

	const syn_flash_port_t *port = syn_store_port();
	uint32_t slot_off;
	uint8_t buf[128];
	uint32_t crc = 0U;
	uint32_t pos = 0U;

	(void)syn_store_slot_bounds(ota.slot, &slot_off, NULL);

	while (pos < ota.hdr.model_size) {
		uint32_t n = MIN(sizeof(buf), ota.hdr.model_size - pos);

		ret = syn_flash_read(port, slot_off + SYN_SYNM_HDR_SIZE + pos,
				     buf, n);
		if (ret != 0) {
			k_mutex_unlock(&ota_lock);
			return enter_error(ret);
		}
		crc = crc32_ieee_update(crc, buf, n);
		pos += n;
	}

	if (crc != ota.hdr.crc32) {
		LOG_ERR("OTA payload CRC mismatch: header 0x%08x flash 0x%08x",
			ota.hdr.crc32, crc);
		k_mutex_unlock(&ota_lock);
		return enter_error(-EILSEQ);
	}

	/* stage: commit the record so the update survives power loss */
	ota.state = SYN_OTA_STATE_STAGING;

	syn_model_info_t info = {0};

	memcpy(info.name, ota.name, sizeof(info.name));
	strncpy(info.version, "ota", sizeof(info.version) - 1);
	info.flash_offset = port->base + slot_off + SYN_SYNM_HDR_SIZE;
	info.flash_size = ota.hdr.model_size;
	info.crc32 = ota.hdr.crc32;

	uint32_t in_elems = 1U, out_elems = 1U;

	for (int i = 0; i < 4; i++) {
		info.input_shape[i] = (uint8_t)MIN(ota.hdr.input_shape[i],
						   (uint16_t)UINT8_MAX);
		info.output_shape[i] = (uint8_t)MIN(ota.hdr.output_shape[i],
						    (uint16_t)UINT8_MAX);
		if (ota.hdr.input_shape[i] != 0U) {
			in_elems *= ota.hdr.input_shape[i];
		}
		if (ota.hdr.output_shape[i] != 0U) {
			out_elems *= ota.hdr.output_shape[i];
		}
	}
	/* stub-NPU convention: INT8 tensors, size = element count */
	info.input_size = in_elems;
	info.output_size = out_elems;
	info.input_dtype = SYN_NPU_DTYPE_INT8;
	info.output_dtype = SYN_NPU_DTYPE_INT8;

	ret = syn_store_mark_staged(ota.slot, &info);
	if (ret != 0) {
		k_mutex_unlock(&ota_lock);
		return enter_error(ret);
	}

	ota.state = SYN_OTA_STATE_READY;
	session_clock();
	k_mutex_unlock(&ota_lock);

	LOG_INF("OTA staged: '%s' %u bytes crc 0x%08x (slot %u), "
		"ready to activate", ota.name, ota.hdr.model_size,
		ota.hdr.crc32, ota.slot);
	return 0;
}

int syn_ota_activate(void)
{
	k_mutex_lock(&ota_lock, K_FOREVER);

	uint8_t slot = ota.slot;

	if (ota.state != SYN_OTA_STATE_READY) {
		/* power-loss recovery: a staged slot from a previous
		 * boot can be activated from IDLE
		 */
		slot = syn_store_staged_slot();
		if (ota.state != SYN_OTA_STATE_IDLE ||
		    slot == SYN_STORE_SLOT_NONE) {
			k_mutex_unlock(&ota_lock);
			return -EPERM;
		}
	}

	int ret = syn_store_activate(slot);

	if (ret != 0) {
		k_mutex_unlock(&ota_lock);
		return enter_error(ret);
	}

	ota.state = SYN_OTA_STATE_IDLE;
	session_clock();
	cpu1_unpark();
	k_mutex_unlock(&ota_lock);

	LOG_INF("OTA activated slot %u (%u us since begin)",
		slot, ota.session_us);
	return 0;
}

int syn_ota_rollback(void)
{
	k_mutex_lock(&ota_lock, K_FOREVER);

	if (ota.state == SYN_OTA_STATE_DOWNLOADING ||
	    ota.state == SYN_OTA_STATE_VALIDATING ||
	    ota.state == SYN_OTA_STATE_STAGING) {
		k_mutex_unlock(&ota_lock);
		return -EBUSY;
	}

	int ret = syn_store_rollback();

	if (ret != 0) {
		k_mutex_unlock(&ota_lock);
		return ret;
	}

	ota.state = SYN_OTA_STATE_IDLE;
	cpu1_unpark();
	k_mutex_unlock(&ota_lock);

	LOG_INF("OTA rollback: previous model active again");
	return 0;
}

syn_ota_state_t syn_ota_get_state(void)
{
	return ota.state;
}

void syn_ota_get_status(syn_ota_status_t *status)
{
	if (status == NULL) {
		return;
	}
	status->state = ota.state;
	status->slot = (ota.state == SYN_OTA_STATE_IDLE) ?
		       SYN_STORE_SLOT_NONE : ota.slot;
	status->total_size = ota.total_size;
	status->received = ota.received;
	status->session_us = ota.session_us;
	status->last_error = ota.last_error;
}

const char *syn_ota_state_str(syn_ota_state_t state)
{
	switch (state) {
	case SYN_OTA_STATE_IDLE:        return "IDLE";
	case SYN_OTA_STATE_DOWNLOADING: return "DOWNLOADING";
	case SYN_OTA_STATE_VALIDATING:  return "VALIDATING";
	case SYN_OTA_STATE_STAGING:     return "STAGING";
	case SYN_OTA_STATE_READY:       return "READY";
	case SYN_OTA_STATE_ERROR:       return "ERROR";
	default:                        return "?";
	}
}

void syn_ota_reset(void)
{
	k_mutex_lock(&ota_lock, K_FOREVER);
	memset(&ota, 0, sizeof(ota));
	ota.slot = SYN_STORE_SLOT_NONE;
	k_mutex_unlock(&ota_lock);
}
