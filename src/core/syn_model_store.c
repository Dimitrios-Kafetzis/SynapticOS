/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file syn_model_store.c
 * @brief SynapticOS - flash-backed model store
 *
 * Persistence for the model registry over the syn_flash_port seam.
 *
 * Registry commit protocol (power-loss safe):
 *  - two fixed registry regions hold alternating COPIES of the whole
 *    store state (header + one record per model slot),
 *  - every commit erases and rewrites only the OLDER copy with a
 *    generation number one above the newest, then verifies it by
 *    read-back,
 *  - boot scan adopts the copy with the highest generation that
 *    passes magic/version/CRC32 checks; a torn write (power loss
 *    mid-commit) simply fails CRC and the previous generation stays
 *    authoritative,
 *  - alternating copies also halves erase wear; each copy records
 *    its own cumulative erase count (wear tracking).
 *
 * Model payloads live in the A/B slots as .synm images (64-byte
 * header + raw model). A record only enters the registry after the
 * payload is fully written and CRC-verified, so a registry entry is
 * a guarantee that its slot content is complete: memory-mapped reads
 * of committed payloads are safe even on ECC flash.
 */

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <zephyr/sys/crc.h>
#include <string.h>

LOG_MODULE_REGISTER(syn_store, CONFIG_SYNAPTIC_LOG_LEVEL);

#include "syn_model_store.h"
#include "syn_model_internal.h"
#include "syn_synm.h"

#define STORE_MAGIC   0x524E5953UL /* "SYNR" read as LE uint32 */
#define STORE_VERSION 1U

struct store_hdr {
	uint32_t magic;
	uint16_t version;
	uint16_t rsvd0;
	uint32_t generation;
	uint32_t wear;          /* erase count of this copy, incl. this write */
	uint8_t  active_slot;   /* SYN_STORE_SLOT_NONE if none */
	uint8_t  staged_slot;
	uint8_t  occupied_mask; /* bit n = slot n holds a complete image */
	uint8_t  rsvd1;
	uint32_t crc32;         /* over the whole image, this field zeroed */
};

#define STORE_IMG_SIZE (sizeof(struct store_hdr) + 2U * sizeof(syn_model_info_t))
/* image padded to page multiple at write time */
#define STORE_BUF_SIZE 512U
BUILD_ASSERT(STORE_BUF_SIZE >= STORE_IMG_SIZE + 128U,
	     "store image buffer too small");

static struct {
	bool ready;
	const syn_flash_port_t *port;
	syn_store_layout_t lay;

	/* adopted state */
	uint32_t generation;
	uint8_t active_slot;
	uint8_t staged_slot;
	uint8_t occupied_mask;
	syn_model_info_t rec[2];

	/* per registry copy */
	uint32_t wear[2];
	bool copy_valid[2];
	uint8_t newest_copy;

	/* stats */
	uint32_t scan_us;
	uint32_t last_commit_us;

	struct k_mutex lock;
} st;

static uint8_t img_buf[STORE_BUF_SIZE];
static uint8_t io_buf[256];

static uint32_t cycles_to_us(uint32_t from)
{
	return k_cyc_to_us_floor32(k_cycle_get_32() - from);
}

static uint32_t round_up(uint32_t v, uint32_t align)
{
	return (v + align - 1U) / align * align;
}

/* Serialize current state into img_buf; returns padded length. */
static uint32_t image_build(uint32_t generation, uint32_t wear)
{
	struct store_hdr hdr = {
		.magic = STORE_MAGIC,
		.version = STORE_VERSION,
		.generation = generation,
		.wear = wear,
		.active_slot = st.active_slot,
		.staged_slot = st.staged_slot,
		.occupied_mask = st.occupied_mask,
		.crc32 = 0U,
	};

	memset(img_buf, 0xFF, sizeof(img_buf));
	memcpy(img_buf, &hdr, sizeof(hdr));
	memcpy(&img_buf[sizeof(hdr)], &st.rec[0], sizeof(st.rec[0]));
	memcpy(&img_buf[sizeof(hdr) + sizeof(st.rec[0])], &st.rec[1],
	       sizeof(st.rec[1]));

	uint32_t crc = crc32_ieee(img_buf, STORE_IMG_SIZE);

	memcpy(&img_buf[offsetof(struct store_hdr, crc32)], &crc,
	       sizeof(crc));
	return round_up(STORE_IMG_SIZE, st.port->page_size);
}

/* Read one registry copy into img_buf and validate it. */
static bool copy_load(uint8_t copy, struct store_hdr *hdr_out)
{
	uint32_t off = st.lay.registry_off[copy];
	struct store_hdr hdr;

	if (syn_flash_is_blank(st.port, off, st.port->page_size) != 0) {
		return false; /* blank or unreadable: invalid */
	}
	if (syn_flash_read(st.port, off, img_buf, STORE_IMG_SIZE) != 0) {
		return false; /* torn pages read as errors on ECC flash */
	}

	memcpy(&hdr, img_buf, sizeof(hdr));
	if (hdr.magic != STORE_MAGIC || hdr.version != STORE_VERSION) {
		return false;
	}
	if ((hdr.active_slot > 1U && hdr.active_slot != SYN_STORE_SLOT_NONE) ||
	    (hdr.staged_slot > 1U && hdr.staged_slot != SYN_STORE_SLOT_NONE) ||
	    (hdr.occupied_mask & ~0x3U) != 0U) {
		return false;
	}

	uint32_t stored_crc = hdr.crc32;

	memset(&img_buf[offsetof(struct store_hdr, crc32)], 0,
	       sizeof(uint32_t));
	if (crc32_ieee(img_buf, STORE_IMG_SIZE) != stored_crc) {
		return false;
	}

	*hdr_out = hdr;
	return true;
}

/* Write current state as generation gen into the given copy. */
static int copy_commit(uint8_t copy, uint32_t gen)
{
	uint32_t t0 = k_cycle_get_32();
	uint32_t off = st.lay.registry_off[copy];
	uint32_t wear = st.wear[copy] + 1U;
	uint32_t len = image_build(gen, wear);
	int ret;

	ret = syn_flash_erase(st.port, off, st.lay.registry_size);
	if (ret != 0) {
		st.copy_valid[copy] = false;
		return ret;
	}
	st.wear[copy] = wear; /* the erase happened; count it */

	ret = syn_flash_write(st.port, off, img_buf, len);
	if (ret != 0) {
		st.copy_valid[copy] = false;
		return ret;
	}

	/* read-back verify */
	uint8_t page[64];
	uint32_t pos = 0;

	while (pos < STORE_IMG_SIZE) {
		uint32_t n = MIN(sizeof(page), STORE_IMG_SIZE - pos);

		ret = syn_flash_read(st.port, off + pos, page, n);
		if (ret != 0 || memcmp(page, &img_buf[pos], n) != 0) {
			st.copy_valid[copy] = false;
			return (ret != 0) ? ret : -EIO;
		}
		pos += n;
	}

	st.copy_valid[copy] = true;
	st.newest_copy = copy;
	st.generation = gen;
	st.last_commit_us = cycles_to_us(t0);
	return 0;
}

/* Persist current RAM state into the older copy. */
static int commit(void)
{
	uint8_t target = st.copy_valid[st.newest_copy] ?
			 (uint8_t)(1U - st.newest_copy) : 0U;

	return copy_commit(target, st.generation + 1U);
}

static const uint8_t *slot_payload_ptr(uint8_t slot)
{
	return syn_flash_mmap(st.port,
			      st.lay.slot_off[slot] + SYN_SYNM_HDR_SIZE);
}

/* CRC32 of a slot payload straight from flash (chunked reads: works
 * on backends without mmap and avoids long bus-read bursts).
 */
static int slot_payload_crc(uint8_t slot, uint32_t size, uint32_t *crc_out)
{
	uint32_t off = st.lay.slot_off[slot] + SYN_SYNM_HDR_SIZE;
	uint32_t crc = 0U;
	uint32_t pos = 0U;

	while (pos < size) {
		uint32_t n = MIN(sizeof(io_buf), size - pos);
		int ret = syn_flash_read(st.port, off + pos, io_buf, n);

		if (ret != 0) {
			return ret;
		}
		crc = crc32_ieee_update(crc, io_buf, n);
		pos += n;
	}
	*crc_out = crc;
	return 0;
}

static void ram_unregister(uint8_t slot)
{
	syn_model_handle_t h;

	if (syn_model_get_by_name(st.rec[slot].name, &h) == 0) {
		syn_model_unregister(h);
	}
}

static int ram_register(uint8_t slot, syn_model_handle_t *handle)
{
	syn_model_handle_t h;
	int ret = syn_model_register(&st.rec[slot], &h);

	if (ret != 0) {
		LOG_ERR("RAM registration of '%s' failed: %d",
			st.rec[slot].name, ret);
		return ret;
	}

	const uint8_t *data = slot_payload_ptr(slot);

	if (data != NULL) {
		syn_model_set_data(h, data, st.rec[slot].flash_size);
	}
	if (handle != NULL) {
		*handle = h;
	}
	return 0;
}

static int layout_validate(const syn_flash_port_t *port,
			   const syn_store_layout_t *lay)
{
	struct { uint32_t off, size; } r[4] = {
		{ lay->registry_off[0], lay->registry_size },
		{ lay->registry_off[1], lay->registry_size },
		{ lay->slot_off[0], lay->slot_size },
		{ lay->slot_off[1], lay->slot_size },
	};

	if (lay->registry_size == 0U || lay->slot_size == 0U) {
		return -EINVAL;
	}
	if (lay->registry_size <
	    round_up(STORE_IMG_SIZE, port->page_size)) {
		return -EINVAL;
	}
	if (lay->slot_size <= SYN_SYNM_HDR_SIZE) {
		return -EINVAL;
	}
	/* chunked writes stream through io_buf in page multiples */
	if (port->page_size > sizeof(io_buf) ||
	    (sizeof(io_buf) % port->page_size) != 0U) {
		return -EINVAL;
	}

	for (int i = 0; i < 4; i++) {
		if ((r[i].off % port->sector_size) != 0U ||
		    (r[i].size % port->sector_size) != 0U ||
		    r[i].off > port->size ||
		    r[i].size > port->size - r[i].off) {
			return -EINVAL;
		}
		for (int j = i + 1; j < 4; j++) {
			if (r[i].off < r[j].off + r[j].size &&
			    r[j].off < r[i].off + r[i].size) {
				return -EINVAL; /* overlap */
			}
		}
	}
	return 0;
}

int syn_store_init(const syn_flash_port_t *port,
		   const syn_store_layout_t *layout)
{
	if (port == NULL || layout == NULL) {
		return -EINVAL;
	}
	if (st.ready) {
		return -EALREADY;
	}

	int ret = layout_validate(port, layout);

	if (ret != 0) {
		return ret;
	}

	uint32_t t0 = k_cycle_get_32();

	memset(&st, 0, sizeof(st));
	k_mutex_init(&st.lock);
	st.port = port;
	st.lay = *layout;
	st.active_slot = SYN_STORE_SLOT_NONE;
	st.staged_slot = SYN_STORE_SLOT_NONE;

	struct store_hdr hdr[2];
	bool valid[2];

	for (uint8_t c = 0; c < 2U; c++) {
		valid[c] = copy_load(c, &hdr[c]);
		if (valid[c]) {
			st.wear[c] = hdr[c].wear;
		}
	}

	uint8_t newest = SYN_STORE_SLOT_NONE;

	if (valid[0] && valid[1]) {
		newest = (hdr[1].generation > hdr[0].generation) ? 1U : 0U;
	} else if (valid[0]) {
		newest = 0U;
	} else if (valid[1]) {
		newest = 1U;
	}

	if (newest != SYN_STORE_SLOT_NONE) {
		/* re-load the adopted copy into img_buf and unpack */
		(void)copy_load(newest, &hdr[newest]);
		st.copy_valid[0] = valid[0];
		st.copy_valid[1] = valid[1];
		st.newest_copy = newest;
		st.generation = hdr[newest].generation;
		st.active_slot = hdr[newest].active_slot;
		st.staged_slot = hdr[newest].staged_slot;
		st.occupied_mask = hdr[newest].occupied_mask;
		memcpy(&st.rec[0], &img_buf[sizeof(struct store_hdr)],
		       sizeof(st.rec[0]));
		memcpy(&st.rec[1],
		       &img_buf[sizeof(struct store_hdr) + sizeof(st.rec[0])],
		       sizeof(st.rec[1]));
		LOG_INF("Registry adopted: copy %u gen %u active %u "
			"staged %u occupied 0x%x",
			newest, st.generation, st.active_slot,
			st.staged_slot, st.occupied_mask);
	} else {
		LOG_INF("Registry empty: starting fresh");
	}

	st.ready = true;

	if (st.active_slot != SYN_STORE_SLOT_NONE) {
		(void)ram_register(st.active_slot, NULL);
	}

	st.scan_us = cycles_to_us(t0);
	return 0;
}

void syn_store_deinit(void)
{
	st.ready = false;
	st.port = NULL;
}

bool syn_store_ready(void)
{
	return st.ready;
}

int syn_store_staging_slot(uint8_t *slot)
{
	if (!st.ready || slot == NULL) {
		return -EINVAL;
	}

	if ((st.occupied_mask & BIT(0)) == 0U) {
		*slot = 0U;
	} else if ((st.occupied_mask & BIT(1)) == 0U) {
		*slot = 1U;
	} else if (st.active_slot == 0U) {
		*slot = 1U;
	} else {
		*slot = 0U;
	}
	return 0;
}

int syn_store_slot_bounds(uint8_t slot, uint32_t *off, uint32_t *size)
{
	if (!st.ready || slot > 1U) {
		return -EINVAL;
	}
	if (off != NULL) {
		*off = st.lay.slot_off[slot];
	}
	if (size != NULL) {
		*size = st.lay.slot_size;
	}
	return 0;
}

int syn_store_mark_staged(uint8_t slot, const syn_model_info_t *info)
{
	if (!st.ready || slot > 1U || info == NULL) {
		return -EINVAL;
	}
	if (slot == st.active_slot) {
		return -EBUSY; /* never stage over the active model */
	}

	k_mutex_lock(&st.lock, K_FOREVER);

	syn_model_info_t saved = st.rec[slot];
	uint8_t s_occ = st.occupied_mask, s_staged = st.staged_slot;

	st.rec[slot] = *info;
	st.occupied_mask |= BIT(slot);
	st.staged_slot = slot;

	int ret = commit();

	if (ret != 0) {
		st.rec[slot] = saved;
		st.occupied_mask = s_occ;
		st.staged_slot = s_staged;
	}
	k_mutex_unlock(&st.lock);
	return ret;
}

int syn_store_activate(uint8_t slot)
{
	if (!st.ready || slot > 1U) {
		return -EINVAL;
	}
	if ((st.occupied_mask & BIT(slot)) == 0U) {
		return -ENOENT;
	}
	if (slot == st.active_slot) {
		return -EALREADY;
	}

	k_mutex_lock(&st.lock, K_FOREVER);

	uint8_t old_active = st.active_slot;
	uint8_t s_staged = st.staged_slot;

	st.active_slot = slot;
	if (st.staged_slot == slot) {
		st.staged_slot = SYN_STORE_SLOT_NONE;
	}

	int ret = commit();

	if (ret != 0) {
		st.active_slot = old_active;
		st.staged_slot = s_staged;
		k_mutex_unlock(&st.lock);
		return ret;
	}

	/* flash state is authoritative; now swing the RAM registry.
	 * If the outgoing model was NPU-resident, hot-load the new one
	 * in its place (waits out any in-flight inference).
	 */
	bool was_loaded = false;

	if (old_active != SYN_STORE_SLOT_NONE) {
		syn_model_handle_t oh;

		if (syn_model_get_by_name(st.rec[old_active].name, &oh) == 0) {
			was_loaded = syn_model_is_loaded(oh);
		}
		ram_unregister(old_active);
	}

	syn_model_handle_t nh = SYN_MODEL_INVALID;

	(void)ram_register(slot, &nh);
	if (was_loaded && nh != SYN_MODEL_INVALID) {
		int lret = syn_model_load(nh);

		if (lret != 0) {
			LOG_WRN("Hot reload of '%s' failed: %d",
				st.rec[slot].name, lret);
		}
	}

	k_mutex_unlock(&st.lock);
	return 0;
}

int syn_store_rollback(void)
{
	if (!st.ready) {
		return -EINVAL;
	}

	k_mutex_lock(&st.lock, K_FOREVER);

	uint8_t cur = st.active_slot;
	uint8_t prev = SYN_STORE_SLOT_NONE;

	/* rollback target: the occupied, non-active, non-staged slot */
	for (uint8_t s = 0; s < 2U; s++) {
		if ((st.occupied_mask & BIT(s)) != 0U &&
		    s != cur && s != st.staged_slot) {
			prev = s;
		}
	}
	if (prev == SYN_STORE_SLOT_NONE) {
		k_mutex_unlock(&st.lock);
		return -ENOENT;
	}

	st.active_slot = prev;

	int ret = commit();

	if (ret != 0) {
		st.active_slot = cur;
		k_mutex_unlock(&st.lock);
		return ret;
	}

	bool was_loaded = false;

	if (cur != SYN_STORE_SLOT_NONE) {
		syn_model_handle_t ch;

		if (syn_model_get_by_name(st.rec[cur].name, &ch) == 0) {
			was_loaded = syn_model_is_loaded(ch);
		}
		ram_unregister(cur);
	}

	syn_model_handle_t ph = SYN_MODEL_INVALID;

	(void)ram_register(prev, &ph);
	if (was_loaded && ph != SYN_MODEL_INVALID) {
		int lret = syn_model_load(ph);

		if (lret != 0) {
			LOG_WRN("Hot reload of '%s' failed: %d",
				st.rec[prev].name, lret);
		}
	}

	k_mutex_unlock(&st.lock);
	return 0;
}

int syn_store_clear_staged(void)
{
	if (!st.ready) {
		return -EINVAL;
	}
	if (st.staged_slot == SYN_STORE_SLOT_NONE) {
		return 0;
	}

	k_mutex_lock(&st.lock, K_FOREVER);

	uint8_t s = st.staged_slot;
	uint8_t s_occ = st.occupied_mask;

	st.occupied_mask &= (uint8_t)~BIT(s);
	st.staged_slot = SYN_STORE_SLOT_NONE;

	int ret = commit();

	if (ret != 0) {
		st.occupied_mask = s_occ;
		st.staged_slot = s;
	}
	k_mutex_unlock(&st.lock);
	return ret;
}

int syn_store_install(const syn_model_info_t *info,
		      const void *data, size_t size,
		      syn_model_handle_t *handle)
{
	if (!st.ready || info == NULL || data == NULL || size == 0U ||
	    info->name[0] == '\0') {
		return -EINVAL;
	}

	uint8_t slot;

	(void)syn_store_staging_slot(&slot);

	uint32_t total = SYN_SYNM_HDR_SIZE + (uint32_t)size;

	if (total > st.lay.slot_size) {
		return -EFBIG;
	}

	uint32_t crc = crc32_ieee(data, size);

	if (info->crc32 != 0U && info->crc32 != crc) {
		return -EILSEQ;
	}

	k_mutex_lock(&st.lock, K_FOREVER);

	uint32_t off = st.lay.slot_off[slot];
	int ret = syn_flash_erase_sectors(st.port, off,
					  round_up(total,
						   st.port->sector_size));

	if (ret != 0) {
		k_mutex_unlock(&st.lock);
		return ret;
	}

	/* .synm image: 64-byte header, then the payload */
	struct syn_synm_hdr hdr = {
		.magic = SYN_SYNM_MAGIC,
		.version = SYN_SYNM_VERSION,
		.model_size = (uint32_t)size,
		.crc32 = crc,
	};

	memcpy(hdr.name, info->name, sizeof(hdr.name));
	for (int i = 0; i < 4; i++) {
		hdr.input_shape[i] = info->input_shape[i];
		hdr.output_shape[i] = info->output_shape[i];
	}

	/* first pages: header + leading payload bytes, page-buffered */
	const uint8_t *src = data;
	uint32_t page = st.port->page_size;
	uint32_t hdr_pages = round_up(SYN_SYNM_HDR_SIZE, page);
	uint32_t lead = MIN(hdr_pages - SYN_SYNM_HDR_SIZE, (uint32_t)size);

	memset(io_buf, 0xFF, hdr_pages);
	memcpy(io_buf, &hdr, sizeof(hdr));
	memcpy(&io_buf[SYN_SYNM_HDR_SIZE], src, lead);

	ret = syn_flash_write(st.port, off, io_buf, hdr_pages);

	/* remaining payload in page multiples, tail padded 0xFF; flash
	 * position of payload byte pos is off + 64 + pos, page-aligned
	 * here because 64 + lead == hdr_pages and io_buf chunks are
	 * page multiples.
	 */
	uint32_t pos = lead;

	while (ret == 0 && pos < size) {
		uint32_t n = MIN(sizeof(io_buf), (uint32_t)size - pos);
		uint32_t padded = round_up(n, page);

		memset(io_buf, 0xFF, padded);
		memcpy(io_buf, &src[pos], n);
		ret = syn_flash_write(st.port,
				      off + SYN_SYNM_HDR_SIZE + pos,
				      io_buf, padded);
		pos += n;
	}

	if (ret != 0) {
		k_mutex_unlock(&st.lock);
		return ret;
	}

	/* verify what actually landed in flash before committing */
	uint32_t flash_crc;

	ret = slot_payload_crc(slot, (uint32_t)size, &flash_crc);
	if (ret != 0 || flash_crc != crc) {
		k_mutex_unlock(&st.lock);
		return (ret != 0) ? ret : -EILSEQ;
	}

	/* record + activate (previous active becomes the rollback) */
	syn_model_info_t rec = *info;

	rec.flash_offset = st.port->base + off + SYN_SYNM_HDR_SIZE;
	rec.flash_size = (uint32_t)size;
	rec.crc32 = crc;

	uint8_t old_active = st.active_slot;
	syn_model_info_t saved = st.rec[slot];
	uint8_t s_occ = st.occupied_mask, s_staged = st.staged_slot;

	st.rec[slot] = rec;
	st.occupied_mask |= BIT(slot);
	if (st.staged_slot == slot) {
		st.staged_slot = SYN_STORE_SLOT_NONE;
	}
	st.active_slot = slot;

	ret = commit();

	if (ret != 0) {
		st.rec[slot] = saved;
		st.occupied_mask = s_occ;
		st.staged_slot = s_staged;
		st.active_slot = old_active;
		k_mutex_unlock(&st.lock);
		return ret;
	}

	if (old_active != SYN_STORE_SLOT_NONE && old_active != slot) {
		ram_unregister(old_active);
	}
	ret = ram_register(slot, handle);

	k_mutex_unlock(&st.lock);

	LOG_INF("Installed '%s' (%u bytes) into slot %u, gen %u",
		rec.name, rec.flash_size, slot, st.generation);
	return ret;
}

uint8_t syn_store_active_slot(void)
{
	return st.ready ? st.active_slot : SYN_STORE_SLOT_NONE;
}

uint8_t syn_store_staged_slot(void)
{
	return st.ready ? st.staged_slot : SYN_STORE_SLOT_NONE;
}

int syn_store_slot_info(uint8_t slot, syn_model_info_t *info)
{
	if (!st.ready || slot > 1U || info == NULL) {
		return -EINVAL;
	}
	if ((st.occupied_mask & BIT(slot)) == 0U) {
		return -ENOENT;
	}
	*info = st.rec[slot];
	return 0;
}

uint32_t syn_store_generation(void)
{
	return st.ready ? st.generation : 0U;
}

uint32_t syn_store_wear(uint8_t copy)
{
	return (st.ready && copy < 2U) ? st.wear[copy] : 0U;
}

uint32_t syn_store_last_commit_us(void)
{
	return st.last_commit_us;
}

uint32_t syn_store_scan_us(void)
{
	return st.scan_us;
}

const syn_flash_port_t *syn_store_port(void)
{
	return st.ready ? st.port : NULL;
}

const syn_store_layout_t *syn_store_layout_get(void)
{
	return st.ready ? &st.lay : NULL;
}

#if defined(CONFIG_SYNAPTIC_STORE_AUTO_INIT) && defined(CONFIG_SOC_FLASH_MCUX)
/* Board auto-init: bring the store up over the partition-map window
 * before main() so every FRDM application (including dual_model's
 * shell-driven OTA) sees the persistent registry. Runs before CPU1
 * is released, so scanning bank 1 is race-free. Never fails boot:
 * a store problem degrades to "no persistent models", logged.
 */
#include <zephyr/init.h>
#include "syn_flash_map.h"

static int syn_store_auto_init(void)
{
	static syn_flash_port_t mcx_port;

	if (syn_flash_port_mcx_init(&mcx_port) != 0) {
		LOG_ERR("Store auto-init skipped: flash port unavailable");
		return 0;
	}

	static const syn_store_layout_t map_layout = {
		.registry_off = {
			SYN_PART_REGISTRY_A_OFFSET - SYN_FLASH_WRITABLE_BASE,
			SYN_PART_REGISTRY_B_OFFSET - SYN_FLASH_WRITABLE_BASE,
		},
		.registry_size = SYN_PART_REGISTRY_A_SIZE,
		.slot_off = {
			SYN_PART_SLOT_A_OFFSET - SYN_FLASH_WRITABLE_BASE,
			SYN_PART_SLOT_B_OFFSET - SYN_FLASH_WRITABLE_BASE,
		},
		.slot_size = SYN_PART_SLOT_A_SIZE,
	};

	int ret = syn_store_init(&mcx_port, &map_layout);

	if (ret != 0) {
		LOG_ERR("Store auto-init failed: %d (no persistence)", ret);
	}
	return 0;
}

SYS_INIT(syn_store_auto_init, APPLICATION, 90);
#endif /* CONFIG_SYNAPTIC_STORE_AUTO_INIT && CONFIG_SOC_FLASH_MCUX */
