/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file test_model_store_multi.c
 * @brief Unit tests for the multi-model store (Phase 6.4)
 *
 * The store generalized past the A/B pair: N slots from the layout,
 * every occupied non-staged slot RESIDENT (registered and runnable
 * by name), activate/rollback moving only the active pivot, staging
 * evicting its victim before the flash erase, and Phase 4 version-1
 * registry images adopted in place.
 *
 * Reuses the shared RAM-flash fixture from test_model_store.c with a
 * four-slot geometry over the same 18 KB array: 2 x 1 KB registry
 * copies + 4 x 4 KB slots.
 */

#include <zephyr/ztest.h>
#include <zephyr/sys/crc.h>
#include <string.h>
#include <synaptic/syn_model.h>
#include <synaptic/syn_hal_npu.h>

#include "syn_model_store.h"
#include "syn_model_internal.h"
#include "syn_flash_port.h"
#include "syn_synm.h"

#define MSECTOR 1024U
#define MPAGE   128U
#define MSLOT   4096U

extern uint8_t syn_test_flash_mem[2U * 1024U + 2U * 8192U];
extern syn_flash_port_t syn_test_port;
extern syn_flash_ram_ctx_t syn_test_ram_ctx;

#define flash_mem syn_test_flash_mem
#define port      syn_test_port
#define ram_ctx   syn_test_ram_ctx

static const syn_store_layout_t mlay = {
	.registry_off = { 0U, MSECTOR },
	.registry_size = MSECTOR,
	.slot_off = { 2U * MSECTOR, 2U * MSECTOR + MSLOT,
		      2U * MSECTOR + 2U * MSLOT, 2U * MSECTOR + 3U * MSLOT },
	.slot_size = MSLOT,
	.slot_count = 4U,
};

static uint8_t mpayload[1000];

static void mpayload_fill(uint32_t seed)
{
	uint32_t x = seed * 2654435761U + 1U;

	for (size_t i = 0; i < sizeof(mpayload); i++) {
		x = x * 1103515245U + 12345U;
		mpayload[i] = (uint8_t)(x >> 16);
	}
}

static syn_model_info_t minfo(const char *name)
{
	syn_model_info_t info = {0};

	strncpy(info.name, name, sizeof(info.name) - 1);
	strncpy(info.version, "1.0", sizeof(info.version) - 1);
	info.input_size = 64;
	info.output_size = 16;
	info.input_shape[0] = 8;
	info.input_shape[1] = 8;
	info.output_shape[0] = 16;
	return info;
}

static void multi_fresh(void)
{
	syn_store_deinit();
	syn_model_reset_all();
	syn_hal_npu_init();
	zassert_ok(syn_flash_port_ram_init(&port, &ram_ctx, flash_mem,
					   sizeof(flash_mem), MSECTOR, MPAGE),
		   "ram port init failed");
	zassert_ok(syn_store_init(&port, &mlay), "store init failed");
}

static void multi_reboot(void)
{
	syn_store_deinit();
	syn_model_reset_all();
	zassert_ok(syn_store_init(&port, &mlay), "store re-init failed");
}

static syn_model_handle_t install_named(const char *name, uint32_t seed)
{
	syn_model_info_t info = minfo(name);
	syn_model_handle_t h = SYN_MODEL_INVALID;

	mpayload_fill(seed);
	zassert_ok(syn_store_install(&info, mpayload, sizeof(mpayload), &h),
		   "install '%s' failed", name);
	return h;
}

static bool resident(const char *name)
{
	syn_model_handle_t h;

	return syn_model_get_by_name(name, &h) == 0;
}

ZTEST_SUITE(syn_store_multi_suite, NULL, NULL, NULL, NULL, NULL);

ZTEST(syn_store_multi_suite, test_three_models_resident)
{
	multi_fresh();
	install_named("alpha", 1);
	install_named("beta", 2);
	install_named("gamma", 3);

	zassert_equal(syn_store_slot_count(), 4U, "slot count");
	zassert_equal(syn_store_active_slot(), 2U, "gamma not active");
	zassert_true(resident("alpha") && resident("beta") &&
		     resident("gamma"), "not all models resident");

	/* every resident model is individually runnable (NPU residency
	 * cycles through them one at a time)
	 */
	const char *names[3] = { "alpha", "beta", "gamma" };

	for (int i = 0; i < 3; i++) {
		syn_model_handle_t h;

		zassert_ok(syn_model_get_by_name(names[i], &h));
		zassert_ok(syn_model_load(h), "'%s' load failed", names[i]);
		zassert_ok(syn_model_unload(h));
	}

	/* residency survives reboot */
	multi_reboot();
	zassert_true(resident("alpha") && resident("beta") &&
		     resident("gamma"), "residency lost on reboot");
	zassert_equal(syn_store_active_slot(), 2U, "active lost on reboot");
}

ZTEST(syn_store_multi_suite, test_activate_by_slot_hot_swap)
{
	multi_fresh();
	install_named("alpha", 4);
	install_named("beta", 5);
	install_named("gamma", 6);

	/* gamma (slot 2) holds the NPU; swing the active pivot to
	 * alpha (slot 0) - everything stays resident
	 */
	syn_model_handle_t hg;

	zassert_ok(syn_model_get_by_name("gamma", &hg));
	zassert_ok(syn_model_load(hg));

	zassert_ok(syn_store_activate(0U), "activate slot 0 failed");
	zassert_equal(syn_store_active_slot(), 0U, "pivot not moved");
	zassert_equal(syn_store_prev_active_slot(), 2U, "prev not gamma");
	zassert_true(resident("alpha") && resident("beta") &&
		     resident("gamma"), "a model fell off the registry");

	/* NPU residency followed the pivot */
	syn_model_handle_t ha;

	zassert_ok(syn_model_get_by_name("alpha", &ha));
	zassert_true(syn_model_is_loaded(ha), "alpha did not inherit NPU");
	zassert_false(syn_model_is_loaded(hg), "gamma still loaded");

	zassert_equal(syn_store_activate(0U), -EALREADY, "re-activate");
}

ZTEST(syn_store_multi_suite, test_rollback_uses_prev_pivot)
{
	multi_fresh();
	install_named("alpha", 7);
	install_named("beta", 8);
	install_named("gamma", 9);

	/* install order left active 2, prev 1 */
	zassert_equal(syn_store_prev_active_slot(), 1U, "prev not beta");

	zassert_ok(syn_store_rollback(), "rollback failed");
	zassert_equal(syn_store_active_slot(), 1U, "pivot not beta");
	zassert_equal(syn_store_prev_active_slot(), 2U, "prev not gamma");
	zassert_true(resident("gamma"), "gamma evicted by rollback");

	/* rollback is a ping-pong between the two pivots */
	zassert_ok(syn_store_rollback());
	zassert_equal(syn_store_active_slot(), 2U, "pivot not gamma");

	multi_reboot();
	zassert_equal(syn_store_active_slot(), 2U, "active lost");
	zassert_equal(syn_store_prev_active_slot(), 1U, "prev lost");
}

ZTEST(syn_store_multi_suite, test_staging_evicts_victim)
{
	multi_fresh();
	install_named("alpha", 10);
	install_named("beta", 11);
	install_named("gamma", 12);
	install_named("delta", 13);

	/* all four slots occupied (active 3 = delta, prev 2 = gamma);
	 * the fifth install must claim a slot: neither the active nor
	 * the rollback pivot - slot 0 (alpha) is the victim
	 */
	zassert_equal(syn_store_active_slot(), 3U, "delta not active");
	install_named("epsilon", 14);

	zassert_false(resident("alpha"), "victim still resident");
	zassert_true(resident("beta") && resident("gamma") &&
		     resident("delta") && resident("epsilon"),
		     "survivor fell off");
	zassert_equal(syn_store_active_slot(), 0U, "epsilon not in slot 0");

	multi_reboot();
	zassert_false(resident("alpha"), "victim back after reboot");
	zassert_true(resident("epsilon"), "epsilon lost on reboot");
}

ZTEST(syn_store_multi_suite, test_powerloss_torn_commit_multi)
{
	multi_fresh();
	install_named("alpha", 15);
	install_named("beta", 16);

	uint32_t gen = syn_store_generation();

	/* corrupt the OLDER registry copy (the next commit target) to
	 * fake a power loss mid-commit, then reboot: the newest valid
	 * generation stays authoritative and both models survive
	 */
	uint8_t torn[MPAGE];

	memset(torn, 0xA5, sizeof(torn));
	zassert_ok(syn_flash_erase(&port, mlay.registry_off[0], MSECTOR));
	zassert_ok(syn_flash_write(&port, mlay.registry_off[0], torn,
				   sizeof(torn)));

	multi_reboot();
	zassert_equal(syn_store_generation(), gen, "generation moved");
	zassert_true(resident("alpha") && resident("beta"),
		     "models lost after torn commit");
	zassert_equal(syn_store_active_slot(), 1U, "active lost");
}

/* Phase 4 version-1 registry header, byte-for-byte */
struct v1_hdr {
	uint32_t magic;
	uint16_t version;
	uint16_t rsvd0;
	uint32_t generation;
	uint32_t wear;
	uint8_t  active_slot;
	uint8_t  staged_slot;
	uint8_t  occupied_mask;
	uint8_t  rsvd1;
	uint32_t crc32;
};

ZTEST(syn_store_multi_suite, test_v1_registry_adopted)
{
	/* two-slot layout, exactly what a Phase 4/5 board carries */
	static const syn_store_layout_t v1lay = {
		.registry_off = { 0U, MSECTOR },
		.registry_size = MSECTOR,
		.slot_off = { 2U * MSECTOR, 2U * MSECTOR + MSLOT },
		.slot_size = MSLOT,
		.slot_count = 2U,
	};

	syn_store_deinit();
	syn_model_reset_all();
	syn_hal_npu_init();
	zassert_ok(syn_flash_port_ram_init(&port, &ram_ctx, flash_mem,
					   sizeof(flash_mem), MSECTOR,
					   MPAGE));

	/* hand-write a v1 image: gen 5, wear 3, slot 0 active with a
	 * payload at slot 0 + synm header offset. Page-aligned stream
	 * from the slot start: 64 bytes of 0xFF stand in for the synm
	 * header (residency maps the payload directly; nothing parses
	 * the header bytes on this path).
	 */
	mpayload_fill(20);
	zassert_ok(syn_flash_erase(&port, v1lay.slot_off[0], MSLOT));
	{
		uint8_t stream[SYN_SYNM_HDR_SIZE + sizeof(mpayload)];
		uint8_t page[MPAGE];
		uint32_t pos = 0;

		memset(stream, 0xFF, SYN_SYNM_HDR_SIZE);
		memcpy(&stream[SYN_SYNM_HDR_SIZE], mpayload,
		       sizeof(mpayload));
		while (pos < sizeof(stream)) {
			uint32_t n = MIN(sizeof(page),
					 (uint32_t)sizeof(stream) - pos);

			memset(page, 0xFF, sizeof(page));
			memcpy(page, &stream[pos], n);
			zassert_ok(syn_flash_write(&port,
						   v1lay.slot_off[0] + pos,
						   page, sizeof(page)));
			pos += (uint32_t)sizeof(page);
		}
	}

	syn_model_info_t rec = minfo("legacy");

	rec.flash_offset = v1lay.slot_off[0] + SYN_SYNM_HDR_SIZE;
	rec.flash_size = sizeof(mpayload);
	rec.crc32 = crc32_ieee(mpayload, sizeof(mpayload));

	uint8_t img[256];
	struct v1_hdr hdr = {
		.magic = 0x524E5953UL,
		.version = 1U,
		.generation = 5U,
		.wear = 3U,
		.active_slot = 0U,
		.staged_slot = SYN_STORE_SLOT_NONE,
		.occupied_mask = 0x1U,
		.crc32 = 0U,
	};
	uint32_t v1_img_size = (uint32_t)(sizeof(struct v1_hdr) +
					  2U * sizeof(syn_model_info_t));

	zassert_true(v1_img_size <= sizeof(img), "v1 image too big");
	memset(img, 0xFF, sizeof(img));
	memcpy(img, &hdr, sizeof(hdr));
	memcpy(&img[sizeof(hdr)], &rec, sizeof(rec));
	/* slot 1 record: erased-flash bytes are what v1 committed for
	 * a never-occupied slot record region... v1 wrote zero-padded
	 * records; keep it simple and zero it
	 */
	memset(&img[sizeof(hdr) + sizeof(rec)], 0, sizeof(rec));

	uint32_t crc = crc32_ieee(img, v1_img_size);

	memcpy(&img[offsetof(struct v1_hdr, crc32)], &crc, sizeof(crc));
	zassert_ok(syn_flash_erase(&port, v1lay.registry_off[0], MSECTOR));
	zassert_ok(syn_flash_write(&port, v1lay.registry_off[0], img,
				   256U));
	zassert_ok(syn_flash_erase(&port, v1lay.registry_off[1], MSECTOR));

	/* adoption: v1 image accepted, state intact, model resident */
	zassert_ok(syn_store_init(&port, &v1lay), "v1 image rejected");
	zassert_equal(syn_store_generation(), 5U, "generation lost");
	zassert_equal(syn_store_active_slot(), 0U, "active lost");
	zassert_equal(syn_store_wear(0), 3U, "wear lost");
	zassert_equal(syn_store_prev_active_slot(), SYN_STORE_SLOT_NONE,
		      "phantom v1 pivot");
	zassert_true(resident("legacy"), "legacy model not resident");

	syn_model_handle_t h;

	zassert_ok(syn_model_get_by_name("legacy", &h));
	zassert_ok(syn_model_load(h), "legacy payload corrupt");
	zassert_ok(syn_model_unload(h));

	/* first commit upgrades to v2 (asserted behaviorally: the
	 * upgraded image round-trips a reboot with the pivot intact)
	 */
	syn_model_info_t nu = minfo("modern");
	syn_model_handle_t hn = SYN_MODEL_INVALID;

	mpayload_fill(21);
	zassert_ok(syn_store_install(&nu, mpayload, sizeof(mpayload), &hn));
	zassert_equal(syn_store_generation(), 6U, "commit did not advance");

	syn_store_deinit();
	syn_model_reset_all();
	zassert_ok(syn_store_init(&port, &v1lay), "v2 re-adopt failed");
	zassert_equal(syn_store_generation(), 6U, "v2 generation lost");
	zassert_equal(syn_store_active_slot(), 1U, "modern not active");
	zassert_equal(syn_store_prev_active_slot(), 0U, "pivot not saved");
	zassert_true(resident("legacy") && resident("modern"),
		     "residency lost after upgrade");
	zassert_ok(syn_store_rollback(), "rollback across upgrade");
	zassert_equal(syn_store_active_slot(), 0U, "rollback missed");
}
