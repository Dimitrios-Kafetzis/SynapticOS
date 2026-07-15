/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file test_model_store.c
 * @brief Unit tests for the flash-backed model store (Phase 4.1)
 *
 * Runs the full persistence logic over the RAM flash port: the same
 * store/registry code the board runs over the MCX ROM API, with a
 * scaled-down layout (QEMU has 64 KB of RAM). "Reboot" = deinit +
 * RAM-registry reset + re-init over unchanged flash contents.
 */

#include <zephyr/ztest.h>
#include <string.h>
#include <synaptic/syn_model.h>
#include <synaptic/syn_hal_npu.h>

#include "syn_model_store.h"
#include "syn_model_internal.h"
#include "syn_flash_port.h"
#include "syn_synm.h"

/* Board-realistic geometry, scaled: same 128-byte program page as
 * the MCX ROM API, 1 KB "sectors", 8 KB model slots (big enough for
 * the OTA 4 KB chunk tests).
 */
#define TSECTOR 1024U
#define TPAGE   128U
#define TSLOT   8192U

/* 2 x 1 KB registry copies + 2 x 8 KB slots */
static uint8_t flash_mem[2U * TSECTOR + 2U * TSLOT];

static const syn_store_layout_t lay = {
    .registry_off = { 0U, TSECTOR },
    .registry_size = TSECTOR,
    .slot_off = { 2U * TSECTOR, 2U * TSECTOR + TSLOT },
    .slot_size = TSLOT,
};

static syn_flash_port_t port;
static syn_flash_ram_ctx_t ram_ctx;

static uint8_t payload[1200];

static void payload_fill(uint32_t seed)
{
    uint32_t x = seed * 2654435761U + 1U;

    for (size_t i = 0; i < sizeof(payload); i++) {
        x = x * 1103515245U + 12345U;
        payload[i] = (uint8_t)(x >> 16);
    }
}

static syn_model_info_t make_info(const char *name, const char *ver)
{
    syn_model_info_t info = {0};

    strncpy(info.name, name, sizeof(info.name) - 1);
    strncpy(info.version, ver, sizeof(info.version) - 1);
    info.input_size = 64;
    info.output_size = 16;
    info.input_shape[0] = 8;
    info.input_shape[1] = 8;
    info.output_shape[0] = 16;
    return info;
}

/* Fresh chip: erased flash, empty registries. */
static void store_fresh(void)
{
    syn_store_deinit();
    syn_model_reset_all();
    syn_hal_npu_init();
    zassert_ok(syn_flash_port_ram_init(&port, &ram_ctx, flash_mem,
                                       sizeof(flash_mem), TSECTOR, TPAGE),
               "ram port init failed");
    zassert_ok(syn_store_init(&port, &lay), "store init failed");
}

/* Reboot: RAM state lost, flash contents preserved. */
static void store_reboot(void)
{
    syn_store_deinit();
    syn_model_reset_all();
    zassert_ok(syn_store_init(&port, &lay), "store re-init failed");
}

ZTEST_SUITE(syn_store_suite, NULL, NULL, NULL, NULL, NULL);

ZTEST(syn_store_suite, test_empty_init)
{
    store_fresh();

    zassert_true(syn_store_ready(), "store not ready");
    zassert_equal(syn_store_active_slot(), SYN_STORE_SLOT_NONE,
                  "unexpected active slot");
    zassert_equal(syn_store_generation(), 0, "generation not 0");

    uint8_t count = 0;
    syn_model_handle_t handles[4];

    syn_model_list(handles, &count, 4);
    zassert_equal(count, 0, "registry not empty");
}

ZTEST(syn_store_suite, test_install_and_load)
{
    store_fresh();
    payload_fill(1);

    syn_model_info_t info = make_info("alpha", "1.0.0");
    syn_model_handle_t h = SYN_MODEL_INVALID;

    zassert_ok(syn_store_install(&info, payload, sizeof(payload), &h),
               "install failed");
    zassert_not_equal(h, SYN_MODEL_INVALID, "invalid handle");
    zassert_equal(syn_store_active_slot(), 0, "expected slot 0 active");

    syn_model_handle_t found;

    zassert_ok(syn_model_get_by_name("alpha", &found), "lookup failed");
    zassert_equal(found, h, "handle mismatch");

    /* load runs the CRC32 gate over the flash payload */
    zassert_ok(syn_model_load(h), "load failed");
    zassert_ok(syn_model_unload(h), "unload failed");
}

ZTEST(syn_store_suite, test_reboot_survival)
{
    store_fresh();
    payload_fill(2);

    syn_model_info_t info = make_info("alpha", "1.0.0");
    syn_model_handle_t h;

    zassert_ok(syn_store_install(&info, payload, sizeof(payload), &h));

    uint32_t gen = syn_store_generation();

    store_reboot();

    zassert_equal(syn_store_generation(), gen, "generation changed");
    zassert_ok(syn_model_get_by_name("alpha", &h),
               "model lost across reboot");

    syn_model_info_t back;

    zassert_ok(syn_model_get_info(h, &back));
    zassert_equal(back.flash_size, sizeof(payload), "size mismatch");
    zassert_ok(syn_model_load(h), "load after reboot failed");
    zassert_ok(syn_model_unload(h));
}

ZTEST(syn_store_suite, test_load_refuses_corrupt_payload)
{
    store_fresh();
    payload_fill(3);

    syn_model_info_t info = make_info("alpha", "1.0.0");
    syn_model_handle_t h;

    zassert_ok(syn_store_install(&info, payload, sizeof(payload), &h));

    /* flip one payload bit behind the store's back */
    flash_mem[lay.slot_off[0] + SYN_SYNM_HDR_SIZE + 100] ^= 0x01U;

    int ret = syn_model_load(h);

    zassert_equal(ret, -EILSEQ, "corrupt model loaded: %d", ret);
    zassert_false(syn_model_is_loaded(h), "marked loaded after refusal");
}

ZTEST(syn_store_suite, test_registry_corruption_falls_back)
{
    store_fresh();
    payload_fill(4);

    syn_model_info_t v1 = make_info("alpha", "1.0.0");
    syn_model_handle_t h;

    zassert_ok(syn_store_install(&v1, payload, sizeof(payload), &h));

    uint32_t gen1 = syn_store_generation();

    payload_fill(5);

    syn_model_info_t v2 = make_info("beta", "2.0.0");

    zassert_ok(syn_store_install(&v2, payload, sizeof(payload), &h));
    zassert_true(syn_store_generation() > gen1, "generation stuck");

    /* corrupt the NEWEST registry copy; the older one must win */
    uint8_t newest = (syn_store_generation() % 2U == 0U) ? 1U : 0U;

    flash_mem[lay.registry_off[newest] + 8] ^= 0xFFU;

    store_reboot();

    zassert_equal(syn_store_generation(), gen1,
                  "did not fall back to older generation");
    zassert_ok(syn_model_get_by_name("alpha", &h),
               "older active model missing");
    zassert_equal(syn_model_get_by_name("beta", &h), -ENOENT,
                  "newer model survived corruption");
}

ZTEST(syn_store_suite, test_pingpong_wear_tracking)
{
    store_fresh();
    payload_fill(6);

    syn_model_info_t info = make_info("alpha", "1.0.0");
    syn_model_handle_t h;

    zassert_ok(syn_store_install(&info, payload, sizeof(payload), &h));

    syn_model_info_t staged = make_info("stage", "0.0.1");
    uint8_t slot;

    for (int i = 0; i < 3; i++) {
        zassert_ok(syn_store_staging_slot(&slot));
        zassert_ok(syn_store_mark_staged(slot, &staged));
        zassert_ok(syn_store_clear_staged());
    }

    uint32_t gen = syn_store_generation();
    uint32_t w0 = syn_store_wear(0);
    uint32_t w1 = syn_store_wear(1);

    zassert_equal(w0 + w1, gen, "wear sum %u != generation %u",
                  w0 + w1, gen);
    zassert_true((w0 > w1 ? w0 - w1 : w1 - w0) <= 1U,
                 "ping-pong unbalanced: %u vs %u", w0, w1);

    /* wear survives reboot */
    store_reboot();
    zassert_equal(syn_store_wear(0) + syn_store_wear(1), gen,
                  "wear lost across reboot");
}

ZTEST(syn_store_suite, test_update_and_rollback)
{
    store_fresh();
    payload_fill(7);

    syn_model_info_t v1 = make_info("alpha", "1.0.0");
    syn_model_handle_t h;

    zassert_ok(syn_store_install(&v1, payload, sizeof(payload), &h));

    payload_fill(8);

    syn_model_info_t v2 = make_info("alpha", "2.0.0");

    zassert_ok(syn_store_install(&v2, payload, sizeof(payload), &h));
    zassert_equal(syn_store_active_slot(), 1, "update not in slot B");

    syn_model_info_t cur;

    zassert_ok(syn_model_get_by_name("alpha", &h));
    zassert_ok(syn_model_get_info(h, &cur));
    zassert_mem_equal(cur.version, "2.0.0", 6, "v2 not active");

    zassert_ok(syn_store_rollback(), "rollback failed");
    zassert_equal(syn_store_active_slot(), 0, "rollback not to slot A");
    zassert_ok(syn_model_get_by_name("alpha", &h));
    zassert_ok(syn_model_get_info(h, &cur));
    zassert_mem_equal(cur.version, "1.0.0", 6, "v1 not restored");
    zassert_ok(syn_model_load(h), "restored model unloadable");
    zassert_ok(syn_model_unload(h));

    /* survives reboot */
    store_reboot();
    zassert_ok(syn_model_get_by_name("alpha", &h));
    zassert_ok(syn_model_get_info(h, &cur));
    zassert_mem_equal(cur.version, "1.0.0", 6, "rollback lost on reboot");
}

ZTEST(syn_store_suite, test_powerloss_during_commit)
{
    store_fresh();
    payload_fill(9);

    syn_model_info_t v1 = make_info("alpha", "1.0.0");
    syn_model_handle_t h;

    zassert_ok(syn_store_install(&v1, payload, sizeof(payload), &h));

    uint32_t gen = syn_store_generation();

    /* die half way through the FIRST page of the registry image, so
     * the tear is strictly inside the CRC-covered bytes (a tear at
     * or past the image end is a legitimately durable commit)
     */
    ram_ctx.fail_after_pages = 1;

    syn_model_info_t staged = make_info("stage", "0.0.1");
    uint8_t slot;

    zassert_ok(syn_store_staging_slot(&slot));
    zassert_not_equal(syn_store_mark_staged(slot, &staged), 0,
                      "commit unexpectedly survived power loss");
    ram_ctx.fail_after_pages = 0;

    store_reboot();

    zassert_equal(syn_store_generation(), gen,
                  "torn commit was adopted");
    zassert_ok(syn_model_get_by_name("alpha", &h),
               "active model lost after torn commit");
    zassert_equal(syn_store_staged_slot(), SYN_STORE_SLOT_NONE,
                  "phantom staged slot after torn commit");
}

ZTEST(syn_store_suite, test_powerloss_during_payload_write)
{
    store_fresh();
    payload_fill(10);

    syn_model_info_t v1 = make_info("alpha", "1.0.0");
    syn_model_handle_t h;

    zassert_ok(syn_store_install(&v1, payload, sizeof(payload), &h));

    uint32_t gen = syn_store_generation();

    payload_fill(11);

    syn_model_info_t v2 = make_info("alpha", "2.0.0");

    ram_ctx.fail_after_pages = 5;
    zassert_not_equal(syn_store_install(&v2, payload, sizeof(payload), &h),
                      0, "install survived power loss");
    ram_ctx.fail_after_pages = 0;

    store_reboot();

    zassert_equal(syn_store_generation(), gen, "generation moved");

    syn_model_info_t cur;

    zassert_ok(syn_model_get_by_name("alpha", &h), "old model lost");
    zassert_ok(syn_model_get_info(h, &cur));
    zassert_mem_equal(cur.version, "1.0.0", 6, "old model replaced");
    zassert_ok(syn_model_load(h), "old model corrupted");
    zassert_ok(syn_model_unload(h));
}

ZTEST(syn_store_suite, test_oversized_install_rejected)
{
    store_fresh();

    syn_model_info_t info = make_info("huge", "1.0.0");
    syn_model_handle_t h;

    /* claim a size the header pushes past the slot; the size check
     * fires before any byte of the buffer is read
     */
    zassert_equal(syn_store_install(&info, payload, TSLOT, &h), -EFBIG,
                  "oversized install accepted");
}

ZTEST(syn_store_suite, test_ops_complete_quickly)
{
    store_fresh();
    payload_fill(12);

    syn_model_info_t info = make_info("alpha", "1.0.0");
    syn_model_handle_t h;

    zassert_ok(syn_store_install(&info, payload, sizeof(payload), &h));

    /* acceptance: registry operations < 50 ms (QEMU figure here;
     * the board figure is measured in the Phase 4 verification)
     */
    zassert_true(syn_store_last_commit_us() < 50000U,
                 "commit took %u us", syn_store_last_commit_us());

    store_reboot();
    zassert_true(syn_store_scan_us() < 50000U,
                 "boot scan took %u us", syn_store_scan_us());
}
