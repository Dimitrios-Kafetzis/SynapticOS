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
#include "syn_synn.h"

/* Board-realistic geometry, scaled: same 128-byte program page as
 * the MCX ROM API, 1 KB "sectors", 8 KB model slots (big enough for
 * the OTA 4 KB chunk tests).
 */
#define TSECTOR 1024U
#define TPAGE   128U
#define TSLOT   8192U

/* 2 x 1 KB registry copies + 2 x 8 KB slots. Shared with
 * test_model_ota.c (same geometry) to stay inside QEMU's 64 KB RAM;
 * the suites run sequentially and every fixture re-initializes it.
 * 16-byte aligned like real flash so slot payload pointers land on
 * the alignment the SYNN sections rely on.
 */
uint8_t syn_test_flash_mem[2U * TSECTOR + 2U * TSLOT] __aligned(16);
syn_flash_port_t syn_test_port;
syn_flash_ram_ctx_t syn_test_ram_ctx;

#define flash_mem syn_test_flash_mem
#define port      syn_test_port
#define ram_ctx   syn_test_ram_ctx

static const syn_store_layout_t lay = {
    .registry_off = { 0U, TSECTOR },
    .registry_size = TSECTOR,
    .slot_off = { 2U * TSECTOR, 2U * TSECTOR + TSLOT },
    .slot_size = TSLOT,
    .slot_count = 2U,
};

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

/* Phase 6.1c: a Neutron SYNN payload rides inside the .synm image
 * like any other model. Round-trip it through install -> commit ->
 * reboot -> flash-mapped load and check the container survives
 * byte-for-byte at the alignment the NPU sections rely on. On this
 * build the NPU HAL is the stub, so load must succeed as an opaque
 * blob (the SYNN header only means something to the Neutron HAL).
 */
ZTEST(syn_store_suite, test_synn_payload_round_trip)
{
    static uint8_t synn[192] __aligned(16);
    struct syn_synn_hdr hdr = {
        .magic          = SYN_SYNN_MAGIC,
        .version        = SYN_SYNN_VERSION,
        .microcode_off  = 48U,
        .microcode_size = 20U,
        .weights_off    = 80U,
        .weights_size   = 40U,
        .kernels_off    = 128U,
        .kernels_size   = 1U,
        .input_size     = 64U,
        .output_size    = 12U,
        .scratch_size   = 2048U,
        .flags          = 0U,
    };

    memset(synn, 0, sizeof(synn));
    memcpy(synn, &hdr, sizeof(hdr));
    for (size_t i = 48; i < sizeof(synn); i++) {
        synn[i] = (uint8_t)(i * 13U);
    }

    store_fresh();

    syn_model_info_t info = make_info("neutron_rt", "1.0.0");
    syn_model_handle_t h;

    zassert_ok(syn_store_install(&info, synn, sizeof(synn), &h),
               "SYNN install failed");
    zassert_ok(syn_model_load(h), "stub load of SYNN payload failed");
    zassert_ok(syn_model_unload(h));

    store_reboot();
    zassert_ok(syn_model_get_by_name("neutron_rt", &h),
               "SYNN model lost across reboot");
    zassert_ok(syn_model_load(h), "load after reboot failed");
    zassert_ok(syn_model_unload(h));

    const uint8_t *slotp =
        &flash_mem[lay.slot_off[syn_store_active_slot()] +
                   SYN_SYNM_HDR_SIZE];

    zassert_equal((uintptr_t)slotp & 0xFU, 0U,
                  "slot payload not 16-byte aligned");
    zassert_mem_equal(slotp, synn, sizeof(synn),
                      "SYNN payload altered by the store");

    struct syn_synn_desc desc;

    zassert_ok(syn_synn_parse(slotp, sizeof(synn), &desc),
               "stored SYNN payload does not parse");
    zassert_equal(desc.input_size, 64U, "input size");
    zassert_equal(desc.output_size, 12U, "output size");
    zassert_equal(desc.scratch_size, 2048U, "scratch size");
    zassert_equal(desc.weights[0], (uint8_t)(80U * 13U), "weights bytes");
}

/* ------------------------------------------------------------------ */
/* Phase 6 S13: negative-path and structural-edge coverage            */
/* ------------------------------------------------------------------ */

#include <zephyr/sys/crc.h>

/** Argument and readiness guards of every introspection entry. */
ZTEST(syn_store_suite, test_accessor_and_guard_probes)
{
    store_fresh();
    payload_fill(40);

    syn_model_info_t info = make_info("probe_a", "1.0");
    syn_model_info_t out;

    zassert_ok(syn_store_install(&info, payload, sizeof(payload), NULL),
               "install failed");

    /* slot_info: bad slot, NULL out, empty slot, then success */
    zassert_equal(syn_store_slot_info(9, &out), -EINVAL, "bad slot");
    zassert_equal(syn_store_slot_info(0, NULL), -EINVAL, "NULL info");
    zassert_equal(syn_store_slot_info(1, &out), -ENOENT, "empty slot");
    zassert_ok(syn_store_slot_info(0, &out), "slot_info failed");
    zassert_equal(strncmp(out.name, "probe_a", sizeof(out.name)), 0,
                  "wrong record");

    /* layout access while ready */
    const syn_store_layout_t *l = syn_store_layout_get();

    zassert_not_null(l, "layout NULL while ready");
    zassert_equal(l->slot_count, 2, "layout slot_count");

    /* slot-guard probes on the mutating entries */
    zassert_equal(syn_store_slot_bounds(9, NULL, NULL), -EINVAL,
                  "bad slot_bounds accepted");
    zassert_equal(syn_store_begin_staging(9), -EINVAL,
                  "bad begin_staging accepted");
    zassert_equal(syn_store_begin_staging(0), -EBUSY,
                  "begin_staging over the active slot accepted");
    zassert_equal(syn_store_mark_staged(9, &info), -EINVAL,
                  "bad mark_staged accepted");
    zassert_equal(syn_store_mark_staged(0, &info), -EBUSY,
                  "mark_staged over the active slot accepted");
    zassert_equal(syn_store_activate(9), -EINVAL, "bad activate accepted");
    zassert_equal(syn_store_activate(1), -ENOENT,
                  "activate of an empty slot accepted");
    zassert_equal(syn_store_install(NULL, payload, 1, NULL), -EINVAL,
                  "NULL install accepted");
    zassert_equal(syn_store_clear_staged(), 0,
                  "clear with nothing staged must be a no-op");

    /* install with a wrong self-declared CRC is refused up front */
    info.crc32 = 0xDEADBEEFU;
    zassert_equal(syn_store_install(&info, payload, sizeof(payload), NULL),
                  -EILSEQ, "bad-CRC install accepted");

    /* not-ready guards */
    syn_store_deinit();
    zassert_equal(syn_store_rollback(), -EINVAL, "rollback while down");
    zassert_equal(syn_store_clear_staged(), -EINVAL, "clear while down");

    uint8_t slot;

    zassert_equal(syn_store_staging_slot(&slot), -EINVAL,
                  "staging_slot while down");
    zassert_is_null(syn_store_layout_get(), "layout while down");
}

/** Every structural reject of layout validation. */
ZTEST(syn_store_suite, test_layout_validation_rejects)
{
    store_fresh();
    syn_store_deinit();

    syn_store_layout_t bad;

    /* NULL arguments and slot_count out of range */
    zassert_equal(syn_store_init(NULL, &lay), -EINVAL, "NULL port");
    zassert_equal(syn_store_init(&port, NULL), -EINVAL, "NULL layout");
    bad = lay;
    bad.slot_count = 1;
    zassert_equal(syn_store_init(&port, &bad), -EINVAL, "slot_count 1");

    /* zero region sizes */
    bad = lay;
    bad.registry_size = 0;
    zassert_equal(syn_store_init(&port, &bad), -EINVAL, "registry 0");

    /* registry too small for one image */
    bad = lay;
    bad.registry_size = 128;
    zassert_equal(syn_store_init(&port, &bad), -EINVAL, "registry tiny");

    /* slot smaller than the .synm header */
    bad = lay;
    bad.slot_size = 64;
    zassert_equal(syn_store_init(&port, &bad), -EINVAL, "slot tiny");

    /* offsets off the sector grid */
    bad = lay;
    bad.slot_off[1] = lay.slot_off[1] + 1U;
    zassert_equal(syn_store_init(&port, &bad), -EINVAL, "unaligned slot");

    /* overlapping regions */
    bad = lay;
    bad.slot_off[1] = bad.slot_off[0];
    zassert_equal(syn_store_init(&port, &bad), -EINVAL, "overlap");

    /* page geometry the chunked writer cannot stream */
    syn_flash_port_t wide = port;

    wide.page_size = 512; /* > sizeof(io_buf) is impossible to buffer */
    zassert_equal(syn_store_init(&wide, &lay), -EINVAL, "page too wide");

    /* recover, then double-init guard */
    zassert_ok(syn_store_init(&port, &lay), "re-init failed");
    zassert_equal(syn_store_init(&port, &lay), -EALREADY, "double init");
}

/** With every slot occupied and the only non-active slot being the
 *  rollback pivot, staging falls through to the last-resort scan.
 */
ZTEST(syn_store_suite, test_staging_slot_last_resort)
{
    store_fresh();
    payload_fill(41);

    syn_model_info_t a = make_info("lr_a", "1.0");
    syn_model_info_t b = make_info("lr_b", "1.0");

    zassert_ok(syn_store_install(&a, payload, sizeof(payload), NULL),
               "install A failed");
    payload_fill(42);
    zassert_ok(syn_store_install(&b, payload, sizeof(payload), NULL),
               "install B failed");

    /* active 1, pivot 0, both occupied: only slot 0 can stage */
    zassert_equal(syn_store_active_slot(), 1, "active slot");
    zassert_equal(syn_store_prev_active_slot(), 0, "pivot slot");

    uint8_t slot = 0xAA;

    zassert_ok(syn_store_staging_slot(&slot), "staging_slot failed");
    zassert_equal(slot, 0, "last-resort scan must pick the pivot");
}

/** A full RAM registry blocks the resident-registration tail of
 *  install; the store still persists the record.
 */
ZTEST(syn_store_suite, test_registry_full_blocks_residency)
{
    store_fresh();
    payload_fill(43);

    /* fill the RAM registry to CONFIG_SYNAPTIC_MAX_MODELS */
    syn_model_handle_t hs[CONFIG_SYNAPTIC_MAX_MODELS];
    char name[16];

    for (int i = 0; i < CONFIG_SYNAPTIC_MAX_MODELS; i++) {
        syn_model_info_t info = {0};

        snprintf(name, sizeof(name), "fill_%d", i);
        strncpy(info.name, name, sizeof(info.name) - 1);
        zassert_ok(syn_model_register(&info, &hs[i]),
                   "filler register %d failed", i);
    }

    syn_model_info_t info = make_info("no_room", "1.0");
    int ret = syn_store_install(&info, payload, sizeof(payload), NULL);

    zassert_equal(ret, -ENOMEM, "full registry not reported: %d", ret);

    /* flash state is authoritative: the record itself landed */
    zassert_equal(syn_store_active_slot(), 0, "record not persisted");

    for (int i = 0; i < CONFIG_SYNAPTIC_MAX_MODELS; i++) {
        zassert_ok(syn_model_unregister(hs[i]), "filler cleanup %d", i);
    }
}

/* Registry image header offsets (struct store_hdr in the store) */
#define REG_HDR_PREV_OFF 7U
#define REG_HDR_GEN_OFF  8U
#define REG_HDR_CRC_OFF  20U
#define REG_HDR_SIZE     24U

/* Patch one byte of the newest registry copy and re-seal its CRC32
 * (test-only surgery on the RAM fixture; real flash cannot do this).
 */
static void registry_patch_newest(uint32_t byte_off, uint8_t value)
{
    uint32_t img = REG_HDR_SIZE + 2U * sizeof(syn_model_info_t);
    uint32_t gen0, gen1;

    memcpy(&gen0, &flash_mem[lay.registry_off[0] + REG_HDR_GEN_OFF], 4);
    memcpy(&gen1, &flash_mem[lay.registry_off[1] + REG_HDR_GEN_OFF], 4);

    uint8_t *copy = &flash_mem[(gen1 > gen0) ? lay.registry_off[1]
                                             : lay.registry_off[0]];

    copy[byte_off] = value;
    memset(&copy[REG_HDR_CRC_OFF], 0, 4);

    uint32_t crc = crc32_ieee(copy, img);

    memcpy(&copy[REG_HDR_CRC_OFF], &crc, 4);
}

/** Without a recorded pivot (pre-v2 image), rollback falls back to
 *  scanning for an occupied, non-active, non-staged slot.
 */
ZTEST(syn_store_suite, test_rollback_fallback_scan)
{
    store_fresh();
    payload_fill(44);

    syn_model_info_t a = make_info("fb_a", "1.0");
    syn_model_info_t b = make_info("fb_b", "1.0");

    zassert_ok(syn_store_install(&a, payload, sizeof(payload), NULL),
               "install A failed");
    payload_fill(45);
    zassert_ok(syn_store_install(&b, payload, sizeof(payload), NULL),
               "install B failed");

    /* erase the pivot from the newest image, like a v1 registry */
    registry_patch_newest(REG_HDR_PREV_OFF, SYN_STORE_SLOT_NONE);
    store_reboot();
    zassert_equal(syn_store_prev_active_slot(), SYN_STORE_SLOT_NONE,
                  "pivot patch did not take");

    zassert_ok(syn_store_rollback(), "fallback rollback failed");
    zassert_equal(syn_store_active_slot(), 0,
                  "fallback scan must find slot 0");
    zassert_equal(syn_store_prev_active_slot(), 1,
                  "demoted slot must become the pivot");
}

/** Structurally invalid registry headers are rejected before the CRC:
 *  impossible record counts and out-of-range slot references.
 */
ZTEST(syn_store_suite, test_registry_structural_rejects)
{
    static const struct {
        uint32_t off;   /* header byte to poison */
        uint8_t  val;
    } cases[] = {
        { 6U, 9U },              /* slot_count > layout capacity  */
        { 16U, 5U },             /* active_slot out of range      */
        { REG_HDR_PREV_OFF, 5U }, /* prev_active out of range     */
    };

    for (size_t i = 0; i < ARRAY_SIZE(cases); i++) {
        store_fresh();

        /* forge a header in copy 0: valid magic/version, poisoned
         * field. Every structural check fires before the CRC, so the
         * image needs no sealing.
         */
        uint8_t hdr[REG_HDR_SIZE] = {
            0x53, 0x59, 0x4E, 0x52,  /* "SYNR" LE magic */
            0x02, 0x00,              /* version 2       */
            0x02,                    /* slot_count      */
            0xFF,                    /* prev_active     */
            0x01, 0x00, 0x00, 0x00,  /* generation 1    */
            0x00, 0x00, 0x00, 0x00,  /* wear            */
            0xFF, 0xFF,              /* active, staged  */
            0x03, 0x00,              /* occupied, rsvd  */
            0x00, 0x00, 0x00, 0x00,  /* crc32 (sealed below) */
        };

        hdr[cases[i].off] = cases[i].val;
        syn_store_deinit();
        memcpy(&flash_mem[lay.registry_off[0]], hdr, sizeof(hdr));

        zassert_ok(syn_store_init(&port, &lay), "init failed");
        zassert_equal(syn_store_generation(), 0,
                      "poisoned image %u adopted", (unsigned)i);
    }
}

/** Corrupted payload of the incoming slot: activation itself sticks
 *  (flash registry is authoritative) but the hot NPU reload of the
 *  new model is refused by the CRC gate.
 */
ZTEST(syn_store_suite, test_activate_hot_reload_refused)
{
    store_fresh();
    payload_fill(46);

    syn_model_handle_t ha = SYN_MODEL_INVALID;
    syn_model_info_t a = make_info("hr_a", "1.0");
    syn_model_info_t b = make_info("hr_b", "1.0");

    zassert_ok(syn_store_install(&a, payload, sizeof(payload), &ha),
               "install A failed");
    payload_fill(47);
    zassert_ok(syn_store_install(&b, payload, sizeof(payload), NULL),
               "install B failed");

    /* cosmic ray on slot 0's payload, then hand it the NPU */
    flash_mem[lay.slot_off[0] + SYN_SYNM_HDR_SIZE + 17] ^= 0x40U;

    zassert_ok(syn_model_get_by_name("hr_a", &ha), "A not resident");
    zassert_ok(syn_model_get_by_name("hr_b", &ha), "B not resident");
    zassert_ok(syn_model_load(ha), "load B failed");

    zassert_ok(syn_store_activate(0), "activate must stick");
    zassert_equal(syn_store_active_slot(), 0, "flash state moved back");

    syn_model_handle_t hnew;

    zassert_ok(syn_model_get_by_name("hr_a", &hnew), "A lost");
    zassert_false(syn_model_is_loaded(hnew),
                  "corrupt model must not be NPU-loaded");
}

/** Same cosmic ray during rollback: the pivot becomes active but the
 *  hot reload of its corrupt payload is refused.
 */
ZTEST(syn_store_suite, test_rollback_hot_reload_refused)
{
    store_fresh();
    payload_fill(48);

    syn_model_info_t a = make_info("rr_a", "1.0");
    syn_model_info_t b = make_info("rr_b", "1.0");
    syn_model_handle_t hb = SYN_MODEL_INVALID;

    zassert_ok(syn_store_install(&a, payload, sizeof(payload), NULL),
               "install A failed");
    payload_fill(49);
    zassert_ok(syn_store_install(&b, payload, sizeof(payload), &hb),
               "install B failed");
    zassert_ok(syn_model_load(hb), "load B failed");

    flash_mem[lay.slot_off[0] + SYN_SYNM_HDR_SIZE + 5] ^= 0x08U;

    zassert_ok(syn_store_rollback(), "rollback must stick");
    zassert_equal(syn_store_active_slot(), 0, "not rolled back");

    syn_model_handle_t hp;

    zassert_ok(syn_model_get_by_name("rr_a", &hp), "pivot model lost");
    zassert_false(syn_model_is_loaded(hp),
                  "corrupt pivot must not be NPU-loaded");
}
