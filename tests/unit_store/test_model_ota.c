/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file test_model_ota.c
 * @brief Unit tests for the A/B OTA engine (Phase 4.2)
 *
 * Full state-machine coverage over the RAM flash port: chunked
 * transfer, integrity rejection at every stage, activate/rollback,
 * power loss mid-transfer and after staging. The .synm images are
 * synthesized in RAM exactly as tools/syn_model_pack.py lays them
 * out on the host.
 */

#include <zephyr/ztest.h>
#include <zephyr/sys/crc.h>
#include <string.h>
#include <synaptic/syn_model.h>
#include <synaptic/syn_model_ota.h>
#include <synaptic/syn_hal_npu.h>

#include "syn_model_store.h"
#include "syn_model_internal.h"
#include "syn_model_ota_internal.h"
#include "syn_flash_port.h"
#include "syn_synm.h"

#define OSECTOR 1024U
#define OPAGE   128U
#define OSLOT   8192U

/* fixture memory shared with test_model_store.c (same geometry) */
extern uint8_t syn_test_flash_mem[2U * OSECTOR + 2U * OSLOT];
extern syn_flash_port_t syn_test_port;
extern syn_flash_ram_ctx_t syn_test_ram_ctx;

#define flash_mem syn_test_flash_mem
#define port      syn_test_port
#define ram_ctx   syn_test_ram_ctx

static const syn_store_layout_t lay = {
    .registry_off = { 0U, OSECTOR },
    .registry_size = OSECTOR,
    .slot_off = { 2U * OSECTOR, 2U * OSECTOR + OSLOT },
    .slot_size = OSLOT,
};

/* synthesized .synm image: header + payload */
#define IMG_PAYLOAD_MAX 6144U
static uint8_t synm_img[SYN_SYNM_HDR_SIZE + IMG_PAYLOAD_MAX];

static uint32_t build_synm(const char *name, uint32_t seed, uint32_t psize)
{
    struct syn_synm_hdr hdr = {
        .magic = SYN_SYNM_MAGIC,
        .version = SYN_SYNM_VERSION,
        .model_size = psize,
        .input_shape = { 8, 8, 0, 0 },
        .output_shape = { 16, 0, 0, 0 },
    };
    uint32_t x = seed * 2654435761U + 7U;

    zassert_true(psize <= IMG_PAYLOAD_MAX, "payload too big for fixture");

    for (uint32_t i = 0; i < psize; i++) {
        x = x * 1103515245U + 12345U;
        synm_img[SYN_SYNM_HDR_SIZE + i] = (uint8_t)(x >> 16);
    }
    hdr.crc32 = crc32_ieee(&synm_img[SYN_SYNM_HDR_SIZE], psize);
    memset(hdr.name, 0, sizeof(hdr.name));
    strncpy(hdr.name, name, sizeof(hdr.name) - 1);
    memcpy(synm_img, &hdr, sizeof(hdr));
    return SYN_SYNM_HDR_SIZE + psize;
}

static void ota_fresh(void)
{
    syn_store_deinit();
    syn_model_reset_all();
    syn_ota_reset();
    syn_hal_npu_init();
    zassert_ok(syn_flash_port_ram_init(&port, &ram_ctx, flash_mem,
                                       sizeof(flash_mem), OSECTOR, OPAGE));
    zassert_ok(syn_store_init(&port, &lay));
}

static void ota_reboot(void)
{
    syn_store_deinit();
    syn_model_reset_all();
    syn_ota_reset();
    zassert_ok(syn_store_init(&port, &lay));
}

/* stream an image through the OTA API in fixed-size chunks */
static int stream_synm(uint32_t total, uint32_t chunk)
{
    uint32_t pos = 0;

    while (pos < total) {
        uint32_t n = MIN(chunk, total - pos);
        int ret = syn_ota_write_chunk(&synm_img[pos], n);

        if (ret != 0) {
            return ret;
        }
        pos += n;
    }
    return 0;
}

ZTEST_SUITE(syn_ota_suite, NULL, NULL, NULL, NULL, NULL);

ZTEST(syn_ota_suite, test_full_cycle_4k_chunks)
{
    ota_fresh();

    uint32_t total = build_synm("m1", 21, IMG_PAYLOAD_MAX);

    zassert_equal(syn_ota_get_state(), SYN_OTA_STATE_IDLE, "not idle");
    zassert_ok(syn_ota_begin("m1", total), "begin failed");
    zassert_equal(syn_ota_get_state(), SYN_OTA_STATE_DOWNLOADING,
                  "not downloading");

    /* plan-reference chunk size: 4 KB */
    zassert_ok(stream_synm(total, 4096), "stream failed");
    zassert_ok(syn_ota_finish(), "finish failed");
    zassert_equal(syn_ota_get_state(), SYN_OTA_STATE_READY, "not ready");

    zassert_ok(syn_ota_activate(), "activate failed");
    zassert_equal(syn_ota_get_state(), SYN_OTA_STATE_IDLE,
                  "not idle after activate");

    /* every chunk byte must be in flash exactly as sent */
    zassert_mem_equal(&flash_mem[lay.slot_off[0]], synm_img, total,
                      "flash content differs from stream");

    syn_model_handle_t h;

    zassert_ok(syn_model_get_by_name("m1", &h), "model not registered");
    zassert_ok(syn_model_load(h), "CRC-gated load failed");
    zassert_ok(syn_model_unload(h));
}

ZTEST(syn_ota_suite, test_odd_chunks_partial_pages)
{
    ota_fresh();

    /* 997 is prime: every page boundary lands mid-chunk */
    uint32_t total = build_synm("m1", 22, 3000);

    zassert_ok(syn_ota_begin("m1", total));
    zassert_ok(stream_synm(total, 997));
    zassert_ok(syn_ota_finish());
    zassert_ok(syn_ota_activate());
    zassert_mem_equal(&flash_mem[lay.slot_off[0]], synm_img, total,
                      "odd chunking corrupted the image");
}

ZTEST(syn_ota_suite, test_bad_magic_rejected)
{
    ota_fresh();

    uint32_t total = build_synm("m1", 23, 1024);

    synm_img[0] ^= 0xFFU; /* break the magic */

    zassert_ok(syn_ota_begin("m1", total));

    int ret = stream_synm(total, 4096);

    zassert_equal(ret, -EILSEQ, "bad magic accepted: %d", ret);
    zassert_equal(syn_ota_get_state(), SYN_OTA_STATE_ERROR, "not in ERROR");

    /* a new session recovers from ERROR */
    total = build_synm("m1", 23, 1024);
    zassert_ok(syn_ota_begin("m1", total), "begin after ERROR failed");
    zassert_ok(stream_synm(total, 4096));
    zassert_ok(syn_ota_finish());
    zassert_ok(syn_ota_activate());
}

ZTEST(syn_ota_suite, test_payload_crc_mismatch_rejected)
{
    ota_fresh();

    uint32_t total = build_synm("m1", 24, 2048);

    /* corrupt one payload byte AFTER the header CRC was computed */
    synm_img[SYN_SYNM_HDR_SIZE + 500] ^= 0x01U;

    zassert_ok(syn_ota_begin("m1", total));
    zassert_ok(stream_synm(total, 1024));

    int ret = syn_ota_finish();

    zassert_equal(ret, -EILSEQ, "corrupt payload staged: %d", ret);
    zassert_equal(syn_ota_get_state(), SYN_OTA_STATE_ERROR, "not in ERROR");
    zassert_equal(syn_store_staged_slot(), SYN_STORE_SLOT_NONE,
                  "corrupt image left staged");
}

ZTEST(syn_ota_suite, test_header_size_and_name_mismatch)
{
    ota_fresh();

    /* model_size disagreeing with the announced total */
    uint32_t total = build_synm("m1", 25, 1024);
    struct syn_synm_hdr *hdr = (struct syn_synm_hdr *)synm_img;

    hdr->model_size = 999;
    zassert_ok(syn_ota_begin("m1", total));
    zassert_equal(stream_synm(total, 512), -EMSGSIZE,
                  "size mismatch accepted");
    zassert_equal(syn_ota_get_state(), SYN_OTA_STATE_ERROR, "not in ERROR");

    /* header name disagreeing with begin() */
    total = build_synm("other", 25, 1024);
    zassert_ok(syn_ota_begin("m1", total));
    zassert_equal(stream_synm(total, 512), -EINVAL,
                  "name mismatch accepted");
}

ZTEST(syn_ota_suite, test_state_machine_guards)
{
    ota_fresh();

    uint8_t junk[16] = {0};

    zassert_equal(syn_ota_write_chunk(junk, sizeof(junk)), -EPERM,
                  "chunk accepted while IDLE");
    zassert_equal(syn_ota_finish(), -EPERM, "finish accepted while IDLE");
    zassert_equal(syn_ota_activate(), -EPERM,
                  "activate accepted with nothing staged");
    zassert_equal(syn_ota_rollback(), -ENOENT,
                  "rollback accepted with no candidate");

    /* oversize and short-stream guards */
    zassert_equal(syn_ota_begin("m1", OSLOT + 1U), -EFBIG,
                  "oversize begin accepted");

    uint32_t total = build_synm("m1", 26, 1024);

    zassert_ok(syn_ota_begin("m1", total));
    zassert_ok(stream_synm(total - 100U, 512));
    zassert_equal(syn_ota_finish(), -ENODATA, "short stream finished");
    zassert_equal(syn_ota_get_state(), SYN_OTA_STATE_ERROR, "not in ERROR");
}

ZTEST(syn_ota_suite, test_update_and_rollback_via_ota)
{
    ota_fresh();

    /* v1 */
    uint32_t total = build_synm("m1", 27, 1500);

    zassert_ok(syn_ota_begin("m1", total));
    zassert_ok(stream_synm(total, 4096));
    zassert_ok(syn_ota_finish());
    zassert_ok(syn_ota_activate());

    syn_model_handle_t h;
    syn_model_info_t v1_info;

    zassert_ok(syn_model_get_by_name("m1", &h));
    zassert_ok(syn_model_get_info(h, &v1_info));

    /* v2 into the other slot */
    total = build_synm("m1", 28, 2500);
    zassert_ok(syn_ota_begin("m1", total));
    zassert_ok(stream_synm(total, 4096));
    zassert_ok(syn_ota_finish());
    zassert_ok(syn_ota_activate());
    zassert_equal(syn_store_active_slot(), 1, "v2 not in slot B");

    syn_model_info_t v2_info;

    zassert_ok(syn_model_get_by_name("m1", &h));
    zassert_ok(syn_model_get_info(h, &v2_info));
    zassert_not_equal(v2_info.crc32, v1_info.crc32, "v2 == v1");
    zassert_ok(syn_model_load(h));
    zassert_ok(syn_model_unload(h));

    /* rollback restores v1 */
    zassert_ok(syn_ota_rollback(), "rollback failed");
    zassert_equal(syn_store_active_slot(), 0, "not back on slot A");
    zassert_ok(syn_model_get_by_name("m1", &h));

    syn_model_info_t back;

    zassert_ok(syn_model_get_info(h, &back));
    zassert_equal(back.crc32, v1_info.crc32, "rollback content differs");
    zassert_ok(syn_model_load(h), "rolled-back model unloadable");
    zassert_ok(syn_model_unload(h));
}

ZTEST(syn_ota_suite, test_powerloss_mid_transfer)
{
    ota_fresh();

    /* baseline model via OTA */
    uint32_t total = build_synm("m1", 29, 1500);

    zassert_ok(syn_ota_begin("m1", total));
    zassert_ok(stream_synm(total, 4096));
    zassert_ok(syn_ota_finish());
    zassert_ok(syn_ota_activate());

    uint32_t gen = syn_store_generation();

    /* second update dies mid-stream: reboot with half a transfer */
    total = build_synm("m1", 30, 4000);
    zassert_ok(syn_ota_begin("m1", total));
    zassert_ok(stream_synm(2000, 1000));

    ota_reboot();

    zassert_equal(syn_ota_get_state(), SYN_OTA_STATE_IDLE,
                  "state survived reboot");
    zassert_equal(syn_store_generation(), gen, "generation moved");
    zassert_equal(syn_store_staged_slot(), SYN_STORE_SLOT_NONE,
                  "phantom staged slot");

    syn_model_handle_t h;

    zassert_ok(syn_model_get_by_name("m1", &h), "old model lost");
    zassert_ok(syn_model_load(h), "old model corrupted");
    zassert_ok(syn_model_unload(h));

    /* the interrupted update can simply be re-run */
    total = build_synm("m1", 30, 4000);
    zassert_ok(syn_ota_begin("m1", total));
    zassert_ok(stream_synm(total, 4096));
    zassert_ok(syn_ota_finish());
    zassert_ok(syn_ota_activate());
}

ZTEST(syn_ota_suite, test_staged_survives_reboot_then_activates)
{
    ota_fresh();

    uint32_t total = build_synm("m1", 31, 2000);

    zassert_ok(syn_ota_begin("m1", total));
    zassert_ok(stream_synm(total, 4096));
    zassert_ok(syn_ota_finish());
    zassert_equal(syn_ota_get_state(), SYN_OTA_STATE_READY, "not READY");

    /* power loss between finish and activate */
    ota_reboot();

    zassert_equal(syn_ota_get_state(), SYN_OTA_STATE_IDLE, "not IDLE");
    zassert_not_equal(syn_store_staged_slot(), SYN_STORE_SLOT_NONE,
                      "staged record lost");

    /* activate() from IDLE picks up the staged slot */
    zassert_ok(syn_ota_activate(), "post-reboot activate failed");

    syn_model_handle_t h;

    zassert_ok(syn_model_get_by_name("m1", &h), "model not active");
    zassert_ok(syn_model_load(h));
    zassert_ok(syn_model_unload(h));
}
