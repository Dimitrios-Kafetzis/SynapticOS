/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file test_store_faults.c
 * @brief Fault-injection tests for the store/OTA flash error paths
 *
 * The RAM flash port only injects write failures (power loss). This
 * suite wraps it in a fault port whose read and erase entry points
 * fail on a countdown, driving the store and OTA engine through the
 * error branches a healthy medium never reaches: registry commit
 * failures (erase, program, read-back verify), boot-scan read
 * errors, payload verification errors, and the state-restore paths
 * behind each of them. The wrapper is possible because the port seam
 * is a struct of function pointers; production code is untouched.
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

#define FSECTOR 1024U
#define FPAGE   128U
#define FSLOT   8192U

/* fixture memory shared with test_model_store.c (same geometry) */
extern uint8_t syn_test_flash_mem[2U * FSECTOR + 2U * FSLOT];
extern syn_flash_port_t syn_test_port;
extern syn_flash_ram_ctx_t syn_test_ram_ctx;

#define flash_mem syn_test_flash_mem
#define base_port syn_test_port
#define ram_ctx   syn_test_ram_ctx

static const syn_store_layout_t lay = {
    .registry_off = { 0U, FSECTOR },
    .registry_size = FSECTOR,
    .slot_off = { 2U * FSECTOR, 2U * FSECTOR + FSLOT },
    .slot_size = FSLOT,
    .slot_count = 2U,
};

/* Fault port: base RAM port with read/erase failing on a countdown
 * (1 = fail the next call, 0 = disarmed). The store captures the
 * port pointer at init, so the wrapper must outlive every test.
 */
static syn_flash_port_t fport;
static int fail_read_in;
static int fail_erase_in;

static int f_read(const syn_flash_port_t *p, uint32_t off,
                  void *buf, size_t len)
{
    if (fail_read_in > 0 && --fail_read_in == 0) {
        return -EIO;
    }
    return base_port.read(p, off, buf, len);
}

static int f_erase(const syn_flash_port_t *p, uint32_t off, size_t len)
{
    if (fail_erase_in > 0 && --fail_erase_in == 0) {
        return -EIO;
    }
    return base_port.erase(p, off, len);
}

static void faults_disarm(void)
{
    fail_read_in = 0;
    fail_erase_in = 0;
    ram_ctx.fail_after_pages = 0;
}

/* Fresh chip behind the fault port, faults disarmed. */
static void fault_fresh(void)
{
    faults_disarm();
    syn_ota_reset();
    syn_store_deinit();
    syn_model_reset_all();
    syn_hal_npu_init();
    zassert_ok(syn_flash_port_ram_init(&base_port, &ram_ctx, flash_mem,
                                       sizeof(flash_mem), FSECTOR, FPAGE),
               "ram port init failed");
    fport = base_port;
    fport.read = f_read;
    fport.erase = f_erase;
    zassert_ok(syn_store_init(&fport, &lay), "store init failed");
}

static syn_model_info_t finfo(const char *name)
{
    syn_model_info_t info = {0};

    strncpy(info.name, name, sizeof(info.name) - 1);
    strncpy(info.version, "1.0.0", sizeof(info.version) - 1);
    info.input_size = 64;
    info.output_size = 16;
    return info;
}

static uint8_t fpayload[384];

static void fpayload_fill(uint32_t seed)
{
    uint32_t x = seed * 2654435761U + 3U;

    for (size_t i = 0; i < sizeof(fpayload); i++) {
        x = x * 1103515245U + 12345U;
        fpayload[i] = (uint8_t)(x >> 16);
    }
}

/* small .synm image for the OTA fault sessions */
static uint8_t fsynm[SYN_SYNM_HDR_SIZE + 448U];

static uint32_t fsynm_build(const char *name, uint32_t psize)
{
    struct syn_synm_hdr hdr = {
        .magic = SYN_SYNM_MAGIC,
        .version = SYN_SYNM_VERSION,
        .model_size = psize,
        .input_shape = { 8, 8, 0, 0 },
        .output_shape = { 16, 0, 0, 0 },
    };

    zassert_true(psize <= sizeof(fsynm) - SYN_SYNM_HDR_SIZE, "img too big");
    fpayload_fill(77);
    memcpy(&fsynm[SYN_SYNM_HDR_SIZE], fpayload, psize);
    hdr.crc32 = crc32_ieee(&fsynm[SYN_SYNM_HDR_SIZE], psize);
    memset(hdr.name, 0, sizeof(hdr.name));
    strncpy(hdr.name, name, sizeof(hdr.name) - 1);
    memcpy(fsynm, &hdr, sizeof(hdr));
    return SYN_SYNM_HDR_SIZE + psize;
}

/* stream the fault-suite image through the OTA API */
static int fsynm_stream(uint32_t total, uint32_t chunk)
{
    uint32_t pos = 0;

    while (pos < total) {
        uint32_t n = MIN(chunk, total - pos);
        int ret = syn_ota_write_chunk(&fsynm[pos], n);

        if (ret != 0) {
            return ret;
        }
        pos += n;
    }
    return 0;
}

ZTEST_SUITE(syn_store_fault_suite, NULL, NULL, NULL, NULL, NULL);

/** Registry commit whose erase fails: begin_staging restores the
 *  occupied/staged bookkeeping it had already updated.
 */
ZTEST(syn_store_fault_suite, test_commit_erase_fail_restores)
{
    fault_fresh();

    syn_model_info_t info = finfo("f_stage");

    zassert_ok(syn_store_mark_staged(1, &info), "mark_staged failed");
    zassert_equal(syn_store_staged_slot(), 1, "not staged");

    fail_erase_in = 1;

    int ret = syn_store_begin_staging(1);

    zassert_equal(ret, -EIO, "erase fault not propagated: %d", ret);

    /* rolled back: still occupied and still the staged slot */
    zassert_equal(syn_store_staged_slot(), 1, "staged flag lost");

    syn_model_info_t chk;

    zassert_ok(syn_store_slot_info(1, &chk), "slot record lost");
    zassert_equal(strncmp(chk.name, "f_stage", sizeof(chk.name)), 0,
                  "record clobbered");
}

/** Registry commit whose read-back verify fails: clear_staged rolls
 *  its state back and reports the error.
 */
ZTEST(syn_store_fault_suite, test_commit_verify_read_fail_restores)
{
    fault_fresh();

    syn_model_info_t info = finfo("f_verify");

    zassert_ok(syn_store_mark_staged(1, &info), "mark_staged failed");

    fail_read_in = 1;

    int ret = syn_store_clear_staged();

    zassert_equal(ret, -EIO, "verify fault not propagated: %d", ret);
    zassert_equal(syn_store_staged_slot(), 1, "staged flag lost on error");

    faults_disarm();
    zassert_ok(syn_store_clear_staged(), "clean clear failed");
    zassert_equal(syn_store_staged_slot(), SYN_STORE_SLOT_NONE,
                  "staged flag survived");
}

/** Boot scan with failing registry reads: the unreadable copy is
 *  treated as invalid and the newest readable copy is adopted.
 */
ZTEST(syn_store_fault_suite, test_scan_survives_read_errors)
{
    fault_fresh();
    fpayload_fill(2);

    /* two commits so BOTH registry copies hold a valid image */
    syn_model_info_t scan_info = finfo("f_scan");

    zassert_ok(syn_store_install(&scan_info, fpayload,
                                 sizeof(fpayload), NULL),
               "install failed");

    syn_model_info_t info = finfo("f_scan2");

    zassert_ok(syn_store_mark_staged(1, &info), "mark_staged failed");

    uint32_t gen = syn_store_generation();

    /* reboot with the copy-0 header read failing */
    syn_store_deinit();
    syn_model_reset_all();
    fail_read_in = 1;
    zassert_ok(syn_store_init(&fport, &lay), "re-init failed");
    faults_disarm();
    zassert_equal(syn_store_generation(), gen,
                  "newest copy not adopted around the bad header read");

    /* reboot again with the copy-0 record-table read failing */
    syn_store_deinit();
    syn_model_reset_all();
    fail_read_in = 2;
    zassert_ok(syn_store_init(&fport, &lay), "re-init 2 failed");
    faults_disarm();
    zassert_equal(syn_store_generation(), gen,
                  "newest copy not adopted around the bad table read");
}

/** Payload read-back verification failing during install: the store
 *  reports the error and records nothing.
 */
ZTEST(syn_store_fault_suite, test_install_payload_read_fail)
{
    fault_fresh();
    fpayload_fill(3);

    /* first read of the whole install is the payload CRC check */
    fail_read_in = 1;

    syn_model_info_t info = finfo("f_pay");
    int ret = syn_store_install(&info, fpayload,
                                sizeof(fpayload), NULL);

    zassert_equal(ret, -EIO, "payload read fault not propagated: %d", ret);
    zassert_equal(syn_store_active_slot(), SYN_STORE_SLOT_NONE,
                  "failed install activated something");

    syn_model_info_t chk;

    zassert_equal(syn_store_slot_info(0, &chk), -ENOENT,
                  "failed install left a record");
}

/** Slot erase failing at the start of install. */
ZTEST(syn_store_fault_suite, test_install_slot_erase_fail)
{
    fault_fresh();
    fpayload_fill(4);
    fail_erase_in = 1;

    syn_model_info_t info = finfo("f_er");
    int ret = syn_store_install(&info, fpayload,
                                sizeof(fpayload), NULL);

    zassert_equal(ret, -EIO, "slot erase fault not propagated: %d", ret);
    zassert_equal(syn_store_active_slot(), SYN_STORE_SLOT_NONE,
                  "failed install activated something");
}

/** Final registry commit failing at the end of install: the record,
 *  active slot and rollback pivot all roll back.
 */
ZTEST(syn_store_fault_suite, test_install_final_commit_fail_restores)
{
    fault_fresh();
    fpayload_fill(5);

    /* erases before the final commit: 384 + 64 bytes fit one sector */
    fail_erase_in = 2;

    syn_model_info_t info = finfo("f_fin");
    int ret = syn_store_install(&info, fpayload,
                                sizeof(fpayload), NULL);

    zassert_equal(ret, -EIO, "commit fault not propagated: %d", ret);
    zassert_equal(syn_store_active_slot(), SYN_STORE_SLOT_NONE,
                  "active slot not rolled back");
    zassert_equal(syn_store_generation(), 0, "generation moved");

    syn_model_info_t chk;

    zassert_equal(syn_store_slot_info(0, &chk), -ENOENT,
                  "record not rolled back");

    /* the medium recovers: the same install then succeeds */
    faults_disarm();
    zassert_ok(syn_store_install(&info, fpayload,
                                 sizeof(fpayload), NULL),
               "recovery install failed");
    zassert_equal(syn_store_active_slot(), 0, "recovery not active");
}

/** Commit failing inside activate: active/staged/pivot state and the
 *  outgoing model's NPU residency are restored.
 */
ZTEST(syn_store_fault_suite, test_activate_commit_fail_restores)
{
    fault_fresh();
    fpayload_fill(6);

    syn_model_handle_t ha = SYN_MODEL_INVALID;
    syn_model_info_t info_a = finfo("f_act_a");

    zassert_ok(syn_store_install(&info_a, fpayload,
                                 sizeof(fpayload), &ha),
               "install A failed");
    zassert_ok(syn_model_load(ha), "load A failed");

    syn_model_info_t info = finfo("f_act_b");

    zassert_ok(syn_store_mark_staged(1, &info), "mark_staged failed");

    fail_erase_in = 1;

    int ret = syn_store_activate(1);

    zassert_equal(ret, -EIO, "activate fault not propagated: %d", ret);
    zassert_equal(syn_store_active_slot(), 0, "active slot not restored");
    zassert_equal(syn_store_staged_slot(), 1, "staged slot not restored");
    zassert_true(syn_model_is_loaded(ha),
                 "outgoing model residency not restored");
}

/** Commit failing inside rollback: the demotion is undone and the
 *  current model is reloaded.
 */
ZTEST(syn_store_fault_suite, test_rollback_commit_fail_restores)
{
    fault_fresh();
    fpayload_fill(7);

    syn_model_info_t info_a = finfo("f_rb_a");

    zassert_ok(syn_store_install(&info_a, fpayload,
                                 sizeof(fpayload), NULL),
               "install A failed");
    fpayload_fill(8);

    syn_model_handle_t hb = SYN_MODEL_INVALID;
    syn_model_info_t info_b = finfo("f_rb_b");

    zassert_ok(syn_store_install(&info_b, fpayload,
                                 sizeof(fpayload), &hb),
               "install B failed");
    zassert_ok(syn_model_load(hb), "load B failed");

    fail_erase_in = 1;

    int ret = syn_store_rollback();

    zassert_equal(ret, -EIO, "rollback fault not propagated: %d", ret);
    zassert_equal(syn_store_active_slot(), 1, "active slot not restored");
    zassert_true(syn_model_is_loaded(hb),
                 "current model residency not restored");
}

/* ------------------------------------------------------------------ */
/* OTA sessions over the fault port                                    */
/* ------------------------------------------------------------------ */

/** OTA begin refused when evicting the staging occupant fails. */
ZTEST(syn_store_fault_suite, test_ota_begin_evict_commit_fail)
{
    fault_fresh();
    fpayload_fill(9);

    /* occupy both slots so begin() must evict its staging victim */
    syn_model_info_t info_a = finfo("f_ob_a");
    syn_model_info_t info_b = finfo("f_ob_b");

    zassert_ok(syn_store_install(&info_a, fpayload,
                                 sizeof(fpayload), NULL), "install A");
    fpayload_fill(10);
    zassert_ok(syn_store_install(&info_b, fpayload,
                                 sizeof(fpayload), NULL), "install B");

    fail_erase_in = 1;

    uint32_t total = fsynm_build("f_ota", 256);
    int ret = syn_ota_begin("f_ota", total);

    zassert_equal(ret, -EIO, "eviction fault not propagated: %d", ret);
    zassert_equal(syn_ota_get_state(), SYN_OTA_STATE_ERROR, "not ERROR");
}

/** OTA begin fails when the staging-slot erase fails. */
ZTEST(syn_store_fault_suite, test_ota_begin_erase_fail)
{
    fault_fresh();
    fail_erase_in = 1;

    uint32_t total = fsynm_build("f_ota", 256);
    int ret = syn_ota_begin("f_ota", total);

    zassert_equal(ret, -EIO, "slot erase fault not propagated: %d", ret);
    zassert_equal(syn_ota_get_state(), SYN_OTA_STATE_ERROR, "not ERROR");
}

/** Mid-stream page program failure ends the session in ERROR. */
ZTEST(syn_store_fault_suite, test_ota_chunk_write_fail)
{
    fault_fresh();

    uint32_t total = fsynm_build("f_ota", 448);

    zassert_ok(syn_ota_begin("f_ota", total), "begin failed");
    ram_ctx.fail_after_pages = 1;

    int ret = fsynm_stream(total, 256);

    zassert_equal(ret, -EIO, "program fault not propagated: %d", ret);
    zassert_equal(syn_ota_get_state(), SYN_OTA_STATE_ERROR, "not ERROR");
}

/** Final-flush page program failure inside finish(). */
ZTEST(syn_store_fault_suite, test_ota_finish_flush_fail)
{
    fault_fresh();

    /* 64 + 100 bytes stay buffered until the final flush */
    uint32_t total = fsynm_build("f_ota", 100);

    zassert_ok(syn_ota_begin("f_ota", total), "begin failed");
    zassert_ok(fsynm_stream(total, 64), "stream failed");
    ram_ctx.fail_after_pages = 1;

    int ret = syn_ota_finish();

    zassert_equal(ret, -EIO, "final flush fault not propagated: %d", ret);
    zassert_equal(syn_ota_get_state(), SYN_OTA_STATE_ERROR, "not ERROR");
}

/** Flash read failure during finish() validation. */
ZTEST(syn_store_fault_suite, test_ota_validate_read_fail)
{
    fault_fresh();

    uint32_t total = fsynm_build("f_ota", 256);

    zassert_ok(syn_ota_begin("f_ota", total), "begin failed");
    zassert_ok(fsynm_stream(total, 256), "stream failed");
    fail_read_in = 1;

    int ret = syn_ota_finish();

    zassert_equal(ret, -EIO, "validate read fault not propagated: %d", ret);
    zassert_equal(syn_ota_get_state(), SYN_OTA_STATE_ERROR, "not ERROR");
}

/** Staging commit failure at the end of finish(). */
ZTEST(syn_store_fault_suite, test_ota_stage_commit_fail)
{
    fault_fresh();

    uint32_t total = fsynm_build("f_ota", 256);

    zassert_ok(syn_ota_begin("f_ota", total), "begin failed");
    zassert_ok(fsynm_stream(total, 256), "stream failed");
    fail_erase_in = 1;

    int ret = syn_ota_finish();

    zassert_equal(ret, -EIO, "stage commit fault not propagated: %d", ret);
    zassert_equal(syn_ota_get_state(), SYN_OTA_STATE_ERROR, "not ERROR");
    zassert_equal(syn_store_staged_slot(), SYN_STORE_SLOT_NONE,
                  "failed staging left a staged slot");
}

/** Activation commit failure after a fully staged session. */
ZTEST(syn_store_fault_suite, test_ota_activate_commit_fail)
{
    fault_fresh();

    uint32_t total = fsynm_build("f_ota", 256);

    zassert_ok(syn_ota_begin("f_ota", total), "begin failed");
    zassert_ok(fsynm_stream(total, 256), "stream failed");
    zassert_ok(syn_ota_finish(), "finish failed");

    fail_erase_in = 1;

    int ret = syn_ota_activate();

    zassert_equal(ret, -EIO, "activate fault not propagated: %d", ret);
    zassert_equal(syn_ota_get_state(), SYN_OTA_STATE_ERROR, "not ERROR");
    zassert_equal(syn_store_active_slot(), SYN_STORE_SLOT_NONE,
                  "failed activation switched the active slot");

    /* the staged record survives: a later activate can still adopt it */
    faults_disarm();
    syn_ota_reset();
    zassert_ok(syn_ota_activate(), "recovery activate failed");
    zassert_equal(syn_store_active_slot(), 0, "staged slot not adopted");
}

/* ------------------------------------------------------------------ */
/* RAM flash port guard probes                                         */
/* ------------------------------------------------------------------ */

/** Range, alignment and argument guards of the port wrappers. */
ZTEST(syn_store_fault_suite, test_flash_port_guards)
{
    fault_fresh();

    uint8_t buf[FPAGE];

    memset(buf, 0xA5, sizeof(buf));

    /* read: bad range, NULL buffer */
    zassert_equal(syn_flash_read(&base_port, base_port.size, buf, 4),
                  -EINVAL, "read past the window accepted");
    zassert_equal(syn_flash_read(&base_port, 0, NULL, 4), -EINVAL,
                  "read into NULL accepted");

    /* write: NULL buffer, misaligned offset and length */
    zassert_equal(syn_flash_write(&base_port, 0, NULL, FPAGE), -EINVAL,
                  "write from NULL accepted");
    zassert_equal(syn_flash_write(&base_port, 1, buf, FPAGE), -EINVAL,
                  "misaligned write offset accepted");
    zassert_equal(syn_flash_write(&base_port, 0, buf, FPAGE - 1U),
                  -EINVAL, "misaligned write length accepted");

    /* erase: bad range, misaligned offset */
    zassert_equal(syn_flash_erase(&base_port, base_port.size, FSECTOR),
                  -EINVAL, "erase past the window accepted");
    zassert_equal(syn_flash_erase(&base_port, 1, FSECTOR), -EINVAL,
                  "misaligned erase accepted");

    /* erase_sectors: bad range, misaligned length */
    zassert_equal(syn_flash_erase_sectors(&base_port, base_port.size,
                                          FSECTOR),
                  -EINVAL, "erase_sectors past the window accepted");
    zassert_equal(syn_flash_erase_sectors(&base_port, 0, FSECTOR - 1U),
                  -EINVAL, "misaligned erase_sectors accepted");

    /* is_blank: bad range; mmap: out of window */
    zassert_equal(syn_flash_is_blank(&base_port, base_port.size, 4),
                  -EINVAL, "is_blank past the window accepted");
    zassert_is_null(syn_flash_mmap(&base_port, base_port.size),
                    "mmap past the window accepted");

    /* NOR program semantics: rewriting a programmed page is refused */
    uint32_t off = lay.slot_off[1];

    zassert_ok(syn_flash_erase(&base_port, off, FSECTOR), "erase failed");
    zassert_ok(syn_flash_write(&base_port, off, buf, FPAGE),
               "first program failed");
    zassert_equal(syn_flash_write(&base_port, off, buf, FPAGE), -EACCES,
                  "reprogram of a non-blank page accepted");

    /* port init argument guard */
    zassert_equal(syn_flash_port_ram_init(&fport, &ram_ctx, NULL,
                                          1024, 1024, 128),
                  -EINVAL, "NULL backing memory accepted");

    /* leave a clean fixture for whoever runs next */
    fault_fresh();
}
