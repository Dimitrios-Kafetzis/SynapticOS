/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file test_dma_ingest.c
 * @brief Unit tests for the DMA HAL stub and the zero-copy ingest
 *        pump (Phase 5.3)
 *
 * The QEMU DMA stub copies on the system workqueue, so these tests
 * verify the async contract and frame integrity, not throughput -
 * the eDMA overlap numbers come from `syn dma bench` on the board.
 */

#include <zephyr/ztest.h>
#include <string.h>

#include <synaptic/syn_hal_dma.h>
#include "syn_ingest.h"

#define XFER_SIZE  256
#define IG_FRAME   128
#define IG_FRAMES  8

static uint8_t src_buf[XFER_SIZE];
static uint8_t dst_buf[XFER_SIZE];

static K_SEM_DEFINE(cb_sem, 0, 10);
static volatile int cb_status;
static volatile int cb_count;

static void xfer_cb(int channel, int status, void *user_data)
{
	ARG_UNUSED(channel);
	ARG_UNUSED(user_data);

	cb_status = status;
	cb_count++;
	k_sem_give(&cb_sem);
}

static void dma_before(void *fixture)
{
	ARG_UNUSED(fixture);

	int ret = syn_hal_dma_init();

	zassert_true(ret == 0 || ret == -EALREADY, "DMA init failed: %d",
		     ret);
	cb_count = 0;
	cb_status = -1;
	k_sem_reset(&cb_sem);
}

ZTEST_SUITE(syn_dma_suite, NULL, NULL, dma_before, NULL, NULL);

/** One-shot transfer: async completion, correct copy, 0 remaining. */
ZTEST(syn_dma_suite, test_dma_oneshot)
{
	for (int i = 0; i < XFER_SIZE; i++) {
		src_buf[i] = (uint8_t)(i * 13 + 5);
	}
	memset(dst_buf, 0, sizeof(dst_buf));

	syn_dma_config_t cfg = {
		.src_periph = SYN_DMA_PERIPH_MEMORY,
		.dst_periph = SYN_DMA_PERIPH_MEMORY,
		.src_addr = src_buf,
		.dst_addr = dst_buf,
		.transfer_size = XFER_SIZE,
	};

	zassert_equal(syn_hal_dma_configure(0, &cfg), 0, "configure failed");
	zassert_equal(syn_hal_dma_start(0, xfer_cb, NULL), 0,
		      "start failed");
	zassert_equal(k_sem_take(&cb_sem, K_MSEC(500)), 0,
		      "completion callback never fired");
	zassert_equal(cb_status, 0, "transfer status %d", cb_status);
	zassert_mem_equal(dst_buf, src_buf, XFER_SIZE, "copy mismatch");

	size_t remaining = 1;

	zassert_equal(syn_hal_dma_get_remaining(0, &remaining), 0,
		      "get_remaining failed");
	zassert_equal(remaining, 0, "remaining should be 0, got %u",
		      (unsigned)remaining);
}

/** Bad channels and configs are rejected. */
ZTEST(syn_dma_suite, test_dma_invalid_args)
{
	syn_dma_config_t cfg = {
		.src_addr = src_buf,
		.dst_addr = dst_buf,
		.transfer_size = 16,
	};

	zassert_equal(syn_hal_dma_configure(-1, &cfg), -EINVAL,
		      "negative channel accepted");
	zassert_equal(syn_hal_dma_configure(99, &cfg), -EINVAL,
		      "out-of-range channel accepted");
	zassert_equal(syn_hal_dma_configure(1, NULL), -EINVAL,
		      "NULL config accepted");

	cfg.transfer_size = 0;
	zassert_equal(syn_hal_dma_configure(1, &cfg), -EINVAL,
		      "zero-size transfer accepted");

	/* Channel 2 was never configured */
	zassert_equal(syn_hal_dma_start(2, xfer_cb, NULL), -EINVAL,
		      "start of unconfigured channel accepted");
}

/** Circular transfers repeat until stopped. */
ZTEST(syn_dma_suite, test_dma_circular)
{
	memset(src_buf, 0xA7, sizeof(src_buf));
	memset(dst_buf, 0, sizeof(dst_buf));

	syn_dma_config_t cfg = {
		.src_periph = SYN_DMA_PERIPH_MEMORY,
		.dst_periph = SYN_DMA_PERIPH_MEMORY,
		.src_addr = src_buf,
		.dst_addr = dst_buf,
		.transfer_size = XFER_SIZE,
		.circular = true,
	};

	zassert_equal(syn_hal_dma_configure(3, &cfg), 0, "configure failed");
	zassert_equal(syn_hal_dma_start(3, xfer_cb, NULL), 0,
		      "start failed");

	/* At least three iterations should complete */
	for (int i = 0; i < 3; i++) {
		zassert_equal(k_sem_take(&cb_sem, K_MSEC(500)), 0,
			      "circular iteration %d never fired", i);
	}
	zassert_equal(syn_hal_dma_stop(3), 0, "stop failed");
	zassert_true(cb_count >= 3, "expected >= 3 callbacks, got %d",
		     cb_count);
	zassert_equal(dst_buf[0], 0xA7, "circular copy missing");

	/* After stop the channel settles; drain any last callback */
	k_msleep(20);
}

/* ------------------------------------------------------------------ */
/* Ingest pump                                                         */
/* ------------------------------------------------------------------ */

static uint8_t ig_src[IG_FRAME];
static uint8_t ig_b0[IG_FRAME];
static uint8_t ig_b1[IG_FRAME];

struct ig_ctx {
	uint32_t mismatches;
	uint32_t frames_seen;
	const void *last_ptr;
	bool alternated;
};

static void ig_fill(void *src, size_t size, uint32_t seq, void *user)
{
	ARG_UNUSED(user);

	uint8_t *p = src;

	for (size_t i = 0; i < size; i++) {
		p[i] = (uint8_t)(seq * 31U + i * 7U);
	}
}

static void ig_process(const void *frame, size_t size, uint32_t seq,
		       void *user)
{
	struct ig_ctx *ctx = user;
	const uint8_t *p = frame;

	for (size_t i = 0; i < size; i++) {
		if (p[i] != (uint8_t)(seq * 31U + i * 7U)) {
			ctx->mismatches++;
			break;
		}
	}
	if (ctx->frames_seen > 0 && frame != ctx->last_ptr) {
		ctx->alternated = true;
	}
	ctx->last_ptr = frame;
	ctx->frames_seen++;
}

/** Full pump run: every frame byte-exact, buffers alternate, and the
 *  consumer reads the DMA-written buffers in place (zero copy).
 */
ZTEST(syn_dma_suite, test_ingest_pump)
{
	struct ig_ctx ctx = {0};
	syn_ingest_config_t cfg = {
		.src = ig_src,
		.bufs = { ig_b0, ig_b1 },
		.frame_size = IG_FRAME,
		.dma_channel = 1,
		.fill = ig_fill,
		.process = ig_process,
		.user = &ctx,
	};

	zassert_equal(syn_ingest_run(&cfg, IG_FRAMES), 0, "pump failed");

	syn_ingest_stats_t st;

	syn_ingest_last_stats(&st);
	zassert_equal(st.frames, IG_FRAMES, "delivered %u of %u frames",
		      st.frames, IG_FRAMES);
	zassert_equal(st.dma_errors, 0, "dma errors: %u", st.dma_errors);
	zassert_equal(ctx.mismatches, 0, "corrupt frames: %u",
		      ctx.mismatches);
	zassert_true(ctx.alternated, "ping/pong buffers did not alternate");
	zassert_true(ctx.last_ptr == ig_b0 || ctx.last_ptr == ig_b1,
		     "consumer did not read the ingest buffers in place");
	zassert_true(st.elapsed_us > 0, "elapsed time missing");
}

/** Bad pump configs are rejected. */
ZTEST(syn_dma_suite, test_ingest_invalid)
{
	struct ig_ctx ctx = {0};
	syn_ingest_config_t cfg = {
		.src = ig_src,
		.bufs = { ig_b0, ig_b1 },
		.frame_size = IG_FRAME,
		.dma_channel = 1,
		.process = ig_process,
		.user = &ctx,
	};

	zassert_equal(syn_ingest_run(NULL, 4), -EINVAL, "NULL cfg");
	zassert_equal(syn_ingest_run(&cfg, 0), -EINVAL, "0 frames");

	cfg.process = NULL;
	zassert_equal(syn_ingest_run(&cfg, 4), -EINVAL, "NULL process");

	cfg.process = ig_process;
	cfg.bufs[1] = NULL;
	zassert_equal(syn_ingest_run(&cfg, 4), -EINVAL, "NULL buffer");
}

/* Phase 6 S13: a bad DMA channel aborts the pump during priming */
ZTEST(syn_dma_suite, test_ingest_bad_channel_aborts)
{
	static uint8_t src[64];
	static uint8_t ping[64], pong[64];
	syn_ingest_config_t cfg = {
		.src = src,
		.bufs = { ping, pong },
		.frame_size = sizeof(src),
		.dma_channel = 42,
		.process = ig_process,
		.user = NULL,
	};

	int ret = syn_ingest_run(&cfg, 2);

	zassert_equal(ret, -EIO, "bad channel not reported: %d", ret);
}
