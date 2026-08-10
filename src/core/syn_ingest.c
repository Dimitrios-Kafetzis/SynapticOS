/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file syn_ingest.c
 * @brief SynapticOS - Double-buffered zero-copy frame ingest
 *
 * See syn_ingest.h for the pump contract.
 */

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <string.h>

LOG_MODULE_REGISTER(syn_ingest, CONFIG_SYNAPTIC_LOG_LEVEL);

#include <synaptic/syn_hal_dma.h>

#include "syn_ingest.h"

#define INGEST_DMA_TIMEOUT_MS 1000

static K_SEM_DEFINE(dma_done, 0, 1);
static volatile int dma_status;
static syn_ingest_stats_t last_stats;

static void dma_cb(int channel, int status, void *user_data)
{
	ARG_UNUSED(channel);
	ARG_UNUSED(user_data);

	dma_status = status;
	k_sem_give(&dma_done);
}

/** Fill (optional) and start the DMA of frame @p seq into @p dst. */
static int kick_transfer(const syn_ingest_config_t *cfg, uint32_t seq,
			 void *dst)
{
	/* Guard against a stale token from a late callback of the
	 * previous transfer. Do NOT stop/abort the channel here: on the
	 * eDMA an abort between back-to-back one-shot transfers leaves
	 * the channel unable to complete the next one (board finding).
	 */
	k_sem_reset(&dma_done);

	if (cfg->fill != NULL) {
		cfg->fill(cfg->src, cfg->frame_size, seq, cfg->user);
	}

	syn_dma_config_t dcfg = {
		.src_periph = SYN_DMA_PERIPH_MEMORY,
		.dst_periph = SYN_DMA_PERIPH_MEMORY,
		.src_addr = cfg->src,
		.dst_addr = dst,
		.transfer_size = cfg->frame_size,
		.circular = false,
	};
	int ret = syn_hal_dma_configure(cfg->dma_channel, &dcfg);

	if (ret != 0) {
		return ret;
	}
	return syn_hal_dma_start(cfg->dma_channel, dma_cb, NULL);
}

static int wait_transfer(void)
{
	if (k_sem_take(&dma_done, K_MSEC(INGEST_DMA_TIMEOUT_MS)) != 0) {
		return -ETIMEDOUT;
	}
	return (dma_status < 0) ? dma_status : 0;
}

int syn_ingest_run(const syn_ingest_config_t *cfg, uint32_t frames)
{
	if (cfg == NULL || cfg->src == NULL || cfg->bufs[0] == NULL ||
	    cfg->bufs[1] == NULL || cfg->frame_size == 0U ||
	    cfg->process == NULL || frames == 0U) {
		return -EINVAL;
	}

	memset(&last_stats, 0, sizeof(last_stats));
	k_sem_reset(&dma_done);

	uint32_t t0 = k_cycle_get_32();

	/* Prime the pipeline with frame 0 */
	int ret = kick_transfer(cfg, 0, cfg->bufs[0]);

	if (ret == 0) {
		ret = wait_transfer();
	}
	if (ret != 0) {
		last_stats.dma_errors++;
		LOG_ERR("Ingest priming failed: %d", ret);
		return -EIO;
	}

	for (uint32_t seq = 0; seq < frames; seq++) {
		bool more = (seq + 1U) < frames;

		if (more) {
			ret = kick_transfer(cfg, seq + 1U,
					    cfg->bufs[(seq + 1U) & 1U]);
			if (ret != 0) {
				last_stats.dma_errors++;
				LOG_ERR("Ingest DMA start failed at frame "
					"%u: %d", seq + 1U, ret);
				return -EIO;
			}
		}

		/* Consume frame N while the DMA writes frame N+1 */
		cfg->process(cfg->bufs[seq & 1U], cfg->frame_size, seq,
			     cfg->user);
		last_stats.frames++;

		if (more) {
			ret = wait_transfer();
			if (ret != 0) {
				last_stats.dma_errors++;
				LOG_ERR("Ingest DMA wait failed at frame "
					"%u: %d", seq + 1U, ret);
				return -EIO;
			}
		}
	}

	last_stats.elapsed_us = k_cyc_to_us_ceil32(k_cycle_get_32() - t0);
	return 0;
}

void syn_ingest_last_stats(syn_ingest_stats_t *stats)
{
	if (stats != NULL) {
		*stats = last_stats;
	}
}
