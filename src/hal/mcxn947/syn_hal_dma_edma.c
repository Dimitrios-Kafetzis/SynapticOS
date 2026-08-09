/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file syn_hal_dma_edma.c
 * @brief SynapticOS - DMA HAL over the MCXN947 eDMA (edma0)
 *
 * Implements the frozen syn_hal_dma.h API on the eDMA v4 engine
 * through Zephyr's DMA driver (nxp,mcux-edma-v4). Memory-to-memory
 * transfers run fully in hardware, so double-buffered ingest
 * genuinely overlaps the copy with CPU work (Phase 5.3).
 *
 * - SynapticOS channels 0..3 map to eDMA channels 8..11, clear of
 *   the FlexComm DMA request lines wired to channels 0..5 in the
 *   device tree.
 * - Completion callbacks fire in ISR context.
 * - Circular transfers are re-armed in software from the completion
 *   callback (reload + start), which is sufficient for the frame
 *   cadences this runtime targets.
 * - The SmartDMA engine stays reserved for the camera path
 *   (deferred with OV7670 bring-up).
 *
 * Builds without CONFIG_DMA fall back to -ENOSYS so samples that do
 * not enable the driver still link.
 */

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>

LOG_MODULE_REGISTER(syn_hal_dma_edma, CONFIG_SYNAPTIC_LOG_LEVEL);

#include <synaptic/syn_hal_dma.h>

#if defined(CONFIG_DMA) && DT_NODE_HAS_STATUS(DT_NODELABEL(edma0), okay)

#include <zephyr/drivers/dma.h>

#define SYN_DMA_CHANNELS   4
#define EDMA_CH_BASE       8   /* first eDMA channel we own */

struct edma_channel {
	syn_dma_config_t cfg;
	syn_dma_cb_t cb;
	void *user_data;
	bool configured;
	volatile bool active;
};

static const struct device *const dma_dev =
	DEVICE_DT_GET(DT_NODELABEL(edma0));
static struct edma_channel channels[SYN_DMA_CHANNELS];
static bool dma_ready;

static struct edma_channel *get_channel(int channel)
{
	if (!dma_ready || channel < 0 || channel >= SYN_DMA_CHANNELS) {
		return NULL;
	}
	return &channels[channel];
}

static void edma_callback(const struct device *dev, void *user_data,
			  uint32_t hw_channel, int status)
{
	ARG_UNUSED(dev);

	struct edma_channel *ch = user_data;
	int idx = (int)(ch - channels);

	if (status < 0 || !ch->cfg.circular) {
		ch->active = false;
	}

	if (ch->cb != NULL) {
		ch->cb(idx, status, ch->user_data);
	}

	if (ch->active && ch->cfg.circular) {
		/* Software re-arm for the next iteration */
		if (dma_reload(dma_dev, hw_channel,
			       (uint32_t)(uintptr_t)ch->cfg.src_addr,
			       (uint32_t)(uintptr_t)ch->cfg.dst_addr,
			       ch->cfg.transfer_size) != 0 ||
		    dma_start(dma_dev, hw_channel) != 0) {
			LOG_ERR("Circular re-arm failed on channel %d", idx);
			ch->active = false;
		}
	}
}

int syn_hal_dma_init(void)
{
	if (dma_ready) {
		return -EALREADY;
	}
	if (!device_is_ready(dma_dev)) {
		LOG_ERR("eDMA device not ready");
		return -ENODEV;
	}

	dma_ready = true;
	LOG_INF("eDMA ready: %d channels (eDMA %d..%d)", SYN_DMA_CHANNELS,
		EDMA_CH_BASE, EDMA_CH_BASE + SYN_DMA_CHANNELS - 1);
	return 0;
}

int syn_hal_dma_configure(int channel, const syn_dma_config_t *config)
{
	struct edma_channel *ch = get_channel(channel);

	if (ch == NULL || config == NULL || config->src_addr == NULL ||
	    config->dst_addr == NULL || config->transfer_size == 0U) {
		return -EINVAL;
	}
	if (config->src_periph != SYN_DMA_PERIPH_MEMORY ||
	    config->dst_periph != SYN_DMA_PERIPH_MEMORY) {
		/* Peripheral endpoints arrive with the camera path */
		return -ENOTSUP;
	}
	if (ch->active) {
		return -EBUSY;
	}

	ch->cfg = *config;
	ch->configured = true;
	return 0;
}

int syn_hal_dma_start(int channel, syn_dma_cb_t callback, void *user_data)
{
	struct edma_channel *ch = get_channel(channel);

	if (ch == NULL || !ch->configured) {
		return -EINVAL;
	}
	if (ch->active) {
		return -EBUSY;
	}

	ch->cb = callback;
	ch->user_data = user_data;

	/* Word transfers when everything is 4-aligned, else bytes */
	uint32_t width = 4;

	if (((uintptr_t)ch->cfg.src_addr & 3U) != 0U ||
	    ((uintptr_t)ch->cfg.dst_addr & 3U) != 0U ||
	    (ch->cfg.transfer_size & 3U) != 0U) {
		width = 1;
	}

	struct dma_block_config block = {
		.source_address = (uint32_t)(uintptr_t)ch->cfg.src_addr,
		.dest_address = (uint32_t)(uintptr_t)ch->cfg.dst_addr,
		.block_size = ch->cfg.transfer_size,
	};
	struct dma_config cfg = {
		.channel_direction = MEMORY_TO_MEMORY,
		.source_data_size = width,
		.dest_data_size = width,
		.source_burst_length = width,
		.dest_burst_length = width,
		.block_count = 1,
		.head_block = &block,
		.dma_callback = edma_callback,
		.user_data = ch,
		.complete_callback_en = 1,
	};

	uint32_t hw_ch = (uint32_t)(EDMA_CH_BASE + channel);
	int ret = dma_config(dma_dev, hw_ch, &cfg);

	if (ret != 0) {
		LOG_ERR("dma_config failed on channel %d: %d", channel, ret);
		return ret;
	}

	ch->active = true;
	ret = dma_start(dma_dev, hw_ch);
	if (ret != 0) {
		ch->active = false;
		LOG_ERR("dma_start failed on channel %d: %d", channel, ret);
	}
	return ret;
}

int syn_hal_dma_stop(int channel)
{
	struct edma_channel *ch = get_channel(channel);

	if (ch == NULL) {
		return -EINVAL;
	}

	ch->active = false;
	return dma_stop(dma_dev, (uint32_t)(EDMA_CH_BASE + channel));
}

int syn_hal_dma_get_remaining(int channel, size_t *remaining)
{
	struct edma_channel *ch = get_channel(channel);

	if (ch == NULL || remaining == NULL) {
		return -EINVAL;
	}

	struct dma_status st;
	int ret = dma_get_status(dma_dev,
				 (uint32_t)(EDMA_CH_BASE + channel), &st);

	if (ret != 0) {
		return ret;
	}
	*remaining = st.pending_length;
	return 0;
}

#else /* !CONFIG_DMA || edma0 disabled */

int syn_hal_dma_init(void)
{
	LOG_WRN("DMA HAL unavailable: CONFIG_DMA not enabled");
	return -ENOSYS;
}

int syn_hal_dma_configure(int channel, const syn_dma_config_t *config)
{
	ARG_UNUSED(channel);
	ARG_UNUSED(config);
	return -ENOSYS;
}

int syn_hal_dma_start(int channel, syn_dma_cb_t callback, void *user_data)
{
	ARG_UNUSED(channel);
	ARG_UNUSED(callback);
	ARG_UNUSED(user_data);
	return -ENOSYS;
}

int syn_hal_dma_stop(int channel)
{
	ARG_UNUSED(channel);
	return -ENOSYS;
}

int syn_hal_dma_get_remaining(int channel, size_t *remaining)
{
	ARG_UNUSED(channel);
	ARG_UNUSED(remaining);
	return -ENOSYS;
}

#endif /* CONFIG_DMA && edma0 */
