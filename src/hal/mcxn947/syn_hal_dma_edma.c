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
#include <fsl_edma.h>

/* BOARD FINDING (Phase 5): Zephyr 3.7's dma_mcux_edma driver never
 * issues the eDMA v4 software START for memory-to-memory transfers -
 * it only enables the hardware request, and with mux source 0 there
 * is no requestor, so the transfer sits forever and the completion
 * callback never fires. We size the minor loop to the WHOLE transfer
 * (burst = transfer_size, so one service request moves everything)
 * and trigger the START bit ourselves after dma_start().
 */
#define EDMA_BASE_PTR ((EDMA_Type *)DT_REG_ADDR(DT_NODELABEL(edma0)))

/* BOARD FINDING (Phase 5): CPU0 runs in the secure world and its
 * pointers carry the TrustZone secure alias (bit 28, e.g. SRAM at
 * 0x30000000). The eDMA issues non-secure transactions and bus-errors
 * on secure-alias addresses, so DMA-visible addresses must be the
 * plain aliases: strip bit 28.
 */
static uint32_t dma_addr(const void *p)
{
	return (uint32_t)(uintptr_t)p & ~BIT(28);
}

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

static void syn_edma_done_cb(const struct device *dev, void *user_data,
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
		/* Software re-arm for the next iteration (no dma_start:
		 * see the ERQ board finding in syn_hal_dma_start)
		 */
		if (dma_reload(dma_dev, hw_channel,
			       dma_addr(ch->cfg.src_addr),
			       dma_addr(ch->cfg.dst_addr),
			       ch->cfg.transfer_size) != 0) {
			LOG_ERR("Circular re-arm failed on channel %d", idx);
			ch->active = false;
		} else {
			EDMA_ClearChannelStatusFlags(EDMA_BASE_PTR,
						     hw_channel,
						     (uint32_t)kEDMA_DoneFlag);
			EDMA_TriggerChannelStart(EDMA_BASE_PTR, hw_channel);
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
		.source_address = dma_addr(ch->cfg.src_addr),
		.dest_address = dma_addr(ch->cfg.dst_addr),
		.block_size = ch->cfg.transfer_size,
	};
	struct dma_config cfg = {
		.channel_direction = MEMORY_TO_MEMORY,
		.source_data_size = width,
		.dest_data_size = width,
		/* Minor loop = whole transfer: one software START moves
		 * everything (see the board finding above).
		 */
		.source_burst_length = ch->cfg.transfer_size,
		.dest_burst_length = ch->cfg.transfer_size,
		.block_count = 1,
		.head_block = &block,
		.dma_callback = syn_edma_done_cb,
		.user_data = ch,
		.complete_callback_en = 1,
	};

	uint32_t hw_ch = (uint32_t)(EDMA_CH_BASE + channel);
	int ret = dma_config(dma_dev, hw_ch, &cfg);

	if (ret != 0) {
		LOG_ERR("dma_config failed on channel %d: %d", channel, ret);
		return ret;
	}

	/* BOARD FINDING (Phase 5): do NOT call dma_start() for
	 * memory-to-memory work. It enables the hardware request (ERQ)
	 * with channel mux source 0, and spurious triggers then race
	 * the software START (channel error flag, transfers dying
	 * mid-stream). Software-paced transfers need only: configure,
	 * clear the latched DONE flag (write-1-clear; START is ignored
	 * while it is set), set START. The completion interrupt is
	 * armed by the configure step.
	 */
	ch->active = true;
	EDMA_ClearChannelStatusFlags(EDMA_BASE_PTR, hw_ch,
				     (uint32_t)kEDMA_DoneFlag);
	EDMA_EnableChannelInterrupts(EDMA_BASE_PTR, hw_ch,
				     (uint32_t)kEDMA_MajorInterruptEnable);
	EDMA_TriggerChannelStart(EDMA_BASE_PTR, hw_ch);
	return 0;
}

/* Diagnostic register dump (printk: synchronous, survives log-buffer
 * pressure). Used by `syn dma regs` while the eDMA bring-up settles.
 */
void syn_hal_dma_dump(int channel)
{
	if (channel < 0 || channel >= SYN_DMA_CHANNELS) {
		return;
	}

	uint32_t hw_ch = (uint32_t)(EDMA_CH_BASE + channel);
	edma_core_channel_t *chan = EDMA_CHANNEL_BASE(EDMA_BASE_PTR, hw_ch);
	edma_core_tcd_t *tcd = EDMA_TCD_BASE(EDMA_BASE_PTR, hw_ch);

	printk("eDMA ch%u: CH_CSR=%08x CH_ES=%08x CH_INT=%08x\n",
	       (unsigned)hw_ch, chan->CH_CSR, chan->CH_ES, chan->CH_INT);
	printk("  TCD: SADDR=%08x DADDR=%08x NBYTES=%08x CSR=%04x "
	       "CITER=%04x BITER=%04x\n", tcd->SADDR, tcd->DADDR,
	       tcd->NBYTES, tcd->CSR, tcd->CITER, tcd->BITER);
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

void syn_hal_dma_dump(int channel)
{
	ARG_UNUSED(channel);
}

#endif /* CONFIG_DMA && edma0 */
