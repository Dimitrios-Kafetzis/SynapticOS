/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file syn_hal_dma_stub.c
 * @brief SynapticOS - DMA stub (software-emulated transfers)
 *
 * Emulates asynchronous DMA on targets without an engine (QEMU):
 * start() queues a work item that performs the copy and fires the
 * completion callback from the system workqueue, preserving the
 * caller-visible async contract. Circular transfers re-arm
 * themselves until stop(). Since the "DMA" is CPU work here, the
 * overlap gain of double-buffered ingest is functional only - real
 * throughput numbers come from the eDMA backend on the board.
 */

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <string.h>

LOG_MODULE_REGISTER(syn_hal_dma_stub, CONFIG_SYNAPTIC_LOG_LEVEL);

#include <synaptic/syn_hal_dma.h>

#define STUB_DMA_CHANNELS 4

struct stub_channel {
	syn_dma_config_t cfg;
	syn_dma_cb_t cb;
	void *user_data;
	struct k_work work;
	bool configured;
	volatile bool active;
	volatile size_t remaining;
};

static struct stub_channel channels[STUB_DMA_CHANNELS];
static bool dma_ready;

static void stub_transfer_work(struct k_work *work)
{
	struct stub_channel *ch = CONTAINER_OF(work, struct stub_channel,
					       work);

	if (!ch->active) {
		return;
	}

	memcpy(ch->cfg.dst_addr, ch->cfg.src_addr, ch->cfg.transfer_size);
	ch->remaining = 0;

	if (!ch->cfg.circular) {
		ch->active = false;
	}

	if (ch->cb != NULL) {
		ch->cb((int)(ch - channels), 0, ch->user_data);
	}

	if (ch->active) {
		/* Circular: immediately re-arm the next iteration */
		ch->remaining = ch->cfg.transfer_size;
		k_work_submit(&ch->work);
	}
}

int syn_hal_dma_init(void)
{
	if (dma_ready) {
		return -EALREADY;
	}
	for (int i = 0; i < STUB_DMA_CHANNELS; i++) {
		memset(&channels[i], 0, sizeof(channels[i]));
		k_work_init(&channels[i].work, stub_transfer_work);
	}
	dma_ready = true;
	LOG_INF("DMA stub initialized (%d channels, software copies)",
		STUB_DMA_CHANNELS);
	return 0;
}

static struct stub_channel *get_channel(int channel)
{
	if (!dma_ready || channel < 0 || channel >= STUB_DMA_CHANNELS) {
		return NULL;
	}
	return &channels[channel];
}

int syn_hal_dma_configure(int channel, const syn_dma_config_t *config)
{
	struct stub_channel *ch = get_channel(channel);

	if (ch == NULL || config == NULL || config->src_addr == NULL ||
	    config->dst_addr == NULL || config->transfer_size == 0U) {
		return -EINVAL;
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
	struct stub_channel *ch = get_channel(channel);

	if (ch == NULL || !ch->configured) {
		return -EINVAL;
	}
	if (ch->active) {
		return -EBUSY;
	}

	ch->cb = callback;
	ch->user_data = user_data;
	ch->remaining = ch->cfg.transfer_size;
	ch->active = true;
	k_work_submit(&ch->work);
	return 0;
}

int syn_hal_dma_stop(int channel)
{
	struct stub_channel *ch = get_channel(channel);

	if (ch == NULL) {
		return -EINVAL;
	}
	ch->active = false;
	return 0;
}

int syn_hal_dma_get_remaining(int channel, size_t *remaining)
{
	struct stub_channel *ch = get_channel(channel);

	if (ch == NULL || remaining == NULL) {
		return -EINVAL;
	}
	*remaining = ch->remaining;
	return 0;
}
