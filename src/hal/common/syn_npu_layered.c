/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file syn_npu_layered.c
 * @brief SynapticOS - Layer-granular synthetic model execution
 *
 * See syn_npu_layered.h for the format and the concurrency contract.
 */

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <string.h>
#include <errno.h>

LOG_MODULE_REGISTER(syn_npu_layered, CONFIG_SYNAPTIC_LOG_LEVEL);

#include "syn_npu_layered.h"

#define ACT_MAX CONFIG_SYNAPTIC_LAYER_ACT_MAX

struct layered_ctx {
	const uint8_t *model;   /* Blob incl. header; NULL = slot free */
	uint16_t layer_count;
	uint16_t next_layer;    /* == layer_count when complete */
	uint16_t act_size;
	uint8_t  act[ACT_MAX];
};

static struct {
	const uint8_t *model;   /* Resident layered model, NULL if none */
	uint16_t layer_count;
	uint16_t input_size;
} resident;

static struct layered_ctx live;
static bool live_active;
static struct layered_ctx slots[SYN_LAYERED_CTX_SLOTS];
static uint8_t tmp[ACT_MAX];

static const uint8_t *layer_desc(const uint8_t *model, uint16_t idx)
{
	return model + SYN_LAYERED_HDR_SIZE + (size_t)idx *
	       SYN_LAYERED_DESC_SIZE;
}

static uint16_t rd16(const uint8_t *p)
{
	return (uint16_t)(p[0] | (p[1] << 8));
}

/** Validate the blob; return layer count or 0 when not layered. */
static uint16_t probe(const uint8_t *data, size_t size)
{
	if (data == NULL || size < SYN_LAYERED_HDR_SIZE) {
		return 0;
	}

	uint32_t magic = (uint32_t)data[0] | ((uint32_t)data[1] << 8) |
			 ((uint32_t)data[2] << 16) |
			 ((uint32_t)data[3] << 24);

	if (magic != SYN_LAYERED_MAGIC) {
		return 0;
	}

	uint16_t n = rd16(data + 4);
	uint16_t in = rd16(data + 6);

	if (n == 0U || in == 0U || in > ACT_MAX) {
		LOG_WRN("Layered magic with bad geometry (n=%u in=%u)", n, in);
		return 0;
	}
	if (size < SYN_LAYERED_HDR_SIZE + (size_t)n * SYN_LAYERED_DESC_SIZE) {
		LOG_WRN("Layered blob truncated (%u layers, %u bytes)",
			n, (unsigned)size);
		return 0;
	}
	for (uint16_t i = 0; i < n; i++) {
		uint16_t out = rd16(layer_desc(data, i));

		if (out == 0U || out > ACT_MAX) {
			LOG_WRN("Layer %u output %u exceeds ACT_MAX %u",
				i, out, ACT_MAX);
			return 0;
		}
	}
	return n;
}

void syn_npu_layered_on_load(const uint8_t *data, size_t size)
{
	uint16_t n = probe(data, size);

	if (n > 0U) {
		resident.model = data;
		resident.layer_count = n;
		resident.input_size = rd16(data + 6);
		LOG_INF("Layered model resident: %u layers, input %u bytes "
			"(synthetic, stub-executed)", n, resident.input_size);
	} else {
		resident.model = NULL;
		resident.layer_count = 0;
		resident.input_size = 0;
	}
}

int syn_npu_layered_begin(const void *input, size_t in_size)
{
	if (resident.model == NULL) {
		return 0;
	}
	if (input == NULL || in_size != resident.input_size) {
		return -EINVAL;
	}

	live.model = resident.model;
	live.layer_count = resident.layer_count;
	live.next_layer = 0;
	live.act_size = resident.input_size;
	memcpy(live.act, input, in_size);
	live_active = true;
	return live.layer_count;
}

int syn_npu_layered_step(void)
{
	if (!live_active || live.next_layer >= live.layer_count) {
		return -EPERM;
	}

	const uint8_t *d = layer_desc(live.model, live.next_layer);
	uint16_t out = rd16(d);
	uint16_t work = rd16(d + 2);
	uint16_t i = live.next_layer;
	uint16_t in = live.act_size;

	/* Order-sensitive chained transform: every output byte mixes two
	 * positions of the full previous activation with the layer index.
	 */
	for (uint16_t j = 0; j < out; j++) {
		uint32_t acc = (uint32_t)live.act[j % in] * 31U +
			       live.act[(j * 7U + i) % in] +
			       (uint32_t)i * 13U + (uint32_t)j * 3U;

		tmp[j] = (uint8_t)acc;
	}
	memcpy(live.act, tmp, out);
	live.act_size = out;

	/* Emulated compute time (volatile loop: QEMU-safe) */
	for (volatile uint32_t w = 0; w < (uint32_t)work * 100U; w++) {
	}

	live.next_layer++;
	return live.layer_count - live.next_layer;
}

int syn_npu_layered_remaining(void)
{
	if (!live_active) {
		return -EPERM;
	}
	return live.layer_count - live.next_layer;
}

int syn_npu_layered_output(void *out, size_t *size)
{
	if (!live_active) {
		return -EPERM;
	}
	if (live.next_layer < live.layer_count) {
		return -EBUSY;
	}
	if (out == NULL || size == NULL || *size < live.act_size) {
		return -ENOMEM;
	}

	memcpy(out, live.act, live.act_size);
	*size = live.act_size;
	live_active = false;
	return 0;
}

int syn_npu_layered_save(void)
{
	if (!live_active) {
		return -EPERM;
	}
	for (int s = 0; s < (int)SYN_LAYERED_CTX_SLOTS; s++) {
		if (slots[s].model == NULL) {
			slots[s] = live;
			live_active = false;
			return s;
		}
	}
	return -ENOSPC;
}

int syn_npu_layered_restore(int slot)
{
	if (slot < 0 || slot >= (int)SYN_LAYERED_CTX_SLOTS ||
	    slots[slot].model == NULL) {
		return -EINVAL;
	}

	live = slots[slot];
	live_active = true;
	slots[slot].model = NULL;
	return 0;
}

void syn_npu_layered_slot_free(int slot)
{
	if (slot >= 0 && slot < (int)SYN_LAYERED_CTX_SLOTS) {
		slots[slot].model = NULL;
	}
}

int syn_npu_layered_make_model(uint8_t *buf, size_t buf_size,
			       uint16_t input_size, uint16_t layer_count,
			       const uint16_t *out_sizes,
			       const uint16_t *work)
{
	if (buf == NULL || out_sizes == NULL || layer_count == 0U ||
	    input_size == 0U || input_size > ACT_MAX) {
		return -EINVAL;
	}

	size_t total = SYN_LAYERED_HDR_SIZE +
		       (size_t)layer_count * SYN_LAYERED_DESC_SIZE;

	if (buf_size < total) {
		return -ENOMEM;
	}

	buf[0] = (uint8_t)(SYN_LAYERED_MAGIC & 0xFFU);
	buf[1] = (uint8_t)((SYN_LAYERED_MAGIC >> 8) & 0xFFU);
	buf[2] = (uint8_t)((SYN_LAYERED_MAGIC >> 16) & 0xFFU);
	buf[3] = (uint8_t)((SYN_LAYERED_MAGIC >> 24) & 0xFFU);
	buf[4] = (uint8_t)(layer_count & 0xFFU);
	buf[5] = (uint8_t)(layer_count >> 8);
	buf[6] = (uint8_t)(input_size & 0xFFU);
	buf[7] = (uint8_t)(input_size >> 8);

	for (uint16_t i = 0; i < layer_count; i++) {
		uint16_t out = out_sizes[i];
		uint16_t w = (work != NULL) ? work[i] : 0U;
		uint8_t *d = buf + SYN_LAYERED_HDR_SIZE +
			     (size_t)i * SYN_LAYERED_DESC_SIZE;

		if (out == 0U || out > ACT_MAX) {
			return -EINVAL;
		}
		d[0] = (uint8_t)(out & 0xFFU);
		d[1] = (uint8_t)(out >> 8);
		d[2] = (uint8_t)(w & 0xFFU);
		d[3] = (uint8_t)(w >> 8);
	}
	return (int)total;
}
