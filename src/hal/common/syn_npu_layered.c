/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file syn_npu_layered.c
 * @brief SynapticOS - Layer-granular synthetic model execution with
 *        memory-optimal activation placement
 *
 * See syn_npu_layered.h for the formats, the planner contract and
 * the concurrency rules.
 */

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <string.h>
#include <errno.h>

LOG_MODULE_REGISTER(syn_npu_layered, CONFIG_SYNAPTIC_LOG_LEVEL);

#include "syn_npu_layered.h"

#define ACT_MAX  CONFIG_SYNAPTIC_LAYER_ACT_MAX
#define PLAN_MAX CONFIG_SYNAPTIC_LAYER_PLAN_MAX

/* Buffer index space: 0 = model input, 1 + i = output of layer i. */
#define NBUF (SYN_LAYERED_MAX_LAYERS + 1U)

struct model_plan {
	const uint8_t *model;
	size_t model_size;
	uint16_t layer_count;
	uint16_t input_size;
	/* Per layer, resolved to buffer indices (src_b < 0 = none) */
	uint16_t out_size[SYN_LAYERED_MAX_LAYERS];
	uint16_t work[SYN_LAYERED_MAX_LAYERS];
	int16_t  src_a[SYN_LAYERED_MAX_LAYERS];
	int16_t  src_b[SYN_LAYERED_MAX_LAYERS];
	/* Per buffer: size, planned offset, lifetime interval in layer
	 * time (buffer b is written at def[b], last read at last[b]).
	 */
	uint16_t buf_size[NBUF];
	uint16_t buf_off[NBUF];
	int16_t  def[NBUF];
	int16_t  last[NBUF];
	uint32_t planned_peak;
	uint32_t naive_peak;
};

struct layered_ctx {
	const uint8_t *model;   /* NULL = slot free */
	size_t model_size;
	uint16_t next_layer;
	uint16_t live_bytes;    /* Valid bytes of area[] (planned peak) */
	uint8_t area[PLAN_MAX];
};

static struct {
	const uint8_t *model;   /* Resident layered model, NULL if none */
	size_t size;
} resident;

static struct model_plan plan;
static uint8_t plan_area[PLAN_MAX];
static uint16_t live_next_layer;
static bool live_active;
static uint32_t last_plan_us;
static struct layered_ctx slots[SYN_LAYERED_CTX_SLOTS];

static uint16_t rd16(const uint8_t *p)
{
	return (uint16_t)(p[0] | (p[1] << 8));
}

static int16_t rds16(const uint8_t *p)
{
	return (int16_t)rd16(p);
}

static uint32_t rd32(const uint8_t *p)
{
	return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
	       ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

/**
 * Resolve a raw source reference of layer @p i to a buffer index.
 * @return Buffer index (>= 0), -1 for "none", -EINVAL on bad refs.
 */
static int resolve_src(int16_t raw, uint16_t i, bool allow_none)
{
	if (raw == SYN_LAYERED_SRC_NONE) {
		return allow_none ? -1 : -EINVAL;
	}
	if (raw == SYN_LAYERED_SRC_PREV) {
		return i; /* buffer i is layer i-1's output, or the input */
	}
	if (raw == SYN_LAYERED_SRC_INPUT) {
		return 0;
	}
	if (raw >= 0 && (uint16_t)raw < i) {
		return raw + 1;
	}
	return -EINVAL;
}

/**
 * Parse and validate a blob into @p p (no placement yet).
 * @return 0 on success, -ENOTSUP when not a layered format, other
 *         negatives on a malformed layered blob.
 */
static int parse_model(const uint8_t *data, size_t size,
		       struct model_plan *p)
{
	if (data == NULL || size < SYN_LAYERED_HDR_SIZE) {
		return -ENOTSUP;
	}

	uint32_t magic = rd32(data);
	bool dag = (magic == SYN_LAYERED_DAG_MAGIC);

	if (!dag && magic != SYN_LAYERED_MAGIC) {
		return -ENOTSUP;
	}

	uint16_t n = rd16(data + 4);
	uint16_t in = rd16(data + 6);
	size_t desc = dag ? SYN_LAYERED_DDESC_SIZE : SYN_LAYERED_DESC_SIZE;

	if (n == 0U || n > SYN_LAYERED_MAX_LAYERS || in == 0U ||
	    in > ACT_MAX) {
		LOG_WRN("Layered blob bad geometry (n=%u in=%u)", n, in);
		return -EINVAL;
	}
	if (size < SYN_LAYERED_HDR_SIZE + (size_t)n * desc) {
		LOG_WRN("Layered blob truncated (%u layers, %u bytes)",
			n, (unsigned)size);
		return -EINVAL;
	}

	p->model = data;
	p->model_size = size;
	p->layer_count = n;
	p->input_size = in;

	for (uint16_t i = 0; i < n; i++) {
		const uint8_t *d = data + SYN_LAYERED_HDR_SIZE +
				   (size_t)i * desc;

		p->out_size[i] = rd16(d);
		p->work[i] = rd16(d + 2);
		if (p->out_size[i] == 0U || p->out_size[i] > ACT_MAX) {
			LOG_WRN("Layer %u output %u exceeds ACT_MAX %u",
				i, p->out_size[i], ACT_MAX);
			return -EINVAL;
		}

		int a = resolve_src(dag ? rds16(d + 4) :
				    (int16_t)SYN_LAYERED_SRC_PREV, i, false);
		int b = resolve_src(dag ? rds16(d + 6) :
				    (int16_t)SYN_LAYERED_SRC_NONE, i, true);

		if (a < 0 || b == -EINVAL) {
			LOG_WRN("Layer %u has a bad source reference", i);
			return -EINVAL;
		}
		p->src_a[i] = (int16_t)a;
		p->src_b[i] = (int16_t)b;
	}
	return 0;
}

/**
 * Greedy first-fit activation placement from buffer lifetimes.
 * Deterministic: identical models always produce identical plans.
 * @return 0 on success, -ENOMEM when the peak exceeds PLAN_MAX.
 */
static int place_buffers(struct model_plan *p)
{
	uint16_t n = p->layer_count;
	uint16_t nbuf = n + 1U;

	/* Buffer sizes and lifetime intervals in layer time: buffer b
	 * is written at def[b] (input: before layer 0) and must stay
	 * until its last consumer; the final output is read after the
	 * last layer (layer time n).
	 */
	p->buf_size[0] = p->input_size;
	p->def[0] = 0;
	p->last[0] = 0;
	for (uint16_t i = 0; i < n; i++) {
		p->buf_size[i + 1U] = p->out_size[i];
		p->def[i + 1U] = (int16_t)i;
		p->last[i + 1U] = (int16_t)i;
	}
	for (uint16_t i = 0; i < n; i++) {
		if (p->last[p->src_a[i]] < (int16_t)i) {
			p->last[p->src_a[i]] = (int16_t)i;
		}
		if (p->src_b[i] >= 0 &&
		    p->last[p->src_b[i]] < (int16_t)i) {
			p->last[p->src_b[i]] = (int16_t)i;
		}
	}
	p->last[n] = (int16_t)n; /* final output read by output() */

	uint32_t naive = 0;

	for (uint16_t b = 0; b < nbuf; b++) {
		naive += p->buf_size[b];
	}
	p->naive_peak = naive;

	/* First-fit in definition order: bump past any already-placed
	 * buffer that overlaps in both lifetime and address space.
	 */
	uint32_t peak = 0;

	for (uint16_t b = 0; b < nbuf; b++) {
		uint32_t off = 0;
		bool moved = true;

		while (moved) {
			moved = false;
			for (uint16_t o = 0; o < b; o++) {
				bool time_overlap =
					p->def[b] <= p->last[o] &&
					p->def[o] <= p->last[b];
				bool addr_overlap =
					off < (uint32_t)p->buf_off[o] +
					      p->buf_size[o] &&
					(uint32_t)p->buf_off[o] <
					      off + p->buf_size[b];

				if (time_overlap && addr_overlap) {
					off = (uint32_t)p->buf_off[o] +
					      p->buf_size[o];
					moved = true;
				}
			}
		}

		if (off + p->buf_size[b] > PLAN_MAX) {
			LOG_WRN("Plan overflow: buffer %u needs %u at %u "
				"(PLAN_MAX %u)", b, p->buf_size[b],
				(unsigned)off, PLAN_MAX);
			return -ENOMEM;
		}
		p->buf_off[b] = (uint16_t)off;
		if (off + p->buf_size[b] > peak) {
			peak = off + p->buf_size[b];
		}
	}
	p->planned_peak = peak;
	return 0;
}

void syn_npu_layered_on_load(const uint8_t *data, size_t size)
{
	struct model_plan probe;
	int ret = parse_model(data, size, &probe);

	if (ret == 0) {
		resident.model = data;
		resident.size = size;
		LOG_INF("Layered model resident: %u layers, input %u bytes "
			"(synthetic, stub-executed)", probe.layer_count,
			probe.input_size);
	} else {
		if (ret != -ENOTSUP) {
			LOG_WRN("Layered blob rejected: %d", ret);
		}
		resident.model = NULL;
		resident.size = 0;
	}
}

/** Parse + place the resident model into the live plan, timed. */
static int plan_live(const uint8_t *model, size_t size)
{
	uint32_t t0 = k_cycle_get_32();
	int ret = parse_model(model, size, &plan);

	if (ret != 0) {
		return ret;
	}
	ret = place_buffers(&plan);
	if (ret != 0) {
		return ret;
	}
	last_plan_us = k_cyc_to_us_ceil32(k_cycle_get_32() - t0);
	return 0;
}

int syn_npu_layered_begin(const void *input, size_t in_size)
{
	if (resident.model == NULL) {
		return 0;
	}

	int ret = plan_live(resident.model, resident.size);

	if (ret != 0) {
		return (ret == -ENOTSUP) ? 0 : ret;
	}
	if (input == NULL || in_size != plan.input_size) {
		return -EINVAL;
	}

	memcpy(plan_area + plan.buf_off[0], input, in_size);
	live_next_layer = 0;
	live_active = true;
	return plan.layer_count;
}

int syn_npu_layered_step(void)
{
	if (!live_active || live_next_layer >= plan.layer_count) {
		return -EPERM;
	}

	uint16_t i = live_next_layer;
	uint16_t out = plan.out_size[i];
	const uint8_t *a = plan_area + plan.buf_off[plan.src_a[i]];
	uint16_t a_size = plan.buf_size[plan.src_a[i]];
	const uint8_t *b = NULL;
	uint16_t b_size = 0;
	uint8_t *o = plan_area + plan.buf_off[i + 1U];

	if (plan.src_b[i] >= 0) {
		b = plan_area + plan.buf_off[plan.src_b[i]];
		b_size = plan.buf_size[plan.src_b[i]];
	}

	/* Order-sensitive chained transform: every output byte mixes
	 * two positions of the full primary source (plus the skip
	 * source when present) with the layer index. The planner
	 * guarantees the output buffer never aliases a live source.
	 */
	for (uint16_t j = 0; j < out; j++) {
		uint32_t acc = (uint32_t)a[j % a_size] * 31U +
			       a[(j * 7U + i) % a_size] +
			       (uint32_t)i * 13U + (uint32_t)j * 3U;

		if (b != NULL) {
			acc += (uint32_t)b[j % b_size] * 17U;
		}
		o[j] = (uint8_t)acc;
	}

	/* Emulated compute time (volatile loop: QEMU-safe) */
	for (volatile uint32_t w = 0; w < (uint32_t)plan.work[i] * 100U;
	     w++) {
	}

	live_next_layer++;
	return plan.layer_count - live_next_layer;
}

int syn_npu_layered_remaining(void)
{
	if (!live_active) {
		return -EPERM;
	}
	return plan.layer_count - live_next_layer;
}

int syn_npu_layered_output(void *out, size_t *size)
{
	if (!live_active) {
		return -EPERM;
	}
	if (live_next_layer < plan.layer_count) {
		return -EBUSY;
	}

	uint16_t final = plan.layer_count; /* buffer index */
	uint16_t final_size = plan.buf_size[final];

	if (out == NULL || size == NULL || *size < final_size) {
		return -ENOMEM;
	}

	memcpy(out, plan_area + plan.buf_off[final], final_size);
	*size = final_size;
	live_active = false;
	return 0;
}

void syn_npu_layered_plan_info(uint32_t *planned_peak,
			       uint32_t *naive_peak, uint32_t *plan_us)
{
	if (planned_peak != NULL) {
		*planned_peak = plan.planned_peak;
	}
	if (naive_peak != NULL) {
		*naive_peak = plan.naive_peak;
	}
	if (plan_us != NULL) {
		*plan_us = last_plan_us;
	}
}

int syn_npu_layered_save(void)
{
	if (!live_active) {
		return -EPERM;
	}
	for (int s = 0; s < (int)SYN_LAYERED_CTX_SLOTS; s++) {
		if (slots[s].model == NULL) {
			slots[s].model = plan.model;
			slots[s].model_size = plan.model_size;
			slots[s].next_layer = live_next_layer;
			slots[s].live_bytes = (uint16_t)plan.planned_peak;
			memcpy(slots[s].area, plan_area,
			       plan.planned_peak);
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

	/* Replanning is deterministic, so the saved offsets are
	 * reproduced exactly.
	 */
	int ret = plan_live(slots[slot].model, slots[slot].model_size);

	if (ret != 0) {
		return -EINVAL;
	}

	memcpy(plan_area, slots[slot].area, slots[slot].live_bytes);
	live_next_layer = slots[slot].next_layer;
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

static void wr16(uint8_t *p, uint16_t v)
{
	p[0] = (uint8_t)(v & 0xFFU);
	p[1] = (uint8_t)(v >> 8);
}

static void write_header(uint8_t *buf, uint32_t magic,
			 uint16_t layer_count, uint16_t input_size)
{
	buf[0] = (uint8_t)(magic & 0xFFU);
	buf[1] = (uint8_t)((magic >> 8) & 0xFFU);
	buf[2] = (uint8_t)((magic >> 16) & 0xFFU);
	buf[3] = (uint8_t)((magic >> 24) & 0xFFU);
	wr16(buf + 4, layer_count);
	wr16(buf + 6, input_size);
}

int syn_npu_layered_make_model(uint8_t *buf, size_t buf_size,
			       uint16_t input_size, uint16_t layer_count,
			       const uint16_t *out_sizes,
			       const uint16_t *work)
{
	if (buf == NULL || out_sizes == NULL || layer_count == 0U ||
	    layer_count > SYN_LAYERED_MAX_LAYERS || input_size == 0U ||
	    input_size > ACT_MAX) {
		return -EINVAL;
	}

	size_t total = SYN_LAYERED_HDR_SIZE +
		       (size_t)layer_count * SYN_LAYERED_DESC_SIZE;

	if (buf_size < total) {
		return -ENOMEM;
	}

	write_header(buf, SYN_LAYERED_MAGIC, layer_count, input_size);
	for (uint16_t i = 0; i < layer_count; i++) {
		uint8_t *d = buf + SYN_LAYERED_HDR_SIZE +
			     (size_t)i * SYN_LAYERED_DESC_SIZE;

		if (out_sizes[i] == 0U || out_sizes[i] > ACT_MAX) {
			return -EINVAL;
		}
		wr16(d, out_sizes[i]);
		wr16(d + 2, (work != NULL) ? work[i] : 0U);
	}
	return (int)total;
}

int syn_npu_layered_make_dag(uint8_t *buf, size_t buf_size,
			     uint16_t input_size, uint16_t layer_count,
			     const syn_layered_dag_layer_t *layers)
{
	if (buf == NULL || layers == NULL || layer_count == 0U ||
	    layer_count > SYN_LAYERED_MAX_LAYERS || input_size == 0U ||
	    input_size > ACT_MAX) {
		return -EINVAL;
	}

	size_t total = SYN_LAYERED_HDR_SIZE +
		       (size_t)layer_count * SYN_LAYERED_DDESC_SIZE;

	if (buf_size < total) {
		return -ENOMEM;
	}

	write_header(buf, SYN_LAYERED_DAG_MAGIC, layer_count, input_size);
	for (uint16_t i = 0; i < layer_count; i++) {
		uint8_t *d = buf + SYN_LAYERED_HDR_SIZE +
			     (size_t)i * SYN_LAYERED_DDESC_SIZE;

		if (layers[i].out_size == 0U ||
		    layers[i].out_size > ACT_MAX ||
		    resolve_src(layers[i].src_a, i, false) < 0 ||
		    resolve_src(layers[i].src_b, i, true) == -EINVAL) {
			return -EINVAL;
		}
		wr16(d, layers[i].out_size);
		wr16(d + 2, layers[i].work);
		wr16(d + 4, (uint16_t)layers[i].src_a);
		wr16(d + 6, (uint16_t)layers[i].src_b);
	}
	return (int)total;
}
