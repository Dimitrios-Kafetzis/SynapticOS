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

#ifdef CONFIG_SYNAPTIC_DMA_ARENA_PROBE
/* ------------------------------------------------------------------
 * Arena eDMA reachability experiments (Phase 6.3 bench diagnostics)
 *
 * Phase 5 board finding: eDMA transfers that touch the tensor arena
 * time out SILENTLY (no fault, no error IRQ on this path), while the
 * same transfers between static buffers complete. Suspected cause:
 * bus security attributes (the eDMA issues non-secure transactions;
 * the arena RAM granules may demand secure access).
 *
 * `syn dma arena` runs a canary+bounded-poll experiment matrix over
 * the available knobs:
 *   - CH_SBR SEC/PAL: per-channel security/privilege attribute of
 *     the eDMA bus transactions.
 *   - AHBSC MASTER_SEC_LEVEL: chip-level security level of the
 *     EDMA0 master (paired anti-pol register must match).
 *   - AHBSC RAMx_MEM_RULE: per-4KB-granule MPC access rules.
 * Every attempt is verified by polling the destination canary bytes
 * (never a DMA flag: the completion path is exactly what fails) with
 * a hard timeout, and the AHBSC security-violation capture registers
 * are dumped after every attempt, so no outcome is silent.
 *
 * Output is printk (synchronous) so a wedged transfer cannot take
 * the evidence down with it.
 */

#include <string.h>
#include <synaptic/syn_mem.h>

#define PROBE_SIZE       4096U
#define PROBE_TIMEOUT_US 200000U
#define PROBE_CH         3                       /* SynapticOS channel */
#define PROBE_HW_CH      (EDMA_CH_BASE + PROBE_CH)

static uint8_t probe_src[PROBE_SIZE] __aligned(4);
static uint8_t probe_dst[PROBE_SIZE] __aligned(4);

/* Each AHBSC RAM rule register covers 32 KB: eight 4 KB granules,
 * one 2-bit rule per granule at 4-bit stride. Map from the MCXN947
 * RAM block layout (RAMA..RAMH contiguous from 0x20000000).
 */
struct ram_rule_reg {
	uint32_t base;                  /* plain-alias start, 32 KB span */
	__IO uint32_t *reg;
	const char *name;
};

static const struct ram_rule_reg ram_rules[] = {
	{ 0x20000000U, &AHBSC->RAMA_MEM_RULE,    "RAMA"   },
	{ 0x20008000U, &AHBSC->RAMB_MEM_RULE,    "RAMB"   },
	{ 0x20010000U, &AHBSC->RAMC_MEM_RULE[0], "RAMC0"  },
	{ 0x20018000U, &AHBSC->RAMC_MEM_RULE[1], "RAMC1"  },
	{ 0x20020000U, &AHBSC->RAMD_MEM_RULE[0], "RAMD0"  },
	{ 0x20028000U, &AHBSC->RAMD_MEM_RULE[1], "RAMD1"  },
	{ 0x20030000U, &AHBSC->RAME_MEM_RULE[0], "RAME0"  },
	{ 0x20038000U, &AHBSC->RAME_MEM_RULE[1], "RAME1"  },
	{ 0x20040000U, &AHBSC->RAMF_MEM_RULE[0], "RAMF0"  },
	{ 0x20048000U, &AHBSC->RAMF_MEM_RULE[1], "RAMF1"  },
	{ 0x20050000U, &AHBSC->RAMG_MEM_RULE[0], "RAMG0"  },
	{ 0x20058000U, &AHBSC->RAMG_MEM_RULE[1], "RAMG1"  },
	{ 0x20060000U, &AHBSC->RAMH_MEM_RULE,    "RAMH"   },
};

static const struct ram_rule_reg *rule_reg_for(uint32_t plain_addr)
{
	for (size_t i = 0; i < ARRAY_SIZE(ram_rules); i++) {
		if (plain_addr >= ram_rules[i].base &&
		    plain_addr < ram_rules[i].base + 0x8000U) {
			return &ram_rules[i];
		}
	}
	return NULL;
}

static void print_rule(const void *p, const char *what)
{
	uint32_t plain = dma_addr(p);
	const struct ram_rule_reg *r = rule_reg_for(plain);

	if (r == NULL) {
		printk("  %s %08x: outside RAMA..RAMH\n", what, plain);
		return;
	}

	uint32_t shift = ((plain - r->base) / 4096U) * 4U;
	uint32_t rule = (*r->reg >> shift) & 0x3U;

	printk("  %s %08x: %s=%08x granule rule %u\n",
	       what, plain, r->name, *r->reg, rule);
}

/* Returns the previous rule bits of the 4 KB granule holding addr */
static uint32_t set_rule(uint32_t plain_addr, uint32_t rule)
{
	const struct ram_rule_reg *r = rule_reg_for(plain_addr);
	uint32_t shift = ((plain_addr - r->base) / 4096U) * 4U;
	uint32_t old = (*r->reg >> shift) & 0x3U;

	*r->reg = (*r->reg & ~(0x3U << shift)) | (rule << shift);
	__DSB();
	return old;
}

static void dump_sec_vio(const char *when)
{
	uint32_t valid = AHBSC->SEC_VIO_INFO_VALID;

	if (valid == 0U) {
		printk("  sec-vio (%s): none\n", when);
		return;
	}
	for (int i = 0; i < 32; i++) {
		if ((valid & BIT(i)) != 0U) {
			printk("  sec-vio (%s) [%d]: addr %08x info %08x\n",
			       when, i, AHBSC->SEC_VIO_ADDR[i],
			       AHBSC->SEC_VIO_MISC_INFO[i]);
		}
	}
	/* W1C so the next attempt reports only fresh violations */
	AHBSC->SEC_VIO_INFO_VALID = valid;
}

/* BOARD FINDING (S5): dma_config() sets INTMAJOR in the TCD, so the
 * driver ISR runs on completion and calls the configured callback
 * UNCONDITIONALLY - a NULL dma_callback is a NULL function call from
 * interrupt context (PC=0 usage fault). Completion is still detected
 * by polling the canary; the callback just has to exist.
 */
static void probe_arena_cb(const struct device *dev, void *user_data,
			   uint32_t hw_channel, int status)
{
	ARG_UNUSED(dev);
	ARG_UNUSED(user_data);
	ARG_UNUSED(hw_channel);
	ARG_UNUSED(status);
}

/* One canary transfer: configure, tweak CH_SBR, software START, poll
 * the destination bytes. sbr_sec: -1 leave as configured, 0 force
 * non-secure, 1 force secure+privileged.
 */
static void probe_xfer(const char *label, const void *src, void *dst,
		       int sbr_sec)
{
	uint8_t *check = (uint8_t *)((uintptr_t)dst & ~BIT(28));
	static uint8_t round = 0x20;

	round += 7U;
	for (uint32_t i = 0; i < PROBE_SIZE; i++) {
		((uint8_t *)((uintptr_t)src & ~BIT(28)))[i] =
			(uint8_t)(round + i);
	}
	memset(check, 0, PROBE_SIZE);
	__DSB();

	struct dma_block_config block = {
		.source_address = (uint32_t)(uintptr_t)src,
		.dest_address = (uint32_t)(uintptr_t)dst,
		.block_size = PROBE_SIZE,
	};
	struct dma_config cfg = {
		.channel_direction = MEMORY_TO_MEMORY,
		.source_data_size = 4,
		.dest_data_size = 4,
		.source_burst_length = PROBE_SIZE,
		.dest_burst_length = PROBE_SIZE,
		.block_count = 1,
		.head_block = &block,
		.dma_callback = probe_arena_cb,
	};

	int ret = dma_config(dma_dev, PROBE_HW_CH, &cfg);

	if (ret != 0) {
		printk("%s: dma_config failed: %d\n", label, ret);
		return;
	}

	edma_core_channel_t *chan = EDMA_CHANNEL_BASE(EDMA_BASE_PTR,
						      PROBE_HW_CH);
	uint32_t sbr = chan->CH_SBR;

	if (sbr_sec == 0) {
		chan->CH_SBR = sbr & ~(DMA_CH_SBR_SEC_MASK |
				       DMA_CH_SBR_PAL_MASK);
	} else if (sbr_sec == 1) {
		chan->CH_SBR = sbr | DMA_CH_SBR_SEC_MASK |
			       DMA_CH_SBR_PAL_MASK;
	}
	printk("%s: src %08x dst %08x SBR %08x -> %08x\n", label,
	       (unsigned)(uintptr_t)src, (unsigned)(uintptr_t)dst,
	       sbr, chan->CH_SBR);

	EDMA_ClearChannelStatusFlags(EDMA_BASE_PTR, PROBE_HW_CH,
				     (uint32_t)kEDMA_DoneFlag);
	EDMA_TriggerChannelStart(EDMA_BASE_PTR, PROBE_HW_CH);

	uint32_t waited = 0;

	while (waited < PROBE_TIMEOUT_US) {
		if (*(volatile uint8_t *)(check + PROBE_SIZE - 1U) ==
		    (uint8_t)(round + PROBE_SIZE - 1U)) {
			break;
		}
		k_busy_wait(20);
		waited += 20;
	}

	bool full = true;

	for (uint32_t i = 0; i < PROBE_SIZE; i++) {
		if (check[i] != (uint8_t)(round + i)) {
			full = false;
			break;
		}
	}

	if (full) {
		printk("%s: OK, canary complete in <= %u us\n", label, waited);
	} else {
		printk("%s: TIMEOUT/PARTIAL after %u us (last byte %02x, "
		       "want %02x)\n", label, waited,
		       check[PROBE_SIZE - 1U],
		       (uint8_t)(round + PROBE_SIZE - 1U));
	}
	printk("  CH_CSR %08x CH_ES %08x\n", chan->CH_CSR, chan->CH_ES);
	dump_sec_vio(label);

	/* Restore the channel bus attributes for whoever runs next */
	chan->CH_SBR = sbr;
}

int syn_dma_arena_probe(void)
{
	int ret = syn_hal_dma_init();

	if (ret != 0 && ret != -EALREADY) {
		printk("arena probe: DMA unavailable: %d\n", ret);
		return ret;
	}
	if (channels[PROBE_CH].active) {
		printk("arena probe: channel %d busy\n", PROBE_CH);
		return -EBUSY;
	}

	uint32_t shape[1] = { PROBE_SIZE };
	syn_tensor_t *t = syn_mem_tensor_alloc(shape, 1, SYN_NPU_DTYPE_INT8,
					       SYN_MEM_EPHEMERAL);

	if (t == NULL) {
		printk("arena probe: arena alloc of %u bytes failed\n",
		       PROBE_SIZE);
		return -ENOMEM;
	}

	void *arena = t->data;
	uint32_t arena_plain = dma_addr(arena);
	uint32_t static_plain = dma_addr(probe_src);

	printk("=== arena eDMA reachability probe (hw ch %d) ===\n",
	       PROBE_HW_CH);
	printk("  MASTER_SEC_LEVEL %08x ANTI_POL %08x LOCK %08x "
	       "MISC_CTRL %08x\n", AHBSC->MASTER_SEC_LEVEL,
	       AHBSC->MASTER_SEC_ANTI_POL_REG, AHBSC->SEC_GP_REG_LOCK,
	       AHBSC->MISC_CTRL_REG);
	print_rule(probe_src, "static buf");
	print_rule(arena, "arena     ");
	dump_sec_vio("start");

	/* Endpoints are passed EXACTLY as the eDMA will see them: the
	 * ns tests use plain aliases (what the HAL normally emits).
	 */
	void *src_plain = (void *)(uintptr_t)static_plain;
	void *dst_plain = (void *)(uintptr_t)dma_addr(probe_dst);
	void *arena_pl = (void *)(uintptr_t)arena_plain;

	/* T0 control, T1/T2 Phase 5 repro, T3/T4 secure channel */
	probe_xfer("T0 static->static ns", src_plain, dst_plain, -1);
	probe_xfer("T1 static->arena  ns", src_plain, arena_pl, -1);
	probe_xfer("T2 arena->static  ns", arena_pl, dst_plain, -1);
	probe_xfer("T3 static->arena  sec (plain)",
		   (void *)(uintptr_t)static_plain,
		   (void *)(uintptr_t)arena_plain, 1);
	probe_xfer("T4 static->arena  sec (secure alias)",
		   (void *)(uintptr_t)(static_plain | BIT(28)),
		   (void *)(uintptr_t)(arena_plain | BIT(28)), 1);

	/* T5: raise the EDMA0 master security level (anti-pol pair) */
	uint32_t msl = AHBSC->MASTER_SEC_LEVEL;
	uint32_t pol = AHBSC->MASTER_SEC_ANTI_POL_REG;
	uint32_t msl_new = (msl & ~AHBSC_MASTER_SEC_LEVEL_EDMA0_MASK) |
			   AHBSC_MASTER_SEC_LEVEL_EDMA0(3U);
	uint32_t pol_new = (pol & ~AHBSC_MASTER_SEC_LEVEL_EDMA0_MASK) |
			   (AHBSC_MASTER_SEC_LEVEL_EDMA0_MASK &
			    ~AHBSC_MASTER_SEC_LEVEL_EDMA0(3U));

	AHBSC->MASTER_SEC_LEVEL = msl_new;
	AHBSC->MASTER_SEC_ANTI_POL_REG = pol_new;
	__DSB();
	printk("T5 master level: MASTER_SEC_LEVEL %08x -> %08x "
	       "(readback %08x)\n", msl, msl_new, AHBSC->MASTER_SEC_LEVEL);
	probe_xfer("T5 static->arena  ns+master-sec", src_plain, arena_pl,
		   -1);
	AHBSC->MASTER_SEC_LEVEL = msl;
	AHBSC->MASTER_SEC_ANTI_POL_REG = pol;
	__DSB();

	/* T6: open the arena granules to non-secure access (MPC rule
	 * 0), transfer, then restore. CPU-side canary reads go through
	 * the plain alias while the rule is lowered.
	 */
	uint32_t old_a = set_rule(arena_plain, 0U);
	uint32_t old_b = set_rule(arena_plain + PROBE_SIZE - 1U, 0U);

	printk("T6 rule: arena granules %u/%u -> 0\n", old_a, old_b);
	probe_xfer("T6 static->arena  ns+rule0", src_plain, arena_pl, -1);
	set_rule(arena_plain, old_a);
	set_rule(arena_plain + PROBE_SIZE - 1U, old_b);

	print_rule(arena, "arena post");

	/* T7-T9: the RAMC0 block (0x20010000-0x20017FFF) boots with
	 * rule 3 (secure+privileged only) while every other RAM block
	 * boots open - dumped live on S5. An arena that starts below
	 * it and is big enough SPANS it, and any tensor the bump
	 * allocator places inside it is exactly the Phase 5 "arena is
	 * not eDMA-reachable" silent timeout. Steer a tensor into
	 * RAMC0 by allocating filler first, then repro + fix.
	 */
	const uint32_t ramc0 = 0x20010000U;

	if (arena_plain < ramc0) {
		uint32_t fill_shape[1] = { ramc0 - (arena_plain +
					   PROBE_SIZE) };
		uint32_t shp[1] = { PROBE_SIZE };
		syn_tensor_t *fill = syn_mem_tensor_alloc(fill_shape, 1,
							  SYN_NPU_DTYPE_INT8,
							  SYN_MEM_EPHEMERAL);
		syn_tensor_t *t2 = (fill != NULL) ?
			syn_mem_tensor_alloc(shp, 1, SYN_NPU_DTYPE_INT8,
					     SYN_MEM_EPHEMERAL) : NULL;

		if (t2 == NULL) {
			printk("T7 skip: cannot steer a tensor into RAMC0 "
			       "(arena too small?)\n");
		} else {
			uint32_t t2_plain = dma_addr(t2->data);
			void *t2_pl = (void *)(uintptr_t)t2_plain;

			print_rule(t2->data, "T7 target ");
			if (t2_plain < ramc0 || t2_plain >= ramc0 + 0x8000U) {
				printk("T7 skip: steered tensor missed "
				       "RAMC0 (%08x)\n", t2_plain);
			} else {
				probe_xfer("T7 static->RAMC0  ns "
					   "(Phase 5 repro)",
					   src_plain, t2_pl, -1);
				probe_xfer("T8 static->RAMC0  sec (fix)",
					   src_plain, t2_pl, 1);

				uint32_t old = set_rule(t2_plain, 0U);

				printk("T9 rule: RAMC0 granule %u -> 0\n",
				       old);
				probe_xfer("T9 static->RAMC0  ns+rule0",
					   src_plain, t2_pl, -1);
				set_rule(t2_plain, old);
				print_rule(t2->data, "T9 post   ");
			}
		}
	} else {
		printk("T7 skip: arena starts above RAMC0\n");
	}

	printk("=== probe done (ephemeral arena reset) ===\n");
	syn_mem_reset_ephemeral();
	return 0;
}
#endif /* CONFIG_SYNAPTIC_DMA_ARENA_PROBE */

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
