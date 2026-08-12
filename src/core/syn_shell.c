/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file syn_shell.c
 * @brief SynapticOS — Zephyr Shell Commands
 *
 * Runtime inspection commands: syn version/mem/model/npu/prof
 */

#include <zephyr/kernel.h>
#include <zephyr/shell/shell.h>
#include <synaptic/syn_api.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../hal/common/syn_dsp_soft.h"
#include "syn_mem_internal.h"
#include "syn_infer_internal.h"

#ifdef CONFIG_SYNAPTIC_MPU_PROTECT
#include "syn_mpu_internal.h"
#endif

#if defined(CONFIG_SYNAPTIC_DUAL_CORE) && !defined(CONFIG_SOC_MCXN947_CPU1)
#include "syn_boot_internal.h"
#include "syn_infer_remote.h"
#include "syn_ipc_internal.h"
#endif

#ifdef CONFIG_SYNAPTIC_OTA
#include <zephyr/sys/util.h> /* hex2bin */
#include <synaptic/syn_model_ota.h>
#include "syn_model_ota_internal.h"
#include "syn_model_store.h"
#endif

/* syn version */
static int cmd_version(const struct shell *sh, size_t argc, char **argv)
{
	ARG_UNUSED(argc);
	ARG_UNUSED(argv);
	shell_print(sh, "SynapticOS v%s", syn_version());
	return 0;
}

/* syn mem stats */
static int cmd_mem_stats(const struct shell *sh, size_t argc, char **argv)
{
	ARG_UNUSED(argc);
	ARG_UNUSED(argv);
	syn_mem_stats_t stats;
	int ret = syn_mem_get_stats(&stats);

	if (ret != 0) {
		shell_error(sh, "Failed to get stats: %d", ret);
		return ret;
	}
	shell_print(sh, "Arena: %u/%u bytes (peak %u)",
		    (unsigned)stats.arena_used,
		    (unsigned)stats.arena_total,
		    (unsigned)stats.arena_peak);
	shell_print(sh, "Scratch: %u/%u bytes",
		    (unsigned)stats.scratch_used,
		    (unsigned)stats.scratch_total);
	shell_print(sh, "Allocations: %u, Resets: %u",
		    stats.alloc_count, stats.reset_count);
	return 0;
}

/* syn mem dump */
static int cmd_mem_dump(const struct shell *sh, size_t argc, char **argv)
{
	ARG_UNUSED(argc);
	ARG_UNUSED(argv);
	syn_mem_layout_t lay;
	int ret = syn_mem_get_layout(&lay);

	if (ret != 0) {
		shell_error(sh, "Arena not initialized: %d", ret);
		return ret;
	}

	const uint8_t *pers_end = lay.base + lay.persistent_used;
	const uint8_t *eph_end = pers_end + lay.ephemeral_used;
	const uint8_t *scratch_base = lay.base + lay.usable;

	shell_print(sh, "Arena layout (%u bytes total):",
		    (unsigned)lay.total);
	shell_print(sh, "  base       %p", (const void *)lay.base);
	shell_print(sh, "  persistent %p - %p (%u bytes)",
		    (const void *)lay.base, (const void *)pers_end,
		    (unsigned)lay.persistent_used);
	shell_print(sh, "  ephemeral  %p - %p (%u bytes)",
		    (const void *)pers_end, (const void *)eph_end,
		    (unsigned)lay.ephemeral_used);
	shell_print(sh, "  free       %u bytes",
		    (unsigned)(lay.usable - lay.persistent_used -
			       lay.ephemeral_used));
	shell_print(sh, "  scratch    %p (%u/%u bytes used)",
		    (const void *)scratch_base, (unsigned)lay.scratch_used,
		    (unsigned)lay.scratch_total);
	shell_print(sh, "First 64 bytes at arena base:");
	shell_hexdump(sh, lay.base, 64);
	return 0;
}

/* syn model list */
static int cmd_model_list(const struct shell *sh, size_t argc, char **argv)
{
	ARG_UNUSED(argc);
	ARG_UNUSED(argv);
	syn_model_handle_t handles[CONFIG_SYNAPTIC_MAX_MODELS];
	uint8_t count = 0;

	int ret = syn_model_list(handles, &count, CONFIG_SYNAPTIC_MAX_MODELS);

	if (ret != 0) {
		shell_error(sh, "Failed: %d", ret);
		return ret;
	}

	shell_print(sh, "Registered models: %u", count);
	for (uint8_t i = 0; i < count; i++) {
		syn_model_info_t info;

		syn_model_get_info(handles[i], &info);
		shell_print(sh, "  [%u] %s v%s %s",
			    handles[i], info.name, info.version,
			    syn_model_is_loaded(handles[i]) ? "(loaded)" : "");
	}
	return 0;
}

static const char *dtype_name(syn_npu_dtype_t dtype)
{
	switch (dtype) {
	case SYN_NPU_DTYPE_INT8:
		return "int8";
	case SYN_NPU_DTYPE_UINT8:
		return "uint8";
	case SYN_NPU_DTYPE_INT16:
		return "int16";
	case SYN_NPU_DTYPE_FLOAT16:
		return "float16";
	case SYN_NPU_DTYPE_FLOAT32:
		return "float32";
	default:
		return "unknown";
	}
}

/* syn model info <name> */
static int cmd_model_info(const struct shell *sh, size_t argc, char **argv)
{
	ARG_UNUSED(argc);

	syn_model_handle_t handle;
	int ret = syn_model_get_by_name(argv[1], &handle);

	if (ret != 0) {
		shell_error(sh, "Model '%s' not found", argv[1]);
		return ret;
	}

	syn_model_info_t info;

	ret = syn_model_get_info(handle, &info);
	if (ret != 0) {
		shell_error(sh, "get_info failed: %d", ret);
		return ret;
	}

	shell_print(sh, "Model '%s' (handle %u):", info.name, handle);
	shell_print(sh, "  version:  %s", info.version);
	shell_print(sh, "  loaded:   %s",
		    syn_model_is_loaded(handle) ? "yes" : "no");
	shell_print(sh, "  input:    %u bytes %s [%u,%u,%u,%u]",
		    info.input_size, dtype_name(info.input_dtype),
		    info.input_shape[0], info.input_shape[1],
		    info.input_shape[2], info.input_shape[3]);
	shell_print(sh, "  output:   %u bytes %s [%u,%u,%u,%u]",
		    info.output_size, dtype_name(info.output_dtype),
		    info.output_shape[0], info.output_shape[1],
		    info.output_shape[2], info.output_shape[3]);
	shell_print(sh, "  sram:     %u bytes required", info.sram_required);
	if (info.flash_size != 0U) {
		shell_print(sh, "  flash:    %u bytes at 0x%08x crc 0x%08x",
			    info.flash_size, info.flash_offset, info.crc32);
	} else {
		shell_print(sh, "  flash:    (RAM-resident model)");
	}
	return 0;
}

/* syn model load <name> */
static int cmd_model_load(const struct shell *sh, size_t argc, char **argv)
{
	ARG_UNUSED(argc);

	syn_model_handle_t handle;
	int ret = syn_model_get_by_name(argv[1], &handle);

	if (ret != 0) {
		shell_error(sh, "Model '%s' not found", argv[1]);
		return ret;
	}

	uint32_t start = k_cycle_get_32();

	ret = syn_model_load(handle);

	uint32_t elapsed_us = k_cyc_to_us_ceil32(k_cycle_get_32() - start);

	if (ret == -EALREADY) {
		shell_print(sh, "Model '%s' is already loaded", argv[1]);
		return 0;
	}
	if (ret != 0) {
		shell_error(sh, "load failed: %d", ret);
		return ret;
	}
	shell_print(sh, "Model '%s' loaded to NPU (%u us)", argv[1],
		    elapsed_us);
	return 0;
}

/* syn model unload <name> */
static int cmd_model_unload(const struct shell *sh, size_t argc, char **argv)
{
	ARG_UNUSED(argc);

	syn_model_handle_t handle;
	int ret = syn_model_get_by_name(argv[1], &handle);

	if (ret != 0) {
		shell_error(sh, "Model '%s' not found", argv[1]);
		return ret;
	}

	ret = syn_model_unload(handle);
	if (ret == -EALREADY) {
		shell_print(sh, "Model '%s' is not loaded", argv[1]);
		return 0;
	}
	if (ret != 0) {
		shell_error(sh, "unload failed: %d", ret);
		return ret;
	}
	shell_print(sh, "Model '%s' unloaded", argv[1]);
	return 0;
}

/* syn npu caps */
static int cmd_npu_caps(const struct shell *sh, size_t argc, char **argv)
{
	ARG_UNUSED(argc);
	ARG_UNUSED(argv);
	syn_npu_caps_t caps;
	int ret = syn_hal_npu_get_caps(&caps);

	if (ret != 0) {
		shell_error(sh, "Failed: %d", ret);
		return ret;
	}
	shell_print(sh, "NPU: %s", caps.name);
	shell_print(sh, "  Max OPS/sec: %u", caps.max_ops_per_sec);
	shell_print(sh, "  Scratch: %u bytes", caps.scratch_size);
	shell_print(sh, "  Async: %s", caps.supports_async ? "yes" : "no");
	return 0;
}

/* syn npu state */
static int cmd_npu_state(const struct shell *sh, size_t argc, char **argv)
{
	ARG_UNUSED(argc);
	ARG_UNUSED(argv);
	static const char *state_names[] = {"IDLE", "BUSY", "ERROR", "SUSPENDED"};
	syn_npu_state_t state = syn_hal_npu_get_state();

	shell_print(sh, "NPU state: %s", state_names[state]);
	return 0;
}

/* syn prof last */
static int cmd_prof_last(const struct shell *sh, size_t argc, char **argv)
{
	ARG_UNUSED(argc);
	ARG_UNUSED(argv);
	syn_prof_result_t result;
	int ret = syn_prof_get_last(&result);

	if (ret != 0) {
		shell_print(sh, "No profiling data available");
		return 0;
	}
	shell_print(sh, "Last inference:");
	shell_print(sh, "  Total:       %u us", result.total_us);
	shell_print(sh, "  Preprocess:  %u us", result.preprocess_us);
	shell_print(sh, "  NPU:         %u us", result.npu_us);
	shell_print(sh, "  Postprocess: %u us", result.postprocess_us);
	shell_print(sh, "  Memory peak: %u bytes", result.mem_peak_bytes);
	return 0;
}

/* syn prof enable */
static int cmd_prof_enable(const struct shell *sh, size_t argc, char **argv)
{
	ARG_UNUSED(argc);
	ARG_UNUSED(argv);
	syn_prof_enable();
	shell_print(sh, "Profiling enabled");
	return 0;
}

/* syn prof disable */
static int cmd_prof_disable(const struct shell *sh, size_t argc, char **argv)
{
	ARG_UNUSED(argc);
	ARG_UNUSED(argv);
	syn_prof_disable();
	shell_print(sh, "Profiling disabled");
	return 0;
}

/* syn dsp bench */
#define BENCH_FFT_N     256
#define BENCH_FFT_ITER  16
#define BENCH_MAT_DIM   16
#define BENCH_MAT_ITER  200

static float bench_fft_in[BENCH_FFT_N * 2];
static float bench_fft_soft[BENCH_FFT_N * 2];
static float bench_fft_hal[BENCH_FFT_N * 2];
static int16_t bench_mat_a[BENCH_MAT_DIM * BENCH_MAT_DIM] __aligned(4);
static int16_t bench_mat_b[BENCH_MAT_DIM] __aligned(4);
static int16_t bench_mat_soft[BENCH_MAT_DIM] __aligned(4);
static int16_t bench_mat_hal[BENCH_MAT_DIM] __aligned(4);

static int cmd_dsp_bench(const struct shell *sh, size_t argc, char **argv)
{
	ARG_UNUSED(argc);
	ARG_UNUSED(argv);

	uint32_t t0, soft_us, hal_us;

	/* --- FFT: two-tone signal, N=256 complex points --- */
	for (int i = 0; i < BENCH_FFT_N; i++) {
		bench_fft_in[2 * i] =
			sinf(2.0f * 3.14159265f * 5.0f * i / BENCH_FFT_N) +
			0.5f * sinf(2.0f * 3.14159265f * 42.0f * i /
				    BENCH_FFT_N);
		bench_fft_in[2 * i + 1] = 0.0f;
	}

	t0 = k_cycle_get_32();
	for (int it = 0; it < BENCH_FFT_ITER; it++) {
		syn_dsp_soft_fft_f32(bench_fft_in, bench_fft_soft,
				     BENCH_FFT_N);
	}
	soft_us = k_cyc_to_us_ceil32(k_cycle_get_32() - t0);

	t0 = k_cycle_get_32();
	for (int it = 0; it < BENCH_FFT_ITER; it++) {
		syn_hal_dsp_fft_f32(bench_fft_in, bench_fft_hal,
				    BENCH_FFT_N);
	}
	hal_us = k_cyc_to_us_ceil32(k_cycle_get_32() - t0);

	float max_err = 0.0f;
	float max_mag = 0.0f;

	for (int i = 0; i < BENCH_FFT_N * 2; i++) {
		float e = fabsf(bench_fft_hal[i] - bench_fft_soft[i]);
		float m = fabsf(bench_fft_soft[i]);

		if (e > max_err) {
			max_err = e;
		}
		if (m > max_mag) {
			max_mag = m;
		}
	}

	shell_print(sh, "FFT f32 %u pts x%u:", BENCH_FFT_N, BENCH_FFT_ITER);
	shell_print(sh, "  soft: %u us (%u us/op)", soft_us,
		    soft_us / BENCH_FFT_ITER);
	shell_print(sh, "  hal:  %u us (%u us/op)", hal_us,
		    hal_us / BENCH_FFT_ITER);
	if (hal_us > 0) {
		shell_print(sh, "  speedup: %u.%02ux",
			    soft_us / hal_us,
			    (soft_us * 100 / hal_us) % 100);
	}
	shell_print(sh, "  max err: %d ppm of peak",
		    max_mag > 0.0f ?
		    (int)(max_err / max_mag * 1000000.0f) : 0);

	/* --- Q15 matmul: 16x16 matrix times vector --- */
	for (int i = 0; i < BENCH_MAT_DIM * BENCH_MAT_DIM; i++) {
		bench_mat_a[i] = (int16_t)(((i * 2654435761U) >> 16) & 0x3FFF)
				 - 8192;
	}
	for (int i = 0; i < BENCH_MAT_DIM; i++) {
		bench_mat_b[i] = (int16_t)(((i * 40503U) & 0x3FFF) - 8192);
	}

	t0 = k_cycle_get_32();
	for (int it = 0; it < BENCH_MAT_ITER; it++) {
		syn_dsp_soft_mat_mult_q15(bench_mat_a, bench_mat_b,
					  bench_mat_soft,
					  BENCH_MAT_DIM, BENCH_MAT_DIM);
	}
	soft_us = k_cyc_to_us_ceil32(k_cycle_get_32() - t0);

	t0 = k_cycle_get_32();
	for (int it = 0; it < BENCH_MAT_ITER; it++) {
		syn_hal_dsp_mat_mult_q15(bench_mat_a, bench_mat_b,
					 bench_mat_hal,
					 BENCH_MAT_DIM, BENCH_MAT_DIM);
	}
	hal_us = k_cyc_to_us_ceil32(k_cycle_get_32() - t0);

	int max_lsb = 0;

	for (int i = 0; i < BENCH_MAT_DIM; i++) {
		int d = abs(bench_mat_hal[i] - bench_mat_soft[i]);

		if (d > max_lsb) {
			max_lsb = d;
		}
	}

	shell_print(sh, "MatMul q15 %ux%u x%u:", BENCH_MAT_DIM,
		    BENCH_MAT_DIM, BENCH_MAT_ITER);
	shell_print(sh, "  soft: %u us (%u ns/op)", soft_us,
		    soft_us * 1000 / BENCH_MAT_ITER);
	shell_print(sh, "  hal:  %u us (%u ns/op)", hal_us,
		    hal_us * 1000 / BENCH_MAT_ITER);
	if (hal_us > 0) {
		shell_print(sh, "  speedup: %u.%02ux",
			    soft_us / hal_us,
			    (soft_us * 100 / hal_us) % 100);
	}
	shell_print(sh, "  max err: %d LSB", max_lsb);

	return 0;
}

static int parse_priority(const char *arg, syn_priority_t *prio)
{
	if (strcmp(arg, "rt") == 0 || strcmp(arg, "realtime") == 0 ||
	    strcmp(arg, "2") == 0) {
		*prio = SYN_PRIORITY_REALTIME;
	} else if (strcmp(arg, "normal") == 0 || strcmp(arg, "1") == 0) {
		*prio = SYN_PRIORITY_NORMAL;
	} else if (strcmp(arg, "be") == 0 || strcmp(arg, "best_effort") == 0 ||
		   strcmp(arg, "0") == 0) {
		*prio = SYN_PRIORITY_BEST_EFFORT;
	} else {
		return -EINVAL;
	}
	return 0;
}

static const char *priority_name(syn_priority_t prio)
{
	switch (prio) {
	case SYN_PRIORITY_REALTIME:
		return "realtime";
	case SYN_PRIORITY_NORMAL:
		return "normal";
	default:
		return "best_effort";
	}
}

/* syn dma bench [frames]: double-buffered zero-copy ingest vs a
 * sequential CPU-copy baseline. The synthetic source stamps a
 * 64-byte header per frame over a static body; processing verifies
 * the stamp and word-checksums the whole frame, so torn or stale
 * frames are caught byte-exactly. On QEMU the DMA stub copies on
 * the CPU: numbers there are functional, not a throughput claim.
 */
#ifdef CONFIG_SOC_SERIES_MCXNX4X
#define DMA_BENCH_FRAME  8192
#else
#define DMA_BENCH_FRAME  512
#endif
#define DMA_BENCH_STAMP  64
#define DMA_BENCH_CH     0

#include "syn_ingest.h"
#include <synaptic/syn_hal_dma.h>

#ifdef CONFIG_SOC_SERIES_MCXNX4X
/* eDMA bring-up diagnostics (board-only, defined in the eDMA HAL) */
void syn_hal_dma_dump(int channel);

/* Shared by `syn dma probe` and `syn dma bench`. The Phase 5 "eDMA
 * cannot reach the tensor arena" finding did NOT survive the Phase 6
 * reachability experiments (`syn dma arena`, S5): with the P5 eDMA
 * fixes in place (software START, no aborts between one-shots) the
 * arena is reachable at every tested address, and the AHBSC boots
 * with secure checking disabled. Latent constraint to remember: RAM
 * block RAMC0 0x20010000-0x20017FFF boots with a secure-only MPC
 * rule, which starts mattering (silently, for non-secure-attributed
 * eDMA transactions) if secure checking is ever enabled; CH_SBR SEC
 * per channel is the proven fix. The bench keeps using statics only
 * so its numbers stay comparable across phases; moving the ingest
 * path onto arena tensors is the 6.2 follow-up.
 */
static uint8_t dma_buf_src[8192] __aligned(4);
static uint8_t dma_buf_a[8192] __aligned(4);
static uint8_t dma_buf_b[8192] __aligned(4);

#endif

struct dma_bench_ctx {
	uint32_t bad_frames;
	uint32_t checksum;
};

static void dma_bench_fill(void *src, size_t size, uint32_t seq,
			   void *user)
{
	ARG_UNUSED(user);

	uint8_t *p = src;
	size_t stamp = MIN((size_t)DMA_BENCH_STAMP, size);

	for (size_t i = 0; i < stamp; i++) {
		p[i] = (uint8_t)(seq * 31U + i * 7U);
	}
}

static void dma_bench_process(const void *frame, size_t size,
			      uint32_t seq, void *user)
{
	struct dma_bench_ctx *ctx = user;
	const uint8_t *p = frame;
	size_t stamp = MIN((size_t)DMA_BENCH_STAMP, size);

	for (size_t i = 0; i < stamp; i++) {
		if (p[i] != (uint8_t)(seq * 31U + i * 7U)) {
			ctx->bad_frames++;
			return;
		}
	}

	const uint32_t *w = frame;

	for (size_t i = 0; i < size / 4U; i++) {
		ctx->checksum += w[i];
	}
}

static int cmd_dma_bench(const struct shell *sh, size_t argc, char **argv)
{
	uint32_t frames = (argc >= 2) ? (uint32_t)strtoul(argv[1], NULL, 0)
				      : 64U;
	bool use_static = false;

	if (frames == 0U) {
		shell_error(sh, "frames must be > 0");
		return -EINVAL;
	}
	if (argc >= 3) {
		if (strcmp(argv[2], "static") == 0) {
			use_static = true;
		} else if (strcmp(argv[2], "arena") != 0) {
			shell_error(sh, "Bad buffer mode '%s' (use arena "
				    "or static)", argv[2]);
			return -EINVAL;
		}
	}

	int ret = syn_hal_dma_init();

	if (ret != 0 && ret != -EALREADY) {
		shell_error(sh, "DMA unavailable: %d", ret);
		return ret;
	}

	/* Default: genuine zero-copy into arena tensors - the arena IS
	 * eDMA-reachable (the Phase 6 `syn dma arena` experiments
	 * revised the Phase 5 finding). The `static` mode keeps the
	 * board-static variant for cross-phase comparability.
	 */
	uint8_t *src = NULL;
	uint8_t *dst0 = NULL;
	uint8_t *dst1 = NULL;
	bool arena_bufs = false;

#ifdef CONFIG_SOC_SERIES_MCXNX4X
	if (use_static) {
		src = dma_buf_src;
		dst0 = dma_buf_a;
		dst1 = dma_buf_b;
	}
#else
	if (use_static) {
		shell_error(sh, "static buffers exist only on the MCXN "
			    "build");
		return -EINVAL;
	}
#endif

	if (src == NULL) {
		src = syn_mem_scratch_acquire(DMA_BENCH_FRAME);

		if (src == NULL) {
			shell_error(sh, "scratch pool too small for a "
				    "%u-byte frame", DMA_BENCH_FRAME);
			return -ENOMEM;
		}

		uint32_t shape[1] = { DMA_BENCH_FRAME };
		syn_tensor_t *b0 = syn_mem_tensor_alloc(shape, 1,
							SYN_NPU_DTYPE_UINT8,
							SYN_MEM_EPHEMERAL);
		syn_tensor_t *b1 = syn_mem_tensor_alloc(shape, 1,
							SYN_NPU_DTYPE_UINT8,
							SYN_MEM_EPHEMERAL);

		if (b0 == NULL || b1 == NULL) {
			shell_error(sh, "arena too small for two %u-byte "
				    "buffers", DMA_BENCH_FRAME);
			syn_mem_scratch_release(src);
			syn_mem_reset_ephemeral();
			return -ENOMEM;
		}

		dst0 = b0->data;
		dst1 = b1->data;
		arena_bufs = true;
	}

	/* Static frame body under the per-frame stamp */
	memset(src, 0x5A, DMA_BENCH_FRAME);

	struct dma_bench_ctx ctx = {0};

	/* CPU-copy baseline: fill -> memcpy -> process, sequential */
	uint32_t t0 = k_cycle_get_32();

	for (uint32_t seq = 0; seq < frames; seq++) {
		dma_bench_fill(src, DMA_BENCH_FRAME, seq, NULL);
		memcpy(dst0, src, DMA_BENCH_FRAME);
		dma_bench_process(dst0, DMA_BENCH_FRAME, seq, &ctx);
	}

	uint32_t cpu_us = k_cyc_to_us_ceil32(k_cycle_get_32() - t0);
	uint32_t cpu_bad = ctx.bad_frames;

	/* Zero-copy path: DMA into ping/pong, processing overlapped */
	ctx.bad_frames = 0;

	syn_ingest_config_t icfg = {
		.src = src,
		.bufs = { dst0, dst1 },
		.frame_size = DMA_BENCH_FRAME,
		.dma_channel = DMA_BENCH_CH,
		.fill = dma_bench_fill,
		.process = dma_bench_process,
		.user = &ctx,
	};

	ret = syn_ingest_run(&icfg, frames);

	syn_ingest_stats_t st;

	syn_ingest_last_stats(&st);
	if (arena_bufs) {
		syn_mem_scratch_release(src);
		syn_mem_reset_ephemeral();
	}

	if (ret != 0) {
		shell_error(sh, "ingest failed: %d (frames %u, dma errors "
			    "%u)", ret, st.frames, st.dma_errors);
#ifdef CONFIG_SOC_SERIES_MCXNX4X
		syn_hal_dma_dump(DMA_BENCH_CH);
#endif
		return ret;
	}

	shell_print(sh, "%u frames of %u bytes (%s buffers):", frames,
		    DMA_BENCH_FRAME, arena_bufs ? "arena" : "static");
	shell_print(sh, "  cpu copy:  %u us (%u us/frame), %u corrupt",
		    cpu_us, cpu_us / frames, cpu_bad);
	shell_print(sh, "  dma ingest:%u us (%u us/frame), %u corrupt, "
		    "%u dma errors", st.elapsed_us, st.elapsed_us / frames,
		    ctx.bad_frames, st.dma_errors);
	if (st.elapsed_us > 0U && st.elapsed_us < cpu_us) {
		shell_print(sh, "  frame rate gain: +%u%%",
			    (unsigned)((cpu_us - st.elapsed_us) * 100U /
				       st.elapsed_us));
	} else {
		shell_print(sh, "  no gain on this target");
	}
	return 0;
}

/* syn infer run <model-name> [priority] */
static int cmd_infer_run(const struct shell *sh, size_t argc, char **argv)
{
	if (argc < 2) {
		shell_error(sh, "Usage: syn infer run <model-name> "
				"[be|normal|rt]");
		return -EINVAL;
	}

	syn_priority_t prio = SYN_PRIORITY_NORMAL;

	if (argc >= 3 && parse_priority(argv[2], &prio) != 0) {
		shell_error(sh, "Bad priority '%s' (use be, normal or rt)",
			    argv[2]);
		return -EINVAL;
	}

	syn_model_handle_t handle;
	int ret = syn_model_get_by_name(argv[1], &handle);

	if (ret != 0) {
		shell_error(sh, "Model '%s' not found", argv[1]);
		return ret;
	}

	syn_model_info_t info;

	syn_model_get_info(handle, &info);

	/* Ephemeral input tensor with a gradient test pattern */
	uint32_t shape[1] = { info.input_size };
	syn_tensor_t *input = syn_mem_tensor_alloc(shape, 1,
						   info.input_dtype,
						   SYN_MEM_EPHEMERAL);

	if (input == NULL) {
		shell_error(sh, "Arena too small for %u-byte input",
			    info.input_size);
		return -ENOMEM;
	}

	uint8_t *data = input->data;

	for (size_t i = 0; i < input->size; i++) {
		data[i] = (uint8_t)(i & 0xFF);
	}

	syn_tensor_t output = {0};
	uint32_t start = k_cycle_get_32();

	ret = syn_infer_run_sync(handle, input, &output, prio);

	uint32_t elapsed_us = k_cyc_to_us_ceil32(k_cycle_get_32() - start);

	if (ret != 0) {
		shell_error(sh, "Inference failed: %d", ret);
		syn_mem_reset_ephemeral();
		return ret;
	}

	uint32_t top_class = 0;

	if (output.dtype == SYN_NPU_DTYPE_INT8 && output.size > 0) {
		syn_hal_dsp_argmax(output.data, output.size, &top_class);
		shell_print(sh, "Model '%s' (%s): class %u (confidence %d), "
			    "%u us", argv[1], priority_name(prio), top_class,
			    ((int8_t *)output.data)[top_class], elapsed_us);
	} else {
		shell_print(sh, "Model '%s' (%s): %u output bytes, %u us",
			    argv[1], priority_name(prio),
			    (unsigned)output.size, elapsed_us);
	}
	shell_print(sh, "Use 'syn prof last' for the stage breakdown.");

	/* Free the input tensor and pipeline intermediates */
	syn_mem_reset_ephemeral();
	return 0;
}

#ifdef CONFIG_SYNAPTIC_LAYER_EXEC
#include "../hal/common/syn_npu_layered.h"

/* syn npu plan */
static int cmd_npu_plan(const struct shell *sh, size_t argc, char **argv)
{
	ARG_UNUSED(argc);
	ARG_UNUSED(argv);

	uint32_t planned, naive, plan_us;

	syn_npu_layered_plan_info(&planned, &naive, &plan_us);
	if (naive == 0U) {
		shell_print(sh, "No layered session has run yet");
		return 0;
	}
	shell_print(sh, "Activation placement of the last layered session:");
	shell_print(sh, "  planned peak: %u bytes", planned);
	shell_print(sh, "  all-live sum: %u bytes (naive baseline)", naive);
	shell_print(sh, "  reduction:    %u%%",
		    (unsigned)(100U - (planned * 100U / naive)));
	shell_print(sh, "  planning:     %u us", plan_us);
	return 0;
}
#endif /* CONFIG_SYNAPTIC_LAYER_EXEC */

/* syn infer stats */
static int cmd_infer_stats(const struct shell *sh, size_t argc, char **argv)
{
	ARG_UNUSED(argc);
	ARG_UNUSED(argv);

	syn_infer_stats_t st;

	syn_infer_get_stats(&st);
	shell_print(sh, "Jobs: %u completed, %u errors, %u cancelled",
		    st.completed, st.errors, st.cancelled);
	shell_print(sh, "Deadline misses: %u", st.deadline_misses);
	shell_print(sh, "Preemptions: %u (resumes %u)",
		    st.preemptions, st.resumes);
	if (st.preemptions > 0U) {
		shell_print(sh, "Context save: last %u us, max %u us",
			    st.last_save_us, st.max_save_us);
	}
	return 0;
}

#ifdef CONFIG_SYNAPTIC_HEALTH
#include "syn_health.h"

/* syn health */
static int cmd_health(const struct shell *sh, size_t argc, char **argv)
{
	ARG_UNUSED(argc);
	ARG_UNUSED(argv);

	shell_print(sh, "Watchdog: %s",
		    syn_health_watchdog_armed() ?
		    "armed (fed by the monitor)" : "absent");
	shell_print(sh, "Stale episodes: %u", syn_health_fault_count());
#if defined(CONFIG_SYNAPTIC_DUAL_CORE) && !defined(CONFIG_SOC_MCXN947_CPU1)
	shell_print(sh, "CPU1 recoveries: %u",
		    syn_health_cpu1_recoveries());
#endif

	syn_health_info_t info;

	for (int i = 0; syn_health_get(i, &info) == 0; i++) {
		shell_print(sh, "  %-8s period %u ms, last kick %lld ms "
			    "ago, %s%s, %u stale episodes", info.name,
			    info.period_ms, info.age_ms,
			    info.busy ? "busy" : "idle",
			    info.stale ? " (STALE)" : "",
			    info.stale_count);
	}
	return 0;
}

/* syn health hang <cpu0|cpu1>: fault-injection for the watchdog and
 * recovery demos. cpu0 stops all scheduling on this core, so the
 * hardware watchdog (when armed) is the only way back.
 */
static int cmd_health_hang(const struct shell *sh, size_t argc, char **argv)
{
	ARG_UNUSED(argc);

	if (strcmp(argv[1], "cpu0") == 0) {
		shell_print(sh, "Hanging CPU0 with the scheduler locked; "
				"only the watchdog can recover this...");
		/* Give the UART a moment to drain the message */
		k_msleep(50);
		k_sched_lock();
		while (1) {
		}
		return 0; /* unreachable */
	}
#if defined(CONFIG_SYNAPTIC_DUAL_CORE) && !defined(CONFIG_SOC_MCXN947_CPU1)
	if (strcmp(argv[1], "cpu1") == 0) {
		syn_shm_region_t *shm_dbg = syn_ipc_region();

		if (shm_dbg == NULL) {
			shell_error(sh, "IPC region not initialized");
			return -ENODEV;
		}
		shm_dbg->ctrl.debug_cmd = SYN_SHM_DEBUG_CPU1_HANG;
		shell_print(sh, "CPU1 hang requested; watch the log for "
				"heartbeat-loss recovery");
		return 0;
	}
#endif
	shell_error(sh, "Usage: syn health hang <cpu0|cpu1>");
	return -EINVAL;
}
#endif /* CONFIG_SYNAPTIC_HEALTH */

#if defined(CONFIG_SYNAPTIC_DUAL_CORE) && !defined(CONFIG_SOC_MCXN947_CPU1)
/* syn ipc status */
static int cmd_ipc_status(const struct shell *sh, size_t argc, char **argv)
{
	ARG_UNUSED(argc);
	ARG_UNUSED(argv);

	shell_print(sh, "CPU1 link: %s",
		    syn_boot_secondary_linked() ? "UP" : "DOWN");
	if (syn_boot_cpu1_boot_us() != 0U) {
		shell_print(sh, "CPU1 boot time: %u us (release to ready)",
			    syn_boot_cpu1_boot_us());
		shell_print(sh, "IPC handshake:  %u us (release to STATUS_REQ)",
			    syn_boot_handshake_us());
	}
	shell_print(sh, "STATUS_REQ answered: %u",
		    syn_boot_status_req_count());
	shell_print(sh, "Inferences served: %u (errors %u, avg %u us)",
		    syn_remote_serve_count(), syn_remote_serve_errors(),
		    syn_remote_serve_avg_us());

	syn_shm_region_t *shm = syn_ipc_region();

	if (shm != NULL && shm->ctrl.rtt_count > 0U) {
		shell_print(sh, "IPC round-trip (CPU1-measured, %u samples): "
				"last %u us, min %u us, max %u us",
			    shm->ctrl.rtt_count, shm->ctrl.rtt_last_us,
			    shm->ctrl.rtt_min_us, shm->ctrl.rtt_max_us);
	}
	return 0;
}

/* syn ipc stats: per-direction ring message counters. The head/tail
 * indices are free-running, so head == total pushed, tail == total
 * popped since ring reset.
 */
static int cmd_ipc_stats(const struct shell *sh, size_t argc, char **argv)
{
	ARG_UNUSED(argc);
	ARG_UNUSED(argv);

	syn_shm_region_t *shm = syn_ipc_region();

	if (shm == NULL || shm->ctrl.magic != SYN_SHM_MAGIC) {
		shell_error(sh, "IPC region not initialized");
		return -ENODEV;
	}

	static const struct {
		const char *name;
		size_t offset;
	} rings[] = {
		{ "cpu0 to cpu1", offsetof(syn_shm_region_t, ring_c0_to_c1) },
		{ "cpu1 to cpu0", offsetof(syn_shm_region_t, ring_c1_to_c0) },
	};

	shell_print(sh, "Ring capacity: %u messages each direction",
		    SYN_IPC_RING_ENTRIES);
	for (size_t i = 0; i < ARRAY_SIZE(rings); i++) {
		const syn_ipc_ring_t *r = (const syn_ipc_ring_t *)
			((const uint8_t *)shm + rings[i].offset);
		uint32_t head = r->head;
		uint32_t tail = r->tail;

		shell_print(sh, "%s: pushed %u, popped %u, queued %u",
			    rings[i].name, head, tail, head - tail);
	}
	return 0;
}
#endif /* CONFIG_SYNAPTIC_DUAL_CORE && !CPU1 */

#ifdef CONFIG_SYNAPTIC_MPU_PROTECT
/* syn mpu test */
static int cmd_mpu_test(const struct shell *sh, size_t argc, char **argv)
{
	ARG_UNUSED(argc);
	ARG_UNUSED(argv);

	shell_print(sh, "Running cross-core MPU self-test "
			"(a MemManage fault dump below is EXPECTED)...");

	int ret = syn_mpu_selftest();

	if (ret == 0) {
		shell_print(sh, "MPU self-test PASS: shared region writable, "
				"cross-core write faulted");
	} else if (ret == -EPERM) {
		shell_error(sh, "MPU self-test FAIL: cross-core write was "
				"NOT blocked");
	} else {
		shell_error(sh, "MPU self-test error: %d", ret);
	}
	return ret;
}
#endif /* CONFIG_SYNAPTIC_MPU_PROTECT */

#ifdef CONFIG_SYNAPTIC_OTA
/* syn store status */
static int cmd_store_status(const struct shell *sh, size_t argc, char **argv)
{
	ARG_UNUSED(argc);
	ARG_UNUSED(argv);

	if (!syn_store_ready()) {
		shell_error(sh, "model store not initialized");
		return -ENODEV;
	}

	shell_print(sh, "generation %u  active slot %u  staged slot %u  "
		    "prev slot %u", syn_store_generation(),
		    syn_store_active_slot(), syn_store_staged_slot(),
		    syn_store_prev_active_slot());
	shell_print(sh, "registry wear: copy0 %u copy1 %u erases",
		    syn_store_wear(0), syn_store_wear(1));
	shell_print(sh, "last commit %u us, boot scan %u us",
		    syn_store_last_commit_us(), syn_store_scan_us());

	for (uint8_t s = 0; s < syn_store_slot_count(); s++) {
		syn_model_info_t info;

		if (syn_store_slot_info(s, &info) == 0) {
			syn_model_handle_t h;
			bool resident =
				(syn_model_get_by_name(info.name, &h) == 0);

			shell_print(sh, "slot %u: '%s' %u bytes crc 0x%08x "
				    "at 0x%08x%s", s, info.name,
				    info.flash_size, info.crc32,
				    info.flash_offset,
				    resident ? " (resident)" : "");
		} else {
			shell_print(sh, "slot %u: empty", s);
		}
	}
	return 0;
}

/* syn store activate <slot> - hot-swap the active model by slot ID */
static int cmd_store_activate(const struct shell *sh, size_t argc,
			      char **argv)
{
	if (argc < 2) {
		shell_error(sh, "Usage: syn store activate <slot>");
		return -EINVAL;
	}
	if (!syn_store_ready()) {
		shell_error(sh, "model store not initialized");
		return -ENODEV;
	}

	uint8_t slot = (uint8_t)strtoul(argv[1], NULL, 0);
	int ret = syn_store_activate(slot);

	if (ret == -EALREADY) {
		shell_print(sh, "slot %u is already active", slot);
		return 0;
	}
	if (ret != 0) {
		shell_error(sh, "activate slot %u failed: %d", slot, ret);
		return ret;
	}
	shell_print(sh, "slot %u active (gen %u); previous slot %u stays "
		    "resident", slot, syn_store_generation(),
		    syn_store_prev_active_slot());
	return 0;
}

/* syn ota status */
static int cmd_ota_status(const struct shell *sh, size_t argc, char **argv)
{
	ARG_UNUSED(argc);
	ARG_UNUSED(argv);

	syn_ota_status_t st;

	syn_ota_get_status(&st);
	shell_print(sh, "state %s  slot %u  received %u/%u bytes",
		    syn_ota_state_str(st.state), st.slot,
		    st.received, st.total_size);
	shell_print(sh, "session %u us  last error %d",
		    st.session_us, st.last_error);
	return 0;
}

/* syn ota begin <name> <total_size> */
static int cmd_ota_begin(const struct shell *sh, size_t argc, char **argv)
{
	ARG_UNUSED(argc);

	size_t total = (size_t)strtoul(argv[2], NULL, 0);
	int ret = syn_ota_begin(argv[1], total);

	if (ret != 0) {
		shell_error(sh, "ota begin failed: %d", ret);
		return ret;
	}
	shell_print(sh, "OTA RX '%s' %u bytes", argv[1], (unsigned)total);
	return 0;
}

/* syn ota data <hex>: one hex-encoded chunk (transport for the UART
 * demo; the host sender is tools/syn_ota_send.py)
 */
static int cmd_ota_data(const struct shell *sh, size_t argc, char **argv)
{
	ARG_UNUSED(argc);

	static uint8_t chunk[CONFIG_SHELL_CMD_BUFF_SIZE / 2];
	size_t hexlen = strlen(argv[1]);

	if (hexlen == 0U || (hexlen % 2U) != 0U ||
	    hexlen / 2U > sizeof(chunk)) {
		shell_error(sh, "bad hex chunk (%u chars)", (unsigned)hexlen);
		return -EINVAL;
	}

	size_t n = hex2bin(argv[1], hexlen, chunk, sizeof(chunk));

	if (n != hexlen / 2U) {
		shell_error(sh, "hex decode failed");
		return -EINVAL;
	}

	int ret = syn_ota_write_chunk(chunk, n);

	if (ret != 0) {
		shell_error(sh, "ota write failed: %d", ret);
		return ret;
	}

	syn_ota_status_t st;

	syn_ota_get_status(&st);
	shell_print(sh, "ok %u/%u", st.received, st.total_size);
	return 0;
}

/* syn ota rawdata <bytes>: binary transport (Phase 5.6). Switches
 * the shell into bypass mode and feeds the next <bytes> raw UART
 * bytes straight into the OTA engine - no hex doubling, no echo, no
 * line parsing. The host waits for the "RAW <n>" line, streams the
 * bytes, then waits for "raw ok". A stalled transfer is abandoned
 * by a watchdog work item so the shell always comes back.
 */
#define OTA_RAW_TIMEOUT_MS 5000

static struct {
	const struct shell *sh;
	size_t expected;
	size_t received;
	int error;
	uint8_t buf[512];
	size_t buffered;
} raw;

static void ota_raw_timeout(struct k_work *work);
static K_WORK_DELAYABLE_DEFINE(raw_watchdog, ota_raw_timeout);

static void ota_raw_finish(const struct shell *sh)
{
	(void)k_work_cancel_delayable(&raw_watchdog);
	shell_set_bypass(sh, NULL);

	if (raw.error != 0) {
		shell_error(sh, "raw transfer failed: %d after %u bytes",
			    raw.error, (unsigned)raw.received);
		return;
	}

	syn_ota_status_t st;

	syn_ota_get_status(&st);
	shell_print(sh, "raw ok %u/%u", st.received, st.total_size);
}

static void ota_raw_timeout(struct k_work *work)
{
	ARG_UNUSED(work);

	if (raw.sh != NULL && raw.received < raw.expected) {
		raw.error = -ETIMEDOUT;
		ota_raw_finish(raw.sh);
	}
}

static void ota_raw_flush(void)
{
	if (raw.buffered == 0U || raw.error != 0) {
		raw.buffered = 0;
		return;
	}

	int ret = syn_ota_write_chunk(raw.buf, raw.buffered);

	if (ret != 0) {
		/* Keep consuming the announced bytes so the stream tail
		 * is never parsed as shell input; report at the end.
		 */
		raw.error = ret;
	}
	raw.buffered = 0;
}

static void ota_raw_bypass(const struct shell *sh, uint8_t *data, size_t len)
{
	while (len > 0U && raw.received < raw.expected) {
		size_t take = MIN(len, sizeof(raw.buf) - raw.buffered);

		take = MIN(take, raw.expected - raw.received);
		memcpy(raw.buf + raw.buffered, data, take);
		raw.buffered += take;
		raw.received += take;
		data += take;
		len -= take;

		if (raw.buffered == sizeof(raw.buf)) {
			ota_raw_flush();
		}
	}

	if (raw.received >= raw.expected) {
		ota_raw_flush();
		ota_raw_finish(sh);
	} else {
		(void)k_work_reschedule(&raw_watchdog,
					K_MSEC(OTA_RAW_TIMEOUT_MS));
	}
}

static int cmd_ota_rawdata(const struct shell *sh, size_t argc, char **argv)
{
	ARG_UNUSED(argc);

	size_t n = (size_t)strtoul(argv[1], NULL, 0);

	if (n == 0U) {
		shell_error(sh, "Usage: syn ota rawdata <bytes>");
		return -EINVAL;
	}

	memset(&raw, 0, sizeof(raw));
	raw.sh = sh;
	raw.expected = n;

	shell_print(sh, "RAW %u", (unsigned)n);
	shell_set_bypass(sh, ota_raw_bypass);
	(void)k_work_reschedule(&raw_watchdog, K_MSEC(OTA_RAW_TIMEOUT_MS));
	return 0;
}

/* syn ota done */
static int cmd_ota_done(const struct shell *sh, size_t argc, char **argv)
{
	ARG_UNUSED(argc);
	ARG_UNUSED(argv);

	int ret = syn_ota_finish();

	if (ret != 0) {
		shell_error(sh, "ota finish failed: %d", ret);
		return ret;
	}
	shell_print(sh, "OTA staged and verified: state %s",
		    syn_ota_state_str(syn_ota_get_state()));
	return 0;
}

/* syn ota activate */
static int cmd_ota_activate(const struct shell *sh, size_t argc, char **argv)
{
	ARG_UNUSED(argc);
	ARG_UNUSED(argv);

	int ret = syn_ota_activate();

	if (ret != 0) {
		shell_error(sh, "ota activate failed: %d", ret);
		return ret;
	}

	syn_ota_status_t st;

	syn_ota_get_status(&st);
	shell_print(sh, "OTA activated (%u us since begin); active slot %u",
		    st.session_us, syn_store_active_slot());
	return 0;
}

/* syn ota rollback */
static int cmd_ota_rollback(const struct shell *sh, size_t argc, char **argv)
{
	ARG_UNUSED(argc);
	ARG_UNUSED(argv);

	int ret = syn_ota_rollback();

	if (ret != 0) {
		shell_error(sh, "ota rollback failed: %d", ret);
		return ret;
	}
	shell_print(sh, "rolled back; active slot %u",
		    syn_store_active_slot());
	return 0;
}
#endif /* CONFIG_SYNAPTIC_OTA */

/* Dynamic tab completion with the registered model names. The shell
 * requires completion candidates in alphabetical order and a stable
 * string for the returned syntax pointer, hence the sorted static
 * snapshot rebuilt on every query.
 */
static void model_name_get(size_t idx, struct shell_static_entry *entry)
{
	static char names[CONFIG_SYNAPTIC_MAX_MODELS][32];
	syn_model_handle_t handles[CONFIG_SYNAPTIC_MAX_MODELS];
	uint8_t count = 0;

	entry->handler = NULL;
	entry->help = NULL;
	entry->subcmd = NULL;
	entry->syntax = NULL;

	if (syn_model_list(handles, &count, CONFIG_SYNAPTIC_MAX_MODELS) != 0 ||
	    idx >= count) {
		return;
	}

	for (uint8_t i = 0; i < count; i++) {
		syn_model_info_t info;

		names[i][0] = '\0';
		if (syn_model_get_info(handles[i], &info) == 0) {
			strncpy(names[i], info.name, sizeof(names[i]) - 1);
			names[i][sizeof(names[i]) - 1] = '\0';
		}
	}

	for (uint8_t i = 1; i < count; i++) {
		char tmp[32];

		strcpy(tmp, names[i]);
		int j = (int)i - 1;

		while (j >= 0 && strcmp(names[j], tmp) > 0) {
			strcpy(names[j + 1], names[j]);
			j--;
		}
		strcpy(names[j + 1], tmp);
	}

	entry->syntax = names[idx];
}

SHELL_DYNAMIC_CMD_CREATE(dsub_model_name, model_name_get);

/* Subcommand trees */
SHELL_STATIC_SUBCMD_SET_CREATE(sub_mem,
	SHELL_CMD(stats, NULL, "Show memory statistics", cmd_mem_stats),
	SHELL_CMD(dump, NULL, "Dump arena layout and header bytes",
		  cmd_mem_dump),
	SHELL_SUBCMD_SET_END
);

SHELL_STATIC_SUBCMD_SET_CREATE(sub_model,
	SHELL_CMD(list, NULL, "List registered models", cmd_model_list),
	SHELL_CMD_ARG(info, &dsub_model_name,
		      "Show model metadata: syn model info <name>",
		      cmd_model_info, 2, 0),
	SHELL_CMD_ARG(load, &dsub_model_name,
		      "Load model to NPU: syn model load <name>",
		      cmd_model_load, 2, 0),
	SHELL_CMD_ARG(unload, &dsub_model_name,
		      "Unload model from NPU: syn model unload <name>",
		      cmd_model_unload, 2, 0),
	SHELL_SUBCMD_SET_END
);

SHELL_STATIC_SUBCMD_SET_CREATE(sub_npu,
	SHELL_CMD(caps, NULL, "Show NPU capabilities", cmd_npu_caps),
	SHELL_CMD(state, NULL, "Show NPU state", cmd_npu_state),
#ifdef CONFIG_SYNAPTIC_LAYER_EXEC
	SHELL_CMD(plan, NULL, "Show layered activation placement",
		  cmd_npu_plan),
#endif
	SHELL_SUBCMD_SET_END
);

SHELL_STATIC_SUBCMD_SET_CREATE(sub_prof,
	SHELL_CMD(last, NULL, "Show last profiling result", cmd_prof_last),
	SHELL_CMD(enable, NULL, "Enable profiling", cmd_prof_enable),
	SHELL_CMD(disable, NULL, "Disable profiling", cmd_prof_disable),
	SHELL_SUBCMD_SET_END
);

SHELL_STATIC_SUBCMD_SET_CREATE(sub_dsp,
	SHELL_CMD(bench, NULL, "Benchmark DSP ops: hardware vs software",
		  cmd_dsp_bench),
	SHELL_SUBCMD_SET_END
);

#ifdef CONFIG_SOC_SERIES_MCXNX4X
static K_SEM_DEFINE(probe_sem, 0, 1);
static volatile int probe_status;

static void probe_cb(int channel, int status, void *user_data)
{
	ARG_UNUSED(channel);
	ARG_UNUSED(user_data);
	probe_status = status;
	k_sem_give(&probe_sem);
}

static int cmd_dma_probe(const struct shell *sh, size_t argc, char **argv)
{
	ARG_UNUSED(argc);
	ARG_UNUSED(argv);

	uint8_t *psrc = dma_buf_src;
	uint8_t *pdst = dma_buf_a;

	int ret = syn_hal_dma_init();

	if (ret != 0 && ret != -EALREADY) {
		shell_error(sh, "DMA unavailable: %d", ret);
		return ret;
	}

	uint8_t *pdst2 = dma_buf_b;
	uint8_t *dsts[2] = { pdst, pdst2 };

	/* Phase 1: fixed destination, start->wait (known good).
	 * Phase 2: the ingest pump's exact pattern at small scale -
	 * ALTERNATING destinations and CPU work between start and wait.
	 */
	for (int round = 0; round < 6; round++) {
		bool pipelined = (round >= 3);
		uint8_t *dst = pipelined ? dsts[round & 1] : pdst;

		memset(psrc, 0x10 + round, 8192);
		memset(dst, 0, 8192);
		k_sem_reset(&probe_sem);

		syn_dma_config_t cfg = {
			.src_periph = SYN_DMA_PERIPH_MEMORY,
			.dst_periph = SYN_DMA_PERIPH_MEMORY,
			.src_addr = psrc,
			.dst_addr = dst,
			.transfer_size = 8192,
		};

		ret = syn_hal_dma_configure(0, &cfg);
		if (ret == 0) {
			ret = syn_hal_dma_start(0, probe_cb, NULL);
		}
		shell_print(sh, "round %d%s: cfg+start=%d", round,
			    pipelined ? " (pump-like)" : "", ret);
		if (ret != 0) {
			syn_hal_dma_dump(0);
			continue;
		}

		if (pipelined) {
			/* Stand-in for frame processing */
			volatile uint32_t acc = 0;

			for (uint32_t i = 0; i < 50000; i++) {
				acc += i;
			}
		}

		if (k_sem_take(&probe_sem, K_MSEC(300)) != 0) {
			shell_error(sh, "round %d: completion TIMEOUT",
				    round);
			syn_hal_dma_dump(0);
			continue;
		}
		shell_print(sh, "round %d: status=%d copy %s", round,
			    probe_status,
			    (memcmp(psrc, dst, 8192) == 0) ?
			    "OK" : "MISMATCH");
	}
	syn_hal_dma_dump(0);
	return 0;
}
#endif /* CONFIG_SOC_SERIES_MCXNX4X */

#ifdef CONFIG_SYNAPTIC_DMA_ARENA_PROBE
/* Phase 6.3 bench diagnostics, defined in the eDMA HAL */
int syn_dma_arena_probe(void);

static int cmd_dma_arena(const struct shell *sh, size_t argc, char **argv)
{
	ARG_UNUSED(argc);
	ARG_UNUSED(argv);

	shell_print(sh, "Arena eDMA experiments (output via printk; "
		    "resets the ephemeral arena)");
	return syn_dma_arena_probe();
}
#endif /* CONFIG_SYNAPTIC_DMA_ARENA_PROBE */

SHELL_STATIC_SUBCMD_SET_CREATE(sub_dma,
	SHELL_CMD_ARG(bench, NULL,
		      "Zero-copy ingest vs CPU copy: syn dma bench "
		      "[frames] [arena|static]", cmd_dma_bench, 1, 2),
#ifdef CONFIG_SOC_SERIES_MCXNX4X
	SHELL_CMD(probe, NULL, "eDMA bring-up probe: 3 transfers + regs",
		  cmd_dma_probe),
#endif
#ifdef CONFIG_SYNAPTIC_DMA_ARENA_PROBE
	SHELL_CMD(arena, NULL,
		  "Arena eDMA reachability experiments (canary+timeout)",
		  cmd_dma_arena),
#endif
	SHELL_SUBCMD_SET_END
);

SHELL_STATIC_SUBCMD_SET_CREATE(sub_infer,
	SHELL_CMD_ARG(run, &dsub_model_name,
		      "Run inference: syn infer run <model-name> "
		      "[be|normal|rt]",
		      cmd_infer_run, 2, 1),
	SHELL_CMD(stats, NULL, "Show scheduler counters", cmd_infer_stats),
	SHELL_SUBCMD_SET_END
);

#ifdef CONFIG_SYNAPTIC_MPU_PROTECT
SHELL_STATIC_SUBCMD_SET_CREATE(sub_mpu,
	SHELL_CMD(test, NULL,
		  "Verify cross-core MPU protection (provokes a fault)",
		  cmd_mpu_test),
	SHELL_SUBCMD_SET_END
);
#endif

#if defined(CONFIG_SYNAPTIC_DUAL_CORE) && !defined(CONFIG_SOC_MCXN947_CPU1)
SHELL_STATIC_SUBCMD_SET_CREATE(sub_ipc,
	SHELL_CMD(status, NULL, "Show CPU1 link status", cmd_ipc_status),
	SHELL_CMD(stats, NULL, "Show IPC ring message counters",
		  cmd_ipc_stats),
	SHELL_SUBCMD_SET_END
);
#endif

#ifdef CONFIG_SYNAPTIC_OTA
SHELL_STATIC_SUBCMD_SET_CREATE(sub_store,
	SHELL_CMD(status, NULL, "Show model store state", cmd_store_status),
	SHELL_CMD_ARG(activate, NULL,
		      "Hot-swap the active model: syn store activate <slot>",
		      cmd_store_activate, 2, 0),
	SHELL_SUBCMD_SET_END
);

SHELL_STATIC_SUBCMD_SET_CREATE(sub_ota,
	SHELL_CMD(status, NULL, "Show OTA session state", cmd_ota_status),
	SHELL_CMD_ARG(begin, NULL, "Start OTA: syn ota begin <name> <bytes>",
		      cmd_ota_begin, 3, 0),
	SHELL_CMD_ARG(data, NULL, "Feed a hex-encoded chunk",
		      cmd_ota_data, 2, 0),
	SHELL_CMD_ARG(rawdata, NULL,
		      "Receive raw binary bytes: syn ota rawdata <bytes>",
		      cmd_ota_rawdata, 2, 0),
	SHELL_CMD(done, NULL, "Finish transfer: validate + stage",
		  cmd_ota_done),
	SHELL_CMD(activate, NULL, "Activate the staged model",
		  cmd_ota_activate),
	SHELL_CMD(rollback, NULL, "Restore the previous model",
		  cmd_ota_rollback),
	SHELL_SUBCMD_SET_END
);
#endif

#ifdef CONFIG_SYNAPTIC_HEALTH
SHELL_STATIC_SUBCMD_SET_CREATE(sub_health,
	SHELL_CMD_ARG(hang, NULL,
		      "Fault injection: syn health hang <cpu0|cpu1>",
		      cmd_health_hang, 2, 0),
	SHELL_SUBCMD_SET_END
);
#endif

SHELL_STATIC_SUBCMD_SET_CREATE(sub_syn,
	SHELL_CMD(version, NULL, "Print SynapticOS version", cmd_version),
#ifdef CONFIG_SYNAPTIC_HEALTH
	SHELL_CMD(health, &sub_health, "Health monitor status", cmd_health),
#endif
	SHELL_CMD(mem, &sub_mem, "Memory management", NULL),
	SHELL_CMD(model, &sub_model, "Model management", NULL),
	SHELL_CMD(npu, &sub_npu, "NPU control", NULL),
	SHELL_CMD(dsp, &sub_dsp, "DSP operations", NULL),
	SHELL_CMD(dma, &sub_dma, "DMA operations", NULL),
	SHELL_CMD(infer, &sub_infer, "Inference control", NULL),
	SHELL_CMD(prof, &sub_prof, "Profiling", NULL),
#ifdef CONFIG_SYNAPTIC_MPU_PROTECT
	SHELL_CMD(mpu, &sub_mpu, "Cross-core memory protection", NULL),
#endif
#if defined(CONFIG_SYNAPTIC_DUAL_CORE) && !defined(CONFIG_SOC_MCXN947_CPU1)
	SHELL_CMD(ipc, &sub_ipc, "Inter-core communication", NULL),
#endif
#ifdef CONFIG_SYNAPTIC_OTA
	SHELL_CMD(store, &sub_store, "Model store", NULL),
	SHELL_CMD(ota, &sub_ota, "OTA model updates", NULL),
#endif
	SHELL_SUBCMD_SET_END
);

SHELL_CMD_REGISTER(syn, &sub_syn, "SynapticOS commands", NULL);
