/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file syn_ingest.h
 * @brief SynapticOS - Double-buffered zero-copy frame ingest (private)
 *
 * Phase 5.3: pumps frames from a source buffer into two destination
 * buffers via the DMA HAL, overlapping each transfer with the
 * consumer's processing of the previous frame:
 *
 *   fill(src, N+1) -> DMA src -> buf[(N+1)%2]   (hardware)
 *   process(buf[N%2], N)                        (CPU, overlapped)
 *
 * The consumer reads the DMA-written buffer in place - no CPU copy
 * touches the frame path. The synthetic source stands in for a
 * camera until OV7670 bring-up (Phase 6); with the QEMU DMA stub
 * the overlap is functional only (the stub copies on the CPU).
 */
#ifndef SYNAPTIC_SYN_INGEST_H_
#define SYNAPTIC_SYN_INGEST_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Fill the source buffer with frame @p seq (synthetic sensor). */
typedef void (*syn_ingest_fill_fn_t)(void *src, size_t size,
				     uint32_t seq, void *user);

/**
 * Consume one frame in place. Runs on the pump caller's thread,
 * overlapped with the DMA of the next frame.
 */
typedef void (*syn_ingest_frame_fn_t)(const void *frame, size_t size,
				      uint32_t seq, void *user);

typedef struct {
	void *src;                     /**< Source (sensor) buffer      */
	void *bufs[2];                 /**< Destination ping/pong pair  */
	size_t frame_size;
	int dma_channel;               /**< syn_hal_dma channel to use  */
	syn_ingest_fill_fn_t fill;     /**< May be NULL (static source) */
	syn_ingest_frame_fn_t process; /**< Required                    */
	void *user;
} syn_ingest_config_t;

typedef struct {
	uint32_t frames;       /**< Frames delivered                    */
	uint32_t dma_errors;   /**< Failed transfers (pump aborts)      */
	uint32_t elapsed_us;   /**< Wall time of the whole pump run     */
} syn_ingest_stats_t;

/**
 * @brief Run the pump for @p frames frames (blocking).
 * @return 0 on success, -EINVAL on bad config, -EIO when a DMA
 *         transfer fails or times out (see stats).
 */
int syn_ingest_run(const syn_ingest_config_t *cfg, uint32_t frames);

/** @brief Stats of the most recent syn_ingest_run(). */
void syn_ingest_last_stats(syn_ingest_stats_t *stats);

#ifdef __cplusplus
}
#endif
#endif /* SYNAPTIC_SYN_INGEST_H_ */
