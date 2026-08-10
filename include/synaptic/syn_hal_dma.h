/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file syn_hal_dma.h
 * @brief SynapticOS — DMA/SmartDMA Hardware Abstraction
 *
 * Backend-neutral DMA channel API used by the zero-copy ingest path.
 * On FRDM-MCXN947 the backend is eDMA (mem-to-mem); on QEMU a stub
 * backend performs asynchronous CPU copies preserving the callback
 * contract. Addresses handed to the DMA must be reachable by the DMA
 * engine (plain, non-secure aliases on MCXN947).
 */
#ifndef SYNAPTIC_SYN_HAL_DMA_H_
#define SYNAPTIC_SYN_HAL_DMA_H_

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/** DMA endpoint kind for a transfer descriptor. */
typedef enum {
    SYN_DMA_PERIPH_CAMERA,    /**< Camera interface (reserved, Phase 6) */
    SYN_DMA_PERIPH_SPI,       /**< SPI peripheral (reserved)            */
    SYN_DMA_PERIPH_MEMORY,    /**< Plain memory endpoint                */
} syn_dma_periph_t;

/** DMA transfer configuration. */
typedef struct {
    syn_dma_periph_t src_periph;      /**< Source endpoint kind          */
    syn_dma_periph_t dst_periph;      /**< Destination endpoint kind     */
    void            *src_addr;        /**< Source address                */
    void            *dst_addr;        /**< Destination address           */
    size_t           transfer_size;   /**< Bytes per transfer            */
    bool             circular;        /**< Re-arm automatically when done */
} syn_dma_config_t;

/**
 * @brief Transfer-completion callback.
 *
 * Invoked when a transfer (or one iteration of a circular transfer)
 * completes. May run in interrupt context — keep it short and do not
 * block.
 *
 * @param channel   Channel the transfer completed on.
 * @param status    0 on success, negative errno on transfer error.
 * @param user_data Pointer registered with syn_hal_dma_start().
 */
typedef void (*syn_dma_cb_t)(int channel, int status, void *user_data);

/**
 * @brief Initialize the DMA HAL backend.
 *
 * @return 0 on success, negative errno on failure.
 */
int  syn_hal_dma_init(void);

/**
 * @brief Configure a DMA channel for a transfer.
 *
 * Must be called before syn_hal_dma_start(). Only memory-to-memory
 * endpoints are supported until the camera path lands (-ENOTSUP for
 * peripheral endpoints).
 *
 * @param channel Channel index (0-based).
 * @param config  Transfer descriptor; copied by the call.
 *
 * @return 0 on success, -EINVAL on bad arguments, -ENOTSUP for
 *         unsupported endpoint kinds, negative errno otherwise.
 */
int  syn_hal_dma_configure(int channel, const syn_dma_config_t *config);

/**
 * @brief Start the configured transfer on a channel.
 *
 * Asynchronous: returns immediately and invokes @p callback on
 * completion (per iteration when circular).
 *
 * @param channel   Channel index previously configured.
 * @param callback  Completion callback (may be NULL for fire-and-forget).
 * @param user_data Opaque pointer passed to the callback.
 *
 * @return 0 on success, negative errno on failure.
 */
int  syn_hal_dma_start(int channel, syn_dma_cb_t callback, void *user_data);

/**
 * @brief Stop an in-flight or circular transfer.
 *
 * @param channel Channel index.
 *
 * @return 0 on success, negative errno on failure.
 */
int  syn_hal_dma_stop(int channel);

/**
 * @brief Query bytes remaining in the current transfer.
 *
 * @param channel   Channel index.
 * @param remaining Out: bytes not yet transferred (0 when idle/done).
 *
 * @return 0 on success, negative errno on failure.
 */
int  syn_hal_dma_get_remaining(int channel, size_t *remaining);

#ifdef __cplusplus
}
#endif
#endif /* SYNAPTIC_SYN_HAL_DMA_H_ */
