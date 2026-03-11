/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file syn_hal_dma.h
 * @brief SynapticOS — DMA/SmartDMA Hardware Abstraction
 */
#ifndef SYNAPTIC_SYN_HAL_DMA_H_
#define SYNAPTIC_SYN_HAL_DMA_H_

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    SYN_DMA_PERIPH_CAMERA,
    SYN_DMA_PERIPH_SPI,
    SYN_DMA_PERIPH_MEMORY,
} syn_dma_periph_t;

typedef struct {
    syn_dma_periph_t src_periph;
    syn_dma_periph_t dst_periph;
    void            *src_addr;
    void            *dst_addr;
    size_t           transfer_size;
    bool             circular;
} syn_dma_config_t;

typedef void (*syn_dma_cb_t)(int channel, int status, void *user_data);

int  syn_hal_dma_init(void);
int  syn_hal_dma_configure(int channel, const syn_dma_config_t *config);
int  syn_hal_dma_start(int channel, syn_dma_cb_t callback, void *user_data);
int  syn_hal_dma_stop(int channel);
int  syn_hal_dma_get_remaining(int channel, size_t *remaining);

#ifdef __cplusplus
}
#endif
#endif /* SYNAPTIC_SYN_HAL_DMA_H_ */
