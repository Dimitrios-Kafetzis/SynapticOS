/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file syn_hal_npu.h
 * @brief SynapticOS — NPU Hardware Abstraction Layer
 */
#ifndef SYNAPTIC_SYN_HAL_NPU_H_
#define SYNAPTIC_SYN_HAL_NPU_H_

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    SYN_NPU_STATE_IDLE,
    SYN_NPU_STATE_BUSY,
    SYN_NPU_STATE_ERROR,
    SYN_NPU_STATE_SUSPENDED,
} syn_npu_state_t;

typedef enum {
    SYN_NPU_DTYPE_INT8,
    SYN_NPU_DTYPE_UINT8,
    SYN_NPU_DTYPE_INT16,
    SYN_NPU_DTYPE_FLOAT16,
    SYN_NPU_DTYPE_FLOAT32,
} syn_npu_dtype_t;

typedef struct {
    const char *name;
    uint32_t    max_ops_per_sec;
    uint32_t    scratch_size;
    uint8_t     supported_dtypes;
    bool        supports_async;
} syn_npu_caps_t;

typedef void (*syn_npu_done_cb_t)(int status, void *user_data);

/* Lifecycle */
int  syn_hal_npu_init(void);
void syn_hal_npu_deinit(void);
int  syn_hal_npu_get_caps(syn_npu_caps_t *caps);
syn_npu_state_t syn_hal_npu_get_state(void);

/* Execution */
int  syn_hal_npu_load_model(const uint8_t *model_data, size_t model_size);
int  syn_hal_npu_set_input(uint8_t index, const void *data, size_t size);
int  syn_hal_npu_invoke(void);
int  syn_hal_npu_invoke_async(syn_npu_done_cb_t cb, void *user_data);
int  syn_hal_npu_get_output(uint8_t index, void *data, size_t *size);

/* Power management */
int  syn_hal_npu_suspend(void);
int  syn_hal_npu_resume(void);

#ifdef __cplusplus
}
#endif
#endif /* SYNAPTIC_SYN_HAL_NPU_H_ */
