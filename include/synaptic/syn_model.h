/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file syn_model.h
 * @brief SynapticOS — Model Lifecycle Management
 */
#ifndef SYNAPTIC_SYN_MODEL_H_
#define SYNAPTIC_SYN_MODEL_H_

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <synaptic/syn_hal_npu.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef uint32_t syn_model_handle_t;

#define SYN_MODEL_INVALID  ((syn_model_handle_t)0)

typedef struct {
    char             name[32];
    char             version[16];
    uint32_t         input_size;
    uint32_t         output_size;
    uint32_t         flash_offset;
    uint32_t         flash_size;
    uint32_t         sram_required;
    uint32_t         crc32;
    syn_npu_dtype_t  input_dtype;
    syn_npu_dtype_t  output_dtype;
    uint8_t          input_shape[4];
    uint8_t          output_shape[4];
} syn_model_info_t;

/* Registry operations */
int  syn_model_register(const syn_model_info_t *info,
                        syn_model_handle_t *handle);
int  syn_model_unregister(syn_model_handle_t handle);
int  syn_model_get_info(syn_model_handle_t handle, syn_model_info_t *info);
int  syn_model_get_by_name(const char *name, syn_model_handle_t *handle);
int  syn_model_list(syn_model_handle_t *handles, uint8_t *count,
                    uint8_t max);

/* Loading */
int  syn_model_load(syn_model_handle_t handle);
int  syn_model_unload(syn_model_handle_t handle);
bool syn_model_is_loaded(syn_model_handle_t handle);

/* Hot-swap */
int  syn_model_swap(syn_model_handle_t old_handle,
                    syn_model_handle_t new_handle);

#ifdef __cplusplus
}
#endif
#endif /* SYNAPTIC_SYN_MODEL_H_ */
