/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file syn_infer.h
 * @brief SynapticOS — Inference Engine and Pipeline Scheduler
 */
#ifndef SYNAPTIC_SYN_INFER_H_
#define SYNAPTIC_SYN_INFER_H_

#include <stdint.h>
#include <stdbool.h>
#include <synaptic/syn_mem.h>
#include <synaptic/syn_model.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    SYN_PRIORITY_BEST_EFFORT = 0,
    SYN_PRIORITY_NORMAL      = 1,
    SYN_PRIORITY_REALTIME    = 2,
} syn_priority_t;

typedef uint32_t syn_job_id_t;

#define SYN_JOB_INVALID  ((syn_job_id_t)0)

typedef void (*syn_infer_cb_t)(syn_job_id_t job, const syn_tensor_t *output,
                               void *user_data);

typedef struct {
    syn_priority_t  priority;
    uint32_t        deadline_us;
    bool            preemptible;
    syn_infer_cb_t  callback;
    void           *user_data;
} syn_infer_params_t;

/** Pre/post-processing function signatures */
typedef int (*syn_preprocess_fn_t)(const syn_tensor_t *in,
                                   syn_tensor_t *out, const void *config);
typedef int (*syn_postprocess_fn_t)(const syn_tensor_t *in,
                                    syn_tensor_t *out, const void *config);

/** Opaque pipeline handle */
typedef struct syn_pipeline syn_pipeline_t;

/* Pipeline construction */
syn_pipeline_t *syn_pipeline_create(const char *name);
int  syn_pipeline_add_preprocess(syn_pipeline_t *pipe,
                                 syn_preprocess_fn_t fn, void *config);
int  syn_pipeline_add_model(syn_pipeline_t *pipe,
                            syn_model_handle_t model);
int  syn_pipeline_add_postprocess(syn_pipeline_t *pipe,
                                  syn_postprocess_fn_t fn, void *config);
int  syn_pipeline_build(syn_pipeline_t *pipe);
void syn_pipeline_destroy(syn_pipeline_t *pipe);

/* Job submission */
syn_job_id_t syn_infer_submit(syn_pipeline_t *pipe,
                              const syn_tensor_t *input,
                              const syn_infer_params_t *params);
int  syn_infer_wait(syn_job_id_t job, uint32_t timeout_ms);
int  syn_infer_cancel(syn_job_id_t job);
int  syn_infer_get_result(syn_job_id_t job, syn_tensor_t *output);

/* Synchronous convenience */
int  syn_infer_run_sync(syn_model_handle_t model,
                        const syn_tensor_t *input,
                        syn_tensor_t *output,
                        syn_priority_t priority);

/* Scheduler control */
int  syn_infer_set_max_concurrent(uint8_t max_jobs);

/* Built-in pre-processors */
typedef struct { uint16_t w; uint16_t h; } syn_resize_config_t;

extern syn_preprocess_fn_t  syn_preprocess_image_resize;
extern syn_preprocess_fn_t  syn_preprocess_image_normalize;
extern syn_preprocess_fn_t  syn_preprocess_quantize_int8;
extern syn_preprocess_fn_t  syn_preprocess_audio_mfcc;

/* Built-in post-processors */
extern syn_postprocess_fn_t syn_postprocess_softmax;
extern syn_postprocess_fn_t syn_postprocess_argmax;
extern syn_postprocess_fn_t syn_postprocess_top_k;
extern syn_postprocess_fn_t syn_postprocess_nms;
extern syn_postprocess_fn_t syn_postprocess_dequantize;

#ifdef __cplusplus
}
#endif
#endif /* SYNAPTIC_SYN_INFER_H_ */
