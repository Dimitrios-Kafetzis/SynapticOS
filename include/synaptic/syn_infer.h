/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file syn_infer.h
 * @brief SynapticOS — Inference Engine and Pipeline Scheduler
 *
 * Pipelines chain optional pre-processing, one model stage and
 * optional post-processing. Jobs are dispatched by priority, then
 * earliest absolute deadline, then submission order; preemptible
 * layered jobs can be suspended at layer boundaries for
 * higher-priority work and resumed bit-exactly.
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

/** Job priority classes (higher value preempts lower). */
typedef enum {
    SYN_PRIORITY_BEST_EFFORT = 0,   /**< Run when nothing else waits  */
    SYN_PRIORITY_NORMAL      = 1,   /**< Default                      */
    SYN_PRIORITY_REALTIME    = 2,   /**< Preempts lower classes       */
} syn_priority_t;

/** Job identifier returned by syn_infer_submit(). */
typedef uint32_t syn_job_id_t;

#define SYN_JOB_INVALID  ((syn_job_id_t)0)

/**
 * @brief Job-completion callback.
 *
 * @param job       Completed job's id.
 * @param output    Job output tensor (valid during the call).
 * @param user_data Pointer from syn_infer_params_t.
 */
typedef void (*syn_infer_cb_t)(syn_job_id_t job, const syn_tensor_t *output,
                               void *user_data);

/** Submission parameters for one inference job. */
typedef struct {
    syn_priority_t  priority;     /**< Priority class                  */
    uint32_t        deadline_us;  /**< Relative deadline (0 = none);
                                       misses are counted, not
                                       enforced                        */
    bool            preemptible;  /**< Allow suspension at layer
                                       boundaries (layered models)     */
    syn_infer_cb_t  callback;     /**< Completion callback (optional)  */
    void           *user_data;    /**< Passed to the callback          */
} syn_infer_params_t;

/** Pre/post-processing function signatures */
typedef int (*syn_preprocess_fn_t)(const syn_tensor_t *in,
                                   syn_tensor_t *out, const void *config);
typedef int (*syn_postprocess_fn_t)(const syn_tensor_t *in,
                                    syn_tensor_t *out, const void *config);

/** Opaque pipeline handle */
typedef struct syn_pipeline syn_pipeline_t;

/* Pipeline construction */

/**
 * @brief Allocate a new, empty pipeline.
 *
 * @param name Display name (copied; used in logs and profiling).
 *
 * @return Pipeline handle, or NULL if no pipeline slot is free.
 */
syn_pipeline_t *syn_pipeline_create(const char *name);

/**
 * @brief Append a pre-processing stage.
 *
 * Stages run in the order added, before the model stage. See
 * syn_process.h for the built-in processors' config structures.
 *
 * @param pipe   Pipeline under construction (not yet built).
 * @param fn     Stage function.
 * @param config Opaque config passed to @p fn on every run.
 *
 * @return 0 on success, -EINVAL on bad arguments, -EPERM after
 *         build, -ENOMEM if the stage table is full.
 */
int  syn_pipeline_add_preprocess(syn_pipeline_t *pipe,
                                 syn_preprocess_fn_t fn, void *config);

/**
 * @brief Set the pipeline's model stage.
 *
 * @param pipe  Pipeline under construction (not yet built).
 * @param model Registered model to execute.
 *
 * @return 0 on success, -EINVAL on bad arguments, -EPERM after
 *         build, -EALREADY if a model stage is already set.
 */
int  syn_pipeline_add_model(syn_pipeline_t *pipe,
                            syn_model_handle_t model);

/**
 * @brief Append a post-processing stage.
 *
 * Stages run in the order added, after the model stage.
 *
 * @param pipe   Pipeline under construction (not yet built).
 * @param fn     Stage function.
 * @param config Opaque config passed to @p fn on every run.
 *
 * @return 0 on success, -EINVAL on bad arguments, -EPERM after
 *         build, -ENOMEM if the stage table is full.
 */
int  syn_pipeline_add_postprocess(syn_pipeline_t *pipe,
                                  syn_postprocess_fn_t fn, void *config);

/**
 * @brief Finalize a pipeline; it can then accept job submissions.
 *
 * @param pipe Pipeline with at least a model stage.
 *
 * @return 0 on success, -EINVAL on bad/incomplete pipeline,
 *         -EALREADY if already built.
 */
int  syn_pipeline_build(syn_pipeline_t *pipe);

/**
 * @brief Destroy a pipeline and release its slot.
 *
 * Pending jobs on the pipeline are cancelled (suspended jobs have
 * their context slots freed).
 *
 * @param pipe Pipeline to destroy (NULL is ignored).
 */
void syn_pipeline_destroy(syn_pipeline_t *pipe);

/* Job submission */

/**
 * @brief Submit an inference job.
 *
 * Asynchronous: the job runs on the scheduler thread. Retrieve the
 * output with syn_infer_wait() + syn_infer_get_result(), or via the
 * completion callback.
 *
 * @param pipe   Built pipeline.
 * @param input  Input tensor (data is consumed during the job).
 * @param params Priority/deadline/callback options (NULL for
 *               defaults).
 *
 * @return Job id, or SYN_JOB_INVALID on bad arguments or if no job
 *         slot is free.
 */
syn_job_id_t syn_infer_submit(syn_pipeline_t *pipe,
                              const syn_tensor_t *input,
                              const syn_infer_params_t *params);

/**
 * @brief Block until a job completes (or fails/cancels).
 *
 * @param job        Job id from syn_infer_submit().
 * @param timeout_ms Maximum wait in milliseconds.
 *
 * @return 0 when the job finished, -ENOENT for an unknown id,
 *         negative errno on timeout or job failure.
 */
int  syn_infer_wait(syn_job_id_t job, uint32_t timeout_ms);

/**
 * @brief Cancel a queued or suspended job.
 *
 * A job that is actively running is not interrupted mid-layer.
 *
 * @param job Job id.
 *
 * @return 0 on success, -ENOENT for an unknown id, negative errno if
 *         the job can no longer be cancelled.
 */
int  syn_infer_cancel(syn_job_id_t job);

/**
 * @brief Fetch a completed job's output and release its slot.
 *
 * Job slots are recycled by this call — every submitted job should
 * eventually be reaped with it.
 *
 * @param job    Job id of a completed job.
 * @param output Out: filled with the output tensor descriptor.
 *
 * @return 0 on success, -EINVAL on NULL output, -ENOENT for an
 *         unknown id, negative errno if the job is not complete.
 */
int  syn_infer_get_result(syn_job_id_t job, syn_tensor_t *output);

/* Synchronous convenience */

/**
 * @brief Run one inference synchronously on a bare model.
 *
 * Wraps submit + wait + get_result on an internal single-model
 * pipeline.
 *
 * @param model    Registered (and loadable) model.
 * @param input    Input tensor.
 * @param output   Out: output tensor descriptor.
 * @param priority Priority class for the job.
 *
 * @return 0 on success, negative errno on failure.
 */
int  syn_infer_run_sync(syn_model_handle_t model,
                        const syn_tensor_t *input,
                        syn_tensor_t *output,
                        syn_priority_t priority);

/* Scheduler control */

/**
 * @brief Limit how many jobs may run concurrently.
 *
 * @param max_jobs 1..CONFIG_SYNAPTIC_MAX_CONCURRENT_JOBS.
 *
 * @return 0 on success, -EINVAL out of range.
 */
int  syn_infer_set_max_concurrent(uint8_t max_jobs);

/* Built-in pre-processors */

/** syn_preprocess_image_resize config: target width/height. */
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
