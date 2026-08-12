/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file syn_infer.c
 * @brief SynapticOS — Inference Pipeline Engine and Job Scheduler
 *
 * Pipeline construction (Phase 2.1):
 *   Pipelines are drawn from a static pool and hold an ordered list of
 *   stages (preprocess* -> model -> postprocess*). Canonical ordering is
 *   enforced at add-time; build() validates and computes a worst-case
 *   memory estimate.
 *
 * Job scheduler (Phase 2.2, extended in 5.1):
 *   Fixed job table of CONFIG_SYNAPTIC_MAX_CONCURRENT_JOBS entries. A
 *   dedicated scheduler thread dequeues jobs in priority order,
 *   earliest-deadline-first within the same priority (FIFO on ties),
 *   executes the pipeline stages sequentially, and signals completion
 *   through a per-job semaphore. Completions later than deadline_us
 *   after submission count as deadline misses.
 *
 * Layer-boundary preemption (Phase 5.1, layered models only):
 *   When the running job is preemptible and executes a layered model
 *   (see syn_npu_layered.h), the scheduler checks for queued
 *   higher-priority jobs between layers. On preemption the layered
 *   context is parked in a suspension slot, the job moves to
 *   JOB_SUSPENDED, and it resumes bit-exactly from the saved context
 *   once it is the best runnable job again. Suspended jobs keep
 *   their pipeline input and model pointer valid; unloading or
 *   overwriting that model while suspended is undefined (the quiesce
 *   gate only drains the RUNNING job).
 *
 * Stage buffer convention:
 *   Each stage output is an ephemeral arena tensor. Its capacity is
 *   MAX(4 * input_size, 64) bytes (worst case: uint8 -> float32
 *   expansion), additionally raised to the model input size for the
 *   stage feeding the model. Stage functions receive the capacity in
 *   out->size and must set the final geometry (shape/ndim/dtype/size)
 *   before returning. Results live in the ephemeral region: they are
 *   valid until syn_mem_reset_ephemeral() is called.
 */

#include <zephyr/kernel.h>
#include <zephyr/init.h>
#include <zephyr/logging/log.h>

LOG_MODULE_REGISTER(syn_infer, CONFIG_SYNAPTIC_LOG_LEVEL);

#include <synaptic/syn_infer.h>
#include <synaptic/syn_process.h>
#include <synaptic/syn_hal_npu.h>
#include <string.h>

#include "syn_prof_internal.h"
#include "syn_infer_internal.h"
#include "syn_health.h"

#ifdef CONFIG_SYNAPTIC_LAYER_EXEC
#include "../hal/common/syn_npu_layered.h"
#endif

/* Health check-in period while executing a job (5.4). Layered jobs
 * kick at every layer boundary; a monolithic invoke must simply
 * finish inside this window.
 */
#define SCHED_HEALTH_PERIOD_MS 5000

static int sched_health_id = -1;

#define SYN_MAX_PIPELINES      4
#define STAGE_MIN_CAPACITY     64
#define SCHED_STACK_SIZE       2048
#define SCHED_THREAD_PRIO      K_PRIO_PREEMPT(8)

enum stage_type {
	STAGE_PREPROCESS,
	STAGE_MODEL,
	STAGE_POSTPROCESS,
};

struct pipeline_stage {
	enum stage_type type;
	union {
		syn_preprocess_fn_t  pre;
		syn_postprocess_fn_t post;
	} fn;
	void *config;
};

struct syn_pipeline {
	bool in_use;
	bool built;
	char name[24];
	uint8_t num_stages;
	syn_model_handle_t model;
	struct pipeline_stage stages[CONFIG_SYNAPTIC_MAX_PIPELINE_STAGES];
	size_t mem_estimate;
};

enum job_state {
	JOB_FREE = 0,
	JOB_QUEUED,
	JOB_RUNNING,
	JOB_SUSPENDED,
	JOB_DONE,
	JOB_ERROR,
	JOB_CANCELLED,
};

struct infer_job {
	syn_job_id_t id;
	enum job_state state;
	syn_pipeline_t *pipe;
	const syn_tensor_t *input;
	syn_infer_params_t params;
	syn_tensor_t output;
	int result;
	uint32_t seq;
	int64_t deadline_at_us;  /* Absolute deadline, INT64_MAX = none */
	uint32_t start_cyc;      /* Submission time for deadline check  */
#ifdef CONFIG_SYNAPTIC_LAYER_EXEC
	int8_t ctx_slot;         /* Suspension slot, -1 = none          */
	uint8_t resume_stage;    /* Pipeline stage to resume at         */
	bool resuming;
#endif
	struct k_sem done;
};

static struct syn_pipeline pipelines[SYN_MAX_PIPELINES];
static struct infer_job jobs[CONFIG_SYNAPTIC_MAX_CONCURRENT_JOBS];

static K_MUTEX_DEFINE(infer_lock);
static K_SEM_DEFINE(sched_wake, 0, CONFIG_SYNAPTIC_MAX_CONCURRENT_JOBS);

static uint32_t next_job_id = 1;
static uint32_t next_seq;
static uint8_t max_concurrent = CONFIG_SYNAPTIC_MAX_CONCURRENT_JOBS;

/* Quiesce gate (4.3): pauses dispatch for NPU residency changes. */
static bool dispatch_paused;
static bool job_running;
static uint32_t deferred_wakes;
static K_SEM_DEFINE(drain_sem, 0, 1);

/* Scheduler counters (5.1); guarded by infer_lock. */
static syn_infer_stats_t stats;

void syn_infer_get_stats(syn_infer_stats_t *out)
{
	if (out == NULL) {
		return;
	}
	k_mutex_lock(&infer_lock, K_FOREVER);
	*out = stats;
	k_mutex_unlock(&infer_lock);
}

void syn_infer_reset_stats(void)
{
	k_mutex_lock(&infer_lock, K_FOREVER);
	memset(&stats, 0, sizeof(stats));
	k_mutex_unlock(&infer_lock);
}

/* ------------------------------------------------------------------ */
/* Pipeline construction (2.1)                                        */
/* ------------------------------------------------------------------ */

syn_pipeline_t *syn_pipeline_create(const char *name)
{
	k_mutex_lock(&infer_lock, K_FOREVER);

	for (int i = 0; i < SYN_MAX_PIPELINES; i++) {
		if (!pipelines[i].in_use) {
			syn_pipeline_t *p = &pipelines[i];

			memset(p, 0, sizeof(*p));
			p->in_use = true;
			strncpy(p->name, (name != NULL) ? name : "unnamed",
				sizeof(p->name) - 1);
			k_mutex_unlock(&infer_lock);
			LOG_DBG("Pipeline '%s' created", p->name);
			return p;
		}
	}

	k_mutex_unlock(&infer_lock);
	LOG_ERR("Pipeline pool exhausted (%d slots)", SYN_MAX_PIPELINES);
	return NULL;
}

static bool pipeline_valid(const syn_pipeline_t *pipe)
{
	return pipe != NULL && pipe >= &pipelines[0] &&
	       pipe < &pipelines[SYN_MAX_PIPELINES] && pipe->in_use;
}

static int pipeline_add_stage(syn_pipeline_t *pipe, enum stage_type type,
			      void *fn, void *config)
{
	if (!pipeline_valid(pipe) || fn == NULL) {
		return -EINVAL;
	}
	if (pipe->built) {
		return -EPERM;
	}
	if (pipe->num_stages >= CONFIG_SYNAPTIC_MAX_PIPELINE_STAGES) {
		return -ENOMEM;
	}

	/* Enforce canonical ordering: preprocess* -> model -> postprocess* */
	bool has_model = (pipe->model != SYN_MODEL_INVALID);

	if (type == STAGE_PREPROCESS && has_model) {
		LOG_ERR("'%s': preprocess must come before the model stage",
			pipe->name);
		return -EINVAL;
	}
	if (type == STAGE_POSTPROCESS && !has_model) {
		LOG_ERR("'%s': postprocess requires a model stage first",
			pipe->name);
		return -EINVAL;
	}

	struct pipeline_stage *s = &pipe->stages[pipe->num_stages];

	s->type = type;
	s->config = config;
	if (type == STAGE_POSTPROCESS) {
		s->fn.post = (syn_postprocess_fn_t)fn;
	} else {
		s->fn.pre = (syn_preprocess_fn_t)fn;
	}
	pipe->num_stages++;
	return 0;
}

int syn_pipeline_add_preprocess(syn_pipeline_t *pipe,
				syn_preprocess_fn_t fn, void *config)
{
	return pipeline_add_stage(pipe, STAGE_PREPROCESS, (void *)fn, config);
}

int syn_pipeline_add_model(syn_pipeline_t *pipe, syn_model_handle_t model)
{
	if (!pipeline_valid(pipe) || model == SYN_MODEL_INVALID) {
		return -EINVAL;
	}
	if (pipe->built) {
		return -EPERM;
	}
	if (pipe->model != SYN_MODEL_INVALID) {
		LOG_ERR("'%s': only one model stage supported", pipe->name);
		return -EALREADY;
	}

	syn_model_info_t info;

	if (syn_model_get_info(model, &info) != 0) {
		LOG_ERR("'%s': model handle %u not registered",
			pipe->name, model);
		return -ENOENT;
	}
	if (pipe->num_stages >= CONFIG_SYNAPTIC_MAX_PIPELINE_STAGES) {
		return -ENOMEM;
	}

	struct pipeline_stage *s = &pipe->stages[pipe->num_stages];

	s->type = STAGE_MODEL;
	s->fn.pre = NULL;
	s->config = NULL;
	pipe->num_stages++;
	pipe->model = model;
	return 0;
}

int syn_pipeline_add_postprocess(syn_pipeline_t *pipe,
				 syn_postprocess_fn_t fn, void *config)
{
	return pipeline_add_stage(pipe, STAGE_POSTPROCESS, (void *)fn, config);
}

/** Worst-case output capacity for a processing stage (see file header). */
static size_t stage_capacity(size_t in_size)
{
	size_t cap = in_size * 4;

	return (cap < STAGE_MIN_CAPACITY) ? STAGE_MIN_CAPACITY : cap;
}

int syn_pipeline_build(syn_pipeline_t *pipe)
{
	if (!pipeline_valid(pipe)) {
		return -EINVAL;
	}
	if (pipe->built) {
		return -EALREADY;
	}
	if (pipe->model == SYN_MODEL_INVALID) {
		LOG_ERR("'%s': pipeline has no model stage", pipe->name);
		return -EINVAL;
	}

	syn_model_info_t info;
	int ret = syn_model_get_info(pipe->model, &info);

	if (ret != 0) {
		return ret;
	}

	/*
	 * Memory estimate: worst-case sum of intermediate stage buffers
	 * (each preprocess output can expand 4x, the model output is
	 * info.output_size, each postprocess output can expand 4x),
	 * plus the model's own SRAM requirement.
	 */
	size_t estimate = info.sram_required;
	size_t cur = info.input_size;

	for (uint8_t i = 0; i < pipe->num_stages; i++) {
		switch (pipe->stages[i].type) {
		case STAGE_PREPROCESS:
			cur = stage_capacity(cur);
			if (cur < info.input_size) {
				cur = info.input_size;
			}
			estimate += cur;
			break;
		case STAGE_MODEL:
			cur = info.output_size;
			estimate += cur;
			break;
		case STAGE_POSTPROCESS:
			cur = stage_capacity(cur);
			estimate += cur;
			break;
		}
	}

	pipe->mem_estimate = estimate;
	pipe->built = true;
	LOG_INF("Pipeline '%s' built: %u stages, est. %u bytes",
		pipe->name, pipe->num_stages, (unsigned)estimate);
	return 0;
}

void syn_pipeline_destroy(syn_pipeline_t *pipe)
{
	if (!pipeline_valid(pipe)) {
		return;
	}

	k_mutex_lock(&infer_lock, K_FOREVER);

	/* Cancel jobs still referencing this pipeline */
	for (int i = 0; i < CONFIG_SYNAPTIC_MAX_CONCURRENT_JOBS; i++) {
		bool waiting = (jobs[i].state == JOB_QUEUED);

#ifdef CONFIG_SYNAPTIC_LAYER_EXEC
		if (jobs[i].state == JOB_SUSPENDED) {
			syn_npu_layered_slot_free(jobs[i].ctx_slot);
			jobs[i].ctx_slot = -1;
			waiting = true;
		}
#endif
		if (waiting && jobs[i].pipe == pipe) {
			jobs[i].state = JOB_CANCELLED;
			jobs[i].result = -ECANCELED;
			stats.cancelled++;
			k_sem_give(&jobs[i].done);
			LOG_WRN("Job %u cancelled: pipeline '%s' destroyed",
				jobs[i].id, pipe->name);
		}
	}

	pipe->in_use = false;
	pipe->built = false;
	pipe->num_stages = 0;
	pipe->model = SYN_MODEL_INVALID;
	k_mutex_unlock(&infer_lock);
	LOG_DBG("Pipeline '%s' destroyed", pipe->name);
}

/* ------------------------------------------------------------------ */
/* Pipeline execution                                                  */
/* ------------------------------------------------------------------ */

/**
 * Exact output size for the built-in processors, computed from the
 * stage config and the runtime input tensor. Returns 0 for custom
 * (unknown) stage functions, which then fall back to the generic
 * 4x worst-case heuristic.
 */
static size_t builtin_stage_capacity(const struct pipeline_stage *s,
				     const syn_tensor_t *in)
{
	if (s->type == STAGE_PREPROCESS) {
		syn_preprocess_fn_t fn = s->fn.pre;

		if (fn == syn_preprocess_image_resize && s->config != NULL) {
			const syn_resize_config_t *c = s->config;
			uint32_t ch = (in->ndim == 4) ? in->shape[3] : 1;

			return (size_t)c->w * c->h * ch;
		}
		if (fn == syn_preprocess_image_normalize) {
			return in->size * sizeof(float);
		}
		if (fn == syn_preprocess_quantize_int8) {
			return in->size / sizeof(float);
		}
		if (fn == syn_preprocess_audio_mfcc && s->config != NULL) {
			const syn_mfcc_config_t *c = s->config;
			size_t frames = (in->size / sizeof(float)) /
					c->frame_len;

			return frames * c->num_coeffs * sizeof(float);
		}
	} else if (s->type == STAGE_POSTPROCESS) {
		syn_postprocess_fn_t fn = s->fn.post;

		if (fn == syn_postprocess_softmax) {
			return (in->dtype == SYN_NPU_DTYPE_FLOAT32) ?
				in->size : in->size * sizeof(float);
		}
		if (fn == syn_postprocess_argmax) {
			return sizeof(syn_classification_t);
		}
		if (fn == syn_postprocess_top_k && s->config != NULL) {
			const syn_topk_config_t *c = s->config;

			return (size_t)c->k * sizeof(syn_classification_t);
		}
		if (fn == syn_postprocess_nms) {
			return in->size;
		}
		if (fn == syn_postprocess_dequantize) {
			return in->size * sizeof(float);
		}
	}

	return 0;
}

/** Allocate an ephemeral stage output tensor with the given capacity. */
static syn_tensor_t *alloc_stage_output(size_t capacity)
{
	uint32_t shape[1] = { (uint32_t)capacity };

	return syn_mem_tensor_alloc(shape, 1, SYN_NPU_DTYPE_INT8,
				    SYN_MEM_EPHEMERAL);
}

#ifdef CONFIG_SYNAPTIC_LAYER_EXEC
/** True when a queued job outranks the running one (dispatch order). */
static bool preempt_pending(const struct infer_job *cur)
{
	bool found = false;

	k_mutex_lock(&infer_lock, K_FOREVER);
	for (int i = 0; i < CONFIG_SYNAPTIC_MAX_CONCURRENT_JOBS; i++) {
		if (jobs[i].state == JOB_QUEUED &&
		    jobs[i].params.priority > cur->params.priority) {
			found = true;
			break;
		}
	}
	k_mutex_unlock(&infer_lock);
	return found;
}

/**
 * Layer-by-layer execution with preemption checks at layer
 * boundaries. Returns 1 when the model completed (output produced),
 * 0 when the resident model is not layered (caller falls back to the
 * monolithic path), -EINTR when the job was preempted and its
 * context parked, other negatives on error.
 */
static int execute_layered(struct infer_job *job,
			   const syn_model_info_t *info,
			   const syn_tensor_t *in, syn_tensor_t **out)
{
	int rem;

	if (job->resuming) {
		job->resuming = false;
		rem = syn_npu_layered_restore(job->ctx_slot);
		if (rem != 0) {
			return rem;
		}
		job->ctx_slot = -1;
		rem = syn_npu_layered_remaining();
	} else {
		rem = syn_npu_layered_begin(in->data, in->size);
		if (rem <= 0) {
			return rem; /* 0: not layered; <0: bad input */
		}
	}

	while (rem > 0) {
		rem = syn_npu_layered_step();
		if (rem < 0) {
			return rem;
		}
		syn_health_kick(sched_health_id);
		if (rem > 0 && job->params.preemptible &&
		    preempt_pending(job)) {
			uint32_t t0 = k_cycle_get_32();
			int slot = syn_npu_layered_save();

			if (slot < 0) {
				/* No free slot: keep running */
				LOG_WRN("Job %u: no suspension slot, "
					"finishing uninterrupted", job->id);
				continue;
			}

			uint32_t save_us =
				k_cyc_to_us_ceil32(k_cycle_get_32() - t0);

			k_mutex_lock(&infer_lock, K_FOREVER);
			job->ctx_slot = (int8_t)slot;
			stats.last_save_us = save_us;
			if (save_us > stats.max_save_us) {
				stats.max_save_us = save_us;
			}
			k_mutex_unlock(&infer_lock);
			return -EINTR;
		}
	}

	size_t cap = info->output_size;

	if (cap < STAGE_MIN_CAPACITY) {
		cap = STAGE_MIN_CAPACITY;
	}

	syn_tensor_t *t = alloc_stage_output(cap);

	if (t == NULL) {
		return -ENOMEM;
	}

	size_t out_size = t->size;
	int ret = syn_npu_layered_output(t->data, &out_size);

	if (ret != 0) {
		return ret;
	}

	t->size = out_size;
	t->dtype = info->output_dtype;
	t->ndim = 1;
	t->shape[0] = (uint32_t)out_size;
	*out = t;
	return 1;
}
#endif /* CONFIG_SYNAPTIC_LAYER_EXEC */

static int execute_model_stage(struct infer_job *job,
			       const syn_model_info_t *info,
			       const syn_tensor_t *in, syn_tensor_t **out)
{
	int ret;

#ifdef CONFIG_SYNAPTIC_LAYER_EXEC
	ret = execute_layered(job, info, in, out);
	if (ret != 0) {
		return (ret == 1) ? 0 : ret;
	}
	/* Not a layered model: monolithic HAL invoke below */
#else
	ARG_UNUSED(job);
#endif

	ret = syn_hal_npu_set_input(0, in->data, in->size);

	if (ret != 0) {
		LOG_ERR("NPU set_input failed: %d", ret);
		return ret;
	}

	ret = syn_hal_npu_invoke();
	if (ret != 0) {
		LOG_ERR("NPU invoke failed: %d", ret);
		return ret;
	}

	size_t cap = info->output_size;

	if (cap < STAGE_MIN_CAPACITY) {
		cap = STAGE_MIN_CAPACITY;
	}

	syn_tensor_t *t = alloc_stage_output(cap);

	if (t == NULL) {
		return -ENOMEM;
	}

	size_t out_size = t->size;

	ret = syn_hal_npu_get_output(0, t->data, &out_size);
	if (ret != 0) {
		LOG_ERR("NPU get_output failed: %d", ret);
		return ret;
	}

	t->size = out_size;
	t->dtype = info->output_dtype;
	t->ndim = 1;
	t->shape[0] = (uint32_t)out_size;
	*out = t;
	return 0;
}

static int execute_pipeline(struct infer_job *job)
{
	syn_pipeline_t *pipe = job->pipe;
	syn_model_info_t info;
	int ret = syn_model_get_info(pipe->model, &info);

	if (ret != 0) {
		return ret;
	}

	const syn_tensor_t *cur = job->input;
	bool npu_marked = false;
	uint8_t first_stage = 0;

#ifdef CONFIG_SYNAPTIC_LAYER_EXEC
	if (job->resuming) {
		/* Continue inside the model stage; earlier stages already
		 * ran and their results live in the saved layer context.
		 * Profiling restarts here, so the stage breakdown of a
		 * preempted job covers only its final leg.
		 */
		first_stage = job->resume_stage;
	}
#endif

	syn_prof_mark_start();

	for (uint8_t i = first_stage; i < pipe->num_stages; i++) {
		struct pipeline_stage *s = &pipe->stages[i];
		syn_tensor_t *out = NULL;

		switch (s->type) {
		case STAGE_PREPROCESS: {
			size_t cap = builtin_stage_capacity(s, cur);

			if (cap == 0) {
				cap = stage_capacity(cur->size);
			}

			/* The stage feeding the model must be able to hold
			 * a full model input.
			 */
			bool feeds_model =
				(i + 1 < pipe->num_stages) &&
				(pipe->stages[i + 1].type == STAGE_MODEL);

			if (feeds_model && cap < info.input_size) {
				cap = info.input_size;
			}

			out = alloc_stage_output(cap);
			if (out == NULL) {
				return -ENOMEM;
			}
			ret = s->fn.pre(cur, out, s->config);
			break;
		}
		case STAGE_MODEL:
			syn_prof_mark_preprocess_done();
			ret = execute_model_stage(job, &info, cur, &out);
#ifdef CONFIG_SYNAPTIC_LAYER_EXEC
			if (ret == -EINTR) {
				job->resume_stage = i;
				return -EINTR;
			}
#endif
			syn_prof_mark_npu_done();
			npu_marked = true;
			break;
		case STAGE_POSTPROCESS: {
			size_t cap = builtin_stage_capacity(s, cur);

			if (cap == 0) {
				cap = stage_capacity(cur->size);
			}

			out = alloc_stage_output(cap);
			if (out == NULL) {
				return -ENOMEM;
			}
			ret = s->fn.post(cur, out, s->config);
			break;
		}
		}

		if (ret != 0) {
			LOG_ERR("'%s': stage %u failed: %d",
				pipe->name, i, ret);
			return ret;
		}
		cur = out;
	}

	if (!npu_marked) {
		syn_prof_mark_preprocess_done();
		syn_prof_mark_npu_done();
	}
	syn_prof_mark_end();

	job->output = *cur;
	return 0;
}

/* ------------------------------------------------------------------ */
/* Job scheduler (2.2)                                                 */
/* ------------------------------------------------------------------ */

static struct infer_job *find_job(syn_job_id_t id)
{
	if (id == SYN_JOB_INVALID) {
		return NULL;
	}
	for (int i = 0; i < CONFIG_SYNAPTIC_MAX_CONCURRENT_JOBS; i++) {
		if (jobs[i].state != JOB_FREE && jobs[i].id == id) {
			return &jobs[i];
		}
	}
	return NULL;
}

/**
 * Dispatch order: priority first, earliest absolute deadline within
 * a priority (jobs without a deadline sort last), FIFO on ties.
 */
static bool job_before(const struct infer_job *a, const struct infer_job *b)
{
	if (a->params.priority != b->params.priority) {
		return a->params.priority > b->params.priority;
	}
	if (a->deadline_at_us != b->deadline_at_us) {
		return a->deadline_at_us < b->deadline_at_us;
	}
	return a->seq < b->seq;
}

/** Pick the best runnable job: queued or suspended. */
static struct infer_job *pick_next_job(void)
{
	struct infer_job *best = NULL;

	for (int i = 0; i < CONFIG_SYNAPTIC_MAX_CONCURRENT_JOBS; i++) {
		struct infer_job *j = &jobs[i];

		if (j->state != JOB_QUEUED && j->state != JOB_SUSPENDED) {
			continue;
		}
		if (best == NULL || job_before(j, best)) {
			best = j;
		}
	}
	return best;
}

static void scheduler_thread(void *p1, void *p2, void *p3)
{
	ARG_UNUSED(p1);
	ARG_UNUSED(p2);
	ARG_UNUSED(p3);

	while (1) {
		syn_health_set_busy(sched_health_id, false);
		k_sem_take(&sched_wake, K_FOREVER);
		syn_health_set_busy(sched_health_id, true);

		k_mutex_lock(&infer_lock, K_FOREVER);

		if (dispatch_paused) {
			/* Residency change in progress: leave the job
			 * queued and replay this wakeup on release.
			 */
			deferred_wakes++;
			k_mutex_unlock(&infer_lock);
			continue;
		}

		struct infer_job *job = pick_next_job();

		if (job == NULL) {
			/* Job was cancelled between wake and dispatch */
			k_mutex_unlock(&infer_lock);
			continue;
		}
#ifdef CONFIG_SYNAPTIC_LAYER_EXEC
		if (job->state == JOB_SUSPENDED) {
			job->resuming = true;
			stats.resumes++;
		}
#endif
		job->state = JOB_RUNNING;
		job_running = true;
		k_mutex_unlock(&infer_lock);

		LOG_DBG("Job %u ('%s', prio %d) running",
			job->id, job->pipe->name, job->params.priority);

		int ret = execute_pipeline(job);

#ifdef CONFIG_SYNAPTIC_LAYER_EXEC
		if (ret == -EINTR) {
			k_mutex_lock(&infer_lock, K_FOREVER);
			job->state = JOB_SUSPENDED;
			job_running = false;
			stats.preemptions++;

			bool drain = dispatch_paused;

			k_mutex_unlock(&infer_lock);

			LOG_DBG("Job %u suspended at a layer boundary",
				job->id);
			if (drain) {
				/* quiesce() only waits for the RUNNING job */
				k_sem_give(&drain_sem);
			}
			/* One extra wakeup pays for the eventual resume */
			k_sem_give(&sched_wake);
			continue;
		}
#endif

		/* Snapshot completion data under the lock: as soon as the
		 * done semaphore is given, the waiter may consume the result
		 * and the slot can be reused by a new submission.
		 */
		k_mutex_lock(&infer_lock, K_FOREVER);
		job->result = ret;
		job->state = (ret == 0) ? JOB_DONE : JOB_ERROR;
		job_running = false;

		if (ret == 0) {
			stats.completed++;
		} else {
			stats.errors++;
		}
		if (job->params.deadline_us != 0U) {
			uint32_t elapsed_us = k_cyc_to_us_ceil32(
				k_cycle_get_32() - job->start_cyc);

			if (elapsed_us > job->params.deadline_us) {
				stats.deadline_misses++;
				LOG_WRN("Job %u missed deadline: %u > %u us",
					job->id, elapsed_us,
					job->params.deadline_us);
			}
		}

		syn_infer_cb_t cb = job->params.callback;
		void *user_data = job->params.user_data;
		syn_job_id_t id = job->id;
		syn_tensor_t output = job->output;
		bool drain = dispatch_paused;

		k_mutex_unlock(&infer_lock);

		if (drain) {
			/* a quiesce() is waiting for this job */
			k_sem_give(&drain_sem);
		}

		/* Callback fires before the done semaphore so completions
		 * are fully reported before any waiter resumes.
		 */
		if (cb != NULL) {
			cb(id, (ret == 0) ? &output : NULL, user_data);
		}

		k_sem_give(&job->done);
	}
}

K_THREAD_DEFINE(syn_sched_tid, SCHED_STACK_SIZE, scheduler_thread,
		NULL, NULL, NULL, SCHED_THREAD_PRIO, 0, 0);

void syn_infer_quiesce(void)
{
	k_mutex_lock(&infer_lock, K_FOREVER);
	dispatch_paused = true;
	k_sem_reset(&drain_sem);

	bool busy = job_running;

	k_mutex_unlock(&infer_lock);

	if (busy) {
		k_sem_take(&drain_sem, K_FOREVER);
	}
}

void syn_infer_release(void)
{
	k_mutex_lock(&infer_lock, K_FOREVER);
	dispatch_paused = false;

	uint32_t replay = deferred_wakes;

	deferred_wakes = 0;
	k_mutex_unlock(&infer_lock);

	while (replay-- > 0U) {
		k_sem_give(&sched_wake);
	}
}

bool syn_infer_model_suspended(syn_model_handle_t model)
{
	bool found = false;

	if (model == SYN_MODEL_INVALID) {
		return false;
	}

	k_mutex_lock(&infer_lock, K_FOREVER);
	for (int i = 0; i < CONFIG_SYNAPTIC_MAX_CONCURRENT_JOBS; i++) {
		if (jobs[i].state == JOB_SUSPENDED &&
		    jobs[i].pipe != NULL && jobs[i].pipe->model == model) {
			found = true;
			break;
		}
	}
	k_mutex_unlock(&infer_lock);
	return found;
}

syn_job_id_t syn_infer_submit(syn_pipeline_t *pipe,
			      const syn_tensor_t *input,
			      const syn_infer_params_t *params)
{
	if (!pipeline_valid(pipe) || !pipe->built || input == NULL ||
	    input->data == NULL) {
		return SYN_JOB_INVALID;
	}

	k_mutex_lock(&infer_lock, K_FOREVER);

	/* Count active jobs against the concurrency limit */
	uint8_t active = 0;

	for (int i = 0; i < CONFIG_SYNAPTIC_MAX_CONCURRENT_JOBS; i++) {
		if (jobs[i].state == JOB_QUEUED ||
		    jobs[i].state == JOB_RUNNING ||
		    jobs[i].state == JOB_SUSPENDED) {
			active++;
		}
	}
	if (active >= max_concurrent) {
		k_mutex_unlock(&infer_lock);
		LOG_WRN("Job queue full (%u active)", active);
		return SYN_JOB_INVALID;
	}

	struct infer_job *job = NULL;

	for (int i = 0; i < CONFIG_SYNAPTIC_MAX_CONCURRENT_JOBS; i++) {
		if (jobs[i].state == JOB_FREE) {
			job = &jobs[i];
			break;
		}
	}
	if (job == NULL) {
		k_mutex_unlock(&infer_lock);
		LOG_WRN("No free job slot (results not yet consumed?)");
		return SYN_JOB_INVALID;
	}

	job->id = next_job_id++;
	if (next_job_id == SYN_JOB_INVALID) {
		next_job_id = 1;
	}
	job->pipe = pipe;
	job->input = input;
	if (params != NULL) {
		job->params = *params;
	} else {
		memset(&job->params, 0, sizeof(job->params));
		job->params.priority = SYN_PRIORITY_NORMAL;
	}
	memset(&job->output, 0, sizeof(job->output));
	job->result = -EINPROGRESS;
	job->seq = next_seq++;
	job->start_cyc = k_cycle_get_32();
	job->deadline_at_us = (job->params.deadline_us != 0U) ?
		(k_ticks_to_us_floor64(k_uptime_ticks()) +
		 job->params.deadline_us) : INT64_MAX;
#ifdef CONFIG_SYNAPTIC_LAYER_EXEC
	job->ctx_slot = -1;
	job->resume_stage = 0;
	job->resuming = false;
#endif
	k_sem_reset(&job->done);
	job->state = JOB_QUEUED;

	syn_job_id_t id = job->id;

	k_mutex_unlock(&infer_lock);
	k_sem_give(&sched_wake);

	LOG_DBG("Job %u queued ('%s', prio %d)",
		id, pipe->name, job->params.priority);
	return id;
}

int syn_infer_wait(syn_job_id_t job_id, uint32_t timeout_ms)
{
	k_mutex_lock(&infer_lock, K_FOREVER);
	struct infer_job *job = find_job(job_id);

	if (job == NULL) {
		k_mutex_unlock(&infer_lock);
		return -ENOENT;
	}

	enum job_state state = job->state;

	k_mutex_unlock(&infer_lock);

	switch (state) {
	case JOB_DONE:
		return 0;
	case JOB_ERROR:
		return job->result;
	case JOB_CANCELLED:
		return -ECANCELED;
	default:
		break;
	}

	int ret = k_sem_take(&job->done, K_MSEC(timeout_ms));

	if (ret != 0) {
		return -EAGAIN;
	}

	/* Put the token back so wait() can be called again */
	k_sem_give(&job->done);

	if (job->state == JOB_CANCELLED) {
		return -ECANCELED;
	}
	return job->result;
}

int syn_infer_cancel(syn_job_id_t job_id)
{
	k_mutex_lock(&infer_lock, K_FOREVER);
	struct infer_job *job = find_job(job_id);

	if (job == NULL) {
		k_mutex_unlock(&infer_lock);
		return -ENOENT;
	}

	int ret;

	switch (job->state) {
	case JOB_QUEUED:
		job->state = JOB_CANCELLED;
		job->result = -ECANCELED;
		stats.cancelled++;
		k_sem_give(&job->done);
		ret = 0;
		LOG_INF("Job %u cancelled", job_id);
		break;
#ifdef CONFIG_SYNAPTIC_LAYER_EXEC
	case JOB_SUSPENDED:
		syn_npu_layered_slot_free(job->ctx_slot);
		job->ctx_slot = -1;
		job->state = JOB_CANCELLED;
		job->result = -ECANCELED;
		stats.cancelled++;
		k_sem_give(&job->done);
		ret = 0;
		LOG_INF("Suspended job %u cancelled", job_id);
		break;
#endif
	case JOB_RUNNING:
		ret = -EBUSY;
		break;
	default:
		ret = -EALREADY;
		break;
	}

	k_mutex_unlock(&infer_lock);
	return ret;
}

int syn_infer_get_result(syn_job_id_t job_id, syn_tensor_t *output)
{
	if (output == NULL) {
		return -EINVAL;
	}

	k_mutex_lock(&infer_lock, K_FOREVER);
	struct infer_job *job = find_job(job_id);

	if (job == NULL) {
		k_mutex_unlock(&infer_lock);
		return -ENOENT;
	}

	int ret;

	switch (job->state) {
	case JOB_DONE:
		*output = job->output;
		job->state = JOB_FREE;
		ret = 0;
		break;
	case JOB_ERROR:
		ret = job->result;
		job->state = JOB_FREE;
		break;
	case JOB_CANCELLED:
		ret = -ECANCELED;
		job->state = JOB_FREE;
		break;
	default:
		ret = -EBUSY; /* Still queued or running */
		break;
	}

	k_mutex_unlock(&infer_lock);
	return ret;
}

int syn_infer_run_sync(syn_model_handle_t model,
		       const syn_tensor_t *input,
		       syn_tensor_t *output,
		       syn_priority_t priority)
{
	if (input == NULL || output == NULL) {
		return -EINVAL;
	}

	syn_pipeline_t *pipe = syn_pipeline_create("run_sync");

	if (pipe == NULL) {
		return -ENOMEM;
	}

	int ret = syn_pipeline_add_model(pipe, model);

	if (ret == 0) {
		ret = syn_pipeline_build(pipe);
	}
	if (ret != 0) {
		syn_pipeline_destroy(pipe);
		return ret;
	}

	syn_infer_params_t params = {
		.priority = priority,
	};

	syn_job_id_t job = syn_infer_submit(pipe, input, &params);

	if (job == SYN_JOB_INVALID) {
		syn_pipeline_destroy(pipe);
		return -EBUSY;
	}

	ret = syn_infer_wait(job, 5000);
	if (ret == 0) {
		syn_tensor_t result;

		ret = syn_infer_get_result(job, &result);
		if (ret == 0) {
			if (output->data != NULL &&
			    output->size >= result.size) {
				/* Caller-provided buffer: copy data out */
				memcpy(output->data, result.data, result.size);
				output->size = result.size;
				output->dtype = result.dtype;
				output->ndim = result.ndim;
				memcpy(output->shape, result.shape,
				       sizeof(output->shape));
			} else {
				/* Hand back the arena-backed descriptor */
				*output = result;
			}
		}
	} else {
		/* Consume the slot so it does not leak */
		syn_tensor_t discard;

		(void)syn_infer_get_result(job, &discard);
	}

	syn_pipeline_destroy(pipe);
	return ret;
}

int syn_infer_set_max_concurrent(uint8_t max_jobs)
{
	if (max_jobs < 1 || max_jobs > CONFIG_SYNAPTIC_MAX_CONCURRENT_JOBS) {
		return -EINVAL;
	}
	max_concurrent = max_jobs;
	return 0;
}

static int syn_infer_sys_init(void)
{
	for (int i = 0; i < CONFIG_SYNAPTIC_MAX_CONCURRENT_JOBS; i++) {
		k_sem_init(&jobs[i].done, 0, 1);
	}

	sched_health_id = syn_health_register("sched",
					      SCHED_HEALTH_PERIOD_MS);
	syn_health_set_busy(sched_health_id, false);
	return 0;
}

SYS_INIT(syn_infer_sys_init, APPLICATION, 90);
