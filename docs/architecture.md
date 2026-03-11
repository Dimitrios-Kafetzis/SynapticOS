# SynapticOS — Software Architecture

## An AI-Native Operating System for Microcontrollers

**Target Platform:** NXP FRDM-MCXN947 (Dual Cortex-M33 @ 150 MHz + eIQ Neutron NPU)
**Foundation:** Zephyr RTOS
**Development Host:** Ubuntu 24.04.4 LTS
**Version:** 0.1.0-draft
**Author:** Dimitris Geortzis

---

## 1. Design Philosophy

SynapticOS treats AI inference as the primary workload, not an afterthought. Where traditional RTOSes schedule threads and manage peripherals, SynapticOS elevates inference pipelines, tensor memory, and neural accelerators to first-class OS primitives.

The core design principles are:

- **Inference-first scheduling**: The fundamental scheduling unit is the inference job, not the thread. Threads still exist (inherited from Zephyr) but serve inference orchestration.
- **Zero-copy tensor pipelines**: Data flows from sensor → pre-processing → NPU → post-processing without unnecessary memory copies.
- **Hardware-agnostic acceleration**: A clean HAL abstracts the NPU, DSP, and DMA so models are portable across silicon.
- **Dual-core asymmetry by design**: CPU0 owns the AI runtime; CPU1 owns the application. They communicate through shared memory and mailbox IPC.
- **OTA-ready model lifecycle**: Models are versioned, updatable, and swappable at runtime without reboot.

---

## 2. System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      APPLICATION LAYER                          │
│  User Application (sensors, comms, business logic)     [CPU1]   │
├─────────────────────────────────────────────────────────────────┤
│                     SYNAPTIC API (syn_*)                        │
│  syn_model_load() · syn_infer() · syn_tensor_alloc()            │
│  syn_pipeline_create() · syn_model_update()                     │
├─────────────────────────────────────────────────────────────────┤
│                    MODEL MANAGEMENT LAYER                       │
│  Model Registry · OTA Updater · A/B Slot Manager                │ 
│  Version Control · Model Validation                             │
├──────────────────────┬──────────────────────────────────────────┤
│   INFERENCE ENGINE   │         MEMORY MANAGER                   │
│  Pipeline Scheduler  │  Tensor Arena Allocator                  │
│  Operator Dispatcher │  Scratch Pool                            │
│  Pre/Post Processor  │  Model Swap Manager                      │
│  Profiler / Tracer   │  Lifetime Tracker                        │
├──────────────────────┴──────────────────────────────────────────┤
│              HARDWARE ABSTRACTION LAYER (syn_hal_*)             │
│  ┌──────────┐  ┌──────────┐  ┌───────────┐  ┌──────────────┐    │
│  │ NPU HAL  │  │ DSP HAL  │  │  DMA HAL  │  │  Sensor HAL  │    │
│  │ (Neutron)│  │(PowerQd) │  │(SmartDMA) │  │  (Camera/    │    │
│  │          │  │          │  │           │  │   Accel/etc) │    │
│  └──────────┘  └──────────┘  └───────────┘  └──────────────┘    │
├─────────────────────────────────────────────────────────────────┤
│                    ZEPHYR RTOS KERNEL                           │
│  Scheduler · IPC (Mailbox/MessageQueue) · Device Drivers        │
│  Flash API · Networking · Logging · Shell                       │
├──────────────┬──────────────────────────────┬───────────────────┤
│    CPU0      │     Shared SRAM Region       │      CPU1         │
│  (AI Runtime)│  (Tensor Buffers, IPC Ring)  │   (Application)   │
└──────────────┴──────────────────────────────┴───────────────────┘
│                  MCXN947 HARDWARE                               │
│  2x CM33 · Neutron NPU · PowerQuad · SmartDMA                   │
│  2MB Flash (dual-bank) · 512KB SRAM · Ethernet · USB-HS         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Dual-Core Partitioning

### 3.1 Core Assignment

| Resource         | CPU0 (AI Core)                      | CPU1 (App Core)                |
|------------------|-------------------------------------|--------------------------------|
| **Role**         | AI runtime, NPU orchestration       | Application logic, I/O         |
| **Zephyr**       | Full kernel + SynapticOS subsystem  | Minimal kernel + app threads   |
| **Boot**         | Primary — boots first, enables CPU1 | Secondary — enabled by CPU0    |
| **NPU access**   | Exclusive ownership                 | Via IPC request to CPU0        |
| **PowerQuad**    | Pre/post-processing acceleration    | Available for app DSP tasks    |
| **Peripherals**  | Flash controller, NPU registers     | UART, SPI, I2C, Ethernet, USB  |
| **SRAM**         | ~256 KB (tensor arena + runtime)    | ~160 KB (app + stack)          |
| **Shared**       | ~96 KB (IPC ring + tensor I/O)      | Same region, opposite access   |

### 3.2 Memory Map (512 KB SRAM)

```
0x2000_0000 ┌────────────────────────┐
            │  CPU0 Private          │  256 KB
            │  - SynapticOS Runtime  │
            │  - Tensor Arena        │
            │  - Scratch Buffers     │
            │  - Model Metadata      │
0x2004_0000 ├────────────────────────┤
            │  Shared Region         │  96 KB
            │  - IPC Ring Buffer     │  (8 KB)
            │  - Input Tensor Buffer │  (40 KB)
            │  - Output Tensor Buffer│  (40 KB)
            │  - Control Structures  │  (8 KB)
0x2005_8000 ├────────────────────────┤
            │  CPU1 Private          │  160 KB
            │  - Application Heap    │
            │  - Thread Stacks       │
            │  - Peripheral Buffers  │
0x2008_0000 └────────────────────────┘
```

### 3.3 Inter-Core Communication

Communication uses Zephyr's MBOX (mailbox) driver for the MCXN947, backed by the MCU's Messaging Unit (MU) hardware. The protocol is a lightweight request-response pattern over a shared-memory ring buffer:

```c
/* IPC message structure in shared SRAM */
struct syn_ipc_msg {
    uint32_t  msg_id;           /* Monotonic sequence number          */
    uint8_t   type;             /* NE_IPC_INFER_REQ, _RESP, _MODEL.. */
    uint8_t   priority;         /* 0=best-effort, 1=normal, 2=urgent  */
    uint16_t  payload_len;      /* Length of payload in shared region  */
    uint32_t  payload_offset;   /* Offset into shared tensor region   */
    uint32_t  timestamp_us;     /* Microsecond timestamp              */
    int32_t   status;           /* 0=OK, negative=error               */
};
```

**Flow for an inference request (CPU1 → CPU0):**

1. CPU1 writes input tensor to shared input buffer
2. CPU1 posts `NE_IPC_INFER_REQ` to IPC ring, triggers MBOX interrupt on CPU0
3. CPU0 reads request, copies/references input from shared region
4. CPU0 dispatches inference pipeline (pre-process → NPU → post-process)
5. CPU0 writes output tensor to shared output buffer
6. CPU0 posts `NE_IPC_INFER_RESP` to IPC ring, triggers MBOX interrupt on CPU1
7. CPU1 reads results from shared output buffer

---

## 4. Hardware Abstraction Layer (syn_hal)

### 4.1 NPU HAL

The NPU HAL provides a uniform interface for neural accelerators. On the MCXN947, it wraps the eIQ Neutron NPU driver. The abstraction is designed so that a future port to, e.g., Arm Ethos-U55 or a RISC-V NPU only requires implementing this interface — no changes to the inference engine above.

```c
/* syn_hal_npu.h — NPU Hardware Abstraction */

typedef enum {
    NE_NPU_STATE_IDLE,
    NE_NPU_STATE_BUSY,
    NE_NPU_STATE_ERROR,
    NE_NPU_STATE_SUSPENDED,
} syn_npu_state_t;

typedef enum {
    NE_NPU_DTYPE_INT8,
    NE_NPU_DTYPE_UINT8,
    NE_NPU_DTYPE_INT16,
    NE_NPU_DTYPE_FLOAT16,
    NE_NPU_DTYPE_FLOAT32,
} syn_npu_dtype_t;

typedef struct {
    const char     *name;              /* "neutron", "ethos-u55", etc.    */
    uint32_t        max_ops_per_sec;   /* Peak INT8 OPS                  */
    uint32_t        scratch_size;      /* Required scratch buffer (bytes) */
    uint8_t         supported_dtypes;  /* Bitmask of syn_npu_dtype_t      */
    bool            supports_async;    /* Can run asynchronously          */
} syn_npu_caps_t;

/* Lifecycle */
int  syn_hal_npu_init(void);
void syn_hal_npu_deinit(void);
int  syn_hal_npu_get_caps(syn_npu_caps_t *caps);
syn_npu_state_t syn_hal_npu_get_state(void);

/* Execution */
int  syn_hal_npu_load_model(const uint8_t *model_data, size_t model_size);
int  syn_hal_npu_set_input(uint8_t index, const void *data, size_t size);
int  syn_hal_npu_invoke(void);                        /* Blocking       */
int  syn_hal_npu_invoke_async(syn_npu_done_cb_t cb);   /* Non-blocking   */
int  syn_hal_npu_get_output(uint8_t index, void *data, size_t *size);

/* Power management */
int  syn_hal_npu_suspend(void);
int  syn_hal_npu_resume(void);
```

### 4.2 DSP HAL (PowerQuad)

Used for pre-processing (normalization, FFT, quantization) and post-processing (softmax, NMS, de-quantization). Wraps the MCXN947 PowerQuad accelerator.

```c
/* syn_hal_dsp.h — DSP Hardware Abstraction */

int syn_hal_dsp_init(void);

/* Vector operations (accelerated on PowerQuad) */
int syn_hal_dsp_normalize_int8(const uint8_t *in, int8_t *out,
                              size_t len, float scale, int32_t zero_point);
int syn_hal_dsp_softmax_f32(const float *in, float *out, size_t len);
int syn_hal_dsp_argmax(const int8_t *data, size_t len, uint32_t *index);
int syn_hal_dsp_fft_f32(const float *in, float *out, size_t len);
int syn_hal_dsp_mat_mult_q15(const int16_t *a, const int16_t *b,
                            int16_t *out, uint16_t rows, uint16_t cols);
```

### 4.3 DMA HAL (SmartDMA)

Manages zero-copy data movement between peripherals and tensor buffers, offloading the CPU. Critical for camera → tensor and tensor → display pipelines.

```c
/* syn_hal_dma.h — DMA/SmartDMA Abstraction */

typedef enum {
    NE_DMA_PERIPH_CAMERA,
    NE_DMA_PERIPH_SPI,
    NE_DMA_PERIPH_MEMORY,
} syn_dma_periph_t;

typedef struct {
    syn_dma_periph_t  src_periph;
    syn_dma_periph_t  dst_periph;
    void            *src_addr;
    void            *dst_addr;
    size_t           transfer_size;
    bool             circular;        /* Auto-restart for streaming */
} syn_dma_config_t;

typedef void (*syn_dma_cb_t)(int channel, int status, void *user_data);

int  syn_hal_dma_init(void);
int  syn_hal_dma_configure(int channel, const syn_dma_config_t *config);
int  syn_hal_dma_start(int channel, syn_dma_cb_t callback, void *user_data);
int  syn_hal_dma_stop(int channel);
int  syn_hal_dma_get_remaining(int channel, size_t *remaining);
```

---

## 5. Memory Manager (syn_mem)

### 5.1 Design Goals

MCU SRAM is the scarcest resource. The memory manager is tensor-shape-aware and optimized for the specific allocation patterns of neural network inference: large contiguous buffers allocated per-layer, reused aggressively across layers, and never fragmented.

### 5.2 Architecture

```
┌──────────────────────────────────────────────────────┐
│                 syn_mem API                          │
├───────────┬──────────────┬──────────────┬────────────┤
│  Tensor   │   Scratch    │   Model      │  Lifetime  │
│  Arena    │   Pool       │   Store      │  Tracker   │
│           │              │              │            │
│  Fixed    │  Per-layer   │  Flash-      │  Ref-count │
│  regions  │  reusable    │  backed      │  based     │
│  for I/O  │  buffers     │  with SRAM   │  auto-free │
│  tensors  │              │  cache       │            │
└───────────┴──────────────┴──────────────┴────────────┘
```

### 5.3 Tensor Arena Allocator

The arena uses a bump allocator within a fixed SRAM region. Each inference invocation resets the bump pointer — no fragmentation, O(1) allocation, zero overhead.

```c
/* syn_mem.h — Tensor Memory Management */

typedef enum {
    NE_MEM_PERSISTENT,    /* Lives across inference calls (weights, biases)  */
    NE_MEM_EPHEMERAL,     /* Freed after each inference (activations)        */
    NE_MEM_SHARED,        /* In shared IPC region (input/output tensors)     */
} syn_mem_lifetime_t;

typedef struct {
    void            *data;
    size_t           size;
    syn_npu_dtype_t   dtype;
    uint8_t          ndim;
    uint32_t         shape[4];    /* Max 4D: [batch, height, width, channels] */
    syn_mem_lifetime_t lifetime;
} syn_tensor_t;

/* Arena management */
int  syn_mem_init(void *arena_base, size_t arena_size);
void syn_mem_reset_ephemeral(void);   /* Called between inference jobs */

/* Tensor allocation */
syn_tensor_t *syn_mem_tensor_alloc(const uint32_t *shape, uint8_t ndim,
                                 syn_npu_dtype_t dtype, syn_mem_lifetime_t lt);
void         syn_mem_tensor_free(syn_tensor_t *tensor);

/* Scratch pool (for intermediate per-layer buffers) */
void *syn_mem_scratch_acquire(size_t size);
void  syn_mem_scratch_release(void *ptr);

/* Stats */
typedef struct {
    size_t  arena_total;
    size_t  arena_used;
    size_t  arena_peak;       /* High-water mark */
    size_t  scratch_total;
    size_t  scratch_used;
    uint32_t alloc_count;
    uint32_t reset_count;
} syn_mem_stats_t;

int syn_mem_get_stats(syn_mem_stats_t *stats);
```

### 5.4 Model Memory Strategy

Models reside in flash but require SRAM for activation tensors during inference. The strategy depends on model size:

| Model Size | Strategy |
|-----------|----------|
| < 128 KB  | Entire model cached in SRAM for fastest inference |
| 128–512 KB | Weights stay in flash (XIP), only activations in SRAM |
| > 512 KB  | Layer-by-layer streaming: load weights per layer from flash |

For the MCXN947 with 2 MB flash (dual-bank), most TinyML models will use the XIP strategy — weights are executed in-place from flash while the 256 KB tensor arena handles activations.

---

## 6. Inference Engine (syn_infer)

### 6.1 Pipeline Architecture

An inference pipeline is a directed graph of stages. The engine schedules these stages across the available compute resources (CPU, NPU, DSP) based on operator support and latency constraints.

```
┌────────────┐    ┌───────────┐    ┌───────────┐    ┌───────────┐
│  Acquire   │───▶│   Pre-    │───▶│    NPU    │───▶│   Post-   │
│  (DMA/     │    │  Process  │    │  Invoke   │    │  Process  │
│   Camera)  │    │  (DSP)    │    │           │    │  (DSP)    │
└────────────┘    └───────────┘    └───────────┘    └───────────┘
    SmartDMA        PowerQuad       Neutron NPU       PowerQuad
```

### 6.2 Inference Scheduler

The scheduler manages multiple inference jobs with priority and deadline awareness. It is the core differentiator of SynapticOS — unlike a general RTOS thread scheduler, it understands model latency profiles and can make preemption decisions based on inference stage boundaries.

```c
/* syn_infer.h — Inference Engine */

typedef enum {
    NE_PRIORITY_BEST_EFFORT = 0,   /* No deadline, runs when idle       */
    NE_PRIORITY_NORMAL      = 1,   /* Standard priority                 */
    NE_PRIORITY_REALTIME    = 2,   /* Hard deadline, preempts others    */
} syn_priority_t;

typedef struct {
    syn_priority_t   priority;
    uint32_t        deadline_us;       /* 0 = no deadline                */
    bool            preemptible;       /* Can be paused between layers   */
    syn_infer_cb_t   callback;          /* Completion callback            */
    void           *user_data;
} syn_infer_params_t;

typedef struct syn_pipeline syn_pipeline_t;

/* Pipeline construction */
syn_pipeline_t *syn_pipeline_create(const char *name);
int  syn_pipeline_add_preprocess(syn_pipeline_t *pipe,
                                syn_preprocess_fn_t fn, void *config);
int  syn_pipeline_add_model(syn_pipeline_t *pipe,
                           syn_model_handle_t model);
int  syn_pipeline_add_postprocess(syn_pipeline_t *pipe,
                                 syn_postprocess_fn_t fn, void *config);
int  syn_pipeline_build(syn_pipeline_t *pipe);  /* Validates & optimizes */

/* Job submission */
typedef uint32_t syn_job_id_t;
syn_job_id_t syn_infer_submit(syn_pipeline_t *pipe,
                            const syn_tensor_t *input,
                            const syn_infer_params_t *params);
int  syn_infer_wait(syn_job_id_t job, uint32_t timeout_ms);
int  syn_infer_cancel(syn_job_id_t job);
int  syn_infer_get_result(syn_job_id_t job, syn_tensor_t *output);

/* Scheduler control */
int  syn_infer_set_max_concurrent(uint8_t max_jobs);
```

### 6.3 Scheduling Policy

The scheduler operates at **layer granularity**, not thread preemption granularity. Between NPU layer invocations, the scheduler can:

1. **Preempt** a low-priority job to run a high-priority one
2. **Time-slice** between equal-priority jobs at layer boundaries
3. **Deadline-check** — if a realtime job risks missing its deadline, it takes over immediately

This is possible because the Neutron NPU processes one layer at a time, and between layers, control returns to the CPU. The scheduler intercepts this return point.

```
Job A (normal):    [Layer 0][Layer 1]...........[Layer 2][Layer 3][Done]
Job B (realtime):                    [Layer 0][Layer 1][Layer 2][Done]
                   ▲                 ▲                          ▲
                   A starts          B preempts A               A resumes
```

### 6.4 Built-in Pre/Post Processors

Common operations provided out-of-the-box, accelerated by PowerQuad DSP:

```c
/* Pre-processing functions */
syn_preprocess_fn_t syn_preprocess_image_resize;    /* Bilinear resize         */
syn_preprocess_fn_t syn_preprocess_image_normalize; /* Scale to [-1,1] or [0,1]*/
syn_preprocess_fn_t syn_preprocess_quantize_int8;   /* Float32 → INT8          */
syn_preprocess_fn_t syn_preprocess_audio_mfcc;      /* Audio → MFCC features   */

/* Post-processing functions */
syn_postprocess_fn_t syn_postprocess_softmax;        /* Classification scores  */
syn_postprocess_fn_t syn_postprocess_argmax;          /* Top-1 class index     */
syn_postprocess_fn_t syn_postprocess_top_k;           /* Top-K results         */
syn_postprocess_fn_t syn_postprocess_nms;             /* Non-max suppression   */
syn_postprocess_fn_t syn_postprocess_dequantize;      /* INT8 → Float32        */
```

---

## 7. Model Management (syn_model)

### 7.1 Model Registry

The model registry maintains metadata about all models stored in flash. It supports multiple models simultaneously, with runtime selection and hot-swapping.

```c
/* syn_model.h — Model Lifecycle Management */

typedef uint32_t syn_model_handle_t;

typedef struct {
    char         name[32];
    char         version[16];      /* Semantic version: "1.2.3"          */
    uint32_t     input_size;       /* Expected input tensor size (bytes) */
    uint32_t     output_size;      /* Output tensor size (bytes)         */
    uint32_t     flash_offset;     /* Offset in flash                    */
    uint32_t     flash_size;       /* Total size in flash                */
    uint32_t     sram_required;    /* Peak SRAM needed for inference     */
    uint32_t     crc32;            /* Integrity check                    */
    syn_npu_dtype_t input_dtype;
    syn_npu_dtype_t output_dtype;
    uint8_t      input_shape[4];
    uint8_t      output_shape[4];
} syn_model_info_t;

/* Registry operations */
int  syn_model_register(const syn_model_info_t *info, syn_model_handle_t *handle);
int  syn_model_unregister(syn_model_handle_t handle);
int  syn_model_get_info(syn_model_handle_t handle, syn_model_info_t *info);
int  syn_model_list(syn_model_handle_t *handles, uint8_t *count, uint8_t max);

/* Loading (flash → NPU-ready state) */
int  syn_model_load(syn_model_handle_t handle);
int  syn_model_unload(syn_model_handle_t handle);
bool syn_model_is_loaded(syn_model_handle_t handle);

/* Hot-swap: atomically replace a running model */
int  syn_model_swap(syn_model_handle_t old_handle, syn_model_handle_t new_handle);
```

### 7.2 OTA Model Updates

Leverages the MCXN947's dual-bank flash for safe model updates:

```
┌─────────────────────────────────────────────┐
│              FLASH BANK 0 (1 MB)            │
│  ┌───────────────────┬─────────────────────┐│
│  │  Firmware (Zephyr │  Model Slot A       ││
│  │  + SynapticOS)    │  (active models)    ││
│  │  ~384 KB          │  ~640 KB            ││
│  └───────────────────┴─────────────────────┘│
├─────────────────────────────────────────────┤
│              FLASH BANK 1 (1 MB)            │
│  ┌───────────────────┬─────────────────────┐│
│  │  Firmware backup  │  Model Slot B       ││
│  │  (for rollback)   │  (staging area)     ││
│  │  ~384 KB          │  ~640 KB            ││
│  └───────────────────┴─────────────────────┘│
└─────────────────────────────────────────────┘
```

**Update flow:**

1. New model arrives over Ethernet/USB → written to Slot B
2. CRC32 validation of Slot B
3. Model metadata registered in registry with `pending` state
4. On command (or auto), atomic swap: Slot B becomes active, Slot A becomes backup
5. If new model fails validation at runtime → automatic rollback to Slot A

```c
/* syn_model_ota.h — Over-The-Air Model Updates */

typedef enum {
    NE_OTA_STATE_IDLE,
    NE_OTA_STATE_DOWNLOADING,
    NE_OTA_STATE_VALIDATING,
    NE_OTA_STATE_STAGING,
    NE_OTA_STATE_READY,
    NE_OTA_STATE_ERROR,
} syn_ota_state_t;

int  syn_ota_begin(const char *model_name, size_t total_size);
int  syn_ota_write_chunk(const uint8_t *data, size_t len);
int  syn_ota_finish(void);                /* Validates CRC & stages */
int  syn_ota_activate(void);              /* Atomic swap            */
int  syn_ota_rollback(void);              /* Revert to previous     */
syn_ota_state_t syn_ota_get_state(void);
```

---

## 8. Profiling & Diagnostics (syn_prof)

Essential for development and optimization. All profiling is zero-overhead when disabled (compile-time flag).

```c
/* syn_prof.h — Profiling and Diagnostics */

typedef struct {
    uint32_t  total_us;            /* Total inference time              */
    uint32_t  preprocess_us;       /* Pre-processing time              */
    uint32_t  npu_us;              /* NPU execution time               */
    uint32_t  postprocess_us;      /* Post-processing time             */
    uint32_t  ipc_overhead_us;     /* IPC latency (if cross-core)      */
    uint32_t  mem_peak_bytes;      /* Peak SRAM usage during inference  */
    uint32_t  npu_utilization_pct; /* NPU busy time as % of total      */
} syn_prof_result_t;

int  syn_prof_enable(void);
int  syn_prof_disable(void);
int  syn_prof_get_last(syn_prof_result_t *result);
void syn_prof_print_summary(void);    /* Dumps to Zephyr shell/log */

/* Per-layer profiling (detailed mode) */
int  syn_prof_enable_layer_trace(void);
int  syn_prof_get_layer_time(uint32_t layer_index, uint32_t *us);
```

---

## 9. Application API (syn_api)

The top-level API that application developers interact with. Designed to be simple enough that a developer can go from zero to running inference in under 20 lines of code.

### 9.1 Minimal Example (CPU1 application)

```c
#include <synaptic/syn_api.h>

/* Callback when inference completes */
static void on_result(syn_job_id_t job, const syn_tensor_t *output, void *ctx)
{
    uint32_t class_index;
    syn_postprocess_argmax(output, &class_index);
    printk("Detected class: %u (confidence: %d)\n",
           class_index, output->data[class_index]);
}

void main(void)
{
    /* Initialize SynapticOS runtime */
    syn_init();

    /* Load a pre-registered model */
    syn_model_handle_t model;
    syn_model_get_by_name("face_detect_v1", &model);
    syn_model_load(model);

    /* Build inference pipeline */
    syn_pipeline_t *pipe = syn_pipeline_create("face_pipeline");
    syn_pipeline_add_preprocess(pipe, syn_preprocess_image_resize,
                               &(syn_resize_config_t){.w=96, .h=96});
    syn_pipeline_add_preprocess(pipe, syn_preprocess_quantize_int8, NULL);
    syn_pipeline_add_model(pipe, model);
    syn_pipeline_add_postprocess(pipe, syn_postprocess_softmax, NULL);
    syn_pipeline_build(pipe);

    /* Continuous inference loop */
    while (1) {
        syn_tensor_t *frame = syn_sensor_capture_frame();  /* Camera DMA */

        syn_infer_params_t params = {
            .priority  = NE_PRIORITY_NORMAL,
            .callback  = on_result,
        };
        syn_infer_submit(pipe, frame, &params);

        k_msleep(100);  /* ~10 FPS */
    }
}
```

### 9.2 Synchronous API (simpler, for prototyping)

```c
/* One-shot synchronous inference */
syn_tensor_t input, output;
syn_mem_tensor_init(&input, (uint32_t[]){1,96,96,3}, 4, NE_NPU_DTYPE_INT8);
syn_mem_tensor_init(&output, (uint32_t[]){1,10}, 2, NE_NPU_DTYPE_INT8);

/* Fill input... */
memcpy(input.data, camera_buffer, input.size);

/* Run inference (blocks until done) */
syn_infer_run_sync(model, &input, &output, NE_PRIORITY_NORMAL);

/* Read result */
int8_t *scores = (int8_t *)output.data;
```

---

## 10. Zephyr Integration

### 10.1 Subsystem Registration

SynapticOS registers as a Zephyr subsystem with proper Kconfig integration:

```kconfig
# Kconfig.synaptic

menuconfig SYNAPTIC
    bool "SynapticOS AI Runtime"
    depends on SOC_SERIES_MCXN
    select HEAP_MEM_POOL_SIZE
    help
      Enable the SynapticOS AI-native runtime for inference
      scheduling, tensor memory management, and NPU orchestration.

if SYNAPTIC

config SYNAPTIC_TENSOR_ARENA_SIZE
    int "Tensor arena size in bytes"
    default 131072
    help
      Size of the SRAM region dedicated to tensor allocations.

config SYNAPTIC_MAX_MODELS
    int "Maximum number of registered models"
    default 4

config SYNAPTIC_MAX_CONCURRENT_JOBS
    int "Maximum concurrent inference jobs"
    default 2

config SYNAPTIC_PROFILING
    bool "Enable inference profiling"
    default y if DEBUG

config SYNAPTIC_OTA
    bool "Enable OTA model updates"
    depends on FLASH
    default y

config SYNAPTIC_DUAL_CORE
    bool "Enable dual-core operation"
    depends on SECOND_CORE_MCUX
    default y

endif # SYNAPTIC
```

### 10.2 Directory Structure

```
synaptic-os/
├── CMakeLists.txt
├── Kconfig
├── west.yml                         # Zephyr manifest with hal_nxp
├── zephyr/                          # Zephyr module integration
│   ├── module.yml
│   └── CMakeLists.txt
├── include/
│   └── synaptic/
│       ├── syn_api.h                 # Top-level application API
│       ├── syn_infer.h               # Inference engine
│       ├── syn_model.h               # Model management
│       ├── syn_model_ota.h           # OTA updates
│       ├── syn_mem.h                 # Memory manager
│       ├── syn_prof.h                # Profiling
│       ├── syn_hal_npu.h             # NPU HAL interface
│       ├── syn_hal_dsp.h             # DSP HAL interface
│       ├── syn_hal_dma.h             # DMA HAL interface
│       └── syn_ipc.h                 # Inter-core communication
├── src/
│   ├── core/
│   │   ├── syn_init.c                # Runtime initialization
│   │   ├── syn_infer.c               # Inference scheduler
│   │   ├── syn_mem.c                 # Tensor arena allocator
│   │   ├── syn_model.c               # Model registry
│   │   ├── syn_model_ota.c           # OTA update logic
│   │   ├── syn_prof.c                # Profiler
│   │   └── syn_ipc.c                 # IPC ring buffer
│   ├── hal/
│   │   ├── mcxn947/
│   │   │   ├── syn_hal_npu_neutron.c # eIQ Neutron NPU driver
│   │   │   ├── syn_hal_dsp_pq.c      # PowerQuad DSP driver
│   │   │   └── syn_hal_dma_sdma.c    # SmartDMA driver
│   │   └── stub/
│   │       ├── syn_hal_npu_stub.c    # CPU-only fallback (no NPU)
│   │       └── syn_hal_dsp_stub.c    # Software DSP fallback
│   ├── preprocess/
│   │   ├── syn_preprocess_image.c
│   │   ├── syn_preprocess_audio.c
│   │   └── syn_preprocess_quant.c
│   └── postprocess/
│       ├── syn_postprocess_classify.c
│       └── syn_postprocess_detect.c
├── boards/
│   └── nxp/
│       └── frdm_mcxn947/
│           ├── frdm_mcxn947_cpu0.overlay  # Device tree overlay (AI core)
│           ├── frdm_mcxn947_cpu1.overlay  # Device tree overlay (App core)
│           └── frdm_mcxn947_cpu0.conf     # Kconfig for AI core
├── models/
│   ├── face_detect_v1.tflite        # Example: face detection
│   ├── keyword_spot_v1.tflite       # Example: keyword spotting
│   └── anomaly_det_v1.tflite        # Example: vibration anomaly
├── samples/
│   ├── hello_inference/             # Minimal inference example
│   ├── face_detection/              # Camera + NPU face detection
│   ├── keyword_spotting/            # Microphone + audio classification
│   ├── dual_model/                  # Two models running concurrently
│   └── ota_update/                  # Model OTA update demo
├── tests/
│   ├── unit/
│   │   ├── test_mem.c               # Memory manager tests
│   │   ├── test_scheduler.c         # Inference scheduler tests
│   │   ├── test_model_registry.c    # Model registry tests
│   │   └── test_ipc.c              # IPC protocol tests
│   └── integration/
│       ├── test_npu_inference.c     # End-to-end NPU test
│       └── test_dual_core.c        # Dual-core IPC test
├── tools/
│   ├── syn_model_pack.py             # Package TFLite → SynapticOS format
│   ├── syn_flash_layout.py           # Generate flash partition map
│   └── syn_profiler_viz.py           # Visualize profiling data
└── docs/
    ├── architecture.md              # This document
    ├── getting-started.md           # Setup guide for Ubuntu 24.04
    ├── porting-guide.md             # How to port to new MCUs
    └── api-reference.md             # Full API docs
```

---

## 11. Build System

### 11.1 West Manifest (west.yml)

```yaml
manifest:
  remotes:
    - name: zephyrproject-rtos
      url-base: https://github.com/zephyrproject-rtos

  projects:
    - name: zephyr
      remote: zephyrproject-rtos
      revision: v3.7.0
      import:
        name-allowlist:
          - hal_nxp
          - cmsis
          - mcuboot

  self:
    path: synaptic-os
```

### 11.2 Build Commands

```bash
# One-time setup
west init -l synaptic-os
west update

# Build AI core (CPU0) firmware
west build -b frdm_mcxn947/mcxn947/cpu0 samples/hello_inference \
    -- -DSHIELD=lcd_par_s035

# Build App core (CPU1) — built via sysbuild when dual-core enabled
west build -b frdm_mcxn947/mcxn947/cpu0 --sysbuild samples/face_detection

# Flash
west flash

# Run unit tests (on-target via Zephyr test runner)
west twister -T tests/ -p frdm_mcxn947/mcxn947/cpu0

# Run unit tests (QEMU for logic-only tests)
west twister -T tests/unit -p qemu_cortex_m3
```

---

## 12. Development Roadmap

### Phase 1: Foundation (Weeks 1–4)
- [ ] Repository setup with west manifest and Zephyr integration
- [ ] Basic project structure and build system
- [ ] syn_mem: Tensor arena allocator (bump allocator + ephemeral reset)
- [ ] syn_hal_npu: Neutron NPU wrapper (load, invoke, get_output)
- [ ] syn_hal_npu_stub: CPU-only fallback for unit testing
- [ ] hello_inference sample: single model, synchronous, CPU0-only
- [ ] Unit tests for memory manager

### Phase 2: Inference Engine (Weeks 5–8)
- [ ] syn_infer: Pipeline construction and execution
- [ ] syn_infer: Job scheduler with priority support
- [ ] syn_hal_dsp: PowerQuad wrapper for pre/post-processing
- [ ] Built-in pre-processors (resize, normalize, quantize)
- [ ] Built-in post-processors (softmax, argmax, top-k)
- [ ] syn_prof: Basic profiling (total time, NPU time, memory peak)
- [ ] face_detection sample with camera + LCD

### Phase 3: Dual-Core & IPC (Weeks 9–12)
- [ ] syn_ipc: Shared memory ring buffer + MBOX integration
- [ ] Dual-core boot sequence (CPU0 enables CPU1)
- [ ] Cross-core inference request/response protocol
- [ ] Memory map finalization and MPU configuration
- [ ] dual_model sample: two models on CPU0, app on CPU1
- [ ] Integration tests for IPC

### Phase 4: Model Management & OTA (Weeks 13–16)
- [ ] syn_model: Model registry with flash-backed metadata
- [ ] syn_model_ota: Dual-bank flash update mechanism
- [ ] syn_model: Hot-swap support
- [ ] syn_model_pack.py: Packaging tool (TFLite → SynapticOS format)
- [ ] ota_update sample
- [ ] Zephyr shell commands for model management

### Phase 5: Optimization & Hardening (Weeks 17–20)
- [ ] syn_hal_dma: SmartDMA zero-copy camera pipeline
- [ ] syn_infer: Layer-granularity preemption
- [ ] syn_mem: Memory-optimal layer scheduling (minimize peak SRAM)
- [ ] Power management (NPU suspend/resume, DVFS hints)
- [ ] Comprehensive test suite
- [ ] Documentation and porting guide

---

## 13. Key Design Decisions Log

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Base OS | Zephyr (not FreeRTOS) | Better modularity, upstream MCXN947 support, RISC-V future path |
| IPC mechanism | Shared SRAM + MBOX | Hardware MU support on MCXN947, lowest latency |
| Memory strategy | Bump allocator + arena | Zero fragmentation, O(1) alloc, matches inference patterns |
| Model storage | Flash XIP + SRAM activations | Best memory efficiency for MCU-class models |
| Scheduling unit | Inference job (not thread) | AI-native abstraction, enables layer-level preemption |
| NPU interface | Synchronous + async callback | Simple default, non-blocking for advanced use |
| Build system | West + CMake (Zephyr native) | No custom tooling, familiar to Zephyr developers |
| Naming prefix | `syn_` (SynapticOS) | Short, unique, avoids collision with Zephyr `z_` / `k_` |
