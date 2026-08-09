# Changelog

All notable changes to SynapticOS are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) with a
0.x.y scheme while the API is still settling.

Inference performance figures below are measured on the software
NPU stub unless explicitly stated otherwise; the eIQ Neutron invoke
path is tracked for a future release.

## [Unreleased]

### Added
- Deadline-aware job dispatch: priority first, earliest deadline
  within a priority, FIFO on ties; deadline misses are counted.
- Layer-granular execution of synthetic layered models with
  preemption at layer boundaries: a higher-priority job suspends a
  preemptible running job, which later resumes bit-exactly from its
  saved context. `syn infer stats` reports the scheduler counters.
- Memory-optimal activation placement: DAG models with skip
  connections share one plan area laid out by a lifetime-based
  first-fit planner; `syn npu plan` reports the planned peak
  against the all-live baseline.
- Health monitor with hardware watchdog integration (WWDT0 on
  FRDM-MCXN947) and CPU1 heartbeat supervision with automatic
  park-and-re-release recovery; `syn health` and fault-injection
  subcommands.
- DMA HAL implemented on the MCXN947 eDMA plus a software stub for
  QEMU; double-buffered zero-copy frame ingest with a synthetic
  source and the `syn dma bench` command.
- Raw binary OTA transport (`syn ota rawdata`, sender `--binary`)
  targeting about twice the hex transport's throughput at 115200.
- OTA begin now drains the in-flight cross-core inference before
  parking CPU1, or refuses the session.
- Shell: `syn mem dump`, `syn model info/load/unload`, priority
  argument on `syn infer run`, `syn ipc stats`, model-name tab
  completion.
- gcov line coverage of `src/core` on QEMU
  (`tools/syn_coverage.sh`); first measurement 83.7% lines over the
  QEMU-buildable subset. Test suite grew from 133 to 158 cases.

## [0.4.0] - 2026-07-15 — Model Lifecycle

### Added
- Flash-backed model store with a ping-pong registry (atomic,
  power-loss-safe commits, wear tracking, CRC-gated loads).
- A/B model OTA updates over the shell transport: staged images
  survive reboot, validation before activation, one-command
  rollback; CPU1 is parked during bank-1 flash work and re-released
  afterwards.
- Model hot-swap under a live inference workload.
- `.synm` packaging and transfer tooling
  (`syn_model_pack.py`, `syn_ota_send.py`, `syn_flash_layout.py`).
- `ota_update` sample and a dedicated store/OTA test application.

### Changed
- Flash map reworked so no OTA range ever touches the CPU1 image
  reserve; the stock `slot1_partition` (which straddled the CPU1
  bank) is deleted from the device tree.

### Fixed
- Multi-sector flash erases are issued sector-by-sector with
  breathing room; a single long erase call wedged the whole chip
  (board finding).
- The NPU stub's arbitrary 256 KB model cap now tracks the flash
  slot capacity (board finding: a valid 432 KB OTA model was
  refused after activation).
- Shell RX ring sized for the OTA transport (board finding: 2 KB
  hex lines dropped bytes with the default 64-byte ring).

## [0.3.0] - 2026-07-14 — Dual-Core & IPC

### Added
- Asymmetric dual-core operation on the MCXN947: CPU0 runs the AI
  runtime, CPU1 boots from flash bank 1 and is supervised by CPU0
  (boot 1514 us, handshake 2554 us measured).
- Lock-free shared-memory IPC: SPSC rings with a fixed exchange
  slot (round-trip 15 us typical / 81 us max measured on hardware).
- Cross-core inference offload: CPU1 requests, CPU0 serves
  (1913-serve soak with zero errors on hardware).
- MPU protection of the shared region with a runtime-programmed
  guard and `syn mpu test`.
- Blank-check of the CPU1 image via the ROM API before releasing
  the core (board finding: releasing CPU1 into erased flash wedges
  the chip).
- Out-of-tree CPU1 board definition (`frdm_mcxn947_cpu1`).

## [0.2.0] - 2026-07-12 — Inference Pipeline

### Added
- Pipeline engine: preprocess -> model -> postprocess stages with
  build-time validation and memory estimation.
- Priority job scheduler (best-effort / normal / realtime) with a
  dedicated scheduler thread and per-job completion semaphores.
- PowerQuad DSP acceleration for FFT and matrix ops (5.51x FFT,
  1.66x matmul over the software fallback; the 10x goal remains
  open) with software fallbacks and `syn dsp bench`.
- Nine built-in pre/post-processors (resize, normalize, quantize,
  MFCC, softmax, argmax, top-k, NMS, dequantize).
- Live profiling wired through the pipeline (`syn prof`).
- `face_detection` sample.

## [0.1.1] - Phase 1 patch

### Fixed
- Iterable-section registrations (shell commands, SYS_INIT) survive
  linker garbage collection by building the runtime as a named
  Zephyr library.

## [0.1.0] - Phase 1 — Foundation

### Added
- Tensor arena allocator with persistent/ephemeral regions and a
  scratch pool.
- NPU/DSP HAL layers with deterministic software stubs; QEMU and
  FRDM-MCXN947 targets.
- Model registry with handle-based lifecycle.
- Profiling core, shell command tree, initial unit test suite.
