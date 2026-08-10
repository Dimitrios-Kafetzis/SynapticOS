# Changelog

All notable changes to SynapticOS are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) with a
0.x.y scheme while the API is still settling.

Inference performance figures below are measured on the software
NPU stub unless explicitly stated otherwise; the eIQ Neutron invoke
path is tracked for a future release.

## [0.5.0] - 2026-08-10 — Production Hardening

All Phase 5 acceptance criteria were verified on the FRDM-MCXN947
board; the figures below are hardware measurements (software NPU
stub, synthetic layered models).

### Added
- Deadline-aware job dispatch: priority first, earliest deadline
  within a priority, FIFO on ties; deadline misses are counted.
- Layer-granular execution of synthetic layered models with
  preemption at layer boundaries: a higher-priority job suspends a
  preemptible running job, which later resumes bit-exactly from its
  saved context (context save measured at 10 us on hardware).
  `syn infer stats` reports the scheduler counters.
- Memory-optimal activation placement: DAG models with skip
  connections share one plan area laid out by a lifetime-based
  first-fit planner; on the 8-layer demo DAG the planned peak is
  43% below the all-live baseline, planned in 18 us. `syn npu plan`
  reports the numbers.
- Health monitor with hardware watchdog integration (WWDT0 on
  FRDM-MCXN947) and CPU1 heartbeat supervision with automatic
  park-and-re-release recovery (hang detected in 600 ms,
  re-release in 1.3 ms on hardware); `syn health` and
  fault-injection subcommands. An induced CPU0 hang produces a
  clean watchdog reboot with the model store re-adopted.
- DMA HAL implemented on the MCXN947 eDMA plus a software stub for
  QEMU; double-buffered zero-copy frame ingest with a synthetic
  source and the `syn dma bench` command. On hardware, 1000 x 8 KB
  frames ingest at 131 us/frame vs 460 us/frame for the CPU-copy
  baseline (+251% frame rate, zero corrupt frames).
- Raw binary OTA transport (`syn ota rawdata`, sender `--binary`):
  a 432 KB model transfers in 38.88 s at 115200 (11.1 KB/s, 2.06x
  the hex transport, 98.7% of the line rate). Power loss injected
  at 81% of a raw transfer left the store intact and the old model
  serving after reboot.
- OTA begin now drains the in-flight cross-core inference before
  parking CPU1, or refuses the session (demonstrated under a live
  ~20 req/s offload with zero serve errors).
- Shell: `syn mem dump`, `syn model info/load/unload`, priority
  argument on `syn infer run`, `syn ipc stats`, model-name tab
  completion.
- gcov line coverage of `src/core` on QEMU
  (`tools/syn_coverage.sh`); first measurement 83.7% lines over the
  QEMU-buildable subset. Test suite grew from 133 to 158 cases.
- Soak: 11,106 inference jobs served cross-core with the watchdog
  armed — zero errors, zero arena growth, zero unintended resets.
- Doxygen documentation for every function in the public headers.

### Changed
- Shared-memory layout version 1 -> 2 (CPU1 heartbeat and debug
  words in the control block): CPU0 and CPU1 images must be
  reflashed together; mismatched images refuse to pair by design.

### Fixed
Five findings from the hardware bring-up sessions:
- Zephyr 3.7's eDMA driver cannot start mem-to-mem transfers on
  eDMA v4 (no software START, hardware request armed on mux source
  0); the HAL now programs the transfer directly, clearing the
  latched W1C DONE flag that gates START.
- DMA-visible addresses must be plain (non-secure) aliases; the
  TrustZone secure alias (bit 28) bus-errors inside the eDMA.
- The tensor arena is not eDMA-reachable under the current bus
  security attributes, and the SoC's shared no-error-IRQ setup
  turns a faulted transfer into a silent timeout. Ingest benchmarks
  use DMA-reachable static buffers; arena DMA visibility is a
  Phase 6 item.
- An eDMA channel abort between back-to-back one-shot transfers
  wedges the next completion; the ingest pump no longer stops the
  channel between frames.
- A preempting submitter must outrank the scheduler thread, or it
  can never inject mid-job (the on-hardware demo boosts itself for
  the duration; QEMU's cooperative test thread masked this).

### Known limitations
- The original 500 KB / 10 s OTA target remains out of reach at
  115200 baud by construction — the transport now runs at 98.7% of
  the line rate, so further gains need a faster link, not better
  software.
- All inference figures are still measured on the software NPU
  stub. The eIQ Neutron invoke path is scoped for Phase 6 as an
  optional module (per-layer invocation is not exposed by the SDK;
  preemption granularity there is per delegated segment).
- Quiesce drains only the running job: unloading or OTA-updating a
  model that still has a suspended layered job is undefined.
- PowerQuad speedups stay at 5.51x FFT / 1.66x matmul against the
  10x goal.

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
