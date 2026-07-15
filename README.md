<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/SynapticOS-logo/svg/synapticos-horizontal-dark.svg">
    <img src="assets/SynapticOS-logo/svg/synapticos-horizontal-light.svg" alt="SynapticOS" width="480">
  </picture>
</p>

<p align="center">
  <strong>AI-Native Operating System for Microcontrollers</strong>
</p>

<p align="center">
  Inference-first runtime for resource-constrained edge devices.<br/>
  Built on <a href="https://zephyrproject.org">Zephyr RTOS</a>. Designed for the <a href="https://www.nxp.com/design/design-center/development-boards-and-designs/general-purpose-mcus/frdm-development-board-for-mcx-n94-n54-mcus:FRDM-MCXN947">NXP FRDM-MCXN947</a>.
</p>

<p align="center">
  <a href="https://github.com/Dimitrios-Kafetzis/SynapticOS/releases"><img src="https://img.shields.io/github/v/release/Dimitrios-Kafetzis/SynapticOS?style=flat-square&color=blue" alt="Version"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache_2.0-green?style=flat-square" alt="License"></a>
  <a href="#roadmap"><img src="https://img.shields.io/badge/phase-4%20of%206-orange?style=flat-square" alt="Phase"></a>
  <a href="#test-suite"><img src="https://img.shields.io/badge/tests-133%20passed-brightgreen?style=flat-square" alt="Tests"></a>
</p>

---

## The Problem

Most RTOS platforms treat AI inference as an afterthought &mdash; a library bolted onto a scheduler that was designed for control loops and sensor polling. On microcontrollers with dedicated NPUs, this mismatch wastes silicon: the NPU sits idle while the CPU copies buffers, the memory allocator fragments the heap, and there is no clean way to manage model lifecycles or pipeline data from sensor to prediction.

## The Solution

**SynapticOS treats inference as the primary workload**, not an add-on. Every subsystem &mdash; memory, scheduling, hardware abstraction, model management &mdash; is designed around the data flow of a neural network inference pipeline.

```
┌─────────────────────────────────────────────────────────────────┐
│                        SynapticOS Runtime                       │
│                                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────────┐   │
│  │  Sensor  │─▶│  Pre-    │─▶│   NPU    │─▶│    Post-      │   │
│  │  Input   │  │ process  │  │ Invoke   │  │   process     │   │
│  └──────────┘  └──────────┘  └──────────┘  └───────────────┘   │
│       │              │             │               │            │
│  ┌────┴──────────────┴─────────────┴───────────────┴────────┐  │
│  │              Tensor-Aware Memory Arena                    │  │
│  │         (16-byte aligned, persistent + ephemeral)        │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Hardware Abstraction Layer                  │   │
│  │    NPU (eIQ Neutron)  ·  DSP (PowerQuad)  ·  DMA       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │
│  │  Model   │  │Profiling │  │  Shell   │  │   IPC        │   │
│  │Store+OTA │  │  Engine  │  │ Commands │  │ (Dual-Core)  │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │   Zephyr RTOS     │
                    │   v3.7.0          │
                    └───────────────────┘
```

## Key Features

| Feature | Description |
|---------|-------------|
| **Tensor-Aware Memory** | Bump allocator with 16-byte DMA alignment. Persistent region for weights, ephemeral region for activations, dedicated scratch pool. Zero fragmentation. |
| **NPU/DSP Abstraction** | Clean HAL with full state machine (idle/busy/suspended). Swap between QEMU stubs and real Neutron NPU without changing application code. |
| **Model Lifecycle** | Register, load, invoke, unload, unregister. Duplicate detection, state guards, metadata tracking. |
| **Persistent Model Store** | Flash-backed A/B model slots with a ping-pong committed registry (generation counter + CRC32, newest-valid-wins), per-copy wear tracking, and a CRC32 gate on every load. Registry commit 1.9&ndash;2.6 ms, boot scan 23&ndash;189 us, measured on the board. |
| **A/B OTA Model Updates** | Power-loss-safe staged updates: stream in chunks, verify from flash, then commit. Activate and rollback (including activating a staged update after a reboot), hot-swap with inference quiescence, and CPU1 park/resume around bank-1 flash writes. Ships with a UART shell transport and host-side sender. |
| **Model Packaging** | `.synm` container tooling: `syn_model_pack.py` (stdlib-only packer with a built-in TFLite flatbuffer reader), `syn_flash_layout.py` (dual-core-safe partition map generator), `syn_ota_send.py` (UART OTA sender). |
| **Inference Profiling** | Cycle-accurate timing of each pipeline stage (preprocess, NPU, postprocess). Memory peak tracking and NPU utilization metrics. |
| **Interactive Shell** | Inspect and drive the runtime over serial: `syn version`, `syn mem stats`, `syn model list`, `syn npu caps`, `syn prof last`, `syn ipc status`, `syn mpu test`, `syn store status`, `syn ota begin/data/done/activate/rollback`. |
| **Dual-Core Offload** | The AI runtime runs on CPU0; CPU1 requests inference as an OS service over lock-free shared-memory rings (15 us typical round-trip, measured on the board). Includes CPU1 boot orchestration with blank-bank fallback and an out-of-tree CPU1 board port for Zephyr 3.7. |
| **Cross-Core Protection** | CPU0's MPU guards CPU1's RAM read-only, with fault containment (offending thread aborted, both cores continue). One-directional by silicon: the MCXN947's CPU1 has no MPU. |
| **Dual-Target Support** | Same codebase builds and runs on real hardware (FRDM-MCXN947) and emulated targets (QEMU Cortex-M3) for CI-friendly development. |

> **Measured honestly.** All inference figures to date run against the **stub NPU HAL** &mdash; every byte travels the real pipeline (UART &rarr; OTA engine &rarr; flash &rarr; CRC-gated load &rarr; serve), but no real network executes until the eIQ Neutron SDK invoke path lands. And the demo OTA transport is deliberately simple ack-paced hex over the 115200-baud shell: a 432 KB slot-max update lands in 80.5 s (5.4 KB/s, transport-bound &mdash; the OTA engine itself is chunk-size-agnostic, so a binary side-channel is a drop-in upgrade).

## Target Hardware

| Spec | Value |
|------|-------|
| **Board** | NXP FRDM-MCXN947 |
| **CPU** | Dual Arm Cortex-M33 @ 150 MHz |
| **NPU** | eIQ Neutron (4.8 GOPS INT8) |
| **SRAM** | 512 KB |
| **Flash** | 2 MB |
| **Price** | ~$15 USD |

## Getting Started

### Prerequisites

- Ubuntu 24.04 (or compatible Linux)
- Python 3.10+ with `west`, `pyocd`
- Zephyr SDK 0.17.0+

See [docs/01-ubuntu-setup.md](docs/01-ubuntu-setup.md) for detailed environment setup.

### Build & Run

```bash
# Clone and initialize workspace
mkdir ~/workspace && cd ~/workspace
git clone https://github.com/Dimitrios-Kafetzis/SynapticOS.git synaptic-os
cd synaptic-os && west init -l . && cd .. && west update
pip install -r zephyr/scripts/requirements.txt

# Run on QEMU (no hardware needed)
west build -b qemu_cortex_m3 synaptic-os/samples/hello_inference --pristine
west build -t run

# Or build for real hardware
west build -b frdm_mcxn947/mcxn947/cpu0 synaptic-os/samples/hello_inference --pristine
west flash
```

### Expected Output

```
[00:00:00.000,000] <inf> hello_inference: === SynapticOS 0.4.0 — Hello Inference ===
[00:00:00.000,000] <inf> hello_inference: Registered model 'test_classify' (handle=1)
[00:00:00.000,000] <inf> hello_inference: Model loaded to NPU
[00:00:00.000,000] <inf> hello_inference: Input tensor: 1x16x16x3 (768 bytes)
[00:00:00.000,000] <inf> syn_infer: Pipeline 'run_sync' built: 1 stages, est. 4106 bytes
[00:00:00.000,000] <inf> hello_inference: Inference completed in 1361 us
[00:00:00.000,000] <inf> hello_inference: Prediction: class 0 (confidence 127)
[00:00:00.000,000] <inf> syn_prof: === Inference Profile ===
[00:00:00.000,000] <inf> hello_inference: === Hello Inference complete ===
```

### Run Tests

```bash
west twister -T synaptic-os/tests -p qemu_cortex_m3
# 133 tests, 16 suites, 100% pass rate (two apps: tests/unit + tests/unit_store)
```

## Build Footprint

| Build | Target | Flash | RAM |
|-------|--------|-------|-----|
| dual_model (CPU0: runtime, shell, serving, OTA) | FRDM-MCXN947 | 108.8 KB | 213.5 KB of 256 KB |
| dual_model (CPU1: remote client) | FRDM-MCXN947 | 32.5 KB | 42.6 KB of 64 KB |
| ota_update (CPU0: runtime, shell, OTA demo) | FRDM-MCXN947 | 103.9 KB | 210.8 KB |
| ota_update | QEMU Cortex-M3 | 80.9 KB | 62.6 KB of 64 KB |
| hello_inference | QEMU Cortex-M3 | 42.7 KB | 30.5 KB |

## Project Structure

```
synaptic-os/
├── include/synaptic/       Public API headers (frozen)
├── src/
│   ├── core/               Runtime: memory, model registry + store, OTA engine,
│   │                       flash map/port, IPC, boot, profiling, init, shell
│   ├── hal/
│   │   ├── mcxn947/        Neutron NPU + PowerQuad DSP drivers
│   │   └── stub/           Software fallbacks for QEMU / CI
│   ├── preprocess/         Image, audio, quantization pipelines
│   └── postprocess/        Classification, detection output processing
├── samples/
│   ├── hello_inference/    End-to-end inference demo
│   ├── face_detection/     Continuous vision-pipeline demo
│   ├── keyword_spotting/   Audio pipeline demo
│   ├── dual_model/         Cross-core inference demo (CPU0 runtime + CPU1 remote client)
│   └── ota_update/         End-to-end OTA-over-UART demo (factory -> update -> rollback)
├── tests/
│   ├── unit/               108 unit tests across 13 suites
│   └── unit_store/         25 store/OTA/hot-swap tests across 3 suites (RAM-emulated flash)
├── tools/                  syn_model_pack.py, syn_flash_layout.py, syn_ota_send.py, profiler viz
├── boards/nxp/
│   ├── frdm_mcxn947/       Device tree overlays, board configs, flash partition map (CPU0)
│   └── frdm_mcxn947_cpu1/  Out-of-tree CPU1 board port (missing from mainline Zephyr 3.7)
├── scripts/                Environment setup and validation
└── docs/                   Guides and specifications
```

## Lessons from Real Hardware

Every phase is verified on the physical board, not just QEMU &mdash; and the board keeps teaching lessons the emulator can't. All of these were found on hardware, fixed, and re-verified:

- **Never release CPU1 into erased flash** (Phase 3). Releasing the second core into a blank bank wedges the whole chip, including the debug port; recovery is ISP-only. SynapticOS blank-checks the bank via the ROM API before releasing CPU1.
- **Never issue one long multi-sector flash erase** (Phase 4). A single ROM-API `FLASH_Erase` call spanning 55 sectors (the 440 KB OTA staging slot) wedged the chip the same way, while every 1-sector erase before and after was fine. OTA and store erases are now sector-wise with 1 ms breathers; the identical 432 KB transfer then passed.
- **Placeholder limits become field bugs** (Phase 4). The stub NPU HAL's arbitrary 256 KB cap refused to load a valid, stored, activated 432 KB model &mdash; and `syn ota rollback` restored service in one command, an involuntary demo of why rollback exists. The cap now tracks flash slot capacity.
- **Size the shell RX ring for the transport** (Phase 4). The default 64-byte UART ring drops bytes of 2 KB hex lines when scheduling jitter delays mid-line reads; OTA over the shell needs a 1 KB ring, and the sender disables echo during transfers.

## Roadmap

SynapticOS is developed in six phases, each building on the previous:

```
  Phase 1        Phase 2         Phase 3        Phase 4       Phase 5        Phase 6
  Foundation     Inference       Dual-Core      Model         Production     Ecosystem
                 Pipeline        & IPC          Lifecycle     Hardening      & Tooling
  ─────●─────────────●────────────●──────────────●─────────────○──────────────○───▶
  v0.1.0         v0.2.0         v0.3.0         v0.4.0        v0.5.0         v1.0.0
  ✓ Complete    ✓ Complete    ✓ Complete    ✓ Complete
```

| Phase | Focus | Version |
|-------|-------|---------|
| **1. Foundation** | Memory, HAL, model registry, profiling, shell, tests | **v0.1.0** ✓ |
| **2. Inference Pipeline** | Pipeline engine, priority job scheduler, PowerQuad DSP, pre/post-processors, live profiling | **v0.2.0** ✓ |
| **3. Dual-Core & IPC** | Asymmetric multiprocessing, lock-free shared-memory IPC, cross-core inference offload, MPU protection, CPU1 board port | **v0.3.0** ✓ |
| **4. Model Lifecycle** | Flash-backed model store, power-loss-safe A/B OTA updates, hot-swap, dual-core-safe flash map, packaging tools | **v0.4.0** ✓ |
| 5. Production Hardening | Watchdog, fault recovery, soak testing, benchmarks | v0.5.0 |
| 6. Ecosystem & Tooling | Model packaging tools, docs site, SDK, v1.0 release | v1.0.0 |

## Documentation

| Document | Description |
|----------|-------------|
| [Ubuntu Environment Setup](docs/01-ubuntu-setup.md) | Full setup guide for development on Ubuntu 24.04 |
| [Project Setup & First Build](docs/02-project-setup.md) | West workspace initialization and first build |
| [Architecture Specification](docs/architecture.md) | System design, data flow, and component overview |

## Contributing

SynapticOS is in active early development. Community contributions will open in an upcoming phase with proper guidelines, issue templates, and a contributor guide.

**In the meantime, you can:**

- Star the repo to follow progress
- Open an issue for bugs, questions, or feature ideas
- Watch releases for milestone announcements

## License

Licensed under the [Apache License 2.0](LICENSE).
