<p align="center">
  <h1 align="center">SynapticOS</h1>
  <p align="center">
    <strong>AI-Native Operating System for Microcontrollers</strong>
  </p>
  <p align="center">
    Inference-first runtime for resource-constrained edge devices.<br/>
    Built on <a href="https://zephyrproject.org">Zephyr RTOS</a>. Designed for the <a href="https://www.nxp.com/design/design-center/development-boards-and-designs/general-purpose-mcus/frdm-development-board-for-mcx-n94-n54-mcus:FRDM-MCXN947">NXP FRDM-MCXN947</a>.
  </p>
  <p align="center">
    <a href="https://github.com/Dimitrios-Kafetzis/SynapticOS/releases"><img src="https://img.shields.io/badge/version-0.1.0-blue?style=flat-square" alt="Version"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache_2.0-green?style=flat-square" alt="License"></a>
    <a href="#roadmap"><img src="https://img.shields.io/badge/phase-1%20of%206-orange?style=flat-square" alt="Phase"></a>
    <a href="#test-suite"><img src="https://img.shields.io/badge/tests-50%20passed-brightgreen?style=flat-square" alt="Tests"></a>
  </p>
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
│  │ Registry │  │  Engine  │  │ Commands │  │ (Dual-Core)  │   │
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
| **Model Lifecycle** | Register, load, invoke, unload, unregister. Duplicate detection, state guards, metadata tracking. Foundation for OTA updates. |
| **Inference Profiling** | Cycle-accurate timing of each pipeline stage (preprocess, NPU, postprocess). Memory peak tracking and NPU utilization metrics. |
| **Interactive Shell** | Inspect runtime state over serial: `syn version`, `syn mem stats`, `syn model list`, `syn npu caps`, `syn prof last`. |
| **Dual-Target Support** | Same codebase builds and runs on real hardware (FRDM-MCXN947) and emulated targets (QEMU Cortex-M3) for CI-friendly development. |

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
[00:00:00.000,000] <inf> hello_inference: === SynapticOS 0.1.0 — Hello Inference ===
[00:00:00.000,000] <inf> hello_inference: Registered model 'test_classify' (handle=0)
[00:00:00.000,000] <inf> hello_inference: Model loaded to NPU
[00:00:00.000,000] <inf> hello_inference: Input tensor: 1x16x16x3 (768 bytes)
[00:00:00.000,000] <inf> hello_inference: Inference completed in 42 us
[00:00:00.000,000] <inf> hello_inference: Prediction: class 0 (confidence 127)
[00:00:00.000,000] <inf> hello_inference: === Hello Inference complete ===
```

### Run Tests

```bash
west twister -T synaptic-os/tests/unit -p qemu_cortex_m3
# 50 tests, 7 suites, 100% pass rate
```

## Build Footprint

| Target | Flash | RAM |
|--------|-------|-----|
| FRDM-MCXN947 | 64 KB | 183 KB |
| QEMU Cortex-M3 | 24 KB | 27 KB |

## Project Structure

```
synaptic-os/
├── include/synaptic/       Public API headers (frozen)
├── src/
│   ├── core/               Runtime: memory, model registry, profiling, init, shell
│   ├── hal/
│   │   ├── mcxn947/        Neutron NPU + PowerQuad DSP drivers
│   │   └── stub/           Software fallbacks for QEMU / CI
│   ├── preprocess/         Image, audio, quantization pipelines
│   └── postprocess/        Classification, detection output processing
├── samples/
│   └── hello_inference/    End-to-end inference demo
├── tests/unit/             50 unit tests across 7 suites
├── boards/nxp/             Device tree overlays and board configs
├── scripts/                Environment setup and validation
└── docs/                   Guides and specifications
```

## Roadmap

SynapticOS is developed in six phases, each building on the previous:

```
  Phase 1        Phase 2         Phase 3        Phase 4       Phase 5        Phase 6
  Foundation     Inference       Dual-Core      Model         Production     Ecosystem
                 Pipeline        & IPC          Lifecycle     Hardening      & Tooling
  ─────●─────────────○────────────○──────────────○─────────────○──────────────○───▶
  v0.1.0         v0.2.0         v0.3.0         v0.4.0        v0.5.0         v1.0.0
  ✓ Complete
```

| Phase | Focus | Version |
|-------|-------|---------|
| **1. Foundation** | Memory, HAL, model registry, profiling, shell, tests | **v0.1.0** ✓ |
| 2. Inference Pipeline | Zero-copy pipelines, layer preemption, TFLite integration | v0.2.0 |
| 3. Dual-Core & IPC | Asymmetric multiprocessing, shared-memory IPC | v0.3.0 |
| 4. Model Lifecycle | OTA updates, A/B flash slots, model versioning | v0.4.0 |
| 5. Production Hardening | Watchdog, fault recovery, soak testing, benchmarks | v0.5.0 |
| 6. Ecosystem & Tooling | Model packaging tools, docs site, SDK, v1.0 release | v1.0.0 |

## Documentation

| Document | Description |
|----------|-------------|
| [Ubuntu Environment Setup](docs/01-ubuntu-setup.md) | Full setup guide for development on Ubuntu 24.04 |
| [Project Setup & First Build](docs/02-project-setup.md) | West workspace initialization and first build |
| [Architecture Specification](docs/architecture.md) | System design, data flow, and component overview |

## Contributing

SynapticOS is in active early development. Community contributions will open in Phase 3 with proper guidelines, issue templates, and a contributor guide.

**In the meantime, you can:**

- Star the repo to follow progress
- Open an issue for bugs, questions, or feature ideas
- Watch releases for milestone announcements

## License

Licensed under the [Apache License 2.0](LICENSE).
