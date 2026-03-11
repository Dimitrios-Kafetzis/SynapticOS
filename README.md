# SynapticOS

**An AI-Native Operating System for Microcontrollers**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Phase](https://img.shields.io/badge/Phase-1_Foundation-green.svg)](#current-status)
[![Version](https://img.shields.io/badge/Version-0.1.0-orange.svg)](https://github.com/Dimitrios-Kafetzis/SynapticOS/releases)
[![Tests](https://img.shields.io/badge/Tests-50_passed-brightgreen.svg)](#current-status)

SynapticOS is a Zephyr-based runtime that treats AI inference as the primary workload on resource-constrained microcontrollers. It provides tensor-aware memory management, hardware-agnostic NPU/DSP abstraction, model lifecycle management, and inference-first scheduling.

## Target Platform

**NXP FRDM-MCXN947** — Dual Cortex-M33 @ 150 MHz, eIQ Neutron NPU (4.8 GOPS INT8), 512 KB SRAM, 2 MB Flash

## Current Status

**Phase 1 — Foundation** is complete (v0.1.0):

| Deliverable | Description | Status |
|-------------|-------------|--------|
| 1.1 | Tensor arena hardening (16-byte alignment, persistent/ephemeral regions, scratch pool) | Done |
| 1.2 | NPU HAL stub (full state machine, deterministic inference, QEMU-compatible) | Done |
| 1.3 | DSP HAL stub (normalize, softmax, argmax, software fallback) | Done |
| 1.4 | NPU HAL Neutron driver (stub with hardware-specific init, TODO markers) | Done |
| 1.5 | Runtime initialization (syn_init with all HAL subsystems) | Done |
| 1.6 | Model registry hardening (duplicate detection, load/unload guards) | Done |
| 1.7 | Basic profiling (cycle counter timing, mark helpers) | Done |
| 1.8 | Shell commands (syn version/mem/model/npu/prof) | Done |
| 1.9 | hello_inference sample (end-to-end demo on QEMU + FRDM) | Done |
| 1.10 | Expanded test suite (50 tests, 100% pass) | Done |

**Build sizes:**
- FRDM-MCXN947: 64 KB flash, 183 KB RAM
- QEMU cortex-m3: 24 KB flash, 27 KB RAM

## Features

- **Inference-first scheduling** — inference jobs are the fundamental scheduling unit
- **Zero-copy tensor pipelines** — sensor → pre-process → NPU → post-process without unnecessary copies
- **Hardware-agnostic acceleration** — clean HAL abstracts NPU, DSP (PowerQuad), and DMA (SmartDMA)
- **Tensor-aware memory** — bump allocator with 16-byte alignment, persistent/ephemeral regions, scratch pool
- **Model lifecycle** — register, load, unload, unregister with duplicate detection
- **Dual-core asymmetry** — CPU0 owns the AI runtime, CPU1 runs application logic (Phase 3)
- **OTA model updates** — dual-bank flash with A/B slot management (Phase 4)

## Quickstart

```bash
# 1. Setup environment (Ubuntu 24.04)
# See docs/01-ubuntu-setup.md for full instructions
source ~/.venvs/synaptic/bin/activate

# 2. Initialize West workspace
mkdir ~/Personal_Projects && cd ~/Personal_Projects
git clone https://github.com/Dimitrios-Kafetzis/SynapticOS.git synaptic-os
cd synaptic-os && west init -l . && cd .. && west update
pip install -r zephyr/scripts/requirements.txt
export ZEPHYR_BASE=~/Personal_Projects/zephyr

# 3. Build for QEMU (no hardware needed)
west build -b qemu_cortex_m3 synaptic-os/samples/hello_inference --pristine
west build -t run

# 4. Build for FRDM-MCXN947
west build -b frdm_mcxn947/mcxn947/cpu0 synaptic-os/samples/hello_inference --pristine
west flash

# 5. Run tests
west twister -T synaptic-os/tests/unit -p qemu_cortex_m3
```

## Project Structure

```
synaptic-os/
├── include/synaptic/       Public API headers (syn_*.h)
├── src/core/               Runtime core (memory, model, profiling, init)
├── src/hal/mcxn947/        NXP MCXN947 hardware drivers (Neutron NPU, PowerQuad DSP)
├── src/hal/stub/           CPU-only fallbacks for testing (QEMU)
├── src/preprocess/         Image, audio, quantization processors
├── src/postprocess/        Classification, detection post-processors
├── samples/                Demo applications
│   └── hello_inference/    End-to-end inference demo
├── tests/unit/             Unit tests (50 test cases)
├── scripts/                Environment setup & validation tools
└── docs/                   Setup guides, phase plans, architecture
```

## Documentation

- [Ubuntu Environment Setup](docs/01-ubuntu-setup.md)
- [Project Setup & First Build](docs/02-project-setup.md)
- [Architecture Specification](docs/architecture.md)

## Contributing

Contributions are welcome starting from Phase 3. Until then, feel free to:
- Star the repository
- Open issues for bugs or feature requests
- Watch for updates

See the [Community Engagement](docs/community-engagement.md) doc for the roadmap.

## License

Apache License 2.0 — see [LICENSE](LICENSE)
