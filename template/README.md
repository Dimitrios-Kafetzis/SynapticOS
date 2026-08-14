# SynapticOS Application Template

A minimal out-of-tree application that consumes SynapticOS as a
Zephyr module: its own west manifest pulls `synaptic-os` (which in
turn brings Zephyr v3.7.0 and the NXP HAL), and the app links the
`synaptic_os` library through the module system without touching the
SynapticOS tree.

```
my-workspace/
├── my-app/            <- this template (the workspace manifest repo)
│   ├── west.yml
│   └── app/           <- the application: CMakeLists, prj.conf, src/
├── synaptic-os/       <- fetched by west
├── zephyr/            <- fetched by west (v3.7.0)
└── modules/           <- fetched by west (NXP HAL, CMSIS, ...)
```

## Prerequisites

- Zephyr SDK 0.16.x installed (`ZEPHYR_SDK_INSTALL_DIR` set or the
  default `~/zephyr-sdk-*` location)
- `west` in a Python 3.10+ environment (`pip install west`)
- The usual Zephyr build tools (CMake >= 3.20, Ninja, dtc)
- A shell WITHOUT `ZEPHYR_BASE` exported: a stale value from another
  Zephyr workspace silently points `west init`/`west update` at that
  workspace instead of the fresh one (`unset ZEPHYR_BASE` first)

## From scratch to a running build

```sh
# 1. A fresh workspace directory
mkdir my-workspace && cd my-workspace

# 2. Take a copy of this template as your app / manifest repo
git clone --depth 1 https://github.com/Dimitrios-Kafetzis/SynapticOS.git /tmp/synaptic-checkout
cp -r /tmp/synaptic-checkout/template my-app

# 3. Initialize the workspace from the template's manifest and fetch
#    everything (synaptic-os, zephyr, modules)
west init -l my-app
west update

# 4. Build for QEMU (Cortex-M3) and run it
west build -b qemu_cortex_m3 my-app/app -d build-qemu
west build -t run -d build-qemu

# 5. Build for the FRDM-MCXN947 (CPU0)
west build -b frdm_mcxn947/mcxn947/cpu0 my-app/app -d build-frdm
```

Expected QEMU output ends with:

```
<inf> synaptic_app: Inference OK: 10 output bytes, top class N
<inf> synaptic_app: Template run complete
```

Flash the FRDM build over ISP (hold SW3, press+release SW1, release
SW3, then):

```sh
blhost -p /dev/ttyACM0 flash-erase-region 0x10000000 0x40000
blhost -p /dev/ttyACM0 write-memory 0x10000000 build-frdm/zephyr/zephyr.bin
```

## Where to go from here

- The app uses the public API only (`#include <synaptic/syn_api.h>`);
  browse `synaptic-os/include/synaptic/` for the full surface.
- Replace the zero blob in `app/src/main.c` with a packed `.synm`
  model (`synaptic-os/tools/syn_model_pack.py`) and enable the
  optional `neutron` west group for real eIQ Neutron NPU inference
  on the FRDM-MCXN947 (see `synaptic-os/west.yml` for the opt-in;
  the Neutron libraries are under NXP's LA_OPT license).
- `synaptic-os/samples/` shows the larger flows: dual-core serving
  (`dual_model`), OTA + model store (`ota_update`), preprocessing
  pipelines (`face_detection`).
- Rename `my-app` freely -- keep `self: path:` in `west.yml` in sync
  with the directory name.
