# SynapticOS — Project Setup and First Build

This guide covers initializing the Zephyr workspace, integrating SynapticOS as a Zephyr module, building the first sample, and flashing to the FRDM-MCXN947 board. Complete [01-ubuntu-setup.md](01-ubuntu-setup.md) before starting this guide.

---

## Step 1: Create the Workspace

SynapticOS uses Zephyr's `west` workspace model. The project sits alongside Zephyr (not inside it) as an out-of-tree module:

```
~/synaptic-workspace/          ← west workspace root
├── .west/                     ← west metadata
├── synaptic-os/               ← THIS REPOSITORY (manifest repo)
│   ├── west.yml               ← manifest file (points to Zephyr + hal_nxp)
│   ├── CMakeLists.txt
│   ├── Kconfig
│   ├── include/synaptic/
│   ├── src/
│   ├── samples/
│   ├── ...
├── zephyr/                    ← pulled by west (Zephyr RTOS)
├── modules/
│   ├── hal/nxp/               ← pulled by west (NXP HAL + Neutron SDK)
│   ├── crypto/mbedtls/        ← pulled by west
│   └── lib/cmsis/             ← pulled by west
├── bootloader/mcuboot/        ← pulled by west
└── tools/                     ← Zephyr host tools
```

### Initialize the workspace:

```bash
# Activate the Python environment
source ~/.venvs/synaptic/bin/activate

# Create workspace root
mkdir -p ~/synaptic-workspace
cd ~/synaptic-workspace

# Clone SynapticOS (or copy the zip contents)
# Option A: From git (when repo is created)
# git clone https://github.com/<your-org>/synaptic-os.git

# Option B: From the zip file
# unzip synaptic-os.zip -d .

# Initialize west with SynapticOS as the manifest repository
cd synaptic-os
west init -l .

# Fetch Zephyr, hal_nxp, cmsis, mcuboot, and mbedtls
cd ..
west update

# Install Zephyr's additional Python dependencies
pip install -r zephyr/scripts/requirements.txt
```

This will take several minutes on the first run as `west update` downloads Zephyr and the NXP HAL (which includes the MCXN947 SDK and eIQ Neutron libraries).

---

## Step 2: Set Environment Variables

Zephyr needs to know where its base directory is:

```bash
# Set ZEPHYR_BASE (required for every terminal session)
export ZEPHYR_BASE=~/synaptic-workspace/zephyr

# Optionally add to your shell profile for persistence
echo 'export ZEPHYR_BASE=~/synaptic-workspace/zephyr' >> ~/.bashrc

# Verify
echo $ZEPHYR_BASE
# Should print: /home/<user>/synaptic-workspace/zephyr
```

---

## Step 3: Build the Hello Inference Sample

This is the smoke test — it compiles SynapticOS, links against Zephyr and the NXP HAL, and produces a flashable binary:

```bash
cd ~/synaptic-workspace

# Clean build
west build -b frdm_mcxn947/mcxn947/cpu0 synaptic-os/samples/hello_inference --pristine
```

**What happens during the build:**

1. CMake configures the build using Zephyr's toolchain abstraction
2. Kconfig merges `prj.conf` + board `.conf` + SynapticOS `Kconfig`
3. Device Tree Compiler processes the board DTS + SynapticOS overlay
4. GCC cross-compiles all source files for Cortex-M33
5. Linker produces `build/zephyr/zephyr.bin` and `build/zephyr/zephyr.elf`

Expected output (last few lines):

```
[148/148] Linking C executable zephyr/zephyr.elf
Memory region         Used Size  Region Size  %age Used
           FLASH:       42356 B         1 MB      4.04%
             RAM:       18432 B       256 KB      7.03%
        IDT_LIST:          0 GB         2 KB      0.00%
```

### Common Build Errors and Fixes

**Error: `Could not find a package configuration file provided by "Zephyr"`**
→ `ZEPHYR_BASE` is not set. Run `export ZEPHYR_BASE=~/synaptic-workspace/zephyr`.

**Error: `Kconfig error: Couldn't parse ... SYNAPTIC`**
→ The `zephyr/module.yml` is not being found. Ensure `west init -l .` was run from inside `synaptic-os/`.

**Error: `fatal error: synaptic/syn_api.h: No such file or directory`**
→ Include path issue. Check that `CMakeLists.txt` has `target_include_directories(... PUBLIC include)`.

---

## Step 4: Flash to Board

Connect the FRDM-MCXN947 via USB-C (MCU-Link port) and flash:

```bash
west flash
```

This uses pyOCD by default with the MCU-Link CMSIS-DAP probe. You should see:

```
-- runners: pyocd
-- Flash file: build/zephyr/zephyr.bin
-- Flash address: 0x0
-- Programming... ████████████████ 100%
-- Verify OK
-- Reset target
```

---

## Step 5: Connect Serial Console

Open a terminal to see the SynapticOS boot output:

```bash
picocom -b 115200 /dev/ttyACM0
```

Expected output:

```
*** Booting Zephyr OS build v3.7.0 ***
[00:00:00.001,000] <inf> hello_inference: SynapticOS 0.1.0 — Hello Inference
[00:00:00.002,000] <inf> hello_inference: Initializing runtime...
[00:00:00.003,000] <inf> hello_inference: Runtime initialized successfully
[00:00:00.004,000] <inf> syn_mem: Arena: 128 KB total, 0 KB used, peak 0 KB
[00:00:00.005,000] <inf> hello_inference: Hello Inference complete. Runtime is operational.
```

Press `Ctrl+A` then `Ctrl+X` to exit picocom.

---

## Step 6: Run Unit Tests

Zephyr's Twister test runner executes tests on QEMU (for CI) and on hardware:

```bash
# Run on QEMU (no board needed — useful for CI)
cd ~/synaptic-workspace
west twister -T synaptic-os/tests/unit -p qemu_cortex_m3

# Run on hardware (board must be connected)
west twister -T synaptic-os/tests/unit -p frdm_mcxn947/mcxn947/cpu0 --device-testing \
    --device-serial /dev/ttyACM0
```

---

## Step 7: Debug with GDB (Optional)

For interactive debugging using the MCU-Link probe:

```bash
# Terminal 1: Start GDB server
pyocd gdbserver -t mcxn947

# Terminal 2: Connect with GDB
~/.local/zephyr-sdk-0.16.8/arm-zephyr-eabi/bin/arm-zephyr-eabi-gdb build/zephyr/zephyr.elf
(gdb) target remote :3333
(gdb) monitor reset halt
(gdb) break main
(gdb) continue
```

For VS Code debugging, install the Cortex-Debug extension and create `.vscode/launch.json`:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "SynapticOS Debug (pyOCD)",
            "type": "cortex-debug",
            "request": "launch",
            "servertype": "pyocd",
            "cwd": "${workspaceFolder}",
            "executable": "${workspaceFolder}/../build/zephyr/zephyr.elf",
            "targetId": "mcxn947",
            "runToEntryPoint": "main",
            "svdFile": ""
        }
    ]
}
```

---

## Step 8: Project Workflow Summary

Day-to-day development commands:

```bash
# Navigate to workspace
cd ~/synaptic-workspace

# Build (incremental — only recompiles changed files)
west build

# Build from scratch
west build -b frdm_mcxn947/mcxn947/cpu0 synaptic-os/samples/hello_inference --pristine

# Flash
west flash

# Serial console
picocom -b 115200 /dev/ttyACM0

# Run tests
west twister -T synaptic-os/tests/unit -p qemu_cortex_m3

# Clean build artifacts
rm -rf build/
```

---

## Directory Reference After Setup

```
~/synaptic-workspace/
├── .west/config
├── synaptic-os/                ← Your code lives here
│   ├── include/synaptic/       ← Public headers
│   ├── src/                    ← Implementation
│   ├── samples/                ← Demo applications
│   ├── tests/                  ← Unit + integration tests
│   ├── boards/                 ← Board overlays
│   ├── tools/                  ← Python utilities
│   └── docs/                   ← Documentation
├── zephyr/                     ← Zephyr RTOS (read-only)
├── modules/hal/nxp/            ← NXP HAL + Neutron SDK (read-only)
├── modules/crypto/mbedtls/     ← TLS library (read-only)
├── modules/lib/cmsis/          ← CMSIS headers (read-only)
├── bootloader/mcuboot/         ← Secure bootloader (read-only)
└── build/                      ← Build output (generated)
    └── zephyr/
        ├── zephyr.elf          ← Debug binary
        ├── zephyr.bin          ← Flash binary
        └── zephyr.map          ← Linker map
```

---

## Next Steps

With the environment validated and the first sample running, proceed to `docs/03-phase-plan.md` to begin implementing the SynapticOS phases.
