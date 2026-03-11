# SynapticOS — Ubuntu 24.04 LTS Development Environment Setup

This guide walks through setting up a complete development environment for SynapticOS on Ubuntu 24.04.4 LTS. After completing these steps you will have a fully functional toolchain capable of building, flashing, debugging, and testing firmware for the NXP FRDM-MCXN947 development board.

---

## Prerequisites

- Ubuntu 24.04.4 LTS (native install or dedicated workstation — not a VM, for reliable USB pass-through)
- Sudo privileges
- Internet connection
- NXP FRDM-MCXN947 board connected via USB-C cable

---

## Step 1: System Packages

Install all required system-level dependencies in one pass:

```bash
sudo apt update && sudo apt upgrade -y

sudo apt install -y --no-install-recommends \
    git cmake ninja-build gperf ccache dfu-util \
    device-tree-compiler wget curl xz-utils file \
    make gcc gcc-multilib g++-multilib \
    python3-dev python3-pip python3-venv python3-setuptools \
    python3-tk python3-wheel \
    libsdl2-dev libmagic1 \
    picocom minicom \
    usbutils \
    dtc
```

**What each group provides:**

- `git cmake ninja-build gperf ccache` — Build system tools required by Zephyr
- `device-tree-compiler dtc` — Device tree compilation for Zephyr board definitions
- `python3-*` — Python environment for west, Zephyr scripts, and SynapticOS tools
- `picocom minicom` — Serial terminal for UART console access to the board
- `usbutils` — `lsusb` for verifying board connection

---

## Step 2: Python Virtual Environment

Create an isolated Python environment to avoid conflicts with system packages:

```bash
python3 -m venv ~/.venvs/synaptic
source ~/.venvs/synaptic/bin/activate

# Add to your shell profile for automatic activation
echo 'source ~/.venvs/synaptic/bin/activate' >> ~/.bashrc
```

Install the west meta-tool and Zephyr dependencies:

```bash
pip install west pyocd intelhex
```

---

## Step 3: Zephyr SDK

The Zephyr SDK provides the ARM cross-compiler toolchain (arm-zephyr-eabi-gcc), OpenOCD, and QEMU. Download and install the latest SDK:

```bash
cd ~
wget https://github.com/zephyrproject-rtos/sdk-ng/releases/download/v0.16.8/zephyr-sdk-0.16.8_linux-x86_64.tar.xz
tar xf zephyr-sdk-0.16.8_linux-x86_64.tar.xz
cd zephyr-sdk-0.16.8
./setup.sh

# Register the CMake package (allows Zephyr to find the SDK)
sudo cp -a cmake/zephyr-sdk.cmake /usr/local/lib/cmake/
```

Verify the ARM toolchain is accessible:

```bash
~/.local/zephyr-sdk-0.16.8/arm-zephyr-eabi/bin/arm-zephyr-eabi-gcc --version
```

**Note:** If you already have the Zephyr SDK installed at a different path, set `ZEPHYR_SDK_INSTALL_DIR` in your environment instead of reinstalling.

---

## Step 4: USB / Debug Probe Permissions

The FRDM-MCXN947's on-board MCU-Link debugger uses CMSIS-DAP over USB. Configure udev rules so you can flash without sudo:

```bash
# NXP MCU-Link / LPC-Link2 CMSIS-DAP
sudo tee /etc/udev/rules.d/99-nxp-mculink.rules << 'UDEV'
# NXP MCU-Link (CMSIS-DAP)
SUBSYSTEM=="usb", ATTR{idVendor}=="1fc9", ATTR{idProduct}=="0143", MODE="0666", GROUP="plugdev"
SUBSYSTEM=="usb", ATTR{idVendor}=="0d28", ATTR{idProduct}=="0204", MODE="0666", GROUP="plugdev"
# NXP LPC-Link2 / MCU-Link Pro
SUBSYSTEM=="usb", ATTR{idVendor}=="1fc9", ATTR{idProduct}=="0090", MODE="0666", GROUP="plugdev"
UDEV

# pyOCD rules
sudo tee /etc/udev/rules.d/99-pyocd.rules << 'UDEV'
# CMSIS-DAP compatible devices
SUBSYSTEM=="usb", ATTR{idVendor}=="0d28", MODE="0666", GROUP="plugdev"
SUBSYSTEM=="usb", ATTR{idVendor}=="1fc9", MODE="0666", GROUP="plugdev"
UDEV

sudo udevadm control --reload-rules
sudo udevadm trigger

# Add yourself to plugdev group
sudo usermod -aG plugdev $USER
```

**Important:** Log out and back in (or reboot) for the group change to take effect.

---

## Step 5: Verify Board Connection

Plug in the FRDM-MCXN947 via the USB-C cable (use the port labeled "MCU-Link"):

```bash
# Check USB enumeration
lsusb | grep -i "nxp\|arm\|dap\|lpc"

# Should show something like:
# Bus 001 Device 005: ID 1fc9:0143 NXP Semiconductors MCU-LINK CMSIS-DAP

# Check the serial port appeared
ls -la /dev/ttyACM*

# Test serial connection (115200 baud, press Ctrl+A then Ctrl+X to exit picocom)
picocom -b 115200 /dev/ttyACM0
```

If the board came with the factory blinky demo, you should see the RGB LED cycling. Press the RESET button and check the serial console for any output.

---

## Step 6: Verify pyOCD Can See the Target

```bash
pyocd list

# Should show:
#   0   1fc9:0143   MCU-LINK CMSIS-DAP   [frdm_mcxn947]
```

If the target is not recognized, update pyOCD's pack support:

```bash
pyocd pack install nxp
pyocd pack find mcxn
```

---

## Step 7: Optional — VS Code Setup

If you plan to use VS Code for development:

```bash
# Install VS Code (if not already installed)
sudo snap install code --classic

# Install recommended extensions
code --install-extension ms-vscode.cpptools
code --install-extension ms-vscode.cmake-tools
code --install-extension marus25.cortex-debug
code --install-extension trond-snekvik.simple-rst
```

Create a VS Code workspace settings file at `.vscode/settings.json` in the project root:

```json
{
    "cmake.configureOnOpen": false,
    "C_Cpp.default.compilerPath": "${env:HOME}/.local/zephyr-sdk-0.16.8/arm-zephyr-eabi/bin/arm-zephyr-eabi-gcc",
    "C_Cpp.default.includePath": [
        "${workspaceFolder}/include",
        "${workspaceFolder}/../zephyr/include",
        "${workspaceFolder}/../modules/hal/nxp/mcux/mcux-sdk/devices/MCXN947"
    ],
    "cortex-debug.JLinkGDBServerPath": "",
    "cortex-debug.openocdPath": ""
}
```

---

## Step 8: Validate the Full Toolchain

Run this quick validation script to confirm everything is in place:

```bash
#!/bin/bash
echo "=== SynapticOS Environment Validation ==="

echo -n "Python 3: "
python3 --version 2>/dev/null || echo "MISSING"

echo -n "west: "
west --version 2>/dev/null || echo "MISSING"

echo -n "cmake: "
cmake --version 2>/dev/null | head -1 || echo "MISSING"

echo -n "ninja: "
ninja --version 2>/dev/null || echo "MISSING"

echo -n "dtc: "
dtc --version 2>/dev/null | head -1 || echo "MISSING"

echo -n "ARM GCC: "
~/.local/zephyr-sdk-0.16.8/arm-zephyr-eabi/bin/arm-zephyr-eabi-gcc --version 2>/dev/null | head -1 || echo "MISSING"

echo -n "pyocd: "
pyocd --version 2>/dev/null || echo "MISSING"

echo -n "picocom: "
which picocom 2>/dev/null || echo "MISSING"

echo -n "Board USB: "
lsusb 2>/dev/null | grep -qi "nxp\|1fc9\|0d28" && echo "DETECTED" || echo "NOT FOUND"

echo -n "Serial port: "
ls /dev/ttyACM0 2>/dev/null && echo "" || echo "NOT FOUND"

echo "=== Done ==="
```

Save this as `scripts/validate_env.sh` and run with `bash scripts/validate_env.sh`.

---

## Troubleshooting

**`west flash` fails with "No target device found":**
- Check USB cable (must be the MCU-Link USB-C port, not the other one)
- Verify udev rules are loaded: `sudo udevadm control --reload-rules`
- Try `pyocd list` to confirm the board is visible
- If using a USB hub, try connecting directly to the host

**Serial console shows garbled text:**
- Ensure baud rate is 115200: `picocom -b 115200 /dev/ttyACM0`
- Check that no other process has the serial port open

**`pip install` fails with "externally-managed-environment":**
- Make sure you activated the venv: `source ~/.venvs/synaptic/bin/activate`

**`cmake` can't find Zephyr SDK:**
- Set the environment variable: `export ZEPHYR_SDK_INSTALL_DIR=~/.local/zephyr-sdk-0.16.8`
- Or re-run `setup.sh` in the SDK directory

---

## Next Steps

Once all validation checks pass, proceed to [02-project-setup.md](02-project-setup.md) to initialize the SynapticOS workspace and build the first sample.
