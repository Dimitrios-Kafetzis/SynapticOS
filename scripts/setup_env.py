#!/usr/bin/env python3
"""SynapticOS — Ubuntu 24.04 Development Environment Setup

Automates every step from docs/01-ubuntu-setup.md.
Requires only Python 3 stdlib (runs before pip/venv exist).

Usage:
    python3 scripts/setup_env.py
"""

import os
import subprocess
import sys
import shutil
import textwrap
from pathlib import Path

# ── Colours ──────────────────────────────────────────────────────────────────

GREEN = "\033[0;32m"
RED = "\033[0;31m"
YELLOW = "\033[0;33m"
CYAN = "\033[0;36m"
BOLD = "\033[1m"
NC = "\033[0m"

ZEPHYR_SDK_VERSION = "0.16.8"
ZEPHYR_SDK_DIR_NAME = f"zephyr-sdk-{ZEPHYR_SDK_VERSION}"
ZEPHYR_SDK_TARBALL = f"{ZEPHYR_SDK_DIR_NAME}_linux-x86_64.tar.xz"
ZEPHYR_SDK_URL = (
    f"https://github.com/zephyrproject-rtos/sdk-ng/releases/download/"
    f"v{ZEPHYR_SDK_VERSION}/{ZEPHYR_SDK_TARBALL}"
)
VENV_DIR = Path.home() / ".venvs" / "synaptic"

SYSTEM_PACKAGES = [
    "git", "cmake", "ninja-build", "gperf", "ccache", "dfu-util",
    "device-tree-compiler", "wget", "curl", "xz-utils", "file",
    "make", "gcc", "gcc-multilib", "g++-multilib",
    "python3-dev", "python3-pip", "python3-venv", "python3-setuptools",
    "python3-tk", "python3-wheel",
    "libsdl2-dev", "libmagic1",
    "picocom", "minicom",
    "usbutils",
]

UDEV_NXP = textwrap.dedent("""\
    # NXP MCU-Link (CMSIS-DAP)
    SUBSYSTEM=="usb", ATTR{idVendor}=="1fc9", ATTR{idProduct}=="0143", MODE="0666", GROUP="plugdev"
    SUBSYSTEM=="usb", ATTR{idVendor}=="0d28", ATTR{idProduct}=="0204", MODE="0666", GROUP="plugdev"
    # NXP LPC-Link2 / MCU-Link Pro
    SUBSYSTEM=="usb", ATTR{idVendor}=="1fc9", ATTR{idProduct}=="0090", MODE="0666", GROUP="plugdev"
""")

UDEV_PYOCD = textwrap.dedent("""\
    # CMSIS-DAP compatible devices
    SUBSYSTEM=="usb", ATTR{idVendor}=="0d28", MODE="0666", GROUP="plugdev"
    SUBSYSTEM=="usb", ATTR{idVendor}=="1fc9", MODE="0666", GROUP="plugdev"
""")

VSCODE_EXTENSIONS = [
    "ms-vscode.cpptools",
    "ms-vscode.cmake-tools",
    "marus25.cortex-debug",
    "trond-snekvik.simple-rst",
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def banner(step: int, title: str) -> None:
    print(f"\n{BOLD}{CYAN}{'═' * 60}{NC}")
    print(f"{BOLD}{CYAN}  Step {step}: {title}{NC}")
    print(f"{BOLD}{CYAN}{'═' * 60}{NC}\n")


def info(msg: str) -> None:
    print(f"  {CYAN}▸{NC} {msg}")


def ok(msg: str) -> None:
    print(f"  {GREEN}✓{NC} {msg}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}!{NC} {msg}")


def fail(msg: str) -> None:
    print(f"  {RED}✗{NC} {msg}")


def ask_yes_no(prompt: str, default: bool = True) -> bool:
    suffix = " [Y/n] " if default else " [y/N] "
    while True:
        answer = input(f"  {YELLOW}?{NC} {prompt}{suffix}").strip().lower()
        if answer == "":
            return default
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("    Please answer y or n.")


def ask_input(prompt: str, default: str = "") -> str:
    suffix = f" [{default}] " if default else " "
    answer = input(f"  {YELLOW}?{NC} {prompt}{suffix}").strip()
    return answer if answer else default


def run(cmd: str | list[str], check: bool = True, capture: bool = False,
        env: dict | None = None, **kwargs) -> subprocess.CompletedProcess:
    """Run a command, streaming output to the terminal by default."""
    shell = isinstance(cmd, str)
    stdout = subprocess.PIPE if capture else None
    stderr = subprocess.PIPE if capture else None
    merged_env = None
    if env:
        merged_env = {**os.environ, **env}
    return subprocess.run(
        cmd, shell=shell, check=check, stdout=stdout, stderr=stderr,
        env=merged_env, **kwargs,
    )


def command_exists(name: str) -> bool:
    return shutil.which(name) is not None


def venv_pip() -> str:
    return str(VENV_DIR / "bin" / "pip")


def venv_python() -> str:
    return str(VENV_DIR / "bin" / "python")


def venv_bin(name: str) -> str:
    return str(VENV_DIR / "bin" / name)


# ── Step implementations ────────────────────────────────────────────────────

def step1_system_packages() -> None:
    banner(1, "System Packages")

    # Check which packages are already installed
    info("Checking installed packages...")
    result = run("dpkg -l", capture=True, check=False)
    installed = result.stdout.decode() if result.stdout else ""

    missing = []
    for pkg in SYSTEM_PACKAGES:
        # dpkg -l lines look like: ii  package-name  version ...
        if f"ii  {pkg} " not in installed and f"ii  {pkg}:" not in installed:
            missing.append(pkg)

    if not missing:
        ok("All system packages are already installed.")
        return

    info(f"{len(missing)} packages to install: {', '.join(missing)}")
    info("Running apt update...")
    run("sudo apt update")
    info("Running apt upgrade...")
    run("sudo apt upgrade -y", check=False)
    info("Installing packages...")
    # Install in a single command with apt-get (more reliable in scripts)
    result = run(
        f"sudo apt-get install -y --no-install-recommends {' '.join(missing)}",
        check=False,
    )
    if result.returncode != 0:
        fail("Some packages failed to install. Trying individually...")
        failed_pkgs = []
        for pkg in missing:
            r = run(f"sudo apt-get install -y --no-install-recommends {pkg}", check=False)
            if r.returncode != 0:
                failed_pkgs.append(pkg)
            else:
                ok(f"Installed {pkg}")
        if failed_pkgs:
            fail(f"Could not install: {', '.join(failed_pkgs)}")
            warn("You may need to install these manually or check your apt sources.")
            if not ask_yes_no("Continue anyway?"):
                sys.exit(1)
        else:
            ok("All packages installed (individually).")
    else:
        ok("System packages installed.")


def step2_python_venv() -> None:
    banner(2, "Python Virtual Environment")

    if VENV_DIR.exists() and (VENV_DIR / "bin" / "activate").exists():
        ok(f"Virtual environment already exists at {VENV_DIR}")
    else:
        info(f"Creating virtual environment at {VENV_DIR}...")
        VENV_DIR.parent.mkdir(parents=True, exist_ok=True)
        run(f"python3 -m venv {VENV_DIR}")
        ok("Virtual environment created.")

    # Install pip packages inside the venv
    info("Installing west, pyocd, intelhex in venv...")
    run(f"{venv_pip()} install --upgrade pip")
    run(f"{venv_pip()} install west pyocd intelhex 'setuptools<81'")
    ok("Python packages installed.")

    # Check .bashrc for activation line
    bashrc = Path.home() / ".bashrc"
    activation_line = f"source {VENV_DIR}/bin/activate"
    already_in_bashrc = False
    if bashrc.exists():
        already_in_bashrc = activation_line in bashrc.read_text()

    if already_in_bashrc:
        ok("Shell profile already sources the venv.")
    else:
        if ask_yes_no(f"Add auto-activation to ~/.bashrc?"):
            with open(bashrc, "a") as f:
                f.write(f"\n# SynapticOS Python venv\n{activation_line}\n")
            ok("Added venv activation to ~/.bashrc")
        else:
            warn("Skipped. Remember to activate manually: "
                 f"source {VENV_DIR}/bin/activate")


def _find_existing_sdk() -> str | None:
    """Look for an existing Zephyr SDK installation."""
    # Check env var first
    env_path = os.environ.get("ZEPHYR_SDK_INSTALL_DIR")
    if env_path and Path(env_path).is_dir():
        return env_path

    # Check common locations
    candidates = [
        Path.home() / ZEPHYR_SDK_DIR_NAME,
        Path.home() / ".local" / ZEPHYR_SDK_DIR_NAME,
        Path("/opt") / ZEPHYR_SDK_DIR_NAME,
    ]
    for p in candidates:
        if p.is_dir() and (p / "setup.sh").exists():
            return str(p)
    return None


# Populated by step3, used by step8 for validation env
_detected_sdk_path: str | None = None


def step3_zephyr_sdk() -> None:
    global _detected_sdk_path
    banner(3, "Zephyr SDK")

    existing = _find_existing_sdk()
    sdk_path: str | None = None

    if existing:
        ok(f"Found existing Zephyr SDK at: {existing}")
        if ask_yes_no("Use this existing installation?"):
            sdk_path = existing
        else:
            info("Will install a fresh copy.")

    if sdk_path is None:
        install_dir = Path.home()
        tarball_path = install_dir / ZEPHYR_SDK_TARBALL
        sdk_path = str(install_dir / ZEPHYR_SDK_DIR_NAME)

        if Path(sdk_path).is_dir():
            ok(f"SDK directory already exists at {sdk_path}, skipping download.")
        else:
            info(f"Downloading Zephyr SDK {ZEPHYR_SDK_VERSION}...")
            run(f"wget -c -P {install_dir} {ZEPHYR_SDK_URL}")
            info("Extracting...")
            run(f"tar xf {tarball_path} -C {install_dir}")
            # Clean up tarball
            tarball_path.unlink(missing_ok=True)
            ok("SDK extracted.")

        info("Running SDK setup (installing ARM toolchain, host tools, CMake package)...")
        run(f"bash {sdk_path}/setup.sh -t arm-zephyr-eabi -h -c")

    # Register cmake package (use the SDK's own export script)
    cmake_export = Path(sdk_path) / "cmake" / "zephyr_sdk_export.cmake"
    if cmake_export.exists():
        info("Registering Zephyr SDK CMake package...")
        run(f"cmake -P {cmake_export}", check=False)
        ok("CMake package registered.")
    else:
        warn(f"CMake export script not found at {cmake_export}, skipping registration.")

    # Export ZEPHYR_SDK_INSTALL_DIR in .bashrc
    bashrc = Path.home() / ".bashrc"
    export_line = f"export ZEPHYR_SDK_INSTALL_DIR={sdk_path}"
    if bashrc.exists() and export_line in bashrc.read_text():
        ok("ZEPHYR_SDK_INSTALL_DIR already in .bashrc")
    else:
        if ask_yes_no(f"Add ZEPHYR_SDK_INSTALL_DIR={sdk_path} to ~/.bashrc?"):
            with open(bashrc, "a") as f:
                f.write(f"\n# Zephyr SDK location\n{export_line}\n")
            ok("Added ZEPHYR_SDK_INSTALL_DIR to ~/.bashrc")
        else:
            warn("Skipped. Set ZEPHYR_SDK_INSTALL_DIR manually if needed.")

    # Verify
    gcc = Path(sdk_path) / "arm-zephyr-eabi" / "bin" / "arm-zephyr-eabi-gcc"
    result = run(f"{gcc} --version", capture=True, check=False)
    if result.returncode == 0:
        version_line = result.stdout.decode().splitlines()[0]
        ok(f"ARM toolchain: {version_line}")
    else:
        fail("ARM toolchain not found. Check the SDK installation.")
        sys.exit(1)

    _detected_sdk_path = sdk_path


def step4_udev_rules() -> None:
    banner(4, "USB / Debug Probe Permissions")

    nxp_rules = Path("/etc/udev/rules.d/99-nxp-mculink.rules")
    pyocd_rules = Path("/etc/udev/rules.d/99-pyocd.rules")

    if nxp_rules.exists() and pyocd_rules.exists():
        ok("udev rules already installed.")
    else:
        info("Writing udev rules...")
        run(f"sudo tee {nxp_rules} > /dev/null", input=UDEV_NXP.encode())
        run(f"sudo tee {pyocd_rules} > /dev/null", input=UDEV_PYOCD.encode())
        info("Reloading udev rules...")
        run("sudo udevadm control --reload-rules")
        run("sudo udevadm trigger")
        ok("udev rules installed and reloaded.")

    # Add user to plugdev
    user = os.environ.get("USER", "")
    result = run(f"id -nG {user}", capture=True, check=False)
    groups = result.stdout.decode().strip() if result.stdout else ""
    if "plugdev" in groups.split():
        ok(f"User '{user}' is already in the plugdev group.")
    else:
        info(f"Adding '{user}' to plugdev group...")
        run(f"sudo usermod -aG plugdev {user}")
        ok(f"Added '{user}' to plugdev.")
        warn("You must log out and back in for the group change to take effect.")


def step5_verify_board() -> None:
    banner(5, "Verify Board Connection")

    info("Checking USB devices...")
    result = run("lsusb", capture=True, check=False)
    usb_output = result.stdout.decode() if result.stdout else ""

    nxp_found = False
    for line in usb_output.splitlines():
        lower = line.lower()
        if any(kw in lower for kw in ["nxp", "1fc9", "0d28", "dap", "lpc"]):
            ok(f"USB: {line.strip()}")
            nxp_found = True

    if not nxp_found:
        warn("No NXP / CMSIS-DAP device detected via USB.")
        warn("Make sure the FRDM-MCXN947 is connected via the MCU-Link USB-C port.")

    # Check serial port
    serial_exists = Path("/dev/ttyACM0").exists()
    if serial_exists:
        ok("Serial port /dev/ttyACM0 is present.")
    else:
        warn("Serial port /dev/ttyACM0 not found.")

    if not nxp_found or not serial_exists:
        if not ask_yes_no("Board not fully detected. Continue anyway?"):
            info("Connect the board and re-run the script.")
            sys.exit(0)


def step6_pyocd() -> None:
    banner(6, "Verify pyOCD")

    pyocd = venv_bin("pyocd")
    info("Running pyocd list...")
    result = run(f"{pyocd} list", capture=True, check=False)
    output = result.stdout.decode() if result.stdout else ""
    print(output)

    if "no connected" in output.lower() or not output.strip():
        warn("No target found by pyOCD (board may not be connected).")
    else:
        ok("pyOCD can see the target.")

    # Install MCXN947 pack support if not already present
    info("Checking MCXN947 target support...")
    result = run(f"{pyocd} list --targets", capture=True, check=False)
    targets_output = result.stdout.decode() if result.stdout else ""
    if "mcxn947" in targets_output.lower():
        ok("MCXN947 target pack already installed.")
    else:
        info("Installing MCXN947 CMSIS pack...")
        run(f"{pyocd} pack install MCXN947VDF", check=False)
        ok("MCXN947 pack installed.")


def step7_vscode() -> None:
    banner(7, "VS Code Setup (Optional)")

    if not ask_yes_no("Install VS Code and recommended extensions?", default=False):
        info("Skipping VS Code setup.")
        return

    if command_exists("code"):
        ok("VS Code is already installed.")
    else:
        info("Installing VS Code via snap...")
        run("sudo snap install code --classic")
        ok("VS Code installed.")

    info("Installing recommended extensions...")
    for ext in VSCODE_EXTENSIONS:
        info(f"  {ext}")
        run(f"code --install-extension {ext}", check=False)
    ok("Extensions installed.")


def step8_validate() -> None:
    banner(8, "Validate Full Toolchain")

    script_dir = Path(__file__).resolve().parent
    validate_script = script_dir / "validate_env.sh"

    if validate_script.exists():
        info("Running validate_env.sh...")
        # Run validation inside the venv so west/pyocd are on PATH
        env = {
            "PATH": f"{VENV_DIR / 'bin'}:{os.environ.get('PATH', '')}",
        }
        if _detected_sdk_path:
            env["ZEPHYR_SDK_INSTALL_DIR"] = _detected_sdk_path
        result = run(f"bash {validate_script}", check=False, env=env)
        if result.returncode == 0:
            ok("All validation checks passed!")
        else:
            warn("Some checks failed — review the output above.")
    else:
        warn(f"Validation script not found at {validate_script}")
        info("Performing inline checks instead...")
        _inline_validate()


def _inline_validate() -> None:
    """Fallback validation if validate_env.sh is missing."""
    checks = [
        ("python3", "python3 --version"),
        ("west", f"{venv_bin('west')} --version"),
        ("cmake", "cmake --version"),
        ("ninja", "ninja --version"),
        ("dtc", "dtc --version"),
        ("pyocd", f"{venv_bin('pyocd')} --version"),
        ("picocom", "which picocom"),
    ]
    for name, cmd in checks:
        result = run(cmd, capture=True, check=False)
        out = result.stdout.decode().strip().splitlines()[0] if result.stdout else ""
        if result.returncode == 0 and out:
            ok(f"{name}: {out}")
        else:
            fail(f"{name}: NOT FOUND")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"\n{BOLD}{'═' * 60}{NC}")
    print(f"{BOLD}  SynapticOS — Ubuntu 24.04 Environment Setup{NC}")
    print(f"{BOLD}{'═' * 60}{NC}")
    print()
    print("  This tool automates the setup described in")
    print("  docs/01-ubuntu-setup.md. It will prompt you when")
    print("  decisions are needed.")
    print()

    if os.geteuid() == 0:
        fail("Do not run this script as root. It will use sudo when needed.")
        sys.exit(1)

    if not ask_yes_no("Ready to begin?"):
        sys.exit(0)

    step1_system_packages()
    step2_python_venv()
    step3_zephyr_sdk()
    step4_udev_rules()
    step5_verify_board()
    step6_pyocd()
    step7_vscode()
    step8_validate()

    print(f"\n{BOLD}{GREEN}{'═' * 60}{NC}")
    print(f"{BOLD}{GREEN}  Setup complete!{NC}")
    print(f"{BOLD}{GREEN}{'═' * 60}{NC}")
    print()
    print("  Next steps:")
    print("    1. Log out and back in (for plugdev group)")
    print("    2. Open a new terminal (for venv activation)")
    print(f"    3. Proceed to docs/02-project-setup.md")
    print()


if __name__ == "__main__":
    main()
