#!/usr/bin/env python3
"""SynapticOS — Project Setup and First Build

Automates every step from docs/02-project-setup.md.
Run this AFTER setup_env.py has completed successfully.

Usage:
    python3 scripts/setup_project.py
"""

import os
import subprocess
import sys
import shutil
from pathlib import Path

# ── Colours ──────────────────────────────────────────────────────────────────

GREEN = "\033[0;32m"
RED = "\033[0;31m"
YELLOW = "\033[0;33m"
CYAN = "\033[0;36m"
BOLD = "\033[1m"
NC = "\033[0m"

# ── Paths ────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
SYNAPTIC_OS_DIR = SCRIPT_DIR.parent  # synaptic-os/
WORKSPACE_ROOT = SYNAPTIC_OS_DIR.parent  # parent directory = workspace root
VENV_DIR = Path.home() / ".venvs" / "synaptic"
ZEPHYR_DIR = WORKSPACE_ROOT / "zephyr"

BOARD = "frdm_mcxn947/mcxn947/cpu0"
DEFAULT_SAMPLE = "synaptic-os/samples/hello_inference"

AVAILABLE_SAMPLES = [
    "synaptic-os/samples/hello_inference",
    "synaptic-os/samples/face_detection",
    "synaptic-os/samples/keyword_spotting",
    "synaptic-os/samples/dual_model",
    "synaptic-os/samples/ota_update",
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


def ask_choice(prompt: str, options: list[str], default: int = 0) -> str:
    print(f"  {YELLOW}?{NC} {prompt}")
    for i, opt in enumerate(options):
        marker = ">" if i == default else " "
        print(f"    {marker} [{i + 1}] {opt}")
    while True:
        answer = input(f"    Choice [default={default + 1}]: ").strip()
        if answer == "":
            return options[default]
        try:
            idx = int(answer) - 1
            if 0 <= idx < len(options):
                return options[idx]
        except ValueError:
            pass
        print(f"    Please enter a number between 1 and {len(options)}.")


def venv_env() -> dict[str, str]:
    """Return env dict with venv activated and workspace as cwd context."""
    env = {
        **os.environ,
        "PATH": f"{VENV_DIR / 'bin'}:{os.environ.get('PATH', '')}",
        "VIRTUAL_ENV": str(VENV_DIR),
    }
    # Ensure ZEPHYR_BASE is set if zephyr exists
    if ZEPHYR_DIR.is_dir():
        env["ZEPHYR_BASE"] = str(ZEPHYR_DIR)
    # Propagate SDK path
    sdk_dir = os.environ.get("ZEPHYR_SDK_INSTALL_DIR")
    if not sdk_dir:
        # Try common locations
        for candidate in [
            Path.home() / f"zephyr-sdk-0.16.8",
            Path.home() / ".local" / f"zephyr-sdk-0.16.8",
        ]:
            if candidate.is_dir():
                sdk_dir = str(candidate)
                break
    if sdk_dir:
        env["ZEPHYR_SDK_INSTALL_DIR"] = sdk_dir
    return env


def run(cmd: str | list[str], check: bool = True, capture: bool = False,
        cwd: Path | None = None, env: dict | None = None,
        **kwargs) -> subprocess.CompletedProcess:
    """Run a command, streaming output to the terminal by default."""
    shell = isinstance(cmd, str)
    stdout = subprocess.PIPE if capture else None
    stderr = subprocess.PIPE if capture else None
    if env is None:
        env = venv_env()
    return subprocess.run(
        cmd, shell=shell, check=check, stdout=stdout, stderr=stderr,
        cwd=cwd, env=env, **kwargs,
    )


# ── Pre-flight checks ───────────────────────────────────────────────────────

def preflight() -> None:
    """Verify setup_env.py has been run."""
    errors = []

    if not VENV_DIR.is_dir():
        errors.append("Python venv not found at ~/.venvs/synaptic (run setup_env.py first)")

    west = VENV_DIR / "bin" / "west"
    if not west.is_file():
        errors.append("west not installed in venv (run setup_env.py first)")

    if not shutil.which("cmake"):
        errors.append("cmake not found (run setup_env.py first)")

    if errors:
        fail("Pre-flight checks failed:")
        for e in errors:
            fail(f"  {e}")
        sys.exit(1)

    ok("Pre-flight checks passed.")
    info(f"Workspace root: {WORKSPACE_ROOT}")
    info(f"SynapticOS dir: {SYNAPTIC_OS_DIR}")


# ── Step implementations ────────────────────────────────────────────────────

def step1_create_workspace() -> None:
    banner(1, "Create the Workspace")

    west_dir = WORKSPACE_ROOT / ".west"

    # Check if already initialized
    if west_dir.is_dir():
        ok(f"West workspace already initialized at {WORKSPACE_ROOT}")
        # Verify the manifest points to synaptic-os
        config = west_dir / "config"
        if config.is_file():
            content = config.read_text()
            if "synaptic-os" in content:
                ok("Manifest correctly points to synaptic-os.")
            else:
                warn("West config exists but may not point to synaptic-os.")
                warn(f"Check {config}")
    else:
        info("Initializing west workspace...")
        info(f"Running 'west init -l .' from {SYNAPTIC_OS_DIR}")
        run("west init -l .", cwd=SYNAPTIC_OS_DIR)
        ok("West workspace initialized.")

    # West update — fetch zephyr, hal_nxp, cmsis, mcuboot, mbedtls
    if ZEPHYR_DIR.is_dir() and (ZEPHYR_DIR / "VERSION").is_file():
        ok("Zephyr already fetched.")
        if ask_yes_no("Run 'west update' anyway to sync?", default=False):
            info("Running west update (this may take a few minutes)...")
            run("west update", cwd=WORKSPACE_ROOT)
            ok("West update complete.")
    else:
        info("Running west update (this will take several minutes on first run)...")
        info("Fetching: Zephyr, hal_nxp, cmsis, mcuboot, mbedtls...")
        run("west update", cwd=WORKSPACE_ROOT)
        ok("West update complete.")

    # Install Zephyr's Python requirements
    zephyr_reqs = ZEPHYR_DIR / "scripts" / "requirements.txt"
    if zephyr_reqs.is_file():
        info("Installing Zephyr Python dependencies...")
        run(f"{VENV_DIR / 'bin' / 'pip'} install -r {zephyr_reqs}")
        ok("Zephyr Python dependencies installed.")
    else:
        warn(f"Zephyr requirements.txt not found at {zephyr_reqs}")


def step2_env_variables() -> None:
    banner(2, "Set Environment Variables")

    bashrc = Path.home() / ".bashrc"
    zephyr_base_value = str(ZEPHYR_DIR)
    export_line = f"export ZEPHYR_BASE={zephyr_base_value}"

    # Check if already set in current env
    current = os.environ.get("ZEPHYR_BASE")
    if current and Path(current).resolve() == ZEPHYR_DIR.resolve():
        ok(f"ZEPHYR_BASE is already set to {current}")
    else:
        info(f"ZEPHYR_BASE will be set to {zephyr_base_value}")

    # Check .bashrc
    if bashrc.is_file() and export_line in bashrc.read_text():
        ok("ZEPHYR_BASE already in ~/.bashrc")
    else:
        if ask_yes_no(f"Add ZEPHYR_BASE={zephyr_base_value} to ~/.bashrc?"):
            with open(bashrc, "a") as f:
                f.write(f"\n# Zephyr RTOS base directory\n{export_line}\n")
            ok("Added ZEPHYR_BASE to ~/.bashrc")
        else:
            warn("Skipped. You'll need to export ZEPHYR_BASE manually each session.")

    # Set for current process (used by subsequent steps)
    os.environ["ZEPHYR_BASE"] = zephyr_base_value
    ok(f"ZEPHYR_BASE set for this session: {zephyr_base_value}")


def step3_build() -> None:
    banner(3, "Build Sample")

    sample = ask_choice(
        "Which sample to build?",
        AVAILABLE_SAMPLES,
        default=0,
    )
    sample_name = sample.split("/")[-1]

    info(f"Building '{sample_name}' for {BOARD}...")
    info(f"Command: west build -b {BOARD} {sample} --pristine")

    result = run(
        f"west build -b {BOARD} {sample} --pristine",
        cwd=WORKSPACE_ROOT,
        check=False,
    )

    if result.returncode != 0:
        fail("Build failed. Review the errors above.")
        if not ask_yes_no("Continue to next steps anyway?", default=False):
            sys.exit(1)
    else:
        ok("Build succeeded!")
        # Show binary info
        elf = WORKSPACE_ROOT / "build" / "zephyr" / "zephyr.elf"
        bin_file = WORKSPACE_ROOT / "build" / "zephyr" / "zephyr.bin"
        if elf.is_file():
            size = elf.stat().st_size
            ok(f"ELF: {elf} ({size:,} bytes)")
        if bin_file.is_file():
            size = bin_file.stat().st_size
            ok(f"BIN: {bin_file} ({size:,} bytes)")


def _flash_via_blhost(bin_file: Path) -> None:
    """Flash using NXP blhost via ISP bootloader (fallback when pyOCD fails)."""
    blhost = shutil.which("blhost")
    if not blhost:
        blhost_venv = VENV_DIR / "bin" / "blhost"
        if blhost_venv.is_file():
            blhost = str(blhost_venv)
        else:
            fail("blhost not found. Install with: pip install spsdk")
            return

    print()
    info("To enter ISP mode on the FRDM-MCXN947:")
    info("  1. Hold the ISP button")
    info("  2. Press and release the Reset button")
    info("  3. Release the ISP button")
    print()

    if not ask_yes_no("Board is in ISP mode and ready?"):
        info("Skipping blhost flash.")
        return

    serial_port = "/dev/ttyACM0"
    info("Erasing flash...")
    result = run(f"sudo {blhost} -p {serial_port} flash-erase-all", check=False)
    if result.returncode != 0:
        fail("Flash erase failed. Check board is in ISP mode and serial port is available.")
        return

    info(f"Writing {bin_file.name} to 0x10000000...")
    result = run(
        f"sudo {blhost} -p {serial_port} write-memory 0x10000000 {bin_file}",
        check=False,
    )
    if result.returncode != 0:
        fail("Flash write failed.")
        return

    ok("Flash complete via blhost!")
    info("Press the Reset button (without ISP) to boot the firmware.")
    input("  Press Enter after resetting the board...")


def step4_flash() -> None:
    banner(4, "Flash to Board")

    if not ask_yes_no("Flash the built binary to the FRDM-MCXN947?"):
        info("Skipping flash.")
        return

    build_dir = WORKSPACE_ROOT / "build"
    if not (build_dir / "zephyr" / "zephyr.bin").is_file():
        fail("No build output found. Run the build step first.")
        return

    bin_file = build_dir / "zephyr" / "zephyr.bin"

    info("Flashing via west flash (pyOCD)...")
    result = run("west flash --runner pyocd", cwd=WORKSPACE_ROOT, check=False)

    if result.returncode != 0:
        fail("pyOCD flash failed.")
        warn("The debug port may be locked or unresponsive.")

        if ask_yes_no("Try flashing via blhost (ISP bootloader) instead?", default=True):
            _flash_via_blhost(bin_file)
        else:
            warn("You can flash manually with:")
            warn("  1. Enter ISP mode: hold ISP button, press reset, release reset, release ISP")
            warn(f"  2. sudo blhost -p /dev/ttyACM0 flash-erase-all")
            warn(f"  3. sudo blhost -p /dev/ttyACM0 write-memory 0x10000000 {bin_file}")
            warn("  4. Press reset to boot")
    else:
        ok("Flash complete!")


def step5_serial_console() -> None:
    banner(5, "Connect Serial Console")

    serial_port = "/dev/ttyACM0"
    if not Path(serial_port).exists():
        warn(f"Serial port {serial_port} not found.")
        warn("Connect the board and check with: ls /dev/ttyACM*")
        return

    if not ask_yes_no(f"Open serial console on {serial_port} at 115200 baud?"):
        info("Skipping serial console.")
        info(f"You can connect manually: picocom -b 115200 {serial_port}")
        return

    # Check if we have permission to open the serial port
    if not os.access(serial_port, os.R_OK | os.W_OK):
        warn(f"No permission to access {serial_port}.")
        warn("You may need to log out and back in for the plugdev group to take effect.")
        if not ask_yes_no("Try with sudo instead?"):
            info(f"Connect manually after re-login: picocom -b 115200 {serial_port}")
            return
        picocom_cmd = f"sudo picocom -b 115200 {serial_port}"
    else:
        picocom_cmd = f"picocom -b 115200 {serial_port}"

    info(f"Opening picocom on {serial_port}...")
    info("Press Ctrl+A then Ctrl+X to exit picocom.")
    print()

    # Use os.system() for full TTY control — subprocess.run() doesn't
    # properly pass stdin/stdout to interactive terminal programs like picocom.
    os.system(picocom_cmd)


def step6_run_tests() -> None:
    banner(6, "Run Unit Tests")

    if not ask_yes_no("Run unit tests on QEMU?"):
        info("Skipping tests.")
        return

    info("Running tests with west twister on qemu_cortex_m3...")
    result = run(
        "west twister -T synaptic-os/tests/unit -p qemu_cortex_m3",
        cwd=WORKSPACE_ROOT,
        check=False,
    )

    if result.returncode != 0:
        warn("Some tests failed. Review the output above.")
    else:
        ok("All tests passed!")

    # Optionally run on hardware
    if Path("/dev/ttyACM0").exists():
        if ask_yes_no("Also run tests on hardware (board must be connected)?", default=False):
            info("Running tests on FRDM-MCXN947...")
            run(
                f"west twister -T synaptic-os/tests/unit -p {BOARD} "
                f"--device-testing --device-serial /dev/ttyACM0",
                cwd=WORKSPACE_ROOT,
                check=False,
            )


def step7_debug_setup() -> None:
    banner(7, "Debug Setup (Optional)")

    if not ask_yes_no("Set up VS Code debug configuration?", default=False):
        info("Skipping debug setup.")
        return

    launch_json = SYNAPTIC_OS_DIR / ".vscode" / "launch.json"
    if launch_json.is_file():
        ok("launch.json already exists.")
        if not ask_yes_no("Overwrite with SynapticOS debug config?", default=False):
            info("Keeping existing launch.json.")
            return

    launch_config = '''{
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
'''
    launch_json.parent.mkdir(parents=True, exist_ok=True)
    launch_json.write_text(launch_config)
    ok(f"Created {launch_json}")

    info("To debug:")
    info("  1. Terminal 1: pyocd gdbserver -t mcxn947")
    info("  2. VS Code: Run 'SynapticOS Debug (pyOCD)' configuration")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"\n{BOLD}{'═' * 60}{NC}")
    print(f"{BOLD}  SynapticOS — Project Setup and First Build{NC}")
    print(f"{BOLD}{'═' * 60}{NC}")
    print()
    print("  This tool automates the setup described in")
    print("  docs/02-project-setup.md. It will prompt you when")
    print("  decisions are needed.")
    print()
    print(f"  Workspace root: {WORKSPACE_ROOT}")
    print(f"  SynapticOS:     {SYNAPTIC_OS_DIR}")
    print()

    if os.geteuid() == 0:
        fail("Do not run this script as root.")
        sys.exit(1)

    preflight()

    if not ask_yes_no("Ready to begin?"):
        sys.exit(0)

    step1_create_workspace()
    step2_env_variables()
    step3_build()
    step4_flash()
    step5_serial_console()
    step6_run_tests()
    step7_debug_setup()

    print(f"\n{BOLD}{GREEN}{'═' * 60}{NC}")
    print(f"{BOLD}{GREEN}  Project setup complete!{NC}")
    print(f"{BOLD}{GREEN}{'═' * 60}{NC}")
    print()
    print("  Day-to-day commands (run from workspace root):")
    print(f"    cd {WORKSPACE_ROOT}")
    print(f"    west build -b {BOARD} synaptic-os/samples/hello_inference --pristine")
    print("    west flash")
    print("    picocom -b 115200 /dev/ttyACM0")
    print()
    print("  Next: proceed to docs/03-phase-plan.md")
    print()


if __name__ == "__main__":
    main()
