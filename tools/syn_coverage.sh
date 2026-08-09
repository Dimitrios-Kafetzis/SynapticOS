#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# syn_coverage.sh - gcov line coverage of src/core on QEMU.
#
# Builds both test apps with library-only gcov instrumentation
# (CONFIG_SYNAPTIC_COVERAGE via each app's coverage.conf), runs them
# under qemu_cortex_m3, harvests the console coverage dumps with
# Zephyr's gen_gcov_files.py, and merges everything into one gcovr
# report (text + HTML).
#
# Why not `west twister --coverage`: global CONFIG_COVERAGE
# instruments the whole image and overflows the 64 KB QEMU target by
# ~245 KB flash / ~128 KB RAM. The library-only scheme fits; see the
# CONFIG_SYNAPTIC_COVERAGE help and the root CMakeLists.
#
# Scope note (honest numbers): the report covers the QEMU-buildable
# subset of src/core. Dual-core/board-only files (syn_boot, syn_ipc,
# syn_mpu, syn_infer_remote, syn_shell, syn_flash_port_mcx hardware
# paths) never build on QEMU and are exercised in the hardware
# sessions instead. syn_flash_port_ram.c is itself QEMU test
# infrastructure.
#
# Usage: from the west workspace root (the parent of synaptic-os/):
#   ./synaptic-os/tools/syn_coverage.sh
# Output: coverage-report/ (index.html + summary.txt) in the
# workspace root.

set -euo pipefail

WS="$(pwd)"
if [ ! -d "$WS/synaptic-os" ] || [ ! -d "$WS/zephyr" ]; then
    echo "error: run from the west workspace root (parent of synaptic-os/)" >&2
    exit 1
fi

QEMU="${QEMU:-$(find "$HOME" -maxdepth 6 -path '*sysroots*/usr/bin/qemu-system-arm' 2>/dev/null | head -1)}"
GCOV="${GCOV:-$(find "$HOME" -maxdepth 4 -path '*arm-zephyr-eabi/bin/arm-zephyr-eabi-gcov' 2>/dev/null | head -1)}"
[ -x "$QEMU" ] || { echo "error: qemu-system-arm not found (set QEMU=)" >&2; exit 1; }
[ -x "$GCOV" ] || { echo "error: arm-zephyr-eabi-gcov not found (set GCOV=)" >&2; exit 1; }

run_app() {
    local app="$1"
    local bdir="$WS/build-cov-$app"
    local log="$bdir/qemu-console.log"

    echo "=== $app: build (instrumented) ==="
    west build -b qemu_cortex_m3 "synaptic-os/tests/$app" --pristine \
        -d "$bdir" -- -DEXTRA_CONF_FILE=coverage.conf

    echo "=== $app: run under QEMU ==="
    timeout 240 "$QEMU" -cpu cortex-m3 -machine lm3s6965evb \
        -display none -monitor none -serial stdio -net none \
        -icount shift=6,align=off,sleep=off -rtc clock=vm \
        -kernel "$bdir/zephyr/zephyr.elf" > "$log" 2>&1 &
    local qpid=$!
    for _ in $(seq 1 48); do
        grep -q "GCOV_COVERAGE_DUMP_END" "$log" 2>/dev/null && break
        sleep 5
    done
    kill "$qpid" 2>/dev/null || true
    wait "$qpid" 2>/dev/null || true

    grep -q "PROJECT EXECUTION SUCCESSFUL" "$log" || {
        echo "error: $app tests did not pass under instrumentation" >&2
        exit 1
    }
    grep -q "GCOV_COVERAGE_DUMP_END" "$log" || {
        echo "error: $app produced no coverage dump" >&2
        exit 1
    }

    echo "=== $app: harvest gcda ==="
    python3 zephyr/scripts/gen_gcov_files.py -i "$log"
}

run_app unit
run_app unit_store

echo "=== merge + report ==="
mkdir -p "$WS/coverage-report"
gcovr --root synaptic-os --filter 'synaptic-os/src/core/' \
    --gcov-executable "$GCOV" \
    "$WS/build-cov-unit" "$WS/build-cov-unit_store" \
    --html-details "$WS/coverage-report/index.html" \
    --print-summary | tee "$WS/coverage-report/summary.txt"

echo "report: $WS/coverage-report/index.html"
