#!/bin/bash
# SynapticOS — Development Environment Validation
# Run: bash scripts/validate_env.sh

set -u

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

pass=0
fail=0
warn=0

check() {
    local name="$1"
    local cmd="$2"
    local result
    result=$(eval "$cmd" 2>/dev/null)
    if [ $? -eq 0 ] && [ -n "$result" ]; then
        echo -e "  ${GREEN}✓${NC} ${name}: ${result}"
        ((pass++))
    else
        echo -e "  ${RED}✗${NC} ${name}: NOT FOUND"
        ((fail++))
    fi
}

check_optional() {
    local name="$1"
    local cmd="$2"
    local result
    result=$(eval "$cmd" 2>/dev/null)
    if [ $? -eq 0 ] && [ -n "$result" ]; then
        echo -e "  ${GREEN}✓${NC} ${name}: ${result}"
        ((pass++))
    else
        echo -e "  ${YELLOW}○${NC} ${name}: not found (optional)"
        ((warn++))
    fi
}

echo ""
echo "═══════════════════════════════════════════════════"
echo "  SynapticOS — Environment Validation"
echo "═══════════════════════════════════════════════════"
echo ""

echo "Build Tools:"
check "Python 3"        "python3 --version | head -1"
check "west"            "west --version 2>&1 | head -1"
check "cmake"           "cmake --version | head -1"
check "ninja"           "ninja --version"
check "dtc"             "dtc --version 2>&1 | head -1"
check "git"             "git --version"
echo ""

echo "Zephyr SDK (ARM toolchain):"
if [ -n "${ZEPHYR_SDK_INSTALL_DIR:-}" ]; then
  SDK_PATH="$ZEPHYR_SDK_INSTALL_DIR"
elif [ -d "$HOME/zephyr-sdk-0.16.8" ]; then
  SDK_PATH="$HOME/zephyr-sdk-0.16.8"
elif [ -d "$HOME/.local/zephyr-sdk-0.16.8" ]; then
  SDK_PATH="$HOME/.local/zephyr-sdk-0.16.8"
elif [ -d "/opt/zephyr-sdk-0.16.8" ]; then
  SDK_PATH="/opt/zephyr-sdk-0.16.8"
else
  SDK_PATH="$HOME/zephyr-sdk-0.16.8"
fi
check "arm-zephyr-eabi-gcc" "${SDK_PATH}/arm-zephyr-eabi/bin/arm-zephyr-eabi-gcc --version | head -1"
echo ""

echo "Debug & Flash:"
check "pyocd"           "pyocd --version 2>&1"
check "picocom"         "which picocom"
echo ""

echo "Zephyr Workspace:"
check_optional "ZEPHYR_BASE"     "echo \${ZEPHYR_BASE:-}"
if [ -n "${ZEPHYR_BASE:-}" ]; then
    check "Zephyr exists"   "ls ${ZEPHYR_BASE}/VERSION 2>/dev/null && cat ${ZEPHYR_BASE}/VERSION | head -3 | tr '\n' '.' | sed 's/\.$//'"
fi
echo ""

echo "Board Connection:"
check_optional "USB (NXP)" "lsusb 2>/dev/null | grep -i 'nxp\|1fc9\|0d28' | head -1"
check_optional "Serial port" "ls /dev/ttyACM0 2>/dev/null"
echo ""

echo "Optional Tools:"
check_optional "VS Code" "code --version 2>/dev/null | head -1"
echo ""

echo "═══════════════════════════════════════════════════"
echo -e "  Results: ${GREEN}${pass} passed${NC}, ${RED}${fail} failed${NC}, ${YELLOW}${warn} optional${NC}"
echo "═══════════════════════════════════════════════════"
echo ""

if [ $fail -gt 0 ]; then
    echo "Fix the failed items above before proceeding."
    exit 1
else
    echo "Environment is ready for SynapticOS development."
    exit 0
fi
