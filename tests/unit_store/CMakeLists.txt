# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.20.0)
find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(synaptic_store_tests)

# Separate test app from tests/unit: the store/OTA fixtures carry a
# RAM-emulated flash (~20 KB) that does not fit next to the other
# suites in QEMU's 64 KB.
target_sources(app PRIVATE
    test_model_store.c
    test_model_ota.c
)

target_link_libraries(app PRIVATE synaptic_os)

# Tests exercise private core internals (flash port, store layout)
target_include_directories(app PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/core
)
