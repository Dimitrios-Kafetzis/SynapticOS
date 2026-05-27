# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.20.0)
find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(synaptic_unit_tests)

target_sources(app PRIVATE
    test_mem.c
    test_mem_bench.c
    test_mem_regions.c
    test_model_registry.c
    test_scheduler.c
    test_ipc.c
    test_npu_hal.c
    test_dsp_hal.c
    test_dsp_verify.c
    test_init.c
)

target_link_libraries(app PRIVATE synaptic_os)
