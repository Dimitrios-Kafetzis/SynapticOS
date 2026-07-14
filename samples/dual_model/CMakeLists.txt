# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.20.0)
find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(dual_model VERSION 0.1.0)
target_sources(app PRIVATE src/main.c)
target_link_libraries(app PRIVATE synaptic_os)

# Boot sequence and shared-layout internals are not part of the
# frozen public API; the sample reaches into src/core deliberately.
target_include_directories(app PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/core
)
