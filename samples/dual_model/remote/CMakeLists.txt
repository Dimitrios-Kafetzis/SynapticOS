# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.20.0)

# The CPU1 board (frdm_mcxn947_cpu1) lives in this repository:
# Zephyr 3.7's in-tree frdm_mcxn947 board has no cpu1 target.
list(APPEND BOARD_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(dual_model_remote VERSION 0.1.0)
target_sources(app PRIVATE src/main.c)
target_link_libraries(app PRIVATE synaptic_os)

# Cross-core inference helpers are internal API, not frozen headers
target_include_directories(app PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../src/core
)
