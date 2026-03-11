# SPDX-License-Identifier: Apache-2.0
# SynapticOS - AI-Native Operating System for Microcontrollers

cmake_minimum_required(VERSION 3.20.0)
find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})

project(synaptic-os VERSION 0.1.0 LANGUAGES C)

# SynapticOS core library
add_library(synaptic_os STATIC)

# Include paths
target_include_directories(synaptic_os PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_include_directories(synaptic_os PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/core
)

# Core sources
target_sources(synaptic_os PRIVATE
    src/core/syn_init.c
    src/core/syn_mem.c
    src/core/syn_model.c
    src/core/syn_prof.c
)

# Conditionally add inference engine
target_sources_ifdef(CONFIG_SYNAPTIC synaptic_os PRIVATE
    src/core/syn_infer.c
)

# Conditionally add IPC (dual-core)
target_sources_ifdef(CONFIG_SYNAPTIC_DUAL_CORE synaptic_os PRIVATE
    src/core/syn_ipc.c
)

# Conditionally add shell commands
target_sources_ifdef(CONFIG_SYNAPTIC_SHELL synaptic_os PRIVATE
    src/core/syn_shell.c
)

# Conditionally add OTA
target_sources_ifdef(CONFIG_SYNAPTIC_OTA synaptic_os PRIVATE
    src/core/syn_model_ota.c
)

# HAL sources - MCXN947 specific
if(CONFIG_SOC_SERIES_MCXNX4X)
    target_sources(synaptic_os PRIVATE
        src/hal/mcxn947/syn_hal_npu_neutron.c
        src/hal/mcxn947/syn_hal_dsp_pq.c
        src/hal/mcxn947/syn_hal_dma_sdma.c
    )
else()
    # Stub HAL for testing / non-NPU targets
    target_sources(synaptic_os PRIVATE
        src/hal/stub/syn_hal_npu_stub.c
        src/hal/stub/syn_hal_dsp_stub.c
    )
endif()

# Pre-processing sources
target_sources(synaptic_os PRIVATE
    src/preprocess/syn_preprocess_image.c
    src/preprocess/syn_preprocess_audio.c
    src/preprocess/syn_preprocess_quant.c
)

# Post-processing sources
target_sources(synaptic_os PRIVATE
    src/postprocess/syn_postprocess_classify.c
    src/postprocess/syn_postprocess_detect.c
)

# Link against Zephyr
target_link_libraries(synaptic_os PUBLIC zephyr_interface)

# Export for samples
zephyr_library_link_libraries(synaptic_os)
