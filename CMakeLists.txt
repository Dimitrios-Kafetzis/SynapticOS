# SPDX-License-Identifier: Apache-2.0
# SynapticOS - AI-Native Operating System for Microcontrollers

cmake_minimum_required(VERSION 3.20.0)
find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})

project(synaptic-os VERSION 0.2.0 LANGUAGES C)

zephyr_library_named(synaptic_os)

zephyr_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
zephyr_library_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/core)

zephyr_library_sources(
    src/core/syn_init.c
    src/core/syn_mem.c
    src/core/syn_model.c
    src/core/syn_prof.c
)

zephyr_library_sources_ifdef(CONFIG_SYNAPTIC
    src/core/syn_infer.c
)

zephyr_library_sources_ifdef(CONFIG_SYNAPTIC_DUAL_CORE
    src/core/syn_ipc.c
)

zephyr_library_sources_ifdef(CONFIG_SYNAPTIC_MPU_PROTECT
    src/core/syn_mpu.c
)

zephyr_library_sources_ifdef(CONFIG_SYNAPTIC_SHELL
    src/core/syn_shell.c
)

zephyr_library_sources_ifdef(CONFIG_SYNAPTIC_OTA
    src/core/syn_model_ota.c
)

zephyr_library_sources(
    src/hal/common/syn_dsp_soft.c
)

if(CONFIG_SOC_SERIES_MCXNX4X)
    zephyr_library_sources(
        src/hal/mcxn947/syn_hal_npu_neutron.c
        src/hal/mcxn947/syn_hal_dsp_pq.c
        src/hal/mcxn947/syn_hal_dma_sdma.c
    )
    if(CONFIG_SYNAPTIC_POWERQUAD)
        # PowerQuad SDK driver has no Zephyr glue in this hal_nxp
        # revision; compile the needed engines directly.
        set(SYN_PQ_DIR
            ${ZEPHYR_HAL_NXP_MODULE_DIR}/mcux/mcux-sdk/drivers/powerquad)
        zephyr_library_sources(
            ${SYN_PQ_DIR}/fsl_powerquad_basic.c
            ${SYN_PQ_DIR}/fsl_powerquad_transform.c
            ${SYN_PQ_DIR}/fsl_powerquad_matrix.c
        )
        zephyr_library_include_directories(${SYN_PQ_DIR})
    endif()
else()
    zephyr_library_sources(
        src/hal/stub/syn_hal_npu_stub.c
        src/hal/stub/syn_hal_dsp_stub.c
    )
endif()

zephyr_library_sources(
    src/preprocess/syn_preprocess_image.c
    src/preprocess/syn_preprocess_audio.c
    src/preprocess/syn_preprocess_quant.c
)

zephyr_library_sources(
    src/postprocess/syn_postprocess_classify.c
    src/postprocess/syn_postprocess_detect.c
)
