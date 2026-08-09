# SPDX-License-Identifier: Apache-2.0
# SynapticOS - AI-Native Operating System for Microcontrollers

cmake_minimum_required(VERSION 3.20.0)
find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})

project(synaptic-os VERSION 0.4.0 LANGUAGES C)

zephyr_library_named(synaptic_os)

if(CONFIG_SYNAPTIC_COVERAGE)
    # Instrument ONLY src/core: blank the toolchain coverage property
    # that arch/common applies tree-wide under CONFIG_COVERAGE (a
    # fully instrumented image overflows the QEMU target by ~245 KB
    # flash / ~128 KB RAM; even whole-library instrumentation is
    # ~26 KB over), then add the real flags per source file. The
    # embedded gcov runtime and console dump still come from Zephyr
    # (CONFIG_COVERAGE_GCOV/DUMP). Coverage reporting targets
    # src/core, so nothing is lost by the narrower scope.
    set_property(TARGET compiler PROPERTY coverage "")
    set_property(TARGET linker PROPERTY coverage "")
    set(SYN_COV_GLOBS ${CONFIG_SYNAPTIC_COVERAGE_FILES})
    separate_arguments(SYN_COV_GLOBS)
    set(SYN_COV_SRCS "")
    foreach(pat ${SYN_COV_GLOBS})
        file(GLOB _syn_cov_m ${CMAKE_CURRENT_SOURCE_DIR}/src/core/${pat})
        list(APPEND SYN_COV_SRCS ${_syn_cov_m})
    endforeach()
    set_source_files_properties(${SYN_COV_SRCS} PROPERTIES
        COMPILE_OPTIONS "-fprofile-arcs;-ftest-coverage")
endif()

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
    src/core/syn_ipc_ring.c
    src/core/syn_drain_gate.c
    src/core/syn_ingest.c
)

zephyr_library_sources_ifdef(CONFIG_SYNAPTIC_HEALTH
    src/core/syn_health.c
)

zephyr_library_sources_ifdef(CONFIG_SYNAPTIC_DUAL_CORE
    src/core/syn_ipc.c
    src/core/syn_boot.c
    src/core/syn_infer_remote.c
)

zephyr_library_sources_ifdef(CONFIG_SYNAPTIC_MPU_PROTECT
    src/core/syn_mpu.c
)

zephyr_library_sources_ifdef(CONFIG_SYNAPTIC_SHELL
    src/core/syn_shell.c
)

zephyr_library_sources_ifdef(CONFIG_SYNAPTIC_MODEL_STORE
    src/core/syn_model_store.c
    src/core/syn_flash_port_ram.c
    src/core/syn_flash_port_mcx.c
)

zephyr_library_sources_ifdef(CONFIG_SYNAPTIC_OTA
    src/core/syn_model_ota.c
)

zephyr_library_sources(
    src/hal/common/syn_dsp_soft.c
)

zephyr_library_sources_ifdef(CONFIG_SYNAPTIC_LAYER_EXEC
    src/hal/common/syn_npu_layered.c
)

if(CONFIG_SOC_SERIES_MCXNX4X)
    zephyr_library_sources(
        src/hal/mcxn947/syn_hal_npu_neutron.c
        src/hal/mcxn947/syn_hal_dsp_pq.c
        src/hal/mcxn947/syn_hal_dma_edma.c
    )
    if(CONFIG_DMA)
        # The DMA HAL software-triggers eDMA v4 mem-to-mem transfers
        # directly (the Zephyr driver never does); it needs the SDK
        # eDMA headers.
        zephyr_library_include_directories(
            ${ZEPHYR_HAL_NXP_MODULE_DIR}/mcux/mcux-sdk/drivers/edma4
            ${ZEPHYR_HAL_NXP_MODULE_DIR}/mcux/mcux-sdk/devices/MCXN947/drivers)
    endif()
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
    if(CONFIG_SYNAPTIC_DUAL_CORE)
        # MAILBOX SDK driver is header-only; no Zephyr glue for the
        # MCXN947 in this hal_nxp revision, so add it directly.
        zephyr_library_include_directories(
            ${ZEPHYR_HAL_NXP_MODULE_DIR}/mcux/mcux-sdk/drivers/mailbox)
    endif()
else()
    zephyr_library_sources(
        src/hal/stub/syn_hal_npu_stub.c
        src/hal/stub/syn_hal_dsp_stub.c
        src/hal/stub/syn_hal_dma_stub.c
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
