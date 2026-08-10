# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.20.0)
find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(neutron_hello VERSION 0.1.0)

# The Neutron blob is produced locally by neutron-converter from a
# LiteRT/TFLite model (see README.md) and is never committed: the
# microcode embeds NXP-generated content that does not belong in an
# Apache-2.0 tree.
set(SYNN_BLOB ${CMAKE_CURRENT_SOURCE_DIR}/../../models/neutron/resnet_cifar10.synn)
if(NOT EXISTS ${SYNN_BLOB})
    message(FATAL_ERROR
        "Missing ${SYNN_BLOB}. Convert and pack a model first -- "
        "see samples/neutron_hello/README.md.")
endif()

set(SYNN_INC ${CMAKE_CURRENT_BINARY_DIR}/model_blob.inc)
add_custom_command(
    OUTPUT ${SYNN_INC}
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/blob2inc.py
            ${SYNN_BLOB} ${SYNN_INC}
    DEPENDS ${SYNN_BLOB} ${CMAKE_CURRENT_SOURCE_DIR}/blob2inc.py
)

target_sources(app PRIVATE src/main.c ${SYNN_INC})
target_include_directories(app PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(app PRIVATE synaptic_os)
