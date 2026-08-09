/* SPDX-License-Identifier: Apache-2.0 */
/** @file test_common.c
 *  @brief Shared fixtures for the unit test app (see test_common.h)
 */

#include <zephyr/toolchain.h>

#include "test_common.h"

uint8_t __aligned(16) test_shared_arena[TEST_SHARED_ARENA_SIZE];
