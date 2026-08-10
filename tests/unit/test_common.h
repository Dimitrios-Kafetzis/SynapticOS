/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file test_common.h
 * @brief Shared fixtures for the unit test app
 *
 * ztest runs suites sequentially and every suite's before() calls
 * syn_mem_init() on its arena, so all suites can share ONE region
 * instead of each carrying its own. On QEMU (64 KB RAM total) the
 * per-suite arenas were the largest single RAM consumer of the app.
 */
#ifndef SYNAPTIC_TESTS_UNIT_TEST_COMMON_H_
#define SYNAPTIC_TESTS_UNIT_TEST_COMMON_H_

#include <stdint.h>

#define TEST_SHARED_ARENA_SIZE (8 * 1024)

extern uint8_t test_shared_arena[TEST_SHARED_ARENA_SIZE];

#endif /* SYNAPTIC_TESTS_UNIT_TEST_COMMON_H_ */
