/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file syn_health.h
 * @brief SynapticOS - Health monitor and watchdog integration (private)
 *
 * Production-hardening core (Phase 5.4): registered software sources
 * must check in (kick) within their declared period while busy. A
 * monitor thread ticks every 100 ms and
 *
 *   - feeds the hardware watchdog (Zephyr `watchdog0` alias, when
 *     CONFIG_WATCHDOG is enabled and the node exists) ONLY while
 *     every busy source is fresh - a hung thread starves the feed
 *     and the watchdog resets the system;
 *   - on targets without a watchdog, reports the stale source once
 *     per episode through an optional fault callback (QEMU tests);
 *   - on dual-core CPU0 builds, watches CPU1's shared-memory
 *     heartbeat and recovers a hung CPU1 by parking and re-releasing
 *     it (the OTA park/resume machinery), without resetting CPU0.
 *
 * Sources idle by default between jobs: mark busy around work so
 * blocking on an empty queue is never mistaken for a hang.
 */
#ifndef SYNAPTIC_SYN_HEALTH_H_
#define SYNAPTIC_SYN_HEALTH_H_

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SYN_HEALTH_MAX_SOURCES 4

typedef struct {
	const char *name;
	uint32_t period_ms;
	int64_t age_ms;        /**< Time since the last kick          */
	bool busy;
	bool stale;            /**< Currently past its period         */
	uint32_t stale_count;  /**< Stale episodes since boot         */
} syn_health_info_t;

typedef void (*syn_health_fault_cb_t)(const char *source, void *user);

#ifdef CONFIG_SYNAPTIC_HEALTH

/**
 * @brief Register a health source (busy by default).
 * @return Source id >= 0, or -ENOSPC when the table is full.
 */
int syn_health_register(const char *name, uint32_t period_ms);

/** @brief Check in: the source is alive right now. */
void syn_health_kick(int id);

/** @brief Idle sources are not checked (and never go stale). */
void syn_health_set_busy(int id, bool busy);

/** @brief Info for `syn health`; @return 0 or -ENOENT past the end. */
int syn_health_get(int idx, syn_health_info_t *info);

/** @brief Fault hook for watchdog-less targets and tests. */
void syn_health_set_fault_cb(syn_health_fault_cb_t cb, void *user);

/** @brief Total stale episodes across all sources. */
uint32_t syn_health_fault_count(void);

/** @brief True when the hardware watchdog is armed and being fed. */
bool syn_health_watchdog_armed(void);

/** @brief CPU1 hang recoveries performed (dual-core CPU0, else 0). */
uint32_t syn_health_cpu1_recoveries(void);

#else /* !CONFIG_SYNAPTIC_HEALTH */

static inline int syn_health_register(const char *name, uint32_t period_ms)
{
	(void)name;
	(void)period_ms;
	return -1;
}

static inline void syn_health_kick(int id)
{
	(void)id;
}

static inline void syn_health_set_busy(int id, bool busy)
{
	(void)id;
	(void)busy;
}

#endif /* CONFIG_SYNAPTIC_HEALTH */

#ifdef __cplusplus
}
#endif
#endif /* SYNAPTIC_SYN_HEALTH_H_ */
