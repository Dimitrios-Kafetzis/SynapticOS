/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file syn_health.c
 * @brief SynapticOS - Health monitor and watchdog integration
 *
 * See syn_health.h for the contract.
 */

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <zephyr/init.h>
#include <string.h>

LOG_MODULE_REGISTER(syn_health, CONFIG_SYNAPTIC_LOG_LEVEL);

#include "syn_health.h"

#if defined(CONFIG_WATCHDOG) && \
	DT_NODE_HAS_STATUS(DT_ALIAS(watchdog0), okay)
#include <zephyr/drivers/watchdog.h>
#define HEALTH_HAS_WDT 1
#endif

#if defined(CONFIG_SYNAPTIC_DUAL_CORE) && !defined(CONFIG_SOC_MCXN947_CPU1)
#include "syn_boot_internal.h"
#include "syn_ipc_internal.h"
#define HEALTH_CPU1_WATCH 1
#define CPU1_RESUME_TIMEOUT_MS 1000U
#endif

#define HEALTH_TICK_MS         100
#define HEALTH_STACK_SIZE      1024
#define HEALTH_THREAD_PRIO     K_PRIO_PREEMPT(4)

struct health_source {
	const char *name;      /* NULL = slot free */
	uint32_t period_ms;
	int64_t last_kick;
	bool busy;
	bool stale;            /* current episode flag */
	uint32_t stale_count;
};

/* Statically initialized: sources register from SYS_INIT hooks that
 * may run before this module's own init.
 */
static struct health_source sources[SYN_HEALTH_MAX_SOURCES];
static K_MUTEX_DEFINE(health_lock);
static syn_health_fault_cb_t fault_cb;
static void *fault_cb_user;
static uint32_t fault_count;
static uint32_t cpu1_recoveries;
static bool wdt_armed;

#ifdef HEALTH_HAS_WDT
static const struct device *const wdt_dev =
	DEVICE_DT_GET(DT_ALIAS(watchdog0));
static int wdt_channel = -1;
#endif

int syn_health_register(const char *name, uint32_t period_ms)
{
	int id = -ENOSPC;

	k_mutex_lock(&health_lock, K_FOREVER);
	for (int i = 0; i < SYN_HEALTH_MAX_SOURCES; i++) {
		if (sources[i].name == NULL) {
			sources[i].name = name;
			sources[i].period_ms = period_ms;
			sources[i].last_kick = k_uptime_get();
			sources[i].busy = true;
			sources[i].stale = false;
			sources[i].stale_count = 0;
			id = i;
			break;
		}
	}
	k_mutex_unlock(&health_lock);

	if (id < 0) {
		LOG_ERR("Health source table full ('%s' not watched)", name);
	}
	return id;
}

void syn_health_kick(int id)
{
	if (id < 0 || id >= SYN_HEALTH_MAX_SOURCES) {
		return;
	}
	sources[id].last_kick = k_uptime_get();
	sources[id].stale = false;
}

void syn_health_set_busy(int id, bool busy)
{
	if (id < 0 || id >= SYN_HEALTH_MAX_SOURCES) {
		return;
	}
	sources[id].busy = busy;
	if (busy) {
		sources[id].last_kick = k_uptime_get();
	}
}

int syn_health_get(int idx, syn_health_info_t *info)
{
	if (idx < 0 || idx >= SYN_HEALTH_MAX_SOURCES || info == NULL ||
	    sources[idx].name == NULL) {
		return -ENOENT;
	}

	k_mutex_lock(&health_lock, K_FOREVER);
	info->name = sources[idx].name;
	info->period_ms = sources[idx].period_ms;
	info->age_ms = k_uptime_get() - sources[idx].last_kick;
	info->busy = sources[idx].busy;
	info->stale = sources[idx].stale;
	info->stale_count = sources[idx].stale_count;
	k_mutex_unlock(&health_lock);
	return 0;
}

void syn_health_set_fault_cb(syn_health_fault_cb_t cb, void *user)
{
	fault_cb = cb;
	fault_cb_user = user;
}

uint32_t syn_health_fault_count(void)
{
	return fault_count;
}

bool syn_health_watchdog_armed(void)
{
	return wdt_armed;
}

uint32_t syn_health_cpu1_recoveries(void)
{
	return cpu1_recoveries;
}

/** @return true when every busy source is fresh. */
static bool check_sources(void)
{
	bool all_ok = true;
	int64_t now = k_uptime_get();

	k_mutex_lock(&health_lock, K_FOREVER);
	for (int i = 0; i < SYN_HEALTH_MAX_SOURCES; i++) {
		struct health_source *s = &sources[i];

		if (s->name == NULL || !s->busy) {
			continue;
		}
		if (now - s->last_kick <= (int64_t)s->period_ms) {
			continue;
		}

		all_ok = false;
		if (!s->stale) {
			/* New episode: report once */
			s->stale = true;
			s->stale_count++;
			fault_count++;
			LOG_ERR("Health source '%s' stale: no kick for "
				"%lld ms (period %u ms)", s->name,
				now - s->last_kick, s->period_ms);
			if (fault_cb != NULL) {
				fault_cb(s->name, fault_cb_user);
			}
		}
	}
	k_mutex_unlock(&health_lock);
	return all_ok;
}

#ifdef HEALTH_CPU1_WATCH
/** Detect a stalled CPU1 heartbeat and park + re-release the core. */
static void check_cpu1(void)
{
	static uint32_t last_hb;
	static int64_t last_change;
	static bool primed;

	if (!syn_boot_secondary_linked()) {
		/* Not booted, parked for OTA, or already lost: the next
		 * link re-primes the tracker.
		 */
		primed = false;
		return;
	}

	syn_shm_region_t *shm = syn_ipc_region();

	if (shm == NULL) {
		return;
	}

	uint32_t hb = shm->ctrl.cpu1_heartbeat;
	int64_t now = k_uptime_get();

	if (!primed || hb != last_hb) {
		last_hb = hb;
		last_change = now;
		primed = true;
		return;
	}

	if (now - last_change <= (int64_t)CONFIG_SYNAPTIC_CPU1_HANG_MS) {
		return;
	}

	LOG_ERR("CPU1 heartbeat lost for %lld ms: parking and "
		"re-releasing", now - last_change);
	shm->ctrl.debug_cmd = 0; /* clear an induced-hang request */

	int ret = syn_boot_secondary_stop();

	if (ret == 0) {
		ret = syn_boot_secondary_resume(CPU1_RESUME_TIMEOUT_MS);
	}
	if (ret == 0) {
		cpu1_recoveries++;
		LOG_WRN("CPU1 recovered (recovery #%u)", cpu1_recoveries);
	} else {
		LOG_ERR("CPU1 recovery failed: %d (continuing "
			"single-core)", ret);
	}
	primed = false;
}
#endif /* HEALTH_CPU1_WATCH */

static void health_thread(void *p1, void *p2, void *p3)
{
	ARG_UNUSED(p1);
	ARG_UNUSED(p2);
	ARG_UNUSED(p3);

	while (1) {
		bool all_ok = check_sources();

#ifdef HEALTH_CPU1_WATCH
		/* CPU1 loss is recovered in software and must not stop
		 * the CPU0 watchdog feed.
		 */
		check_cpu1();
#endif

#ifdef HEALTH_HAS_WDT
		if (wdt_armed && all_ok) {
			(void)wdt_feed(wdt_dev, wdt_channel);
		}
#else
		ARG_UNUSED(all_ok);
#endif
		k_msleep(HEALTH_TICK_MS);
	}
}

K_THREAD_STACK_DEFINE(health_stack, HEALTH_STACK_SIZE);
static struct k_thread health_thread_data;

static int syn_health_init(void)
{
#ifdef HEALTH_HAS_WDT
	if (device_is_ready(wdt_dev)) {
		struct wdt_timeout_cfg cfg = {
			.window.min = 0U,
			.window.max = CONFIG_SYNAPTIC_WATCHDOG_TIMEOUT_MS,
			.flags = WDT_FLAG_RESET_SOC,
		};

		wdt_channel = wdt_install_timeout(wdt_dev, &cfg);
		if (wdt_channel >= 0 &&
		    wdt_setup(wdt_dev, WDT_OPT_PAUSE_HALTED_BY_DBG) == 0) {
			wdt_armed = true;
			LOG_INF("Watchdog armed: %u ms, fed by the health "
				"monitor", CONFIG_SYNAPTIC_WATCHDOG_TIMEOUT_MS);
		} else {
			LOG_ERR("Watchdog setup failed (channel %d)",
				wdt_channel);
		}
	}
#endif

	k_thread_create(&health_thread_data, health_stack,
			K_THREAD_STACK_SIZEOF(health_stack), health_thread,
			NULL, NULL, NULL, HEALTH_THREAD_PRIO, 0, K_NO_WAIT);
	k_thread_name_set(&health_thread_data, "syn_health");

	LOG_INF("Health monitor started (tick %d ms, watchdog %s)",
		HEALTH_TICK_MS, wdt_armed ? "armed" : "absent");
	return 0;
}

SYS_INIT(syn_health_init, APPLICATION, 95);
