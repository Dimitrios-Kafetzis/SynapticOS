/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file test_health.c
 * @brief Unit tests for the health monitor (Phase 5.4)
 *
 * QEMU has no watchdog device, so the monitor's fault callback is
 * the observable outcome here; the hardware-watchdog reset and the
 * CPU1 heartbeat recovery are board-session items. The monitor
 * thread ticks every 100 ms.
 */

#include <zephyr/ztest.h>
#include <string.h>

#include "syn_health.h"

static volatile int faults_seen;
static const char *last_fault_name;

static void fault_hook(const char *source, void *user)
{
	ARG_UNUSED(user);
	last_fault_name = source;
	faults_seen++;
}

static void health_before(void *fixture)
{
	ARG_UNUSED(fixture);
	faults_seen = 0;
	last_fault_name = NULL;
	syn_health_set_fault_cb(fault_hook, NULL);
}

static void health_after(void *fixture)
{
	ARG_UNUSED(fixture);
	syn_health_set_fault_cb(NULL, NULL);
}

ZTEST_SUITE(syn_health_suite, NULL, NULL, health_before, health_after,
	    NULL);

/** A kicked source stays fresh; a neglected one goes stale once per
 *  episode and recovers on the next kick.
 */
ZTEST(syn_health_suite, test_stale_detection_and_recovery)
{
	int id = syn_health_register("t_stale", 150);

	zassert_true(id >= 0, "register failed: %d", id);

	/* Kicked: no faults while we keep checking in */
	for (int i = 0; i < 4; i++) {
		syn_health_kick(id);
		k_msleep(50);
	}
	zassert_equal(faults_seen, 0, "kicked source must not fault");

	/* Neglected past its period: exactly one episode */
	k_msleep(400);
	zassert_equal(faults_seen, 1,
		      "expected one stale episode, got %d", faults_seen);
	zassert_not_null(last_fault_name, "fault name missing");
	zassert_equal(strcmp(last_fault_name, "t_stale"), 0,
		      "wrong source reported: %s", last_fault_name);

	syn_health_info_t info;
	bool found = false;

	for (int i = 0; syn_health_get(i, &info) == 0; i++) {
		if (strcmp(info.name, "t_stale") == 0) {
			zassert_true(info.stale, "must be flagged stale");
			zassert_equal(info.stale_count, 1, "one episode");
			found = true;
		}
	}
	zassert_true(found, "source missing from introspection");

	/* Recovery: a kick ends the episode; staying fresh adds none */
	syn_health_kick(id);
	k_msleep(120);
	syn_health_kick(id);
	zassert_equal(faults_seen, 1, "no new episode after recovery");

	/* Neglect again: a second episode is counted separately */
	k_msleep(400);
	zassert_equal(faults_seen, 2, "second episode expected");

	/* Idle the source so later suites see no more faults from it */
	syn_health_set_busy(id, false);
}

/** Idle sources are never checked. */
ZTEST(syn_health_suite, test_idle_source_skipped)
{
	int id = syn_health_register("t_idle", 100);

	zassert_true(id >= 0, "register failed: %d", id);
	syn_health_set_busy(id, false);

	k_msleep(400);
	zassert_equal(faults_seen, 0, "idle source must never fault");

	/* Marking busy re-arms from now, not from registration time */
	syn_health_set_busy(id, true);
	k_msleep(50);
	zassert_equal(faults_seen, 0, "fresh busy source must not fault");
	syn_health_set_busy(id, false);
}

/** No watchdog on QEMU; the counters exist and are consistent. */
ZTEST(syn_health_suite, test_introspection)
{
	zassert_false(syn_health_watchdog_armed(),
		      "QEMU must not report an armed watchdog");
	zassert_equal(syn_health_cpu1_recoveries(), 0,
		      "no CPU1 on this target");
	zassert_true(syn_health_fault_count() >= (uint32_t)faults_seen,
		     "global counter must cover this suite's episodes");
}
