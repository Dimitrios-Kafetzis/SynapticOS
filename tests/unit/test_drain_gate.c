/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file test_drain_gate.c
 * @brief Unit tests for the single-server drain gate (Phase 5.7)
 *
 * The gate itself is platform-independent, so the OTA-vs-remote-serve
 * drain semantics are exercised here on QEMU even though the serve
 * path only builds on dual-core targets. A worker thread stands in
 * for the IPC dispatch thread.
 */

#include <zephyr/ztest.h>
#include <errno.h>

#include "syn_drain_gate.h"

static syn_drain_gate_t gate;

#define WORKER_HOLD_MS 50

static K_THREAD_STACK_DEFINE(worker_stack, 1024);
static struct k_thread worker_thread;
static volatile bool worker_admitted;
static volatile bool worker_done;

static void worker_fn(void *p1, void *p2, void *p3)
{
	ARG_UNUSED(p1);
	ARG_UNUSED(p2);
	ARG_UNUSED(p3);

	worker_admitted = syn_drain_gate_enter(&gate);
	if (worker_admitted) {
		k_msleep(WORKER_HOLD_MS);
		syn_drain_gate_exit(&gate);
	}
	worker_done = true;
}

static void start_worker(void)
{
	worker_admitted = false;
	worker_done = false;
	k_thread_create(&worker_thread, worker_stack,
			K_THREAD_STACK_SIZEOF(worker_stack), worker_fn,
			NULL, NULL, NULL, K_PRIO_PREEMPT(5), 0, K_NO_WAIT);
}

ZTEST_SUITE(syn_drain_gate_suite, NULL, NULL, NULL, NULL, NULL);

/** Draining an idle gate returns immediately and pauses admission. */
ZTEST(syn_drain_gate_suite, test_drain_idle)
{
	syn_drain_gate_init(&gate);

	zassert_equal(syn_drain_gate_drain(&gate, 100), 0,
		      "drain of an idle gate must succeed immediately");
	zassert_false(syn_drain_gate_enter(&gate),
		      "admission must stay paused after a drain");

	syn_drain_gate_resume(&gate);
	zassert_true(syn_drain_gate_enter(&gate),
		     "admission must reopen after resume");
	syn_drain_gate_exit(&gate);
}

/** A drain waits for the busy worker, then keeps the gate closed. */
ZTEST(syn_drain_gate_suite, test_drain_waits_for_worker)
{
	syn_drain_gate_init(&gate);
	start_worker();

	/* Let the worker enter the gate and start its hold */
	k_msleep(10);
	zassert_true(worker_admitted, "worker must have been admitted");
	zassert_false(worker_done, "worker must still be inside");

	int64_t t0 = k_uptime_get();

	zassert_equal(syn_drain_gate_drain(&gate, 1000), 0,
		      "drain must succeed once the worker exits");

	int64_t waited = k_uptime_get() - t0;

	/* The drain returns the instant the worker exits the gate; the
	 * worker thread may not have run its final statements yet, so
	 * join before checking its completion flag.
	 */
	zassert_equal(k_thread_join(&worker_thread, K_MSEC(500)), 0,
		      "worker did not finish");
	zassert_true(worker_done, "worker must have completed its exit");
	zassert_true(waited >= WORKER_HOLD_MS / 2,
		     "drain returned too early (%lld ms)", waited);
	zassert_false(syn_drain_gate_enter(&gate),
		      "admission must stay paused after the drain");

	syn_drain_gate_resume(&gate);
}

/** A worker that outlives the timeout produces -ETIMEDOUT, and a
 *  later drain of the now-idle gate succeeds.
 */
ZTEST(syn_drain_gate_suite, test_drain_timeout)
{
	syn_drain_gate_init(&gate);

	/* Occupy the gate from this thread: nobody will exit it */
	zassert_true(syn_drain_gate_enter(&gate), "enter failed");
	zassert_equal(syn_drain_gate_drain(&gate, 30), -ETIMEDOUT,
		      "drain must time out while the gate is held");

	/* The worker finishes late; a fresh drain now succeeds */
	syn_drain_gate_exit(&gate);
	zassert_equal(syn_drain_gate_drain(&gate, 30), 0,
		      "drain of the released gate must succeed");

	/* Paused: workers are rejected until resume */
	zassert_false(syn_drain_gate_enter(&gate), "must be paused");
	syn_drain_gate_resume(&gate);
	zassert_true(syn_drain_gate_enter(&gate), "must reopen");
	syn_drain_gate_exit(&gate);
}
