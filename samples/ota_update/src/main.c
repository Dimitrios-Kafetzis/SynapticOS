/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file main.c
 * @brief SynapticOS — ota_update sample (placeholder)
 *
 * TODO: Implementation pending.
 */

#include <zephyr/kernel.h>
#include <synaptic/syn_api.h>
#include <zephyr/logging/log.h>
LOG_MODULE_REGISTER(ota_update, LOG_LEVEL_INF);

int main(void)
{
    LOG_INF("SynapticOS %s — ota_update", syn_version());
    /* TODO: Implement ota_update sample */
    return 0;
}
