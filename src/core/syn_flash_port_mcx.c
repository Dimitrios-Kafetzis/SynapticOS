/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file syn_flash_port_mcx.c
 * @brief SynapticOS - flash port over the MCX ROM API (FRDM-MCXN947)
 *
 * Manages exactly the OTA-writable window of bank 1
 * ([SYN_FLASH_WRITABLE_BASE, SYN_FLASH_WRITABLE_END), i.e. registry
 * copies + model slots). The window is established at init from the
 * generated partition map, so by construction no operation can reach
 * bank 0 (CPU0 XIP) or the CPU1 image reserve; the base/size checks
 * in the syn_flash_* wrappers then bound every call to the window.
 *
 * All access goes through the ROM API rather than bus reads:
 * reading erased ECC flash through the AHB raises a bus error (the
 * Phase 3 chip-wedge lesson), so reads use FLASH_Read, blank checks
 * use FLASH_VerifyErase (FMC margin command), and mmap pointers are
 * handed out only for committed, CRC-verified content.
 *
 * The caller (OTA engine) is responsible for quiescing CPU1 before
 * erase/program: CPU1 executes in place from this same bank.
 */

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>

#ifdef CONFIG_SOC_FLASH_MCUX

#include "fsl_flash.h"

#include "syn_flash_port.h"
#include "syn_flash_map.h"

LOG_MODULE_REGISTER(syn_flash_mcx, CONFIG_SYNAPTIC_LOG_LEVEL);

static flash_config_t fcfg;
static bool fcfg_ready;

static int mcx_status(status_t st)
{
	return (st == kStatus_Success) ? 0 : -EIO;
}

static int mcx_read(const syn_flash_port_t *p, uint32_t off,
		    void *buf, size_t len)
{
	return mcx_status(FLASH_Read(&fcfg, p->base + off,
				     (uint8_t *)buf, (uint32_t)len));
}

static int mcx_write(const syn_flash_port_t *p, uint32_t off,
		     const void *buf, size_t len)
{
	/* FLASH_Program takes a non-const src pointer but does not
	 * modify it; the ROM copies from the buffer.
	 */
	return mcx_status(FLASH_Program(&fcfg, p->base + off,
					(uint8_t *)(uintptr_t)buf,
					(uint32_t)len));
}

static int mcx_erase(const syn_flash_port_t *p, uint32_t off, size_t len)
{
	return mcx_status(FLASH_Erase(&fcfg, p->base + off, (uint32_t)len,
				      (uint32_t)kFLASH_ApiEraseKey));
}

static int mcx_is_blank(const syn_flash_port_t *p, uint32_t off, size_t len)
{
	status_t st = FLASH_VerifyErase(&fcfg, p->base + off, (uint32_t)len);

	return (st == kStatus_Success) ? 1 : 0;
}

static const uint8_t *mcx_mmap(const syn_flash_port_t *p, uint32_t off)
{
	return (const uint8_t *)SYN_FLASH_XIP_ADDR(p->base + off);
}

int syn_flash_port_mcx_init(syn_flash_port_t *port)
{
	if (port == NULL) {
		return -EINVAL;
	}

	if (!fcfg_ready) {
		if (FLASH_Init(&fcfg) != kStatus_Success) {
			LOG_ERR("ROM flash API init failed");
			return -EIO;
		}
		fcfg_ready = true;
	}

	port->base = (uint32_t)SYN_FLASH_WRITABLE_BASE;
	port->size = (uint32_t)(SYN_FLASH_WRITABLE_END -
				SYN_FLASH_WRITABLE_BASE);
	port->sector_size = (uint32_t)SYN_FLASH_SECTOR_SIZE;
	port->page_size = (uint32_t)SYN_FLASH_PAGE_SIZE;
	port->read = mcx_read;
	port->write = mcx_write;
	port->erase = mcx_erase;
	port->is_blank = mcx_is_blank;
	port->mmap = mcx_mmap;
	port->ctx = NULL;

	LOG_INF("Flash port: bank 1 window 0x%08x-0x%08x",
		port->base, port->base + port->size - 1U);
	return 0;
}

#endif /* CONFIG_SOC_FLASH_MCUX */
