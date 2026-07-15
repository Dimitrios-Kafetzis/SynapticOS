/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file syn_flash_port.h
 * @brief SynapticOS - flash access seam for the model store and OTA
 *
 * The model registry and OTA engine never touch a flash driver
 * directly; they go through this port. Two backends exist:
 *
 *  - syn_flash_port_mcx.c: MCX ROM API (board), managing the
 *    OTA-writable window of bank 1. Every operation is range-checked
 *    against the partition map so no erase or write can ever reach
 *    bank 0 or the CPU1 image reserve (last line of defence behind
 *    the BUILD_ASSERTed layout).
 *  - syn_flash_port_ram.c: RAM-backed emulation with NOR program
 *    semantics (program only clears bits, page must be blank) and
 *    write fault injection, so all store/OTA logic is unit-testable
 *    on qemu_cortex_m3.
 *
 * Offsets are relative to the managed window (port->base). Like the
 * IPC ring in Phase 3, keeping the seam this narrow is what makes
 * the logic above it host-agnostic.
 */
#ifndef SYNAPTIC_SYN_FLASH_PORT_H_
#define SYNAPTIC_SYN_FLASH_PORT_H_

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct syn_flash_port syn_flash_port_t;

struct syn_flash_port {
	uint32_t base;        /* absolute plain address of the window */
	uint32_t size;        /* window size in bytes */
	uint32_t sector_size; /* erase granularity */
	uint32_t page_size;   /* program granularity */

	int (*read)(const syn_flash_port_t *p, uint32_t off,
		    void *buf, size_t len);
	/* len and off must be page-aligned; target must be erased */
	int (*write)(const syn_flash_port_t *p, uint32_t off,
		     const void *buf, size_t len);
	/* off and len must be sector-aligned */
	int (*erase)(const syn_flash_port_t *p, uint32_t off, size_t len);
	/* 1 = blank, 0 = programmed, negative = error */
	int (*is_blank)(const syn_flash_port_t *p, uint32_t off, size_t len);
	/* Direct-read pointer for committed (CRC-verified) content only;
	 * NULL when the backend is not memory-mapped.
	 */
	const uint8_t *(*mmap)(const syn_flash_port_t *p, uint32_t off);

	void *ctx;
};

/* Range/alignment-checked wrappers; all callers use these. */
int syn_flash_read(const syn_flash_port_t *p, uint32_t off,
		   void *buf, size_t len);
int syn_flash_write(const syn_flash_port_t *p, uint32_t off,
		    const void *buf, size_t len);
int syn_flash_erase(const syn_flash_port_t *p, uint32_t off, size_t len);
/* Erase one sector per flash command with a scheduler breather in
 * between. Large single-call erases (hundreds of KB) wedged the
 * MCXN947 whole-chip during Phase 4 board bring-up (flash XIP and
 * debug port dead, ISP-only recovery); every multi-sector erase
 * must go through this instead of syn_flash_erase.
 */
int syn_flash_erase_sectors(const syn_flash_port_t *p, uint32_t off,
			    size_t len);
int syn_flash_is_blank(const syn_flash_port_t *p, uint32_t off, size_t len);
const uint8_t *syn_flash_mmap(const syn_flash_port_t *p, uint32_t off);

/* RAM backend (tests, QEMU samples) */
typedef struct {
	uint8_t *mem;
	uint32_t size;
	/* fault injection: fail after this many more programmed pages
	 * (0 = disabled). The failing write programs half the request
	 * then returns -EIO, emulating power loss mid-write.
	 */
	uint32_t fail_after_pages;
	/* stats */
	uint32_t pages_written;
	uint32_t sectors_erased;
} syn_flash_ram_ctx_t;

int syn_flash_port_ram_init(syn_flash_port_t *port, syn_flash_ram_ctx_t *ctx,
			    uint8_t *mem, uint32_t size,
			    uint32_t sector_size, uint32_t page_size);

#ifdef CONFIG_SOC_FLASH_MCUX
/* MCX ROM API backend over the OTA-writable window of bank 1 */
int syn_flash_port_mcx_init(syn_flash_port_t *port);
#endif

#ifdef __cplusplus
}
#endif
#endif /* SYNAPTIC_SYN_FLASH_PORT_H_ */
