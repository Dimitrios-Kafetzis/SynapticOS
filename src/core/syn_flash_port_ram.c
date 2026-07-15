/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file syn_flash_port_ram.c
 * @brief SynapticOS - flash port over a RAM buffer
 *
 * Emulates NOR flash strictly enough to catch real driver-misuse
 * bugs: erase sets 0xFF, programming a page that is not blank fails
 * (the MCX ECC flash rejects reprogramming too), and reads of the
 * window are always allowed (RAM has no ECC-on-erased hazard, so
 * is_blank just inspects bytes). Write fault injection emulates
 * power loss mid-write for the OTA robustness tests.
 */

#include <zephyr/kernel.h>
#include <string.h>

#include "syn_flash_port.h"

/* Shared range/alignment guards for every backend. These are the
 * runtime counterpart of the partition-map BUILD_ASSERTs: nothing
 * outside [0, size) of the managed window is ever touched.
 */
static int check_range(const syn_flash_port_t *p, uint32_t off, size_t len)
{
	if (p == NULL || len == 0U || off > p->size ||
	    len > p->size - off) {
		return -EINVAL;
	}
	return 0;
}

int syn_flash_read(const syn_flash_port_t *p, uint32_t off,
		   void *buf, size_t len)
{
	int ret = check_range(p, off, len);

	if (ret != 0 || buf == NULL) {
		return ret != 0 ? ret : -EINVAL;
	}
	return p->read(p, off, buf, len);
}

int syn_flash_write(const syn_flash_port_t *p, uint32_t off,
		    const void *buf, size_t len)
{
	int ret = check_range(p, off, len);

	if (ret != 0 || buf == NULL) {
		return ret != 0 ? ret : -EINVAL;
	}
	if ((off % p->page_size) != 0U || (len % p->page_size) != 0U) {
		return -EINVAL;
	}
	return p->write(p, off, buf, len);
}

int syn_flash_erase(const syn_flash_port_t *p, uint32_t off, size_t len)
{
	int ret = check_range(p, off, len);

	if (ret != 0) {
		return ret;
	}
	if ((off % p->sector_size) != 0U || (len % p->sector_size) != 0U) {
		return -EINVAL;
	}
	return p->erase(p, off, len);
}

int syn_flash_is_blank(const syn_flash_port_t *p, uint32_t off, size_t len)
{
	int ret = check_range(p, off, len);

	if (ret != 0) {
		return ret;
	}
	return p->is_blank(p, off, len);
}

const uint8_t *syn_flash_mmap(const syn_flash_port_t *p, uint32_t off)
{
	if (p == NULL || off >= p->size || p->mmap == NULL) {
		return NULL;
	}
	return p->mmap(p, off);
}

/* RAM backend */

static int ram_read(const syn_flash_port_t *p, uint32_t off,
		    void *buf, size_t len)
{
	const syn_flash_ram_ctx_t *c = p->ctx;

	memcpy(buf, &c->mem[off], len);
	return 0;
}

static int ram_write(const syn_flash_port_t *p, uint32_t off,
		     const void *buf, size_t len)
{
	syn_flash_ram_ctx_t *c = p->ctx;
	const uint8_t *src = buf;
	size_t pages = len / p->page_size;

	for (size_t i = 0; i < pages; i++) {
		uint8_t *dst = &c->mem[off + i * p->page_size];

		if (c->fail_after_pages != 0U) {
			c->fail_after_pages--;
			if (c->fail_after_pages == 0U) {
				/* power loss: half a page lands */
				memcpy(dst, &src[i * p->page_size],
				       p->page_size / 2U);
				return -EIO;
			}
		}

		/* NOR semantics: target page must be erased */
		for (uint32_t b = 0; b < p->page_size; b++) {
			if (dst[b] != 0xFFU) {
				return -EACCES;
			}
		}
		memcpy(dst, &src[i * p->page_size], p->page_size);
		c->pages_written++;
	}
	return 0;
}

static int ram_erase(const syn_flash_port_t *p, uint32_t off, size_t len)
{
	syn_flash_ram_ctx_t *c = p->ctx;

	memset(&c->mem[off], 0xFF, len);
	c->sectors_erased += len / p->sector_size;
	return 0;
}

static int ram_is_blank(const syn_flash_port_t *p, uint32_t off, size_t len)
{
	const syn_flash_ram_ctx_t *c = p->ctx;

	for (size_t i = 0; i < len; i++) {
		if (c->mem[off + i] != 0xFFU) {
			return 0;
		}
	}
	return 1;
}

static const uint8_t *ram_mmap(const syn_flash_port_t *p, uint32_t off)
{
	const syn_flash_ram_ctx_t *c = p->ctx;

	return &c->mem[off];
}

int syn_flash_port_ram_init(syn_flash_port_t *port, syn_flash_ram_ctx_t *ctx,
			    uint8_t *mem, uint32_t size,
			    uint32_t sector_size, uint32_t page_size)
{
	if (port == NULL || ctx == NULL || mem == NULL ||
	    sector_size == 0U || page_size == 0U ||
	    size == 0U || (size % sector_size) != 0U ||
	    (sector_size % page_size) != 0U) {
		return -EINVAL;
	}

	memset(ctx, 0, sizeof(*ctx));
	ctx->mem = mem;
	ctx->size = size;
	memset(mem, 0xFF, size);

	port->base = 0U;
	port->size = size;
	port->sector_size = sector_size;
	port->page_size = page_size;
	port->read = ram_read;
	port->write = ram_write;
	port->erase = ram_erase;
	port->is_blank = ram_is_blank;
	port->mmap = ram_mmap;
	port->ctx = ctx;
	return 0;
}
