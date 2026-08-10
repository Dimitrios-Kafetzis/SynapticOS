/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file syn_synn.c
 * @brief SynapticOS - SYNN Neutron payload parsing
 *
 * Structural validation only: the HAL applies its own policy limits
 * (i/o caps, scratch buffer capacity) on top of a successful parse.
 * Compiled on every target so the container logic is covered by the
 * QEMU test suites even though only the MCXN Neutron HAL acts on it.
 */

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <string.h>

#include "syn_synn.h"

LOG_MODULE_REGISTER(syn_synn, CONFIG_SYNAPTIC_LOG_LEVEL);

int syn_synn_parse(const uint8_t *data, size_t size,
		   struct syn_synn_desc *desc)
{
	struct syn_synn_hdr hdr;

	if (data == NULL || desc == NULL) {
		return -EINVAL;
	}
	if (size < sizeof(hdr)) {
		return -ENOTSUP;
	}

	/* The payload pointer is only guaranteed 16-byte aligned as a
	 * whole; copy the header out rather than aliasing it.
	 */
	memcpy(&hdr, data, sizeof(hdr));

	if (hdr.magic != SYN_SYNN_MAGIC) {
		return -ENOTSUP;
	}
	if (hdr.version != SYN_SYNN_VERSION) {
		/* Also catches the interim headerless-version blobs of
		 * the first-light bring-up (their microcode offset
		 * lands in this field).
		 */
		LOG_ERR("SYNN version %u unsupported (want %u)",
			hdr.version, SYN_SYNN_VERSION);
		return -EINVAL;
	}
	if (hdr.flags != 0U) {
		LOG_ERR("SYNN reserved flags set (0x%08x)", hdr.flags);
		return -EINVAL;
	}
	if ((uint64_t)hdr.microcode_off + hdr.microcode_size > size ||
	    (uint64_t)hdr.weights_off + hdr.weights_size > size ||
	    (uint64_t)hdr.kernels_off + hdr.kernels_size > size) {
		LOG_ERR("SYNN section out of bounds (blob %zu B)", size);
		return -EINVAL;
	}
	if ((hdr.microcode_off | hdr.weights_off | hdr.kernels_off) & 0xFU) {
		LOG_ERR("SYNN section misaligned (16-byte required)");
		return -EINVAL;
	}
	if (hdr.input_size == 0U || hdr.output_size == 0U) {
		LOG_ERR("SYNN i/o sizes zero (%u/%u)",
			hdr.input_size, hdr.output_size);
		return -EINVAL;
	}

	desc->microcode      = data + hdr.microcode_off;
	desc->weights        = data + hdr.weights_off;
	desc->kernels        = data + hdr.kernels_off;
	desc->microcode_size = hdr.microcode_size;
	desc->weights_size   = hdr.weights_size;
	desc->kernels_size   = hdr.kernels_size;
	desc->input_size     = hdr.input_size;
	desc->output_size    = hdr.output_size;
	desc->scratch_size   = hdr.scratch_size;
	return 0;
}
