/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file syn_synn.h
 * @brief SynapticOS - "SYNN" Neutron model payload format
 *
 * A Neutron model travels inside a .synm image (see syn_synm.h) as a
 * payload that starts with this 48-byte header followed by the
 * neutron-converter section triple (microcode / weights / kernels),
 * each 16-byte aligned relative to the payload start. Relative
 * alignment is sufficient: every payload source (model store slots,
 * OTA staging, build-time embedding) hands out 16-byte-aligned base
 * pointers.
 *
 * tools/syn_model_pack.py --neutron-* produces this layout on the
 * host; this header is the single C-side definition. Little-endian,
 * like .synm.
 *
 * scratch_size carries the converter memory report so the HAL can
 * refuse a model whose scratch requirement exceeds its buffer
 * instead of letting the NPU overrun it.
 */
#ifndef SYNAPTIC_SYN_SYNN_H_
#define SYNAPTIC_SYN_SYNN_H_

#include <stddef.h>
#include <stdint.h>
#include <zephyr/toolchain.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SYN_SYNN_MAGIC    0x4E4E5953UL /* "SYNN" read as LE uint32 */
#define SYN_SYNN_VERSION  1U
#define SYN_SYNN_HDR_SIZE 48U

struct syn_synn_hdr {
	uint32_t magic;
	uint32_t version;
	uint32_t microcode_off;
	uint32_t microcode_size;
	uint32_t weights_off;
	uint32_t weights_size;
	uint32_t kernels_off;
	uint32_t kernels_size;
	uint32_t input_size;    /* single input, bytes */
	uint32_t output_size;   /* single output, bytes (NPU-padded) */
	uint32_t scratch_size;  /* converter-reported requirement, bytes */
	uint32_t flags;         /* reserved, must be zero */
};

BUILD_ASSERT(sizeof(struct syn_synn_hdr) == SYN_SYNN_HDR_SIZE,
	     "SYNN header must be exactly 48 bytes");

/* Parsed view of a SYNN payload: section pointers into the blob. */
struct syn_synn_desc {
	const uint8_t *microcode;
	const uint8_t *weights;
	const uint8_t *kernels;
	uint32_t microcode_size;
	uint32_t weights_size;
	uint32_t kernels_size;
	uint32_t input_size;
	uint32_t output_size;
	uint32_t scratch_size;
};

/* Validate a SYNN payload and fill @p desc with section views.
 *
 * Returns 0 on success, -ENOTSUP when the blob does not carry the
 * SYNN magic (caller falls back to its non-Neutron path), -EINVAL
 * when the blob claims to be SYNN but is malformed (unsupported
 * version, reserved flags set, section out of bounds or misaligned,
 * zero-sized i/o).
 */
int syn_synn_parse(const uint8_t *data, size_t size,
		   struct syn_synn_desc *desc);

#ifdef __cplusplus
}
#endif
#endif /* SYNAPTIC_SYN_SYNN_H_ */
