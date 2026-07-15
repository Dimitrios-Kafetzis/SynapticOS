/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file syn_synm.h
 * @brief SynapticOS - .synm packaged-model format (on-flash / on-wire)
 *
 * A .synm image is a 64-byte header followed immediately by the raw
 * model binary. tools/syn_model_pack.py produces the same layout on
 * the host; this header is the single C-side definition. The format
 * is little-endian (both the host tool and the MCU are LE).
 *
 * Works for real .tflite payloads and for the stub blobs used with
 * the stub NPU HAL alike: the header carries everything the registry
 * needs, the payload is opaque to the store.
 */
#ifndef SYNAPTIC_SYN_SYNM_H_
#define SYNAPTIC_SYN_SYNM_H_

#include <stdint.h>
#include <zephyr/toolchain.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SYN_SYNM_MAGIC    0x4D4E5953UL /* "SYNM" read as LE uint32 */
#define SYN_SYNM_VERSION  1U
#define SYN_SYNM_HDR_SIZE 64U

/* magic(4) version(4) name(32) model_size(4) crc32(4)
 * input_shape(8 = 4 x u16) output_shape(8 = 4 x u16)
 */
struct syn_synm_hdr {
	uint32_t magic;
	uint32_t version;
	char     name[32];      /* not necessarily NUL-terminated */
	uint32_t model_size;    /* payload bytes after the header */
	uint32_t crc32;         /* IEEE CRC32 of the payload only */
	uint16_t input_shape[4];
	uint16_t output_shape[4];
};

BUILD_ASSERT(sizeof(struct syn_synm_hdr) == SYN_SYNM_HDR_SIZE,
	     "synm header must be exactly 64 bytes");

#ifdef __cplusplus
}
#endif
#endif /* SYNAPTIC_SYN_SYNM_H_ */
