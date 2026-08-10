/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file test_synn.c
 * @brief Unit tests for the SYNN Neutron payload parser (Phase 6.1c)
 *
 * The parser is structural only (the Neutron HAL applies its policy
 * limits on top), so the whole container contract is testable on
 * QEMU without the NPU: magic/version gates, section bounds and
 * alignment, i/o sanity, and the fall-through contract (-ENOTSUP)
 * that keeps non-SYNN models on the stub path.
 */

#include <zephyr/ztest.h>
#include <string.h>

#include "syn_synn.h"

/* A minimal well-formed blob: header + one 16-byte-aligned section
 * per member, mirroring the packer layout (first section right after
 * the 48-byte header).
 */
#define T_UCODE_SIZE   20U
#define T_UCODE_OFF    48U
#define T_WEIGHTS_SIZE 40U
#define T_WEIGHTS_OFF  80U  /* align16(48 + 20) */
#define T_KERNELS_SIZE 1U
#define T_KERNELS_OFF  128U /* align16(80 + 40) */
#define T_BLOB_SIZE    144U /* align16(128 + 1) */

static uint8_t blob[T_BLOB_SIZE] __aligned(16);

static void blob_build(void)
{
	struct syn_synn_hdr hdr = {
		.magic          = SYN_SYNN_MAGIC,
		.version        = SYN_SYNN_VERSION,
		.microcode_off  = T_UCODE_OFF,
		.microcode_size = T_UCODE_SIZE,
		.weights_off    = T_WEIGHTS_OFF,
		.weights_size   = T_WEIGHTS_SIZE,
		.kernels_off    = T_KERNELS_OFF,
		.kernels_size   = T_KERNELS_SIZE,
		.input_size     = 64U,
		.output_size    = 12U,
		.scratch_size   = 1024U,
		.flags          = 0U,
	};

	memset(blob, 0, sizeof(blob));
	memcpy(blob, &hdr, sizeof(hdr));
	for (size_t i = 0; i < T_UCODE_SIZE; i++) {
		blob[T_UCODE_OFF + i] = (uint8_t)(0xA0U + i);
	}
	for (size_t i = 0; i < T_WEIGHTS_SIZE; i++) {
		blob[T_WEIGHTS_OFF + i] = (uint8_t)(0x40U + i);
	}
	blob[T_KERNELS_OFF] = 0x77U;
}

static void hdr_patch(size_t offset, uint32_t value)
{
	memcpy(&blob[offset], &value, sizeof(value));
}

ZTEST_SUITE(syn_synn_suite, NULL, NULL, NULL, NULL, NULL);

ZTEST(syn_synn_suite, test_parse_valid)
{
	struct syn_synn_desc desc;

	blob_build();
	zassert_ok(syn_synn_parse(blob, sizeof(blob), &desc), "parse failed");
	zassert_equal_ptr(desc.microcode, &blob[T_UCODE_OFF], "ucode ptr");
	zassert_equal_ptr(desc.weights, &blob[T_WEIGHTS_OFF], "weights ptr");
	zassert_equal_ptr(desc.kernels, &blob[T_KERNELS_OFF], "kernels ptr");
	zassert_equal(desc.microcode_size, T_UCODE_SIZE, "ucode size");
	zassert_equal(desc.weights_size, T_WEIGHTS_SIZE, "weights size");
	zassert_equal(desc.kernels_size, T_KERNELS_SIZE, "kernels size");
	zassert_equal(desc.input_size, 64U, "input size");
	zassert_equal(desc.output_size, 12U, "output size");
	zassert_equal(desc.scratch_size, 1024U, "scratch size");
	zassert_equal(desc.microcode[0], 0xA0U, "ucode bytes");
	zassert_equal(desc.weights[0], 0x40U, "weights bytes");
	zassert_equal(desc.kernels[0], 0x77U, "kernels bytes");
}

ZTEST(syn_synn_suite, test_parse_rejects_bad_args)
{
	struct syn_synn_desc desc;

	blob_build();
	zassert_equal(syn_synn_parse(NULL, sizeof(blob), &desc), -EINVAL,
		      "NULL data accepted");
	zassert_equal(syn_synn_parse(blob, sizeof(blob), NULL), -EINVAL,
		      "NULL desc accepted");
}

ZTEST(syn_synn_suite, test_parse_not_synn_falls_through)
{
	struct syn_synn_desc desc;

	blob_build();
	/* Non-SYNN payloads must return -ENOTSUP so the HAL keeps its
	 * stub path (this is how every pre-Neutron model still loads).
	 */
	hdr_patch(0, 0x33464C54UL); /* "TLF3": not SYNN */
	zassert_equal(syn_synn_parse(blob, sizeof(blob), &desc), -ENOTSUP,
		      "bad magic not -ENOTSUP");

	/* Shorter than a header: cannot even carry the magic. */
	blob_build();
	zassert_equal(syn_synn_parse(blob, SYN_SYNN_HDR_SIZE - 1U, &desc),
		      -ENOTSUP, "short blob not -ENOTSUP");
}

ZTEST(syn_synn_suite, test_parse_rejects_bad_version)
{
	struct syn_synn_desc desc;

	blob_build();
	hdr_patch(4, SYN_SYNN_VERSION + 1U);
	zassert_equal(syn_synn_parse(blob, sizeof(blob), &desc), -EINVAL,
		      "future version accepted");

	/* The interim first-light header had no version field: its
	 * microcode offset (48) lands here and must be rejected.
	 */
	hdr_patch(4, 48U);
	zassert_equal(syn_synn_parse(blob, sizeof(blob), &desc), -EINVAL,
		      "interim headerless blob accepted");
}

ZTEST(syn_synn_suite, test_parse_rejects_reserved_flags)
{
	struct syn_synn_desc desc;

	blob_build();
	hdr_patch(44, 0x1U);
	zassert_equal(syn_synn_parse(blob, sizeof(blob), &desc), -EINVAL,
		      "reserved flags accepted");
}

ZTEST(syn_synn_suite, test_parse_rejects_out_of_bounds)
{
	struct syn_synn_desc desc;

	blob_build();
	hdr_patch(12, T_BLOB_SIZE); /* microcode_size: off 48 + this > blob */
	zassert_equal(syn_synn_parse(blob, sizeof(blob), &desc), -EINVAL,
		      "oversized microcode accepted");

	/* Overflow attempt: off + size wraps 32 bits. */
	blob_build();
	hdr_patch(16, 0xFFFFFFF0UL); /* weights_off */
	hdr_patch(20, 0x20U);        /* weights_size */
	zassert_equal(syn_synn_parse(blob, sizeof(blob), &desc), -EINVAL,
		      "wrapping section accepted");
}

ZTEST(syn_synn_suite, test_parse_rejects_misalignment)
{
	struct syn_synn_desc desc;

	blob_build();
	hdr_patch(24, T_KERNELS_OFF + 4U); /* kernels_off off 16-B grid */
	zassert_equal(syn_synn_parse(blob, sizeof(blob), &desc), -EINVAL,
		      "misaligned section accepted");
}

ZTEST(syn_synn_suite, test_parse_rejects_zero_io)
{
	struct syn_synn_desc desc;

	blob_build();
	hdr_patch(32, 0U); /* input_size */
	zassert_equal(syn_synn_parse(blob, sizeof(blob), &desc), -EINVAL,
		      "zero input size accepted");

	blob_build();
	hdr_patch(36, 0U); /* output_size */
	zassert_equal(syn_synn_parse(blob, sizeof(blob), &desc), -EINVAL,
		      "zero output size accepted");
}
