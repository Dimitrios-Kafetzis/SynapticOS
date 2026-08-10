#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
syn_model_pack.py - package a model binary into SynapticOS .synm format.

A .synm image is a 64-byte little-endian header followed by the raw
model payload; the C-side definition is src/core/syn_synm.h:

  magic        4B  "SYNM"
  version      4B  format version (1)
  name        32B  model name, NUL-padded
  model_size   4B  payload bytes after the header
  crc32        4B  IEEE CRC32 of the payload (zlib.crc32)
  input_shape  8B  4 x u16 (0 = unused dim)
  output_shape 8B  4 x u16

For real .tflite files the input/output tensor shapes and dtypes are
extracted directly from the flatbuffer (minimal reader below, no
external dependencies). For stub blobs (the models used with the stub
NPU HAL are opaque byte blobs) pass --input-shape/--output-shape
explicitly.

Neutron models: the neutron-converter dump triple (microcode /
weights / kernels) is packed into a "SYNN" payload (48-byte header +
16-byte-aligned sections; C-side definition src/core/syn_synn.h)
which then travels inside the .synm image like any other payload.
The i/o byte sizes default to the shape element counts (INT8, one
byte per element); pass --input-size/--output-size to override.
--scratch-size is the scratch requirement from the converter memory
report; the device HAL refuses models needing more scratch than its
buffer. --raw-output additionally writes the bare SYNN blob (no
.synm header) for build-time embedding (samples/neutron_hello).

Usage:
  python3 syn_model_pack.py --input model.tflite --name face_detect \
      --output face_detect.synm
  python3 syn_model_pack.py --input blob.bin --name stub_model \
      --input-shape 1,96,96,3 --output-shape 1,10 --output stub.synm
  python3 syn_model_pack.py --neutron-microcode ucode.bin \
      --neutron-weights weights.bin --neutron-kernels kernels.bin \
      --scratch-size 49152 --name resnet_cifar10 \
      --input-shape 1,32,32,3 --output-shape 1,12 \
      --output resnet_cifar10.synm --raw-output resnet_cifar10.synn
  python3 syn_model_pack.py --inspect face_detect.synm
  python3 syn_model_pack.py --selftest

Python 3.10+, stdlib only (struct, zlib, argparse).
"""

import argparse
import struct
import sys
import zlib

SYNM_MAGIC = b"SYNM"
SYNM_VERSION = 1
SYNM_HDR = struct.Struct("<4sI32sII4H4H")
assert SYNM_HDR.size == 64

# SYNN Neutron payload: magic, version, 3 x (off, size), input_size,
# output_size, scratch_size, flags — see src/core/syn_synn.h.
SYNN_MAGIC = b"SYNN"
SYNN_VERSION = 1
SYNN_HDR = struct.Struct("<4s11I")
assert SYNN_HDR.size == 48

TFLITE_DTYPES = {
    0: "FLOAT32", 1: "FLOAT16", 2: "INT32", 3: "UINT8", 4: "INT64",
    5: "STRING", 6: "BOOL", 7: "INT16", 8: "COMPLEX64", 9: "INT8",
    10: "FLOAT64", 11: "COMPLEX128", 12: "UINT64", 13: "RESOURCE",
    14: "VARIANT", 15: "UINT32", 16: "UINT16", 17: "INT4",
}


# --------------------------------------------------------------------
# Minimal flatbuffer reader (just enough for the TFLite schema)
# --------------------------------------------------------------------

class FB:
    def __init__(self, buf):
        self.buf = buf

    def u16(self, pos):
        return struct.unpack_from("<H", self.buf, pos)[0]

    def i32(self, pos):
        return struct.unpack_from("<i", self.buf, pos)[0]

    def u32(self, pos):
        return struct.unpack_from("<I", self.buf, pos)[0]

    def indirect(self, pos):
        """Follow a table/vector offset stored at pos."""
        return pos + self.u32(pos)

    def field(self, table, field_id):
        """Absolute position of a table field, or None if absent."""
        vtable = table - self.i32(table)
        vsize = self.u16(vtable)
        entry = 4 + 2 * field_id
        if entry >= vsize:
            return None
        off = self.u16(vtable + entry)
        if off == 0:
            return None
        return table + off

    def vector(self, table, field_id):
        """(element_base, length) of a vector field, or (None, 0)."""
        pos = self.field(table, field_id)
        if pos is None:
            return None, 0
        vec = self.indirect(pos)
        return vec + 4, self.u32(vec)


def parse_tflite(buf):
    """Return (input_shape, output_shape, in_dtype, out_dtype).

    Shapes are tuples of ints for the FIRST subgraph's first input and
    first output tensor. Raises ValueError if buf is not a TFLite
    flatbuffer.
    """
    if len(buf) < 8 or buf[4:8] != b"TFL3":
        raise ValueError("not a TFLite flatbuffer (missing TFL3 identifier)")

    fb = FB(buf)
    model = fb.indirect(0)

    # Model.subgraphs = field 2
    sg_base, sg_len = fb.vector(model, 2)
    if sg_base is None or sg_len == 0:
        raise ValueError("TFLite model has no subgraphs")
    subgraph = fb.indirect(sg_base)  # first subgraph

    # SubGraph: tensors=0, inputs=1, outputs=2
    tens_base, tens_len = fb.vector(subgraph, 0)
    in_base, in_len = fb.vector(subgraph, 1)
    out_base, out_len = fb.vector(subgraph, 2)
    if in_len == 0 or out_len == 0 or tens_len == 0:
        raise ValueError("TFLite subgraph lacks inputs/outputs")

    def tensor_info(index):
        if index >= tens_len:
            raise ValueError(f"tensor index {index} out of range")
        tensor = fb.indirect(tens_base + 4 * index)
        shape_base, shape_len = fb.vector(tensor, 0)  # Tensor.shape
        shape = tuple(fb.i32(shape_base + 4 * i) for i in range(shape_len))
        tpos = fb.field(tensor, 1)  # Tensor.type (byte)
        dtype = buf[tpos] if tpos is not None else 0
        return shape, TFLITE_DTYPES.get(dtype, f"?{dtype}")

    in_shape, in_dtype = tensor_info(fb.i32(in_base))
    out_shape, out_dtype = tensor_info(fb.i32(out_base))
    return in_shape, out_shape, in_dtype, out_dtype


# --------------------------------------------------------------------
# SYNN (Neutron payload) building / inspection
# --------------------------------------------------------------------

def align16(n):
    return (n + 15) & ~15


def build_synn(microcode, weights, kernels,
               input_size, output_size, scratch_size):
    """Concatenate the converter triple behind a SYNN header, every
    section 16-byte aligned relative to the blob start."""
    sections = (microcode, weights, kernels)
    offs = []
    pos = align16(SYNN_HDR.size)
    for data in sections:
        offs.append(pos)
        pos = align16(pos + len(data))

    blob = bytearray(pos)
    SYNN_HDR.pack_into(blob, 0, SYNN_MAGIC, SYNN_VERSION,
                       offs[0], len(microcode),
                       offs[1], len(weights),
                       offs[2], len(kernels),
                       input_size, output_size, scratch_size, 0)
    for off, data in zip(offs, sections):
        blob[off:off + len(data)] = data
    return bytes(blob)


def inspect_synn(payload, indent=""):
    """Print and validate a SYNN payload; returns True when valid."""
    if len(payload) < SYNN_HDR.size:
        print(f"{indent}SYNN payload shorter than a header")
        return False
    (magic, version, uc_off, uc_size, w_off, w_size, k_off, k_size,
     in_size, out_size, scratch, flags) = SYNN_HDR.unpack_from(payload, 0)

    ok = True
    print(f"{indent}SYNN version {version} "
          f"{'ok' if version == SYNN_VERSION else 'UNSUPPORTED'}")
    sections = (("microcode", uc_off, uc_size),
                ("weights", w_off, w_size),
                ("kernels", k_off, k_size))
    for name, off, size in sections:
        bad = []
        if off % 16:
            bad.append("MISALIGNED")
        if off + size > len(payload):
            bad.append("OUT OF BOUNDS")
        print(f"{indent}{name:<10} {size:>8} B @ {off}"
              + ("".join(" " + b for b in bad) if bad else ""))
        ok = ok and not bad
    print(f"{indent}io {in_size}/{out_size} B  scratch {scratch} B  "
          f"flags 0x{flags:08x}")
    if version != SYNN_VERSION or flags != 0 or in_size == 0 or out_size == 0:
        ok = False
    return ok


# --------------------------------------------------------------------
# Packing
# --------------------------------------------------------------------

def shape4(shape, what):
    """Normalize a shape to exactly 4 u16 dims (0-padded)."""
    dims = list(shape)
    if len(dims) > 4:
        raise SystemExit(f"error: {what} has {len(dims)} dims; "
                         f"the .synm header carries at most 4: {shape}")
    for d in dims:
        if not 0 <= d <= 0xFFFF:
            raise SystemExit(f"error: {what} dim {d} out of u16 range")
    return tuple(dims) + (0,) * (4 - len(dims))


def parse_shape_arg(text):
    try:
        return tuple(int(x) for x in text.split(",") if x.strip() != "")
    except ValueError:
        raise SystemExit(f"error: bad shape '{text}' (want e.g. 1,96,96,3)")


def pack(args):
    with open(args.input, "rb") as f:
        payload = f.read()
    if len(payload) == 0:
        raise SystemExit("error: input file is empty")
    if len(args.name.encode()) > 31:
        raise SystemExit("error: name exceeds 31 bytes")

    in_shape = out_shape = None
    try:
        in_shape, out_shape, in_dtype, out_dtype = parse_tflite(payload)
        note = (f"tflite: input {in_shape} {in_dtype}, "
                f"output {out_shape} {out_dtype}")
        if in_dtype != "INT8" or out_dtype != "INT8":
            print(f"warning: tensor dtypes are {in_dtype}/{out_dtype}; "
                  f"the device runtime assumes INT8 (stub-NPU convention)")
    except ValueError as e:
        note = f"payload treated as opaque blob ({e})"
        if args.input_shape is None or args.output_shape is None:
            raise SystemExit(
                f"error: {e}; pass --input-shape and --output-shape "
                f"to pack an opaque blob")

    if args.input_shape is not None:
        in_shape = parse_shape_arg(args.input_shape)
    if args.output_shape is not None:
        out_shape = parse_shape_arg(args.output_shape)

    crc = zlib.crc32(payload) & 0xFFFFFFFF
    hdr = SYNM_HDR.pack(SYNM_MAGIC, SYNM_VERSION,
                        args.name.encode().ljust(32, b"\0"),
                        len(payload), crc,
                        *shape4(in_shape, "input shape"),
                        *shape4(out_shape, "output shape"))
    with open(args.output, "wb") as f:
        f.write(hdr)
        f.write(payload)

    print(f"packed {args.input} -> {args.output}")
    print(f"  {note}")
    print(f"  name '{args.name}'  payload {len(payload)} bytes  "
          f"crc32 0x{crc:08x}")
    print(f"  header shapes: in {shape4(in_shape, 'i')} "
          f"out {shape4(out_shape, 'o')}")
    return 0


def shape_elements(shape):
    n = 1
    for d in shape:
        n *= d
    return n


def pack_neutron(args):
    if args.input_shape is None or args.output_shape is None:
        raise SystemExit("error: --input-shape and --output-shape are "
                         "required for a Neutron pack")
    if len(args.name.encode()) > 31:
        raise SystemExit("error: name exceeds 31 bytes")

    sections = []
    for path in (args.neutron_microcode, args.neutron_weights,
                 args.neutron_kernels):
        with open(path, "rb") as f:
            sections.append(f.read())
    if len(sections[0]) == 0:
        raise SystemExit("error: microcode file is empty")

    in_shape = parse_shape_arg(args.input_shape)
    out_shape = parse_shape_arg(args.output_shape)
    # INT8: one byte per element. The NPU output may be lane-padded
    # (e.g. 10 classes padded to 12) — the shape passed here must be
    # the padded one, or --output-size must override.
    input_size = args.input_size or shape_elements(in_shape)
    output_size = args.output_size or shape_elements(out_shape)
    if input_size == 0 or output_size == 0:
        raise SystemExit("error: zero input/output size")

    payload = build_synn(sections[0], sections[1], sections[2],
                         input_size, output_size, args.scratch_size)

    crc = zlib.crc32(payload) & 0xFFFFFFFF
    hdr = SYNM_HDR.pack(SYNM_MAGIC, SYNM_VERSION,
                        args.name.encode().ljust(32, b"\0"),
                        len(payload), crc,
                        *shape4(in_shape, "input shape"),
                        *shape4(out_shape, "output shape"))
    with open(args.output, "wb") as f:
        f.write(hdr)
        f.write(payload)
    if args.raw_output:
        with open(args.raw_output, "wb") as f:
            f.write(payload)

    print(f"packed Neutron triple -> {args.output}"
          + (f" (+ raw {args.raw_output})" if args.raw_output else ""))
    print(f"  name '{args.name}'  payload {len(payload)} bytes  "
          f"crc32 0x{crc:08x}")
    print(f"  ucode {len(sections[0])} B  weights {len(sections[1])} B  "
          f"kernels {len(sections[2])} B")
    print(f"  io {input_size}/{output_size} B  "
          f"scratch {args.scratch_size} B")
    return 0


def inspect(path):
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] == SYNN_MAGIC:
        # bare SYNN blob (no .synm wrapper), e.g. the build-time
        # embedding artifact
        print(f"bare SYNN blob, {len(data)} bytes")
        return 0 if inspect_synn(data) else 1
    if len(data) < SYNM_HDR.size:
        raise SystemExit("error: file shorter than a .synm header")
    (magic, version, name, model_size, crc,
     i0, i1, i2, i3, o0, o1, o2, o3) = SYNM_HDR.unpack_from(data, 0)
    payload = data[SYNM_HDR.size:]
    actual_crc = zlib.crc32(payload) & 0xFFFFFFFF

    ok_magic = magic == SYNM_MAGIC
    ok_version = version == SYNM_VERSION
    ok_size = model_size == len(payload)
    ok_crc = crc == actual_crc

    print(f"magic        {magic!r} {'ok' if ok_magic else 'BAD'}")
    print(f"version      {version} {'ok' if ok_version else 'UNSUPPORTED'}")
    print(f"name         '{name.rstrip(bytes(1)).decode(errors='replace')}'")
    print(f"model_size   {model_size} " +
          ("ok" if ok_size else f"MISMATCH (file has {len(payload)})"))
    print(f"crc32        0x{crc:08x} " +
          ("ok" if ok_crc else f"MISMATCH (computed 0x{actual_crc:08x})"))
    print(f"input_shape  {(i0, i1, i2, i3)}")
    print(f"output_shape {(o0, o1, o2, o3)}")

    ok_payload = True
    if payload[:4] == SYNN_MAGIC:
        print("payload      SYNN (Neutron)")
        ok_payload = inspect_synn(payload, indent="  ")

    return 0 if (ok_magic and ok_version and ok_size and ok_crc and
                 ok_payload) else 1


def selftest():
    """Round-trip an opaque blob and verify every header field."""
    import os
    import tempfile

    blob = bytes((i * 37 + 11) % 256 for i in range(5000))
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "blob.bin")
        dst = os.path.join(td, "blob.synm")
        with open(src, "wb") as f:
            f.write(blob)

        ns = argparse.Namespace(input=src, name="selftest", output=dst,
                                input_shape="1,96,96,3", output_shape="1,10")
        pack(ns)

        with open(dst, "rb") as f:
            data = f.read()
        assert len(data) == 64 + len(blob)
        assert data[64:] == blob
        (magic, version, name, model_size, crc, *shapes) = \
            SYNM_HDR.unpack_from(data, 0)
        assert magic == SYNM_MAGIC
        assert version == SYNM_VERSION
        assert name.rstrip(bytes(1)) == b"selftest"
        assert model_size == len(blob)
        assert crc == (zlib.crc32(blob) & 0xFFFFFFFF)
        assert tuple(shapes) == (1, 96, 96, 3, 1, 10, 0, 0)
        assert inspect(dst) == 0

        # Neutron triple round-trip
        ucode = bytes(range(100))
        weights = bytes((i * 3) % 256 for i in range(1000))
        kernels = b"\x00"
        paths = {}
        for tag, blob_data in (("u", ucode), ("w", weights), ("k", kernels)):
            paths[tag] = os.path.join(td, f"{tag}.bin")
            with open(paths[tag], "wb") as f:
                f.write(blob_data)
        synm = os.path.join(td, "neutron.synm")
        synn = os.path.join(td, "neutron.synn")
        ns = argparse.Namespace(neutron_microcode=paths["u"],
                                neutron_weights=paths["w"],
                                neutron_kernels=paths["k"],
                                scratch_size=49152, name="neutron_st",
                                input_shape="1,32,32,3",
                                output_shape="1,12",
                                input_size=None, output_size=None,
                                output=synm, raw_output=synn)
        pack_neutron(ns)

        with open(synm, "rb") as f:
            data = f.read()
        with open(synn, "rb") as f:
            raw = f.read()
        payload = data[64:]
        assert payload == raw
        (magic, version, uc_off, uc_size, w_off, w_size, k_off, k_size,
         in_size, out_size, scratch, flags) = SYNN_HDR.unpack_from(payload, 0)
        assert (magic, version, flags) == (SYNN_MAGIC, SYNN_VERSION, 0)
        assert uc_off % 16 == 0 and w_off % 16 == 0 and k_off % 16 == 0
        assert payload[uc_off:uc_off + uc_size] == ucode
        assert payload[w_off:w_off + w_size] == weights
        assert payload[k_off:k_off + k_size] == kernels
        assert (in_size, out_size, scratch) == (3072, 12, 49152)
        (_, _, _, model_size, crc, *shapes) = SYNM_HDR.unpack_from(data, 0)
        assert model_size == len(payload)
        assert crc == (zlib.crc32(payload) & 0xFFFFFFFF)
        assert tuple(shapes) == (1, 32, 32, 3, 1, 12, 0, 0)
        assert inspect(synm) == 0
        assert inspect(synn) == 0
    print("selftest ok")
    return 0


def main():
    ap = argparse.ArgumentParser(
        description="Package a model binary into SynapticOS .synm format")
    ap.add_argument("--input", help="input .tflite or opaque model blob")
    ap.add_argument("--name", help="model name (max 31 bytes)")
    ap.add_argument("--output", help="output .synm file")
    ap.add_argument("--input-shape",
                    help="override/provide input shape, e.g. 1,96,96,3")
    ap.add_argument("--output-shape",
                    help="override/provide output shape, e.g. 1,10")
    ap.add_argument("--neutron-microcode", metavar="BIN",
                    help="neutron-converter microcode dump (SYNN pack)")
    ap.add_argument("--neutron-weights", metavar="BIN",
                    help="neutron-converter weights dump (SYNN pack)")
    ap.add_argument("--neutron-kernels", metavar="BIN",
                    help="neutron-converter kernels dump (SYNN pack)")
    ap.add_argument("--scratch-size", type=int,
                    help="scratch bytes from the converter memory report")
    ap.add_argument("--input-size", type=int,
                    help="override input bytes (default: shape elements)")
    ap.add_argument("--output-size", type=int,
                    help="override output bytes (default: shape elements)")
    ap.add_argument("--raw-output", metavar="SYNN",
                    help="also write the bare SYNN blob (for embedding)")
    ap.add_argument("--inspect", metavar="FILE",
                    help="print and verify a .synm or .synn file, then exit")
    ap.add_argument("--selftest", action="store_true",
                    help="run the built-in round-trip test, then exit")
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if args.inspect:
        return inspect(args.inspect)

    neutron_args = (args.neutron_microcode, args.neutron_weights,
                    args.neutron_kernels, args.scratch_size)
    if any(a is not None for a in neutron_args):
        if any(a is None for a in neutron_args):
            ap.error("a Neutron pack needs --neutron-microcode, "
                     "--neutron-weights, --neutron-kernels and "
                     "--scratch-size together")
        if not (args.name and args.output):
            ap.error("--name and --output are required to pack")
        return pack_neutron(args)

    if not (args.input and args.name and args.output):
        ap.error("--input, --name and --output are required to pack")
    return pack(args)


if __name__ == "__main__":
    sys.exit(main())
