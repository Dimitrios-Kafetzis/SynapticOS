#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
syn_ota_send.py - stream a .synm model to a SynapticOS device over UART.

Drives the device's shell OTA transport:

  syn ota begin <name> <total_bytes>
  syn ota data <hex>          (one line per chunk, ack-paced)
  syn ota done                (device validates CRC32 and stages)
  syn ota activate            (unless --no-activate)

The transfer is flow-controlled by the device's per-line responses,
so no hardware handshaking is needed. Model name defaults to the one
embedded in the .synm header.

Usage:
  python3 syn_ota_send.py --port /dev/ttyACM0 demo.synm
  python3 syn_ota_send.py --port /dev/ttyACM0 --no-activate demo.synm

Linux only (termios), Python 3.10+, stdlib only.
"""

import argparse
import os
import select
import struct
import sys
import termios
import time

SYNM_HDR = struct.Struct("<4sI32sII4H4H")


def open_port(path, baud):
    fd = os.open(path, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    attrs = termios.tcgetattr(fd)
    speed = getattr(termios, f"B{baud}")
    # raw 8N1, no flow control
    attrs[0] = 0                                  # iflag
    attrs[1] = 0                                  # oflag
    attrs[2] = (termios.CS8 | termios.CREAD | termios.CLOCAL)  # cflag
    attrs[3] = 0                                  # lflag
    attrs[4] = speed                              # ispeed
    attrs[5] = speed                              # ospeed
    termios.tcsetattr(fd, termios.TCSANOW, attrs)
    termios.tcflush(fd, termios.TCIOFLUSH)
    return fd


class Shell:
    def __init__(self, fd, verbose=False):
        self.fd = fd
        self.verbose = verbose
        self.pending = b""

    def read_line(self, timeout):
        """One line of device output, or None on timeout."""
        deadline = time.monotonic() + timeout
        while True:
            nl = self.pending.find(b"\n")
            if nl >= 0:
                line, self.pending = self.pending[:nl], self.pending[nl+1:]
                text = line.decode(errors="replace").strip("\r\x1b[m ")
                if self.verbose and text:
                    print(f"    <- {text}")
                return text
            remain = deadline - time.monotonic()
            if remain <= 0:
                return None
            r, _, _ = select.select([self.fd], [], [], remain)
            if self.fd in r:
                try:
                    self.pending += os.read(self.fd, 4096)
                except BlockingIOError:
                    pass

    def command(self, cmd, expect, deny=("failed", "error"), timeout=5.0):
        """Send a shell line, wait for a line containing `expect`."""
        os.write(self.fd, cmd.encode() + b"\n")
        deadline = time.monotonic() + timeout
        while True:
            line = self.read_line(max(0.05, deadline - time.monotonic()))
            if line is None:
                raise SystemExit(f"error: timeout waiting for '{expect}' "
                                 f"after: {cmd[:60]}...")
            if expect in line:
                return line
            if any(d in line for d in deny):
                raise SystemExit(f"error: device reported: {line}")


def main():
    ap = argparse.ArgumentParser(
        description="Stream a .synm model to a SynapticOS device")
    ap.add_argument("synm", help=".synm file (from syn_model_pack.py)")
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--name", help="override the model name from the header")
    ap.add_argument("--chunk", type=int, default=1024,
                    help="payload bytes per shell line (default 1024; the "
                         "device's SHELL_CMD_BUFF_SIZE bounds this)")
    ap.add_argument("--no-activate", action="store_true",
                    help="stage only; activate later on the device shell")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="echo device output")
    args = ap.parse_args()

    with open(args.synm, "rb") as f:
        image = f.read()
    if len(image) < SYNM_HDR.size or image[:4] != b"SYNM":
        raise SystemExit("error: not a .synm file")

    name = args.name
    if name is None:
        name = SYNM_HDR.unpack_from(image, 0)[2].rstrip(b"\0").decode()

    fd = open_port(args.port, args.baud)
    sh = Shell(fd, args.verbose)

    # Echoing kilobyte-long hex lines back (with the shell's full-line
    # ANSI redraws) saturates the device TX at UART speed and stalls
    # RX ingestion; disable echo for the duration of the transfer.
    os.write(fd, b"shell echo off\n")
    time.sleep(0.3)
    termios.tcflush(fd, termios.TCIFLUSH)
    sh.pending = b""

    print(f"sending '{name}' ({len(image)} bytes) to {args.port} "
          f"in {args.chunk}-byte chunks")

    t0 = time.monotonic()
    # begin erases the whole staging area sector by sector; scale the
    # wait with the image size (~1 s per 50 KB is generous)
    begin_timeout = max(10.0, len(image) / 51200)
    sh.command(f"syn ota begin {name} {len(image)}", "OTA RX",
               timeout=begin_timeout)

    sent = 0
    while sent < len(image):
        chunk = image[sent:sent + args.chunk]
        sh.command(f"syn ota data {chunk.hex()}", "ok ", timeout=10.0)
        sent += len(chunk)
        pct = 100 * sent // len(image)
        print(f"\r  {sent}/{len(image)} bytes ({pct}%)", end="", flush=True)
    print()

    sh.command("syn ota done", "OTA staged", timeout=15.0)
    t_staged = time.monotonic() - t0
    print(f"staged and CRC-verified in {t_staged:.2f} s "
          f"({len(image)/max(t_staged, 1e-9)/1024:.1f} KB/s)")

    if not args.no_activate:
        line = sh.command("syn ota activate", "OTA activated", timeout=10.0)
        print(line)
        print(f"total update time {time.monotonic() - t0:.2f} s")
    else:
        print("staged only; run 'syn ota activate' on the device")

    os.write(fd, b"shell echo on\n")
    os.close(fd)
    return 0


if __name__ == "__main__":
    sys.exit(main())
