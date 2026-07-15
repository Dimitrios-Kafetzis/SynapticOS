# ota_update

End-to-end model OTA demo: boot with a factory model serving a
periodic inference tick, stream an update over the shell UART, watch
the tick switch models without interruption, roll back from the shell.

## Build and run

QEMU (RAM-emulated flash; full flow minus true reboot persistence):

```sh
west build -b qemu_cortex_m3 synaptic-os/samples/ota_update --pristine
west build -t run
```

FRDM-MCXN947 (persistent store in the bank 1 tail, per the generated
partition map; never touches bank 0 or the CPU1 image reserve):

```sh
west build -b frdm_mcxn947/mcxn947/cpu0 synaptic-os/samples/ota_update --pristine
# flash via ISP (blhost), see the project docs
```

## Pushing an update

Pack any model binary (a real `.tflite` or an opaque stub blob) and
send it. With a second serial terminal closed (one reader at a time):

```sh
head -c 2048 /dev/urandom > blob.bin
python3 tools/syn_model_pack.py --input blob.bin --name demo_model \
    --input-shape 1,8,8 --output-shape 1,16 --output demo_v2.synm
python3 tools/syn_ota_send.py --port /dev/ttyACM0 demo_v2.synm
```

The device log shows the staging slot being validated and activated;
the inference tick line switches from `(factory) slot 0` to the new
model. Roll back on the device shell:

```
syn ota rollback
syn store status
```

Useful shell commands: `syn store status`, `syn ota status`,
`syn model list`.

## Notes

- Transfer rides the shell as ack-paced hex lines (default 1024-byte
  chunks; the OTA engine page-buffers 128-byte flash pages under any
  chunking). At 115200 baud the hex encoding halves throughput to
  roughly 5 KB/s; large models are transport-bound, which is reported
  honestly by the sender's timing output.
- Dual-core: this sample is single-core. On a dual-core-flashed board
  the same `syn ota` commands inside the dual_model sample exercise
  the full interaction: the OTA session parks CPU1 (its XIP bank is
  being written), resumes it afterwards, and offload pauses meanwhile.
- Models are stub blobs until the Neutron invoke path lands; the
  `.synm` pipeline is identical for real `.tflite` files.
