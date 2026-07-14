# dual_model

Asymmetric dual-core demo for the FRDM-MCXN947: CPU0 runs the
SynapticOS AI runtime and serves inference; CPU1 runs the
application, alternating two models over IPC.

```
CPU0 (AI runtime)                    CPU1 (application)
 syn_init + models                    attach shared region
 syn_ipc_init  ── shared 96 KB ──►    syn_ipc_init
 serve INFER_REQ ◄── SPSC rings ──►   STATUS_REQ handshake
 release CPU1 via SYSCON              alternate face_detect /
 shell on VCOM                        keyword_spot requests
```

- face_detect: 96x96x3 INT8 frame, REALTIME priority
- keyword_spot: 49x10 INT8 MFCC window, NORMAL priority
- models run against the stub NPU path (honest labeling: real
  compiled Neutron models arrive in a later phase)

## Build

Two separate images. CPU0 uses the stock Zephyr board; CPU1 uses the
out-of-tree board in `boards/nxp/frdm_mcxn947_cpu1` (Zephyr 3.7 has
no in-tree cpu1 target).

```sh
# CPU0 (AI runtime + shell), flash bank 0
west build -b frdm_mcxn947/mcxn947/cpu0 synaptic-os/samples/dual_model \
    --pristine -d build-dm-cpu0

# CPU1 (application), flash bank 1
west build -b frdm_mcxn947_cpu1/mcxn947/cpu1 synaptic-os/samples/dual_model/remote \
    --pristine -d build-dm-cpu1
```

## Flash (ISP / blhost)

Enter ISP mode (hold SW3, press+release SW1, release SW3), then:

```sh
blhost -p /dev/ttyACM0 flash-erase-all
blhost -p /dev/ttyACM0 write-memory 0x10000000 build-dm-cpu0/zephyr/zephyr.bin
blhost -p /dev/ttyACM0 write-memory 0x10100000 build-dm-cpu1/zephyr/zephyr.bin
```

Reset (press SW1). CPU0 boots, publishes the shared region, releases
CPU1 via SYSCON CPBOOT/CPUCTRL, and logs the handshake timing.

## Observe

Only CPU0 has a console (the board's single VCOM):

```
uart:~$ syn ipc status      # CPU1 link, boot/handshake times, serve stats
uart:~$ syn mpu test        # cross-core write protection demo
uart:~$ syn model list      # both models registered
```

Every 100th served inference is logged with model, priority and
latency. If CPU1 is absent or unresponsive, CPU0 continues
single-core with the full shell.

## Notes

- CPU1 is a Cortex-M33 without FPU/DSP/TrustZone/MPU. It uses plain
  (non-secure) addresses and cannot fault its own accesses; MPU
  write-protection is enforced on CPU0's side only.
- Ring indices live in shared SRAM, so a CPU1 reset mid-run rejoins
  the existing ring state; stale responses drain via the heartbeat.
- One cross-core request is in flight at a time; CPU1 callers
  serialize on a mutex.
