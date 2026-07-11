# face_detection

Continuous vision-pipeline demo for the SynapticOS Phase 2 inference
engine.

Every frame runs the full production path:

```
synthetic frame (24x24x3)
  -> resize (bilinear, 12x12)
  -> normalize (per-channel mean/std, float32)
  -> quantize (int8)
  -> detector model (NPU HAL)
  -> grid decode -> non-maximum suppression
  -> console report + per-frame profiling
```

The frame source is a deterministic synthetic generator (a bright blob
moving across a gradient background) standing in for the OV7670
camera, and results are reported on the console instead of an LCD
overlay. DVP camera capture and LCD-PAR-S035 rendering are the
remaining hardware bring-up steps for the FRDM-MCXN947 target; the
pipeline, scheduler, and post-processing here are unchanged by them.

## Build & run (QEMU)

```sh
west build -b qemu_cortex_m3 synaptic-os/samples/face_detection --pristine
west build -t run
```

## Build & flash (FRDM-MCXN947)

```sh
west build -b frdm_mcxn947/mcxn947/cpu0 synaptic-os/samples/face_detection --pristine
```

Flash `build/zephyr/zephyr.bin` with `blhost` (ISP mode) as described
in the top-level README.

## What to look for

- Detections vary deterministically with the frame content (the stub
  NPU derives its winning cell from a hash of the input; on Neutron
  hardware the model's real detections appear here instead)
- `syn_mem_reset_ephemeral()` after each frame keeps the arena at a
  constant footprint (zero fragmentation across frames)
- The final summary prints average frame time, effective FPS, and the
  profiling breakdown of the last inference
