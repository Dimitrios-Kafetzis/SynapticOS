# Hello Inference

Minimal SynapticOS sample that initializes the runtime, prints
memory statistics, and validates the build chain.

## Build & Flash

```bash
west build -b frdm_mcxn947/mcxn947/cpu0 samples/hello_inference
west flash
```

## Expected Output

```
[00:00:00.001,000] <inf> hello_inference: SynapticOS 0.1.0 — Hello Inference
[00:00:00.002,000] <inf> hello_inference: Initializing runtime...
[00:00:00.003,000] <inf> hello_inference: Runtime initialized successfully
[00:00:00.004,000] <inf> syn_mem: Arena: 128 KB total, 0 KB used, peak 0 KB
```
