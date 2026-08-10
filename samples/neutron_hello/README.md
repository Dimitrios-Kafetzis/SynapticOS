# Neutron NPU Hello

First-light sample for the real eIQ Neutron NPU invoke path on the
FRDM-MCXN947 (CPU0). Runs a neutron-converter-compiled INT8
classifier through `syn_infer_run_sync` on the NPU and prints raw
logits, the top-1 class and latency statistics.

This sample only builds for `frdm_mcxn947/mcxn947/cpu0` with the
optional eIQ Neutron module fetched (`CONFIG_SYNAPTIC_NEUTRON=y`
fails the build otherwise, by design):

```sh
west config manifest.group-filter -- +neutron
west update mcuxsdk-middleware-eiq
```

## Model blob

The build embeds `models/neutron/resnet_cifar10.synn`, which is NOT
committed (converter output embeds NXP-generated content that does
not belong in an Apache-2.0 tree). Regenerate it locally:

1. Install the converter (Linux x86_64, Python 3.10-3.12):

   ```sh
   python3 -m venv ~/.venvs/neutron
   ~/.venvs/neutron/bin/pip install neutron_converter_sdk_25_12 \
       --extra-index-url https://eiq.nxp.com/repository
   ```

2. Convert an INT8 LiteRT/TFLite model for target `mcxn94x` with
   `dumpMicrocodeFile/dumpWeightsFile/dumpKernelsFile` enabled
   (`convertModel` from the `neutron_converter` Python module). The
   MLPerf Tiny ResNet-8 CIFAR-10 model converts cleanly (23/24 ops
   on the NPU, softmax stays on the CPU, so the NPU output is the
   pre-softmax logits; top-1 is unaffected).

3. Pack the dumped microcode / weights / kernels triple into the
   SYNN blob and place it at `models/neutron/resnet_cifar10.synn`
   (input 3072 B, output 10 B for this model).

Set `CONFIG_SYNAPTIC_NEUTRON_SCRATCH_SIZE` to the scratch size from
the converter's memory report (49,152 B for this model).

## Build and run

```sh
west build -b frdm_mcxn947/mcxn947/cpu0 synaptic-os/samples/neutron_hello --pristine
```

Flash over ISP (`blhost`) and watch the UART at 115200. Expected:
top-1 class 3 for the built-in deterministic test pattern, matching
the host LiteRT reference.
