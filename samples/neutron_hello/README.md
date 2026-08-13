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
   ~/.venvs/neutron/bin/pip install neutron_converter_SDK_26_03 \
       --extra-index-url https://eiq.nxp.com/repository
   ```

   The converter release and the on-target driver are a locked
   pair: the microcode carries the driver generation hash and the
   driver rejects any other. The `west.yml` eiq pin tracks Neutron
   Software 3.0.0, which pairs with SDK_26_03 output.

2. Convert an INT8 LiteRT/TFLite model for target `mcxn94x` with
   `dumpMicrocodeFile/dumpWeightsFile/dumpKernelsFile` enabled
   (`convertModel` from the `neutron_converter` Python module). The
   MLPerf Tiny ResNet-8 CIFAR-10 model converts cleanly (24/26 ops
   on the NPU; softmax and the output-trim slice stay on the CPU,
   so the NPU output is the pre-softmax logits padded to 12 lanes;
   top-1 is unaffected).

3. Pack the dumped microcode / weights / kernels triple with
   `tools/syn_model_pack.py` (input 3072 B, output 12 B and
   49,152 B scratch for this model):

   ```sh
   python3 tools/syn_model_pack.py \
       --neutron-microcode ucode.bin --neutron-weights weights.bin \
       --neutron-kernels kernels.bin --scratch-size 49152 \
       --name resnet_cifar10 --input-shape 1,32,32,3 \
       --output-shape 1,12 \
       --output models/neutron/resnet_cifar10.synm \
       --raw-output models/neutron/resnet_cifar10.synn
   ```

   The `.synn` file is the bare SYNN payload this sample embeds;
   the `.synm` file is the same payload in the model-store/OTA
   container.

Set `CONFIG_SYNAPTIC_NEUTRON_SCRATCH_SIZE` to at least the scratch
size from the converter's memory report; the HAL refuses models
whose SYNN header asks for more than the configured buffer.

## Build and run

```sh
west build -b frdm_mcxn947/mcxn947/cpu0 synaptic-os/samples/neutron_hello --pristine
```

Flash over ISP (`blhost`) and watch the UART at 115200. Expected:
top-1 class 3 for the built-in deterministic test pattern, matching
the host LiteRT reference.
