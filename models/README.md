# SynapticOS Model Store

Place `.tflite` model files here. They are excluded from git
(see `.gitignore`) and should be managed separately.

## Required Models

| Model | File | Purpose | Phase |
|-------|------|---------|-------|
| Face Detection | `face_detect_v1.tflite` | Vision demo with camera + LCD | Phase 2 |
| Keyword Spotting | `keyword_spot_v1.tflite` | Audio classification demo | Phase 2 |
| Anomaly Detection | `anomaly_det_v1.tflite` | Vibration anomaly (industrial) | Phase 4 |

## Obtaining Models

Models can be obtained from:
- NXP eIQ Model Zoo (optimized for Neutron NPU)
- TensorFlow Lite Model Zoo (requires Neutron Conversion Tool)
- Custom training via eIQ Portal

## Converting for Neutron NPU

```bash
# Use NXP's Neutron Conversion Tool (part of eIQ Toolkit)
neutron_converter --input model.tflite --output model_neutron.tflite
```
