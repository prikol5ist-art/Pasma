# Deployment notes

## Input and output

All bundled ONNX models use opset 17 and **float32 input/output**, including those with FP16 or INT8 internal weights. Input `image` is static batch-one RGB NCHW: `[1,3,320,320]` for Small, `[1,3,256,256]` for Nano/Pico.

Correct EXIF orientation, convert to RGB, scale the longest side to the model size with bilinear interpolation, then center-pad with RGB(128,128,128). Divide by 255 and transpose to NCHW. Do not apply ImageNet normalization yourself: normalization, temperature-calibrated softmax, material sigmoid and category aggregation are already inside the graph. Odd padding adds the spare pixel to the right/bottom. The shared preprocessing.json describes Small's default; read each graph shape for other variants.

| Output | Shape | Meaning |
|---|---|---|
| prob_type | [1,20] | Calibrated, mutually exclusive garment type probabilities |
| prob_coarse | [1,6] | Category probabilities summed from types |
| prob_material | [1,21] | Independent visual-material scores |

Decode with models/labels_order.json. For a consistent displayed pair derive category from the selected type via type_to_coarse. Independent category/type argmax choices can disagree. Do not apply another softmax, or interpret material scores as percentages of fibre content.

Material decision thresholds in models/material_thresholds.json are keyed by internal run: Small `r4_edgenext_320`, Nano `r3_edgenext_s`, Pico `r3_convnext_pico`. Classes with fitted=false lack supported threshold calibration; the example omits those suggestions.

## Choose a format

FP32 ONNX is the portable CPU starting point. FP16 ONNX is included for runtime-specific GPU/NPU evaluation; the Small graph is plain FP16, not the separate fp16_safe experiment. The .mlpackage.zip files contain FP16 CoreML MLPrograms with an iOS 17 target; unzip and add to an Xcode app target. The CoreML image input is an RGB float32 multi-array with the same preprocessing contract, not a CVPixelBuffer ImageType. Android and browser execution require target-device validation; no LiteRT/TFLite artifact is supplied.

Pico conv-only INT8 is **experimental**: on the recorded 249-image parity run, type Top-1 agreement was 92.77% and max probability difference 0.807988. It is not a drop-in numerical equivalent. Pico FP16 ONNX max type probability difference was 0.005546; CoreML CPU+NE differences were 0.011022–0.018128. Those configurations exceed the original 0.005 tolerance. FP32 ONNX matched the recorded reference type output exactly. Validate all outputs on your target runtime; saved parity experiments primarily cover type probabilities.

The PyTorch safetensors files contain raw network state. Build the model in training/model.py with the matching timm backbone and load via safetensors.torch.load_file. Raw network inputs require ImageNet normalization and outputs are logits. For calibrated deployment probabilities use the packaged ONNX/CoreML graphs. Export temperatures are 0.887703/0.891193/0.907134 for Small/Nano/Pico; do not substitute local evaluation fits.

## Integrity

From the repository root:

```bash
(cd models && shasum -a 256 -c checksums.sha256)
```

The checksum manifest covers the distributed CoreML ZIPs, all model files, labels and calibration metadata. Startup, image decoding, end-to-end latency, memory and thermal behavior must be measured in the consuming app.
