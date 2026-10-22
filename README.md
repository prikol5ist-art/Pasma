<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/pasma-light.svg">
    <img src="assets/pasma.svg" width="132" height="132" alt="Pasma — a gathered skein of five threads">
  </picture>
</p>
<h1 align="center">Pasma</h1>
<p align="center"><strong>Clothing recognition that stays on your device.</strong></p>
<p align="center">20 garment types · 6 categories · 21 visual material scores</p>
<p align="center">By <a href="https://vestesarchive.dev/">Vestes Archive</a> · <a href="LICENSE">Apache-2.0</a> · Python / ONNX / CoreML</p>

Pasma is a family of compact vision models for recognizing a single clothing item in an everyday photograph. It helps wardrobe and fashion applications suggest the garment type, category and likely visual materials, without a network request or a separate background-removal model.

Built by **[Vestes Archive](https://vestesarchive.dev/)**, the digital atelier behind this project, Pasma brings a small part of our clothing intelligence to the open-source community. Its name refers to a counted bundle of threads: individual visual clues brought together into a structured result.

**This is the v1.0 candidate release.** Weights and a runnable inference example are included. Suggestions should remain user-correctable; phone-specific validation and some export-parity work are still pending.

## At a glance

- **Small enough to bundle:** the Small and Nano FP16 ONNX graphs are about **10.32 MiB** each.
- **Made for wardrobe photos:** **90.75%** type Top-1 on the 854-image `test_real` slice; **89.12%** on all 1,103 held-out real images for Small-320.
- **Useful alternatives:** Small-320 reaches **97.10% Top-3** on the combined real set, allowing an app to ask the user to choose when the first suggestion is uncertain.
- **Local inference:** after downloading the model, no API key, image upload or per-image API fee is needed.
- **Permissive release:** project code and fitted weights are distributed under Apache-2.0 with retained upstream MIT/Apache notices.

## Quick start

Python 3.11+ is recommended. No PyTorch installation is needed for ONNX inference.

```bash
git clone https://github.com/prikol5ist-art/Pasma.git
cd Pasma
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python examples/classify.py /path/to/garment.jpg
```

The example uses Small-320 FP32 on CPU and returns the top three types, a category consistent with the selected type, and thresholded visual-material suggestions. To use a smaller-resolution graph:

```bash
python examples/classify.py garment.jpg --model models/pasma_nano_256_fp32.onnx
```

Minimal inference, with already prepared RGB float32 NCHW input:

```python
import onnxruntime as ort

session = ort.InferenceSession(
    "models/pasma_small_320_fp32.onnx",
    providers=["CPUExecutionProvider"],
)
coarse, types, materials = session.run(
    ["prob_coarse", "prob_type", "prob_material"], {"image": image_tensor}
)
```

The [complete example](examples/classify.py) handles EXIF orientation, aspect-ratio-preserving bilinear letterbox, gray padding and conversion to `[0,1]`. **ImageNet normalization is inside the exported graph.** Do not apply it again. See [deployment notes](docs/DEPLOYMENT.md) for the exact contract.

## Choose your model

All three variants expose the same labels and output schema. MiB means 1,048,576 bytes. Sizes below are measured ONNX files, rather than estimates from parameter count.

| Variant | Input | Parameters | FP16 / FP32 | Combined Top-1 | Combined Top-3 |
|---|---|---:|---:|---:|---:|
| **Pasma-Small-320** | 320 × 320 | 5.296M | 10.32 / 20.37 MiB | **89.12%** | **97.10%** |
| Pasma-Nano-256 | 256 × 256 | 5.296M | 10.32 / 20.36 MiB | 87.94% | 97.01% |
| Pasma-Pico-256 | 256 × 256 | 8.557M | 16.40 / 32.69 MiB | 87.49% | 96.83% |

Start with **Small-320** for recognition quality. Nano lowers input resolution using the same backbone family. Pico uses ConvNeXt and was the fastest native CoreML candidate on the measured M2 host; actual device/runtime results may differ.

[Browse weights and metadata](models/) · [Model card](MODEL_CARD.md) · [Training source and lineage](docs/TRAINING.md)

The bundle includes FP32/FP16 ONNX, PyTorch safetensors and zipped FP16 CoreML packages. Pico also includes an **experimental** 8.71 MiB conv-only INT8 graph with significant probability drift; it is not the default. Check downloads with:

```bash
(cd models && shasum -a 256 -c checksums.sha256)
```

## How it compares

These are recorded configurations evaluated against the same Pasma type taxonomy. `test_real` contains 854 held-out source-derived real photos; `bench_test` contains 249 curated home/catalogue images. **Combined** is their concatenation, not an independent third test. Scores are type Top-1 unless stated otherwise.

| Model / configuration | test_real | bench_test | Combined | Combined macro F1 |
|---|---:|---:|---:|---:|
| **Pasma-Small-320, fine-tuned + distilled** | **90.75%** | 83.53% | **89.12%** | **87.01%** |
| Pasma-Nano-256, fine-tuned + distilled | 90.05% | 80.72% | 87.94% | 85.44% |
| Pasma-Pico-256, fine-tuned + distilled | 89.11% | 81.93% | 87.49% | 84.84% |
| SigLIP-Base-Patch16-256, zero-shot | 78.22% | **87.15%** | 80.24% | 76.95% |
| FashionCLIP ViT-B/32, zero-shot | 75.53% | 82.33% | 77.07% | 73.96% |
| MobileCLIP-S1, zero-shot | 76.93% | 80.72% | 77.79% | 75.27% |
| OpenCLIP ViT-B/32, zero-shot | 65.69% | 64.66% | 65.46% | 64.31% |
| SigLIP2-SO400M, fine-tuned teacher | 94.38% | 92.37% | 93.93% | 92.70% |

Small-320 leads the strongest tested zero-shot baseline by **8.88 percentage points on the combined set**. SigLIP-Base wins on `bench_test`, so the ranking depends on the image slice. These comparisons measure task-specific training against zero-shot inference, not an equal-training ablation. The teacher is a scale reference, not a mobile deployment option. Third-party comparison weights are not redistributed.

Small-320's combined Top-1 95% interval is **87.4–90.9%**, from 2,000 image-level bootstrap resamples. This interval does not capture correlated items, label uncertainty or training-seed variation. Headline Pasma scores are local PyTorch FP32, without TTA or the exploratory BLOUSE bias; exported precision variants have separate parity results.

### Local versus cloud

On `bench_test`, the recorded Qwen3-VL-30B-A3B endpoint scored **91.97% Top-1**, compared with Small-320's **83.53%**. Its measured p50 request time was **1,796 ms**, including network and provider processing. Small-320's native CoreML p50 was **5.85 ms on an M2 MacBook Air**, excluding image preparation and startup.

The cloud model saw 1024-pixel JPEGs; Small used a 320-pixel letterbox. This illustrates an accuracy/operations trade-off, not equivalent input or timing conditions. Pasma is useful when offline execution, local photos and avoiding recurring API calls matter.

### Measured Apple performance

| Model | CoreML CPU + Neural Engine p50 / p95 |
|---|---:|
| Small-320 | 5.85 / 6.49 ms |
| Nano-256 | 4.21 / 4.97 ms |
| Pico-256 | 2.27 / 2.96 ms |

**These are Mac M2 measurements, not iPhone or Android benchmarks.** Batch one, 5 warm-ups, 30 timed model calls; decoding, preprocessing and model loading are excluded. CPU fallback is allowed. Android/Web execution, phone latency and peak activation RAM still require validation. Some FP16/CoreML/INT8 configurations exceed the original probability-parity tolerance; see [format-specific caveats](docs/DEPLOYMENT.md).

This README contains selected, user-facing comparisons. The [Hugging Face release](https://huggingface.co/Prikol5ist/pasma) includes the full benchmark report, per-class metrics, confusion matrices, calibration, export-parity results and machine-readable aggregate evidence.

## What the outputs mean

| Output | Vocabulary | Interpretation |
|---|---|---|
| Garment type | 20 types | T-shirt, shirt, blouse, sweater, hoodie, pants, jeans, shorts, skirt, jacket, coat, vest, dress, shoes, boots, sneakers, sandals, accessory, hat, scarf |
| Category | 6 categories | Top, bottom, outerwear, footwear, full body, accessory |
| Visual material | 21 independent scores | Appearance suggestions such as denim, fleece, leather-like surfaces and other material families |

Use [labels_order.json](models/labels_order.json) as the index authority. Category probabilities are sums of type probabilities; derive the displayed category from the selected type to keep the pair consistent. Material scores are independent and do not sum to one.

**Visual appearance is not fibre composition.** Material mAP for Small is **0.664** on the curated 249-image benchmark, averaging the 17 of 21 classes with positive support. It cannot certify cotton content, fibre percentages, authenticity or care instructions.

## Known limits

- **BLOUSE versus SHIRT is difficult:** Small recalls 11 of 23 combined held-out blouses (47.8%); 21 of 23 are in Top-3. Keep alternatives and manual correction available.
- Closed-set predictions have no explicit unknown class. Multiple garments, unusual crops and unfamiliar items can produce confident mistakes.
- Type labels are source/filename-derived; material targets include curated expected sets. No complete independent human-adjudication record or demographic/device fairness study is supplied.
- Source images, private manifests and teacher checkpoints are not included. Full retraining/evaluation cannot be replicated from a bare checkout alone. The historical headline evaluation used EXIF correction off; the deployment example corrects EXIF.
- Pasma v3 predicts type/category/material appearance. It does not provide segmentation, weather, style, dating or 3D fit predictions.

## About Vestes Archive

[Vestes Archive](https://vestesarchive.dev/) is our digital atelier for clothing, personal archives and machine-assisted curation. We build tools that help people understand and care for what they already own. Pasma is the open-source clothing-perception model within that work.

Website: **[vestesarchive.dev](https://vestesarchive.dev/)** · Contact: **[team@vestesarchive.dev](mailto:team@vestesarchive.dev)**

## License and contributions

Project code and fitted weights are released under the **[Apache License 2.0](LICENSE)**, retaining upstream notices. Small/Nano derive from MIT-licensed EdgeNeXt weights; Pico derives from Apache-2.0 ConvNeXt weights. See [third-party notices](THIRD_PARTY_NOTICES.md) and [lineage](docs/TRAINING.md). The project license does not grant rights to source datasets or third-party baseline weights.

For bug reports, include the model filename/hash, runtime, device and preprocessing steps. Share non-sensitive reproductions, and use validation data for tuning rather than repeatedly selecting changes against the published test set. Open an [issue](https://github.com/prikol5ist-art/Pasma/issues) or submit a pull request.
