# Pasma v1.0 candidate — model card

**Developer:** [Vestes Archive](https://vestesarchive.dev/). **Release date:** 2026-10-22. **License:** Apache-2.0, retaining upstream MIT/Apache notices. **Task:** single-item garment classification and multi-label visual-material suggestions.

The [README](README.md) contains the selected model comparison and quick start. Full benchmark reports are reserved for Hugging Face; no unverified Hub URL is supplied.

## Architecture and intended use

Small/Nano use `edgenext_small.usi_in1k`, at 320/256 pixels; Pico uses `convnext_pico.d1_in1k`, at 256 pixels. Training has type/coarse/material heads. Exports derive category probabilities from the 20-type distribution and a fixed hierarchy matrix, alongside 21 material sigmoids. There are no weather/style or segmentation heads.

Use for user-correctable wardrobe entry, catalogue assistance and local applications. A model score should not determine chemical material composition, authenticity, care safety or dating. There is no explicit out-of-distribution/unknown class.

## Training and evaluation

See [training notes](docs/TRAINING.md) for the original source, fine-tuning recipe and upstream lineage. Students use real-photo training and online type distillation from a fine-tuned SigLIP2-SO400M teacher. Source photos, private manifests and the teacher are not redistributed.

| Variant | Combined Top-1 [95% CI] | Top-3 | Type macro F1 | Material mAP on bench_test |
|---|---|---:|---:|---:|
| Small-320 | 89.12% [87.4,90.9] | 97.10% | 87.01% | 0.664 |
| Nano-256 | 87.94% [86.0,89.8] | 97.01% | 85.44% | 0.654 |
| Pico-256 | 87.49% [85.5,89.4] | 96.83% | 84.84% | 0.638 |

Combined = test_real (854) + bench_test (249). Type intervals use 2,000 image resamples; they do not capture correlated items or label uncertainty. Material mAP averages supported classes (17/21 on bench_test); thresholds are selected on bench_dev, not bench_test. Source type labels and expected material sets are not claimed to form a fully independently adjudicated human gold benchmark. No demographic/geographic/device fairness audit is supplied.

Headline evaluation uses local FP32 without TTA or the exploratory BLOUSE bias and with EXIF correction disabled to match training. Export calibration fits differ slightly; deployment corrects EXIF. Precision-specific parity and device performance must be evaluated separately.

## Practical limitations

Small's BLOUSE recall is 47.8% on 23 held-out blouses, with 91.3% Top-3 recall. Offer alternatives and permit correction. Material outputs concern visual appearance, not fibre identity or percentages. Multiple garments and taxonomy ambiguities can cause errors; confidence is not a correctness guarantee.

Native CoreML timing is from an M2 MacBook Air, not a phone. Android, browser, actual iPhone performance and peak activation RAM remain unvalidated. CoreML and some FP16/INT8 configurations exceed the original max probability error requirement. Pico INT8 is experimental with 92.77% type agreement in the saved parity test. See [deployment notes](docs/DEPLOYMENT.md).

Local inference does not upload photos. Consuming applications may add their own telemetry or uploads. No dataset redistribution rights or third-party baseline-weight rights are granted by this release. Maintain source-data rights/consent records for any further training or dataset publication.
