# Training and lineage

The code in training/train_v3_modal.py is the original Modal v3 research pipeline; training/model.py is the matching raw PyTorch architecture. This public release is usable for inference without training infrastructure or cloud credentials.

The three supplied students train on real photos with online type distillation from fine-tuned SigLIP2-SO400M. Shared settings: seed 42, batch 64, 40 epochs, head learning rate 5e-4, backbone multiplier 0.5, AdamW weight decay 0.05, one warm-up epoch followed by cosine decay, EMA 0.999, label smoothing 0.1, distillation alpha 0.9 at T=2, auxiliary coarse/material weights 0.3 each. The checkpoint maximizes validation Top-1 plus macro F1. Unknown/low-confidence material targets are masked.

Type/coarse/material heads share a backbone. Deployment derives coarse probabilities from calibrated types and the hierarchy matrix; the auxiliary coarse training head is not used as an independent deployment decision. There are no weather/style heads in this v3 release.

| Variant | Internal run | Backbone | Best epoch |
|---|---|---|---:|
| Small-320 | r4_edgenext_320 | edgenext_small.usi_in1k | 27 |
| Nano-256 | r3_edgenext_s | edgenext_small.usi_in1k | 30 |
| Pico-256 | r3_convnext_pico | convnext_pico.d1_in1k | 39 |

The original script expects a private manifest at training/data/v3/manifest.csv and pre-provisioned Modal data/run volumes. Images, manifests, teacher checkpoints and benchmark labels are not distributed, so a bare checkout cannot reproduce the training run. Preparing your own data and teacher volumes is required; running the Modal pipeline incurs cloud usage. Use fixed splits, preserve physical-item groups and letterbox aspect ratios. Do not call the split-building function to regenerate the published test split.

Small/Nano pretrained weights: timm/edgenext_small.usi_in1k, MIT; recorded revision 81bd8f2ee56330bdf6a7a2e47e1b2b82eb34f978. Pico: timm/convnext_pico.d1_in1k, Apache-2.0, checkpoint convnext_pico_d1-10ad7f0d.pth. Teacher: timm/vit_so400m_patch16_siglip_256.v2_webli, Apache-2.0; teacher weights are not included. Original upstream hashes/revisions and data-rights provenance are not independently re-audited by this packaging step.

Dependencies for raw architecture loading: torch, timm and safetensors. The research script declares its remote Modal environment and also requires the Modal CLI locally. No segmentation network or Ultralytics dependency is needed for the published inference example.

Code and fitted-weight release: Apache-2.0 with retained upstream notices. Upstream model cards: [EdgeNeXt](https://huggingface.co/timm/edgenext_small.usi_in1k), [ConvNeXt-Pico](https://huggingface.co/timm/convnext_pico.d1_in1k), [SigLIP2](https://huggingface.co/timm/vit_so400m_patch16_siglip_256.v2_webli). Source dataset redistribution rights are separate; no license to those images is granted here.
