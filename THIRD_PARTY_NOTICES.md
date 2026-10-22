# Third-party notices

Pasma project code is distributed under the Apache License, Version 2.0; see [LICENSE](LICENSE). The intended fitted-weight release retains the notices of its upstream pretrained sources. See [training and lineage notes](docs/TRAINING.md) for exact model identifiers and outstanding revision evidence.

## EdgeNeXt

Source: [mmaaz60/EdgeNeXt](https://github.com/mmaaz60/EdgeNeXt), [license text](https://github.com/mmaaz60/EdgeNeXt/blob/main/LICENSE), inspected 2026-10-21.

```text
MIT License

Copyright (c) 2022 Muhammad Maaz
Copyright (c) Meta Platforms, Inc. and affiliates.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## timm / ConvNeXt and SigLIP2 weights

Architecture implementation: [huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models), Apache-2.0. Pico's `convnext_pico.d1_in1k` and the teacher's `vit_so400m_patch16_siglip_256.v2_webli` model cards declare Apache-2.0. A copy of the Apache-2.0 license text is included at the repository root.

These acknowledgements do not replace any additional copyright/NOTICE files in the exact upstream revisions. Preserve those when assembling the public weight archive. The teacher and third-party comparison weights are not distributed in `models/`.
