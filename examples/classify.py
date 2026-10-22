"""Run a bundled Pasma ONNX graph on one local image (no image upload)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path)
    parser.add_argument('--model', type=Path, default=Path(__file__).resolve().parents[1] / 'models/pasma_small_320_fp32.onnx')
    args = parser.parse_args()
    metadata = args.model.parent
    labels = json.loads((metadata / 'labels_order.json').read_text())
    session = ort.InferenceSession(str(args.model), providers=['CPUExecutionProvider'])
    size = session.get_inputs()[0].shape[-1]
    with Image.open(args.image) as source:
        image = ImageOps.exif_transpose(source).convert('RGB')
    scale = size / max(image.size)
    image = image.resize(tuple(max(1, round(v * scale)) for v in image.size), Image.Resampling.BILINEAR)
    canvas = Image.new('RGB', (size, size), (128, 128, 128))
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    x = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
    coarse, types, materials = session.run(['prob_coarse', 'prob_type', 'prob_material'], {'image': x})
    top3 = np.argsort(-types[0])[:3]
    chosen = labels['garment_types'][int(top3[0])]
    # Marginal argmax may differ; use this mapping for the displayed pair.
    category = labels['type_to_coarse'][chosen]
    prefix_to_run = {'pasma_small_320': 'r4_edgenext_320', 'pasma_nano_256': 'r3_edgenext_s',
                     'pasma_pico_256': 'r3_convnext_pico'}
    run = next(run for prefix, run in prefix_to_run.items() if args.model.stem.startswith(prefix))
    thresholds = json.loads((metadata / 'material_thresholds.json').read_text())['thresholds'][run]
    suggestions = [{'label': name, 'score': float(materials[0, i])}
                   for i, name in enumerate(labels['materials'])
                   if thresholds[name]['fitted'] and materials[0, i] >= thresholds[name]['threshold']]
    print(json.dumps({'top3': [{'type': labels['garment_types'][int(i)], 'score': float(types[0, i])}
                              for i in top3], 'category_for_selected_type': category,
                      'coarse_marginal_argmax': labels['coarse_categories'][int(coarse[0].argmax())],
                      'visual_material_suggestions': sorted(suggestions, key=lambda v: -v['score'])}, indent=2))


if __name__ == '__main__':
    main()
