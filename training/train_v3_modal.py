"""Fashion Vision v3 - real-photo training pipeline on Modal.

Why v3: the v2 model reached 93.8% type top-1 on its own test split but only 61%
on real wardrobe photos (fresh_wardrobe_500). The v1/v2 training images are almost
all synthetic, uniformly staged renders, and ~7k real wardrobe photos were never
used. v3 trains on the real pool plus the synthetic set, removes near-duplicates
of the benchmark, distils from a large teacher and bakes the type->coarse
hierarchy into the exported graph.

Usage (all compute runs on Modal):
    modal run train_v3_modal.py::upload_hint            # prints the volume put command
    modal run train_v3_modal.py::build_splits
    modal run train_v3_modal.py::sweep --name <sweep>   # see SWEEPS below
    modal run train_v3_modal.py::final_eval --runs a,b
    modal run train_v3_modal.py::export --run <run>
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any

import modal

APP_NAME = "fashion-vision-v3"
LOCAL_ROOT = Path(__file__).resolve().parent

data_volume = modal.Volume.from_name("fv3-data", create_if_missing=True)
runs_volume = modal.Volume.from_name("fv3-runs", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "timm==1.0.15",
        "huggingface_hub>=0.25,<0.36",
        "safetensors>=0.4",
        "numpy<2.2",
        "pillow>=10.4",
        "scikit-learn>=1.5",
        "onnx==1.17.0",
        "onnxruntime==1.20.1",
        "onnxconverter-common==1.14.0",
        "onnxscript>=0.1",
    )
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "HF_HOME": "/data/hf"})
    .add_local_file(str(LOCAL_ROOT / "data" / "v3" / "manifest.csv"), "/root/manifest.csv")
)

app = modal.App(APP_NAME)

TYPES = [
    "T_SHIRT", "SHIRT", "BLOUSE", "SWEATER", "HOODIE", "PANTS", "JEANS", "SHORTS", "SKIRT",
    "JACKET", "COAT", "VEST", "DRESS", "SHOES", "BOOTS", "SNEAKERS", "SANDALS", "ACCESSORY",
    "HAT", "SCARF",
]
COARSE = ["TOP", "BOTTOM", "OUTERWEAR", "FOOTWEAR", "FULL_BODY", "ACCESSORY"]
TYPE_TO_COARSE = {
    "T_SHIRT": "TOP", "SHIRT": "TOP", "BLOUSE": "TOP", "SWEATER": "TOP", "HOODIE": "TOP",
    "PANTS": "BOTTOM", "JEANS": "BOTTOM", "SHORTS": "BOTTOM", "SKIRT": "BOTTOM",
    "JACKET": "OUTERWEAR", "COAT": "OUTERWEAR", "VEST": "OUTERWEAR",
    "DRESS": "FULL_BODY",
    "SHOES": "FOOTWEAR", "BOOTS": "FOOTWEAR", "SNEAKERS": "FOOTWEAR", "SANDALS": "FOOTWEAR",
    "ACCESSORY": "ACCESSORY", "HAT": "ACCESSORY", "SCARF": "ACCESSORY",
}
MATERIALS = [
    "cotton", "linen", "wool", "cashmere", "silk", "denim", "corduroy", "tweed", "polyester",
    "nylon", "spandex_elastane", "acrylic", "rayon_viscose", "fleece", "technical_synthetic",
    "leather_smooth", "leather_suede", "leather_patent", "rubber", "metal_hardware",
    "blend_unspecified",
]
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
PAD = (128, 128, 128)
SPLITS_PATH = "/data/splits_v3.csv"
LOCAL_IMAGES = "/tmp/images"

# Permissive-licence backbones only (AGENTS.md). Teacher weights are never shipped.
BACKBONES = {
    "mnv4_hm": "mobilenetv4_hybrid_medium.e200_r256_in12k_ft_in1k",  # Apache-2.0, 11.1M
    "mnv4_cm": "mobilenetv4_conv_medium.e500_r256_in1k",  # Apache-2.0, 9.7M
    "convnext_pico": "convnext_pico.d1_in1k",  # Apache-2.0, 9.0M
    "edgenext_s": "edgenext_small.usi_in1k",  # MIT, 5.6M
    "effformerv2_s2": "efficientformerv2_s2.snap_dist_in1k",  # Apache-2.0, 12.7M
    "siglip2_b": "vit_base_patch16_siglip_256.v2_webli",  # Apache-2.0 teacher
    "siglip2_so400m": "vit_so400m_patch16_siglip_256.v2_webli",  # Apache-2.0 teacher
    "dinov2_b": "vit_base_patch14_reg4_dinov2.lvd142m",  # Apache-2.0 teacher
}


# ---------------------------------------------------------------------------
# Shared helpers (run inside containers)
# ---------------------------------------------------------------------------


def _ensure_images() -> Path:
    """Extract the image tar from the volume onto fast local disk once per container."""
    import tarfile

    dst = Path(LOCAL_IMAGES)
    marker = dst / ".ready"
    if marker.exists() and (dst / "real" / "athletic_shoes_001.jpg").exists() and (dst / "synthetic" / "BOTTOM" / "SHORTS" / "SHORTS_042.jpg").exists():
        return dst
    dst.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with tarfile.open("/data/fv3_images.tar", ignore_zeros=True) as tar:
        tar.extractall(dst, filter="data")
    patch = Path("/data/patch.tar")
    if patch.exists():
        with tarfile.open(patch) as tar:
            tar.extractall(dst, filter="data")
    marker.touch()
    print(f"Extracted images in {time.time() - t0:.1f}s")
    return dst


def _read_csv(path: str) -> list[dict[str, str]]:
    import csv

    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _letterbox(img, size: int):
    from PIL import Image

    w, h = img.size
    scale = min(size / w, size / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = img.resize((nw, nh), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size, size), PAD)
    canvas.paste(resized, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def _material_target(row: dict[str, str]):
    import torch

    vec = torch.zeros(len(MATERIALS))
    known = row.get("primary_material", "unknown") in MATERIALS and row.get("material_confidence") != "low"
    if known:
        for key in ("primary_material", "secondary_material"):
            if row.get(key) in MATERIALS:
                vec[MATERIALS.index(row[key])] = 1.0
    return vec, torch.tensor(1.0 if known else 0.0)


class WardrobeDataset:
    def __init__(self, rows, root: Path, size: int, train: bool, aug: str = "strong"):
        self.rows = rows
        self.root = root
        self.size = size
        self.train = train
        self.aug = aug

    def __len__(self) -> int:
        return len(self.rows)

    def _augment(self, img):
        import random

        from PIL import Image, ImageFilter
        from torchvision.transforms import ColorJitter

        w, h = img.size
        if random.random() < 0.6:
            s = random.uniform(0.55 if self.aug == "strong" else 0.75, 1.0)
            r = math.exp(random.uniform(math.log(0.8), math.log(1.25)))
            cw = min(w, max(8, int(w * math.sqrt(s * r))))
            ch = min(h, max(8, int(h * math.sqrt(s / r))))
            x0 = random.randint(0, w - cw)
            y0 = random.randint(0, h - ch)
            img = img.crop((x0, y0, x0 + cw, y0 + ch))
        if random.random() < 0.5:
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if random.random() < 0.3:
            img = img.rotate(random.uniform(-20, 20), resample=Image.Resampling.BILINEAR, expand=True, fillcolor=PAD)
        if random.random() < 0.8:
            img = ColorJitter(0.35, 0.35, 0.3, 0.03)(img)
        if random.random() < 0.05:
            img = img.convert("L").convert("RGB")
        if random.random() < 0.15:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.5)))
        if self.aug == "strong" and random.random() < 0.2:
            # Simulate a low-quality phone capture: downscale then upscale.
            f = random.uniform(0.3, 0.6)
            small = img.resize((max(8, int(img.width * f)), max(8, int(img.height * f))), Image.Resampling.BILINEAR)
            img = small.resize(img.size, Image.Resampling.BILINEAR)
        return img

    def __getitem__(self, i: int):
        import torch
        from PIL import Image
        from torchvision.transforms import functional as TF

        row = self.rows[i]
        with Image.open(self.root / row["file"]) as im:
            img = im.convert("RGB")
        if self.train:
            img = self._augment(img)
        img = _letterbox(img, self.size)
        x = TF.normalize(TF.to_tensor(img), MEAN, STD)
        if self.train and self.aug == "strong" and torch.rand(1).item() < 0.25:
            # Gray-patch erasing (background/occlusion invariance, ML plan 4.2).
            hh, ww = x.shape[1:]
            eh, ew = int(hh * torch.empty(1).uniform_(0.1, 0.35)), int(ww * torch.empty(1).uniform_(0.1, 0.35))
            y0, x0 = torch.randint(0, hh - eh, (1,)).item(), torch.randint(0, ww - ew, (1,)).item()
            x[:, y0:y0 + eh, x0:x0 + ew] = 0.0  # 0 after normalisation ~= mean colour
        mat, mat_mask = _material_target(row)
        return {
            "x": x,
            "type": TYPES.index(row["garment_type"]),
            "coarse": COARSE.index(row["coarse_category"]),
            "mat": mat,
            "mat_mask": mat_mask,
            "real": 1 if row["domain"] == "real" else 0,
            "idx": i,
        }


def _build_model(backbone: str, pretrained: bool = True, drop: float = 0.2, drop_path: float = 0.1):
    import timm
    import torch
    import torch.nn as nn

    class WardrobeNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            kwargs: dict[str, Any] = {"pretrained": pretrained, "num_classes": 0}
            if drop_path > 0:
                kwargs["drop_path_rate"] = drop_path
            self.backbone = timm.create_model(backbone, **kwargs)
            nf = getattr(self.backbone, "head_hidden_size", None) or self.backbone.num_features
            self.drop = nn.Dropout(drop)
            self.type_head = nn.Linear(nf, len(TYPES))
            self.coarse_head = nn.Linear(nf, len(COARSE))
            self.material_head = nn.Linear(nf, len(MATERIALS))

        def forward(self, x: torch.Tensor, return_feats: bool = False):
            feats = self.backbone(x)
            f = self.drop(feats)
            out = (self.type_head(f), self.coarse_head(f), self.material_head(f))
            return (*out, feats) if return_feats else out

    return WardrobeNet()


def _hierarchy_matrix():
    import torch

    m = torch.zeros(len(TYPES), len(COARSE))
    for t, c in TYPE_TO_COARSE.items():
        m[TYPES.index(t), COARSE.index(c)] = 1.0
    return m


def _metrics(type_logits, type_true, coarse_logits=None, temp: float = 1.0) -> dict[str, Any]:
    import numpy as np
    from sklearn.metrics import f1_score

    logits = np.asarray(type_logits, dtype=np.float64) / temp
    p = np.exp(logits - logits.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    y = np.asarray(type_true)
    pred = p.argmax(1)
    top3 = np.argsort(-p, 1)[:, :3]
    h = _hierarchy_matrix().numpy()
    coarse_true = h[y].argmax(1)
    coarse_from_type = (p @ h).argmax(1)
    conf = p.max(1)
    # 15-bin expected calibration error
    bins = np.linspace(0, 1, 16)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = (conf > lo) & (conf <= hi)
        if sel.any():
            ece += sel.mean() * abs((pred[sel] == y[sel]).mean() - conf[sel].mean())
    out = {
        "n": int(len(y)),
        "type_top1": float((pred == y).mean()),
        "type_top3": float((top3 == y[:, None]).any(1).mean()),
        "type_macro_f1": float(f1_score(y, pred, average="macro", labels=sorted(set(y.tolist())), zero_division=0)),
        "coarse_acc": float((coarse_from_type == coarse_true).mean()),
        "coarse_macro_f1": float(f1_score(coarse_true, coarse_from_type, average="macro", zero_division=0)),
        "ece": float(ece),
        "per_class_recall": {
            TYPES[c]: round(float((pred[y == c] == c).mean()), 4) for c in sorted(set(y.tolist()))
        },
    }
    # Selective prediction: accuracy when only answering with confidence >= 0.5 / 0.7.
    for thr in (0.5, 0.7):
        sel = conf >= thr
        out[f"coverage@{thr}"] = float(sel.mean())
        out[f"acc@{thr}"] = float((pred[sel] == y[sel]).mean()) if sel.any() else 0.0
    if coarse_logits is not None:
        out["coarse_head_acc"] = float((np.asarray(coarse_logits).argmax(1) == coarse_true).mean())
    return out


def _predict(model, rows, root: Path, size: int, device, tta: bool = False, bs: int = 128):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    ds = WardrobeDataset(rows, root, size, train=False)
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=8)
    tl, cl, ml = [], [], []
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for b in dl:
            x = b["x"].to(device, non_blocking=True)
            t, c, m = model(x)
            if tta:
                t2, c2, m2 = model(torch.flip(x, dims=[3]))
                t, c, m = (t + t2) / 2, (c + c2) / 2, (m + m2) / 2
            tl.append(t.float().cpu())
            cl.append(c.float().cpu())
            ml.append(m.float().cpu())
    return torch.cat(tl).numpy(), torch.cat(cl).numpy(), torch.cat(ml).numpy()


def _material_hit3(mat_logits, rows) -> float | None:
    """Top-3 material hit rate against the benchmark's expected material sets."""
    import numpy as np

    expected = [r.get("expected_materials", "") for r in rows]
    if not any(expected):
        return None
    top3 = np.argsort(-np.asarray(mat_logits), 1)[:, :3]
    hits = [
        any(MATERIALS[j] in set(e.split(",")) for j in t3) for t3, e in zip(top3, expected) if e
    ]
    return float(np.mean(hits))


def _load_split_rows() -> dict[str, list[dict[str, str]]]:
    import csv

    rows = _read_csv(SPLITS_PATH)
    bench_mats: dict[str, str] = {}
    bench_csv = Path("/data/bench_annotations.csv")
    if bench_csv.exists():
        with open(bench_csv, newline="") as f:
            for r in csv.DictReader(f):
                bench_mats[Path(r["file"]).name] = r["expected_materials"]
    out: dict[str, list[dict[str, str]]] = {}
    for r in rows:
        if r["split"].startswith("bench"):
            r["expected_materials"] = bench_mats.get(Path(r["file"]).name, "")
        out.setdefault(r["split"], []).append(r)
    return out


# ---------------------------------------------------------------------------
# Step 1: splits with near-duplicate removal against the benchmark
# ---------------------------------------------------------------------------


@app.function(image=image, gpu="L4", cpu=8, memory=32768, timeout=3600, volumes={"/data": data_volume})
def build_splits(seed: int = 42, dup_cos: float = 0.92, group_cos: float = 0.95) -> dict[str, Any]:
    """Assign every manifest row to train/val/test/bench_dev/bench_test/excluded.

    Real photos: any image whose DINOv2 embedding has cosine >= dup_cos (or dHash
    distance <= 6) to a benchmark image is excluded. Remaining real photos are
    grouped by near-duplicate clusters (cosine >= group_cos) and split 80/10/10
    per source category with whole groups kept together. Synthetic rows keep their
    v2 split assignment.
    """
    import random

    import numpy as np
    import timm
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    root = _ensure_images()
    rows = _read_csv("/root/manifest.csv")
    v2_split: dict[str, str] = {}
    for split in ("train", "val", "test"):
        p = Path(f"/data/v2_splits/{split}.csv")
        for r in _read_csv(str(p)):
            v2_split[r["file"]] = split

    # dHash for all rows
    def dhash(path: Path) -> int:
        with Image.open(path) as im:
            g = np.asarray(im.convert("L").resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int16)
        bits = (g[:, 1:] > g[:, :-1]).flatten()
        return int(sum(1 << i for i, b in enumerate(bits) if b))

    hashes = np.array([dhash(root / r["file"]) for r in rows], dtype=np.uint64)

    # DINOv2 embeddings (Apache-2.0) for semantic near-duplicate detection
    device = torch.device("cuda")
    enc = timm.create_model("vit_small_patch14_reg4_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=224)
    enc.eval().to(device)

    class Plain(Dataset):
        def __len__(self):
            return len(rows)

        def __getitem__(self, i):
            from torchvision.transforms import functional as TF

            with Image.open(root / rows[i]["file"]) as im:
                img = _letterbox(im.convert("RGB"), 224)
            return TF.normalize(TF.to_tensor(img), MEAN, STD)

    embs = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for x in DataLoader(Plain(), batch_size=256, num_workers=8):
            e = enc(x.to(device)).float()
            embs.append(torch.nn.functional.normalize(e, dim=1).cpu())
    emb = torch.cat(embs)

    bench_idx = [i for i, r in enumerate(rows) if r["bench_split"]]
    other_idx = [i for i, r in enumerate(rows) if not r["bench_split"]]
    sims = emb[other_idx] @ emb[bench_idx].T
    max_sim = sims.max(1).values.numpy()

    def popcount(a: np.ndarray) -> np.ndarray:
        a = a.copy()
        c = np.zeros(a.shape, dtype=np.int64)
        while a.any():
            c += (a & np.uint64(1)).astype(np.int64)
            a >>= np.uint64(1)
        return c

    bh = hashes[bench_idx]
    min_ham = np.array([popcount(hashes[i] ^ bh).min() for i in other_idx])
    dup_of_bench = {other_idx[k] for k in range(len(other_idx)) if max_sim[k] >= dup_cos or min_ham[k] <= 6}

    # Union-find grouping of near-duplicate real photos (not benchmark, not dup-of-bench)
    real_idx = [i for i in other_idx if rows[i]["domain"] == "real" and i not in dup_of_bench]
    parent = {i: i for i in real_idx}

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    re_ = emb[real_idx]
    s = re_ @ re_.T
    s.fill_diagonal_(0)
    pairs = (s >= group_cos).nonzero().tolist()
    for a, b in pairs:
        ra, rb = find(real_idx[a]), find(real_idx[b])
        if ra != rb:
            parent[ra] = rb
    groups: dict[int, list[int]] = {}
    for i in real_idx:
        groups.setdefault(find(i), []).append(i)

    # Group-aware split per source category (category of the group's first member)
    rng = random.Random(seed)
    by_cat: dict[str, list[list[int]]] = {}
    for members in groups.values():
        by_cat.setdefault(rows[members[0]]["source_category"], []).append(members)
    assign: dict[int, str] = {}
    for cat, glist in sorted(by_cat.items()):
        rng.shuffle(glist)
        total = sum(len(g) for g in glist)
        acc = 0
        for g in glist:
            frac = acc / total
            split = "val" if frac < 0.10 else ("test" if frac < 0.20 else "train")
            for i in g:
                assign[i] = split
            acc += len(g)

    # Synthetic: v2 assignment; also drop synthetic near-dups of the benchmark
    for i, r in enumerate(rows):
        if r["bench_split"]:
            assign[i] = "bench_dev" if r["bench_split"] == "dev" else "bench_test"
        elif i in dup_of_bench:
            assign[i] = "excluded_bench_dup"
        elif r["domain"] == "synthetic":
            assign[i] = v2_split.get(r["file"].removeprefix("synthetic/"), "train")

    import csv

    fields = list(rows[0].keys()) + ["split", "group"]
    with open(SPLITS_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, r in enumerate(rows):
            w.writerow({**r, "split": assign[i], "group": find(i) if i in parent else i})
    data_volume.commit()

    from collections import Counter

    stats = {
        "splits": dict(Counter((rows[i]["domain"], assign[i]) for i in range(len(rows))).items()),
        "excluded_bench_dups": len(dup_of_bench),
        "real_groups": len(groups),
        "real_groups_multi": sum(1 for g in groups.values() if len(g) > 1),
    }
    stats["splits"] = {f"{k[0]}:{k[1]}": v for k, v in stats["splits"].items()}
    print(json.dumps(stats, indent=2))
    return stats


# ---------------------------------------------------------------------------
# Step 2: training
# ---------------------------------------------------------------------------

DEFAULT_CFG: dict[str, Any] = {
    "backbone": "mnv4_hm",
    "img_size": 256,
    "epochs": 30,
    "batch_size": 64,
    "lr": 1e-3,  # head lr; backbone gets lr * backbone_lr_mult
    "backbone_lr_mult": 0.3,
    "weight_decay": 0.05,
    "warmup_epochs": 1,
    "label_smoothing": 0.1,
    "drop": 0.2,
    "drop_path": 0.1,
    "aug": "strong",
    "domains": "real+synthetic",  # or "synthetic", "real"
    "real_frac": 0.65,  # share of real photos in each epoch's sampling
    "epoch_size": 12000,
    "coarse_w": 0.3,
    "material_w": 0.3,
    "class_weighting": True,
    "ema": 0.999,
    "teacher": "",  # run name of a trained teacher for online KD
    "kd_alpha": 0.7,
    "kd_T": 2.0,
    "mixup": 0.0,
    "feat_kd": 0.0,  # weight of cosine feature distillation from the teacher embedding
    "train_on": "train",  # "train+val" for final fits
    "seed": 42,
}


def _train_impl(cfg: dict[str, Any], run_name: str) -> dict[str, Any]:
    import copy
    import random

    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, WeightedRandomSampler

    cfg = {**DEFAULT_CFG, **cfg}
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    root = _ensure_images()
    splits = _load_split_rows()
    domains = set(cfg["domains"].split("+"))
    train_splits = cfg["train_on"].split("+")
    train_rows = [r for s in train_splits for r in splits.get(s, []) if r["domain"] in domains]
    oof_rows: list[dict[str, str]] = []
    if cfg.get("oof_fold") is not None:
        # k-fold over real train groups: hold one fold out, predict it at the end (label audit).
        k, fold = cfg["oof_k"], cfg["oof_fold"]
        held = lambda r: r["domain"] == "real" and r["split"] == "train" and int(r["group"]) % k == fold  # noqa: E731
        oof_rows = [r for r in train_rows if held(r)]
        train_rows = [r for r in train_rows if not held(r)]
    val_real = [r for r in splits["val"] if r["domain"] == "real"]
    val_syn = [r for r in splits["val"] if r["domain"] == "synthetic"]
    bench_dev = splits["bench_dev"]
    print(f"[{run_name}] train={len(train_rows)} val_real={len(val_real)} val_syn={len(val_syn)} bench_dev={len(bench_dev)}")

    # Sampling weights: balance domains, then classes (sqrt) within domain
    types = np.array([TYPES.index(r["garment_type"]) for r in train_rows])
    is_real = np.array([r["domain"] == "real" for r in train_rows])
    w = np.ones(len(train_rows))
    if len(domains) > 1:
        w[is_real] = cfg["real_frac"] / is_real.sum()
        w[~is_real] = (1 - cfg["real_frac"]) / (~is_real).sum()
    counts = np.bincount(types, minlength=len(TYPES)).astype(float)
    sampler = WeightedRandomSampler(torch.tensor(w), num_samples=cfg["epoch_size"], replacement=True)
    ds = WardrobeDataset(train_rows, root, cfg["img_size"], train=True, aug=cfg["aug"])
    dl = DataLoader(ds, batch_size=cfg["batch_size"], sampler=sampler, num_workers=14, pin_memory=True,
                    drop_last=True, persistent_workers=True, prefetch_factor=4)

    class_w = None
    if cfg["class_weighting"]:
        cw = 1.0 / np.sqrt(np.maximum(counts, 1))
        class_w = torch.tensor(cw / cw.mean(), dtype=torch.float32, device=device)

    model = _build_model(BACKBONES.get(cfg["backbone"], cfg["backbone"]), True, cfg["drop"], cfg["drop_path"]).to(device)
    model = model.to(memory_format=torch.channels_last)
    ema = copy.deepcopy(model).eval() if cfg["ema"] else None
    if ema is not None:
        for p in ema.parameters():
            p.requires_grad_(False)

    teacher = None
    if cfg["teacher"]:
        members = []
        for tname in cfg["teacher"].split("+"):
            tck = torch.load(f"/runs/{tname}/best.pt", map_location="cpu", weights_only=False)
            tm = _build_model(BACKBONES.get(tck["cfg"]["backbone"], tck["cfg"]["backbone"]), False, 0.0, 0.0)
            tm.load_state_dict(tck["model"])
            members.append(tm.eval().to(device))
            teacher_size = tck["cfg"]["img_size"]
            print(f"[{run_name}] KD teacher {tname} ({tck['cfg']['backbone']} @ {teacher_size})")

        class TeacherEnsemble(torch.nn.Module):
            def __init__(self, ms):
                super().__init__()
                self.ms = torch.nn.ModuleList(ms)

            def forward(self, x, return_feats=False):
                outs = [m(x, return_feats=True) for m in self.ms]
                # average log-probabilities; features from the first member
                t = torch.stack([torch.log_softmax(o[0].float(), 1) for o in outs]).mean(0)
                res = (t, outs[0][1], outs[0][2])
                return (*res, outs[0][3]) if return_feats else res

        teacher = TeacherEnsemble(members).eval()

    proj = None
    if teacher is not None and cfg["feat_kd"] > 0:
        with torch.no_grad():
            d_s = model(torch.zeros(1, 3, cfg["img_size"], cfg["img_size"], device=device), return_feats=True)[3].shape[1]
            d_t = teacher(torch.zeros(1, 3, teacher_size, teacher_size, device=device), return_feats=True)[3].shape[1]
        proj = torch.nn.Linear(d_s, d_t).to(device)  # training-only projector, never exported
    bb_params = list(model.backbone.parameters())
    bb_ids = {id(p) for p in bb_params}
    head_params = [p for p in model.parameters() if id(p) not in bb_ids] + (list(proj.parameters()) if proj else [])
    opt = torch.optim.AdamW(
        [
            {"params": bb_params, "lr": cfg["lr"] * cfg["backbone_lr_mult"]},
            {"params": head_params, "lr": cfg["lr"]},
        ],
        weight_decay=cfg["weight_decay"],
    )
    steps_per_epoch = len(dl)
    total = steps_per_epoch * cfg["epochs"]
    warm = steps_per_epoch * cfg["warmup_epochs"]
    base_lrs = [g["lr"] for g in opt.param_groups]

    def set_lr(step: int) -> None:
        f = step / max(warm, 1) if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(total - warm, 1)))
        for g, b in zip(opt.param_groups, base_lrs):
            g["lr"] = b * max(f, 0.01 if step >= warm else f)

    run_dir = Path("/runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best = -1.0
    step = 0
    t_start = time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        tot, n = 0.0, 0
        for b in dl:
            set_lr(step)
            x = b["x"].to(device, non_blocking=True).to(memory_format=torch.channels_last)
            yt = b["type"].to(device)
            yc = b["coarse"].to(device)
            ym = b["mat"].to(device)
            mm = b["mat_mask"].to(device)
            lam = 1.0
            if cfg["mixup"] > 0 and random.random() < 0.5:
                lam = float(np.random.beta(cfg["mixup"], cfg["mixup"]))
                perm = torch.randperm(x.size(0), device=device)
                x = lam * x + (1 - lam) * x[perm]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lt, lc, lm, fs = model(x, return_feats=True)
                if lam < 1.0:
                    l_type = lam * F.cross_entropy(lt.float(), yt, weight=class_w, label_smoothing=cfg["label_smoothing"]) + (
                        1 - lam) * F.cross_entropy(lt.float(), yt[perm], weight=class_w, label_smoothing=cfg["label_smoothing"])
                else:
                    l_type = F.cross_entropy(lt.float(), yt, weight=class_w, label_smoothing=cfg["label_smoothing"])
                l_coarse = F.cross_entropy(lc.float(), yc, label_smoothing=cfg["label_smoothing"])
                bce = F.binary_cross_entropy_with_logits(lm.float(), ym, reduction="none").mean(1)
                l_mat = (bce * mm).sum() / mm.sum().clamp(min=1.0)
                if teacher is not None:
                    with torch.no_grad():
                        xt = x if teacher_size == cfg["img_size"] else F.interpolate(x, size=teacher_size, mode="bilinear", align_corners=False)
                        tt, _, _, ft = teacher(xt, return_feats=True)
                    T = cfg["kd_T"]
                    kd = F.kl_div(F.log_softmax(lt.float() / T, 1), F.softmax(tt.float() / T, 1), reduction="batchmean") * T * T
                    l_type = (1 - cfg["kd_alpha"]) * l_type + cfg["kd_alpha"] * kd
                    if proj is not None:
                        l_type = l_type + cfg["feat_kd"] * (1 - F.cosine_similarity(proj(fs.float()), ft.float(), dim=1)).mean()
                loss = l_type + cfg["coarse_w"] * l_coarse + cfg["material_w"] * l_mat
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            if ema is not None:
                d = min(cfg["ema"], (1 + step) / (10 + step))
                with torch.no_grad():
                    for pe, pm in zip(ema.state_dict().values(), model.state_dict().values()):
                        if pe.dtype.is_floating_point:
                            pe.mul_(d).add_(pm.detach(), alpha=1 - d)
                        else:
                            pe.copy_(pm)
            tot += loss.item()
            n += 1
            step += 1

        eval_model = ema if ema is not None else model
        rec: dict[str, Any] = {"epoch": epoch, "train_loss": round(tot / max(n, 1), 4), "min": round((time.time() - t_start) / 60, 1)}
        for name, rows in (("val_real", val_real), ("bench_dev", bench_dev)):
            tl, cl, ml = _predict(eval_model, rows, root, cfg["img_size"], device)
            m = _metrics(tl, [TYPES.index(r["garment_type"]) for r in rows], cl)
            rec[name] = {k: round(v, 4) for k, v in m.items() if isinstance(v, float)}
        if val_syn and (epoch == cfg["epochs"] or epoch % 10 == 0):
            tl, cl, _ = _predict(eval_model, val_syn, root, cfg["img_size"], device)
            rec["val_syn"] = {k: round(v, 4) for k, v in _metrics(tl, [TYPES.index(r["garment_type"]) for r in val_syn], cl).items() if isinstance(v, float)}
        history.append(rec)
        score = rec["val_real"]["type_top1"] + rec["val_real"]["type_macro_f1"]
        print(f"[{run_name}] ep {epoch:02d} loss {rec['train_loss']:.3f} | val_real top1 {rec['val_real']['type_top1']:.4f} f1 {rec['val_real']['type_macro_f1']:.4f} | bench_dev top1 {rec['bench_dev']['type_top1']:.4f} top3 {rec['bench_dev']['type_top3']:.4f} coarse {rec['bench_dev']['coarse_acc']:.4f} | {rec['min']}m", flush=True)
        if score > best or cfg["train_on"] != "train":
            best = score
            torch.save({"model": eval_model.state_dict(), "cfg": cfg, "epoch": epoch, "metrics": rec}, run_dir / "best.pt")
        with open(run_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)
        runs_volume.commit()

    # Final evaluation of the selected checkpoint (bench_test stays locked; see final_eval)
    ck = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    eval_model = _build_model(BACKBONES.get(cfg["backbone"], cfg["backbone"]), False, 0.0, 0.0)
    eval_model.load_state_dict(ck["model"])
    eval_model.to(device)
    result: dict[str, Any] = {"run": run_name, "cfg": cfg, "best_epoch": ck["epoch"]}
    for name, rows in (("val_real", val_real), ("bench_dev", bench_dev)):
        for tta in (False, True):
            tl, cl, ml = _predict(eval_model, rows, root, cfg["img_size"], device, tta=tta)
            m = _metrics(tl, [TYPES.index(r["garment_type"]) for r in rows], cl)
            hit = _material_hit3(ml, rows)
            if hit is not None:
                m["material_top3_hit"] = hit
            result[f"{name}{'_tta' if tta else ''}"] = m
            if name == "bench_dev" and not tta:
                np.savez(run_dir / "bench_dev_logits.npz", type=tl, coarse=cl, mat=ml)
    if oof_rows:
        tl, _, _ = _predict(eval_model, oof_rows, root, cfg["img_size"], device, tta=True)
        np.savez(run_dir / "oof_logits.npz", type=tl, files=np.array([r["file"] for r in oof_rows]))
        result["oof_n"] = len(oof_rows)
    for name in ("bench_test", "test"):
        # Logits are stored for the label audit only; metrics on these splits are computed in final_eval.
        if cfg.get("oof_fold") is not None and name in splits:
            rows = splits[name] if name == "bench_test" else [r for r in splits[name] if r["domain"] == "real"]
            tl, _, _ = _predict(eval_model, rows, root, cfg["img_size"], device, tta=True)
            np.savez(run_dir / f"{name}_audit_logits.npz", type=tl, files=np.array([r["file"] for r in rows]))
    with open(run_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    runs_volume.commit()
    return result


# ---------------------------------------------------------------------------
# Step 3: locked final evaluation (real test + bench_test), bootstrap CIs
# ---------------------------------------------------------------------------


def _load_eval_model(run: str, device):
    import torch

    ck = torch.load(f"/runs/{run}/best.pt", map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    model = _build_model(BACKBONES.get(cfg["backbone"], cfg["backbone"]), False, 0.0, 0.0)
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), cfg


def _fit_temperature(logits, y) -> float:
    import torch

    lg = torch.tensor(logits, dtype=torch.float64)
    yt = torch.tensor(y)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(lg / log_t.exp(), yt)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def _bootstrap(type_logits, y, n: int = 2000, seed: int = 0) -> dict[str, list[float]]:
    import numpy as np

    rng = np.random.default_rng(seed)
    p = np.asarray(type_logits)
    y = np.asarray(y)
    pred = p.argmax(1)
    top3 = (np.argsort(-p, 1)[:, :3] == y[:, None]).any(1)
    h = _hierarchy_matrix().numpy()
    sm = np.exp(p - p.max(1, keepdims=True))
    sm /= sm.sum(1, keepdims=True)
    coarse_ok = (sm @ h).argmax(1) == h[y].argmax(1)
    stats = {"type_top1": [], "type_top3": [], "coarse_acc": []}
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        stats["type_top1"].append((pred[i] == y[i]).mean())
        stats["type_top3"].append(top3[i].mean())
        stats["coarse_acc"].append(coarse_ok[i].mean())
    return {k: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)] for k, v in stats.items()}


@app.function(image=image, gpu="L4", cpu=8, memory=32768, timeout=3600,
              volumes={"/data": data_volume, "/runs": runs_volume})
def final_eval(runs: list[str], tta: bool = False) -> dict[str, Any]:
    """Evaluate selected runs once on the locked splits. Ensembles: pass 'a+b'."""
    import numpy as np

    device = "cuda"
    root = _ensure_images()
    splits = _load_split_rows()
    sets = {
        "val_real": [r for r in splits["val"] if r["domain"] == "real"],
        "test_real": [r for r in splits["test"] if r["domain"] == "real"],
        "bench_dev": splits["bench_dev"],
        "bench_test": splits["bench_test"],
        "test_syn": [r for r in splits["test"] if r["domain"] == "synthetic"],
    }
    sets["real_all_heldout"] = sets["test_real"] + sets["bench_test"]
    out: dict[str, Any] = {}
    for spec in runs:
        members = spec.split("+")
        logits: dict[str, Any] = {}
        mats: dict[str, Any] = {}
        for run in members:
            model, cfg = _load_eval_model(run, device)
            for name, rows in sets.items():
                if name == "real_all_heldout":
                    continue
                tl, _, ml = _predict(model, rows, root, cfg["img_size"], device, tta=tta)
                lp = tl - np.log(np.exp(tl - tl.max(1, keepdims=True)).sum(1, keepdims=True)) - tl.max(1, keepdims=True)
                logits[name] = logits.get(name, 0) + lp / len(members)
                mats[name] = mats.get(name, 0) + ml / len(members)
        logits["real_all_heldout"] = np.concatenate([logits["test_real"], logits["bench_test"]])
        mats["real_all_heldout"] = np.concatenate([mats["test_real"], mats["bench_test"]])
        y = {k: [TYPES.index(r["garment_type"]) for r in v] for k, v in sets.items()}
        temp = _fit_temperature(logits["val_real"], y["val_real"])
        res: dict[str, Any] = {"temperature": round(temp, 4)}
        for name in sets:
            m = _metrics(logits[name], y[name], temp=temp)
            m["ci95"] = _bootstrap(logits[name], y[name])
            hit = _material_hit3(mats[name], sets[name])
            if hit is not None:
                m["material_top3_hit"] = hit
            pred = np.asarray(logits[name]).argmax(1)
            cm = np.zeros((len(TYPES), len(TYPES)), dtype=int)
            for t, p in zip(y[name], pred):
                cm[t, p] += 1
            m["confusion"] = cm.tolist()
            res[name] = m
        out[spec + ("_tta" if tta else "")] = res
        Path("/runs/final_eval").mkdir(exist_ok=True)
        with open(f"/runs/final_eval/{spec.replace('+', '__')}{'_tta' if tta else ''}.json", "w") as f:
            json.dump(res, f, indent=2)
    runs_volume.commit()
    return out


# ---------------------------------------------------------------------------
# Step 4: mobile export (normalisation + calibrated softmax + hierarchy in-graph)
# ---------------------------------------------------------------------------


@app.function(image=image, gpu="L4", cpu=8, memory=32768, timeout=3600,
              volumes={"/data": data_volume, "/runs": runs_volume})
def export(run: str, calib_n: int = 512) -> dict[str, Any]:
    import random

    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    import torch.nn as nn
    from onnxconverter_common import float16
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process
    from PIL import Image

    device = "cuda"
    root = _ensure_images()
    splits = _load_split_rows()
    model, cfg = _load_eval_model(run, device)
    size = cfg["img_size"]
    val_real = [r for r in splits["val"] if r["domain"] == "real"]
    tl, _, _ = _predict(model, val_real, root, size, device)
    temp = _fit_temperature(tl, [TYPES.index(r["garment_type"]) for r in val_real])

    class Exportable(nn.Module):
        """Input: RGB float32 in [0,1], letterboxed to size x size with RGB(128,128,128) padding."""

        def __init__(self, net: nn.Module, t: float) -> None:
            super().__init__()
            self.net = net
            self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))
            self.register_buffer("hierarchy", _hierarchy_matrix())
            self.inv_t = 1.0 / t

        def forward(self, image: torch.Tensor):
            t, _, m = self.net((image - self.mean) / self.std)
            prob_type = torch.softmax(t * self.inv_t, dim=-1)
            prob_coarse = prob_type @ self.hierarchy
            return prob_coarse, prob_type, torch.sigmoid(m)

    wrapper = Exportable(model.float().cpu().eval(), temp).eval()
    out_dir = Path(f"/runs/{run}/export")
    out_dir.mkdir(parents=True, exist_ok=True)
    fp32 = out_dir / "model_fp32.onnx"
    torch.onnx.export(
        wrapper, torch.rand(1, 3, size, size), str(fp32), input_names=["image"],
        output_names=["prob_coarse", "prob_type", "prob_material"], opset_version=17, do_constant_folding=True,
    )
    onnx.save(onnx.load(str(fp32)), str(fp32))  # inline any external data
    fp16 = out_dir / "model_fp16.onnx"
    onnx.save(float16.convert_float_to_float16(onnx.load(str(fp32)), keep_io_types=True), str(fp16))
    # Normalisation/softmax kept in FP32: plain FP16 broke the 0.005 parity budget on EdgeNeXt (0.0125).
    fp16_safe = out_dir / "model_fp16_safe.onnx"
    block = list(float16.DEFAULT_OP_BLOCK_LIST) + [
        "LayerNormalization", "ReduceMean", "Pow", "Sqrt", "Softmax", "Erf", "Sigmoid"]
    onnx.save(float16.convert_float_to_float16(onnx.load(str(fp32)), keep_io_types=True, op_block_list=block), str(fp16_safe))

    def load01(row) -> np.ndarray:
        with Image.open(root / row["file"]) as im:
            img = _letterbox(im.convert("RGB"), size)
        return (np.asarray(img, dtype=np.float32) / 255.0).transpose(2, 0, 1)[None]

    rng = random.Random(0)
    calib_rows = rng.sample([r for r in splits["train"] if r["domain"] == "real"], calib_n)

    class Reader(CalibrationDataReader):
        def __init__(self) -> None:
            self.it = iter(calib_rows)

        def get_next(self):
            r = next(self.it, None)
            return None if r is None else {"image": load01(r)}

    pre = out_dir / "model_fp32_pre.onnx"
    quant_pre_process(str(fp32), str(pre), skip_symbolic_shape=True)
    from onnxruntime.quantization import CalibrationMethod

    # Attention/softmax/LayerNorm and the classifier stay float; MNv4-Hybrid attention does not
    # survive full INT8 (72% -> 49% top-1 in the first test).
    int8_variants = {
        "int8_qdq_all_minmax": {"op_types_to_quantize": None, "calibrate_method": CalibrationMethod.MinMax},
        "int8_qdq_conv_minmax": {"op_types_to_quantize": ["Conv"], "calibrate_method": CalibrationMethod.MinMax},
        "int8_qdq_conv_percentile": {"op_types_to_quantize": ["Conv"], "calibrate_method": CalibrationMethod.Percentile},
    }
    int8_paths = {}
    for vname, kw in int8_variants.items():
        path = out_dir / f"model_{vname}.onnx"
        extra = {"ActivationSymmetric": False, "WeightSymmetric": True, "CalibMovingAverage": True}
        if kw["calibrate_method"] == CalibrationMethod.Percentile:
            extra = {"CalibPercentile": 99.99, "WeightSymmetric": True}
        quantize_static(
            str(pre), str(path), Reader(), quant_format=QuantFormat.QDQ, per_channel=True,
            activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, op_types_to_quantize=kw["op_types_to_quantize"],
            calibrate_method=kw["calibrate_method"], extra_options=extra,
        )
        int8_paths[vname] = path
    pre.unlink(missing_ok=True)

    # Parity + accuracy per format on bench_dev and val_real
    eval_rows = {"bench_dev": splits["bench_dev"], "val_real": val_real}
    report: dict[str, Any] = {"run": run, "temperature": temp, "img_size": size, "formats": {}}
    ref: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for name, rows in eval_rows.items():
            ref[name] = np.concatenate([wrapper(torch.from_numpy(load01(r)))[1].numpy() for r in rows])
    for fmt, path in (("fp32", fp32), ("fp16", fp16), ("fp16_safe", fp16_safe), *int8_paths.items()):
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        fr: dict[str, Any] = {"size_mb": round(path.stat().st_size / 2**20, 2)}
        for name, rows in eval_rows.items():
            probs = np.concatenate([sess.run(None, {"image": load01(r)})[1] for r in rows])
            y = np.array([TYPES.index(r["garment_type"]) for r in rows])
            fr[name] = {
                "type_top1": round(float((probs.argmax(1) == y).mean()), 4),
                "top1_agreement_vs_torch": round(float((probs.argmax(1) == ref[name].argmax(1)).mean()), 4),
                "max_abs_prob_diff": round(float(np.abs(probs - ref[name]).max()), 5),
            }
        x = load01(eval_rows["bench_dev"][0])
        for _ in range(5):
            sess.run(None, {"image": x})
        t0 = time.time()
        for _ in range(30):
            sess.run(None, {"image": x})
        fr["cpu_1thread_ms"] = round((time.time() - t0) / 30 * 1000, 1)
        report["formats"][fmt] = fr
    with open(out_dir / "labels_order.json", "w") as f:
        json.dump({"coarse_categories": COARSE, "garment_types": TYPES, "materials": MATERIALS,
                   "type_to_coarse": TYPE_TO_COARSE}, f, indent=2)
    with open(out_dir / "preprocessing.json", "w") as f:
        json.dump({"input_name": "image", "input_size": [size, size], "layout": "NCHW", "color": "RGB",
                   "range": [0.0, 1.0], "resize": "letterbox", "pad_color": list(PAD), "exif_transpose": True,
                   "normalization": "in-graph (ImageNet mean/std)",
                   "outputs": {"prob_coarse": "P(type) aggregated by hierarchy", "prob_type": "calibrated softmax",
                               "prob_material": "independent sigmoids"}}, f, indent=2)
    with open(out_dir / "export_report.json", "w") as f:
        json.dump(report, f, indent=2)
    runs_volume.commit()
    return report


@app.function(image=image, gpu="L40S", cpu=16, memory=65536, timeout=3600 * 6,
              volumes={"/data": data_volume, "/runs": runs_volume})
def train(cfg: dict[str, Any], run_name: str) -> dict[str, Any]:
    return _train_impl(cfg, run_name)


SWEEPS: dict[str, dict[str, dict[str, Any]]] = {
    # Round 1: quantify the data problem with the current production backbone.
    "r1_data": {
        "r1_mnv4_syn_only": {"domains": "synthetic", "epochs": 25},
        "r1_mnv4_real_only": {"domains": "real", "epochs": 25},
        "r1_mnv4_real_syn": {"epochs": 25},
        "r1_siglip2b_teacher": {"backbone": "siglip2_b", "lr": 3e-4, "backbone_lr_mult": 0.1,
                                "drop_path": 0.1, "weight_decay": 0.05, "epochs": 15, "batch_size": 48},
    },
    # Round 2a: stronger teacher + out-of-fold teachers for the label audit (Stage 1.3).
    "r2_teachers": {
        "r2_so400m_teacher": {"backbone": "siglip2_so400m", "lr": 2e-4, "backbone_lr_mult": 0.1,
                              "drop_path": 0.15, "epochs": 12, "batch_size": 32},
        **{f"r2_oof_f{k}": {"backbone": "siglip2_b", "lr": 3e-4, "backbone_lr_mult": 0.1, "epochs": 10,
                            "batch_size": 48, "oof_k": 5, "oof_fold": k} for k in range(5)},
    },
    # Round 2b: distil the So400m teacher into the MobileNetV4 student.
    "r2_kd": {
        "r2_kd_real_a07": {"domains": "real", "teacher": "r2_so400m_teacher", "kd_alpha": 0.7, "kd_T": 2.0, "epochs": 40},
        "r2_kd_real_a09_T4": {"domains": "real", "teacher": "r2_so400m_teacher", "kd_alpha": 0.9, "kd_T": 4.0, "epochs": 40},
        "r2_kd_realsyn_a09": {"teacher": "r2_so400m_teacher", "kd_alpha": 0.9, "kd_T": 2.0, "epochs": 40, "real_frac": 0.6},
        "r2_kd_real_a09_light": {"domains": "real", "teacher": "r2_so400m_teacher", "kd_alpha": 0.9, "kd_T": 2.0,
                                 "epochs": 40, "aug": "light", "drop_path": 0.05},
        "r2_real_light_noKD": {"domains": "real", "epochs": 40, "aug": "light", "drop_path": 0.05},
    },
    # Round 3: student backbones under the same distillation recipe (all <= 25 MB FP16, permissive).
    "r3_backbones": {
        f"r3_{bb}": {"backbone": bb, "domains": "real", "teacher": "r2_so400m_teacher", "kd_alpha": 0.9,
                     "kd_T": 2.0, "epochs": 40, **extra}
        for bb, extra in {
            "mnv4_cm": {},
            "convnext_pico": {"lr": 5e-4, "backbone_lr_mult": 0.5},
            "edgenext_s": {"lr": 5e-4, "backbone_lr_mult": 0.5},
            "effformerv2_s2": {"lr": 5e-4, "backbone_lr_mult": 0.5},
        }.items()
    },
    # Round 4: tune the R3 leaders (resolution, schedule, KD strength) + MNv4-Hybrid LR diagnosis.
    "r4_tune": {
        "r4_edgenext_e60": {"backbone": "edgenext_s", "domains": "real", "teacher": "r2_so400m_teacher",
                            "kd_alpha": 0.9, "kd_T": 2.0, "epochs": 60, "lr": 5e-4, "backbone_lr_mult": 0.5},
        "r4_edgenext_320": {"backbone": "edgenext_s", "img_size": 320, "domains": "real", "teacher": "r2_so400m_teacher",
                            "kd_alpha": 0.9, "kd_T": 2.0, "epochs": 40, "lr": 5e-4, "backbone_lr_mult": 0.5},
        "r4_edgenext_kdsyn": {"backbone": "edgenext_s", "teacher": "r2_so400m_teacher", "real_frac": 0.7,
                              "kd_alpha": 1.0, "kd_T": 2.0, "epochs": 40, "lr": 5e-4, "backbone_lr_mult": 0.5},
        "r4_convnext_pico_e60": {"backbone": "convnext_pico", "domains": "real", "teacher": "r2_so400m_teacher",
                                 "kd_alpha": 0.9, "kd_T": 2.0, "epochs": 60, "lr": 5e-4, "backbone_lr_mult": 0.5},
        "r4_mnv4hm_lowlr": {"backbone": "mnv4_hm", "domains": "real", "teacher": "r2_so400m_teacher",
                            "kd_alpha": 0.9, "kd_T": 2.0, "epochs": 40, "lr": 5e-4, "backbone_lr_mult": 0.2, "drop_path": 0.0},
    },
    # Teacher #2 on the cleaned splits for ensemble distillation.
    "r6_teacher2": {
        "r6_so400m_clean_s7": {"backbone": "siglip2_so400m", "lr": 2e-4, "backbone_lr_mult": 0.1,
                               "drop_path": 0.15, "epochs": 9, "batch_size": 32, "seed": 7},
    },
    # Round 5: "patient & consistent" function matching (Beyer et al. 2022): pure KD, mixup,
    # long schedule, synthetic + real images as the transfer set; plus feature distillation.
    "r5_patient": {
        "r5_edge_patient256": {"backbone": "edgenext_s", "teacher": "r2_so400m_teacher", "real_frac": 0.7,
                               "kd_alpha": 1.0, "kd_T": 1.0, "mixup": 0.8, "epochs": 150, "lr": 5e-4, "backbone_lr_mult": 0.5},
        "r5_edge_patient320": {"backbone": "edgenext_s", "img_size": 320, "teacher": "r2_so400m_teacher", "real_frac": 0.7,
                               "kd_alpha": 1.0, "kd_T": 1.0, "mixup": 0.8, "epochs": 120, "lr": 5e-4, "backbone_lr_mult": 0.5},
        "r5_cnx_patient256": {"backbone": "convnext_pico", "teacher": "r2_so400m_teacher", "real_frac": 0.7,
                              "kd_alpha": 1.0, "kd_T": 1.0, "mixup": 0.8, "epochs": 150, "lr": 5e-4, "backbone_lr_mult": 0.5},
        "r5_edge_featkd": {"backbone": "edgenext_s", "domains": "real", "teacher": "r2_so400m_teacher",
                           "kd_alpha": 0.9, "kd_T": 2.0, "feat_kd": 1.0, "epochs": 60, "lr": 5e-4, "backbone_lr_mult": 0.5},
    },
}


@app.local_entrypoint()
def sweep(name: str, only: str = "") -> None:
    configs = SWEEPS[name]
    if only:
        configs = {k: v for k, v in configs.items() if k in only.split(",")}
    print(f"Launching {len(configs)} runs on Modal: {list(configs)}")
    calls = [(k, train.spawn(v, k)) for k, v in configs.items()]
    for k, call in calls:
        try:
            res = call.get()
            bd = res["bench_dev"]
            print(f"{k}: bench_dev top1={bd['type_top1']:.4f} top3={bd['type_top3']:.4f} coarse={bd['coarse_acc']:.4f} f1={bd['type_macro_f1']:.4f} | val_real top1={res['val_real']['type_top1']:.4f}")
        except Exception as e:  # keep collecting other runs
            print(f"{k}: FAILED {e!r}")


@app.local_entrypoint()
def evaluate(runs: str, tta: bool = False) -> None:
    res = final_eval.remote(runs.split(","), tta)
    for spec, r in res.items():
        for name in ("val_real", "bench_dev", "test_real", "bench_test", "real_all_heldout", "test_syn"):
            m = r[name]
            print(f"{spec:40s} {name:17s} n={m['n']:4d} top1={m['type_top1']:.4f} {m['ci95']['type_top1']} "
                  f"top3={m['type_top3']:.4f} f1={m['type_macro_f1']:.4f} coarse={m['coarse_acc']:.4f} "
                  f"cF1={m['coarse_macro_f1']:.4f} ece={m['ece']:.4f} acc@0.7={m['acc@0.7']:.4f}/cov {m['coverage@0.7']:.3f}")


@app.local_entrypoint()
def export_run(run: str) -> None:
    print(json.dumps(export.remote(run), indent=2))


@app.local_entrypoint()
def splits() -> None:
    print(json.dumps(build_splits.remote(), indent=2))
