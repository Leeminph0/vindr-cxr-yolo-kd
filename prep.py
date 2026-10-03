"""Chuan bi du lieu va goi y cho tai lap SP-Det (notebooks/01_reproduce_spdet.ipynb).

  1. dicom_to_png / convert_all  : DICOM VinDr-CXR -> PNG 8 bit, canh dai 1024, ghi kich thuoc goc
  2. fuse_radiologists           : gop hop cua 3 bac si bang WBF (bai goc khong noi cach gop, review co loi so 3)
  3. split_811                   : chia 8:1:1 tren 4,394 anh co ton thuong (nhu bai goc)
  4. write_yolo_dataset / write_gt_csv
  5. clean_report, extract_dbp   : hau xu ly bao cao CheXagent, trich ten benh (muc 3.1.1, 3.1.2)
  6. encode_prompts              : SCP -> BERT-base (token), DBP -> CLIP ViT-B/32, luu <image_id>.pt
"""
from __future__ import annotations

import csv
import json
import os
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

VINDR_CLASSES = [
    "Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly", "Consolidation", "ILD", "Infiltration",
    "Lung Opacity", "Nodule/Mass", "Other lesion", "Pleural effusion", "Pleural thickening", "Pneumothorax",
    "Pulmonary fibrosis",
]
NO_FINDING = 14


# ----------------------------------------------------------------------------- 1. DICOM -> PNG
def dicom_to_png(dcm_path, out_path, long_side=1024):
    import cv2
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut

    ds = pydicom.dcmread(str(dcm_path))
    arr = apply_voi_lut(ds.pixel_array, ds).astype(np.float32)
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = arr.max() - arr
    arr = (arr - arr.min()) / max(float(arr.max() - arr.min()), 1e-6) * 255.0
    h, w = arr.shape[:2]
    s = long_side / max(h, w)
    img = cv2.resize(arr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else arr
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img.astype(np.uint8))
    return w, h


def _convert_one(a):
    iid, dcm_dir, png_dir, long_side = a
    dcm, out = Path(dcm_dir) / f"{iid}.dicom", Path(png_dir) / f"{iid}.png"
    if out.exists():  # da chuyen: chi doc header de lay kich thuoc goc
        import pydicom

        ds = pydicom.dcmread(str(dcm), stop_before_pixels=True)
        return iid, int(ds.Columns), int(ds.Rows)
    w, h = dicom_to_png(dcm, out, long_side)
    return iid, w, h


def convert_all(image_ids, dcm_dir, png_dir, meta_csv, long_side=1024, workers=8):
    """Chuyen DICOM sang PNG (bo qua anh da chuyen) va ghi meta_csv: image_id,width,height (kich thuoc goc)."""
    meta = read_meta(meta_csv) if Path(meta_csv).exists() else {}
    todo = [i for i in image_ids if i not in meta or not (Path(png_dir) / f"{i}.png").exists()]
    with ProcessPoolExecutor(workers) as ex:
        for k, (iid, w, h) in enumerate(ex.map(_convert_one, [(i, dcm_dir, png_dir, long_side) for i in todo], chunksize=8)):
            meta[iid] = (w, h)
            if k % 500 == 0:
                print(f"  {k}/{len(todo)} anh")
    with open(meta_csv, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["image_id", "width", "height"])
        for k, (w, h) in sorted(meta.items()):
            wr.writerow([k, w, h])
    return meta


def read_meta(meta_csv):
    with open(meta_csv) as f:
        return {r["image_id"]: (int(r["width"]), int(r["height"])) for r in csv.DictReader(f)}


# ----------------------------------------------------------------------------- 2. gop hop bac si
def read_kaggle_train(train_csv):
    """train.csv cua cuoc thi VinBigData: image_id,class_name,class_id,rad_id,x_min,y_min,x_max,y_max."""
    rows = defaultdict(list)
    with open(train_csv) as f:
        for r in csv.DictReader(f):
            c = int(r["class_id"])
            if c == NO_FINDING:
                rows[r["image_id"]]  # giu anh binh thuong voi danh sach rong
                continue
            rows[r["image_id"]].append((r["rad_id"], c, *(float(r[k]) for k in ("x_min", "y_min", "x_max", "y_max"))))
    return rows


def fuse_radiologists(rows, meta, iou_thr=0.4):
    """WBF moi anh, moi bac si la mot 'mo hinh', trong so bang nhau. Tra ve {image_id: [(c, x1, y1, x2, y2), ...]}."""
    from ensemble_boxes import weighted_boxes_fusion

    fused = {}
    for iid, boxes in rows.items():
        if not boxes:
            fused[iid] = []
            continue
        w, h = meta[iid]
        by_rad = defaultdict(list)
        for rad, c, x1, y1, x2, y2 in boxes:
            by_rad[rad].append((c, x1 / w, y1 / h, x2 / w, y2 / h))
        bl = [[np.clip(b[1:], 0, 1).tolist() for b in v] for v in by_rad.values()]
        sl = [[1.0] * len(v) for v in by_rad.values()]
        ll = [[b[0] for b in v] for v in by_rad.values()]
        b, _, lab = weighted_boxes_fusion(bl, sl, ll, iou_thr=iou_thr, skip_box_thr=0.0)
        fused[iid] = [(int(c), x1 * w, y1 * h, x2 * w, y2 * h) for (x1, y1, x2, y2), c in zip(b, lab)]
    return fused


# ----------------------------------------------------------------------------- 3-4. chia va ghi du lieu
def split_811(image_ids, seed=42):
    ids = sorted(image_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n = len(ids)
    a, b = round(0.8 * n), round(0.9 * n)
    return {"train": sorted(ids[:a]), "val": sorted(ids[a:b]), "test": sorted(ids[b:])}


def class_counts(fused, ids):
    cnt = np.zeros(len(VINDR_CLASSES), int)
    for i in ids:
        for c, *_ in fused[i]:
            cnt[c] += 1
    return cnt


def write_yolo_dataset(root, splits, fused, meta, png_dir):
    """root/images/<split>/<id>.png (symlink) + root/labels/<split>/<id>.txt + root/data.yaml."""
    import yaml

    root, png_dir = Path(root), Path(png_dir).resolve()
    for split, ids in splits.items():
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for iid in ids:
            dst = root / "images" / split / f"{iid}.png"
            if not dst.exists():
                os.symlink(png_dir / f"{iid}.png", dst)
            w, h = meta[iid]
            with open(root / "labels" / split / f"{iid}.txt", "w") as f:
                for c, x1, y1, x2, y2 in fused[iid]:
                    f.write(f"{c} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}\n")
    data = {"path": str(root.resolve()), "train": "images/train", "val": "images/val", "test": "images/test",
            "names": dict(enumerate(VINDR_CLASSES))}
    with open(root / "data.yaml", "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return root / "data.yaml"


def write_gt_csv(fused, ids, path):
    """GT cho eval/cxr_eval.py, toa do pixel anh goc."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["image_id", "class_id", "x_min", "y_min", "x_max", "y_max"])
        for iid in ids:
            for c, x1, y1, x2, y2 in fused[iid]:
                wr.writerow([iid, c, round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)])
    with open(Path(path).with_name(Path(path).stem + "_ids.txt"), "w") as f:
        f.write("\n".join(ids) + "\n")


# ----------------------------------------------------------------------------- 5. bao cao va ten benh
def clean_report(text):
    """Muc 3.1.1: bo cau dang do o cuoi, bo doan lap lai."""
    text = re.sub(r"\s+", " ", text.replace("</s>", " ")).strip()
    sents = re.split(r"(?<=[.!?])\s+", text)
    if sents and not re.search(r"[.!?]$", sents[-1]):
        sents = sents[:-1]  # cau cuoi bi cat
    out, seen = [], set()
    for s in sents:
        key = s.lower().strip()
        if key and key not in seen:
            seen.add(key)
            out.append(s.strip())
    return " ".join(out)


# Tu dong nghia cho tung lop VinDr (muc 3.1.2: "opacity va consolidation co the cung chi mot ton thuong").
SYNONYMS = {
    "Aortic enlargement": ["aortic enlargement", "enlarged aorta", "aorta is enlarged", "tortuous aorta", "aortic knob",
                           "dilated aorta", "aortic ectasia", "unfolded aorta", "unfolding of the aorta"],
    "Atelectasis": ["atelectasis", "atelectatic", "collapse", "volume loss"],
    "Calcification": ["calcification", "calcified", "calcific", "granuloma"],
    "Cardiomegaly": ["cardiomegaly", "enlarged heart", "cardiac enlargement", "enlarged cardiac silhouette",
                     "enlarged cardiomediastinal silhouette",
                     r"(heart|heart size|cardiac silhouette|cardiac size) (is |appears )?(mildly |moderately |severely )?enlarged"],
    "Consolidation": ["consolidation", "consolidative", "pneumonia", "airspace disease", "air space disease"],
    "ILD": ["interstitial lung disease", "interstitial", "ild", "reticular", "reticulonodular", "honeycombing"],
    "Infiltration": ["infiltrate", "infiltration", "infiltrates"],
    "Lung Opacity": ["opacity", "opacities", "opacification", "haziness", "density", "densities"],
    "Nodule/Mass": ["nodule", "nodular", "mass", "masses", "lesion"],
    "Other lesion": ["fracture", "emphysema", "bulla", "cavity", "cyst", "hernia", "device", "pacemaker", "line",
                     "tube", "catheter", "edema"],
    "Pleural effusion": ["pleural effusion", "effusion", "effusions", "blunting of the costophrenic",
                         "blunted costophrenic", "fluid"],
    "Pleural thickening": ["pleural thickening", "thickened pleura", "pleural scarring", "apical cap"],
    "Pneumothorax": ["pneumothorax", "pneumothoraces"],
    "Pulmonary fibrosis": ["fibrosis", "fibrotic", "scarring", "fibrous"],
}
NEG_CUES = ["no ", "not ", "without ", "negative for ", "free of ", "absence of ", "absent ", "no evidence of ",
            "rule out ", "ruled out ", "resolved ", "clear of ", "unremarkable"]
NEG_STOP = [" but ", " however", " although", " except", "; ", " which ", " with the exception"]
GENERIC = {"the image", "the chest x-ray", "the chest x - ray", "chest x-ray", "this image", "the patient", "the x-ray",
           "the lungs", "the heart", "the chest", "it", "there", "this", "which", "that", "findings", "the findings"}


def _negated(sentence_lower, start_char):
    pre = sentence_lower[:start_char]
    for stop in NEG_STOP:
        k = pre.rfind(stop)
        if k >= 0:
            pre = pre[k + len(stop):]
    pre = " " + pre
    return any(cue in pre[-60:] for cue in NEG_CUES)


def _find_classes(text_lower):
    """Moi lan xuat hien tu dong nghia trong cau: (vi tri, lop). Tu dong nghia la regex."""
    hits = []
    for cls, syns in SYNONYMS.items():
        for s in syns:
            for m in re.finditer(r"(?<![a-z])(?:" + s + r")(?![a-z])", text_lower):
                hits.append((m.start(), cls))
    return hits


def match_class(phrase):
    hits = _find_classes(phrase.lower())
    return hits[0][1] if hits else None


def extract_dbp(report, nlp, max_prompts=16):
    """Muc 3.1.2: tim ten benh va cum danh tu trong tung cau, loai phan bi phu dinh (NegEx don gian).

    Tra ve dict: prompts (chuoi dua vao CLIP: ten lop VinDr cho phan khop = goi y duong, cum danh tu
    khong khop lop nao = goi y am), positives, negatives, negated.
    """
    doc = nlp(report)
    pos, neg, negated = [], [], []
    for sent in doc.sents:
        sl = sent.text.lower()
        for start, cls in sorted(_find_classes(sl)):
            (negated if _negated(sl, start) else pos).append(cls)
        for ch in sent.noun_chunks:
            phrase = ch.text.strip().lower()
            if not phrase or phrase in GENERIC or match_class(phrase):
                continue
            if not _negated(sl, ch.start_char - sent.start_char + len(ch.text)):
                neg.append(phrase)
    uniq = lambda xs: list(dict.fromkeys(xs))
    pos = uniq(pos)
    negated = [c for c in uniq(negated) if c not in pos]
    neg = uniq(neg)
    return {"prompts": (pos + neg)[:max_prompts], "positives": pos, "negatives": neg, "negated": negated}


# ----------------------------------------------------------------------------- 6. ma hoa goi y
def encode_prompts(reports, dbps, out_dir, device="cuda", bert="bert-base-uncased", max_tokens=256, batch=64):
    """reports: {image_id: text}; dbps: {image_id: [chuoi]}. Ghi out_dir/<image_id>.pt = {scp, dbp} (fp16)."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    from ultralytics.nn.text_model import build_text_model

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(bert)
    enc = AutoModel.from_pretrained(bert).to(device).eval()
    clip = build_text_model("clip:ViT-B/32", device=torch.device(device))
    ids = [i for i in reports if not (out_dir / f"{i}.pt").exists()]
    for k in range(0, len(ids), batch):
        chunk = ids[k:k + batch]
        t = tok([reports[i] for i in chunk], padding=True, truncation=True, max_length=max_tokens, return_tensors="pt").to(device)
        with torch.no_grad():
            hs = enc(**t).last_hidden_state
        for j, iid in enumerate(chunk):
            n = int(t["attention_mask"][j].sum())
            scp = hs[j, :n].half().cpu()
            texts = dbps.get(iid, [])
            if texts:
                with torch.no_grad():
                    dbp = clip.encode_text(clip.tokenize(texts)).half().cpu()
            else:
                dbp = torch.zeros(0, 512, dtype=torch.float16)
            torch.save({"scp": scp, "dbp": dbp}, out_dir / f"{iid}.pt")


def load_jsonl(path):
    out = {}
    if Path(path).exists():
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                out[d["image_id"]] = d
    return out
