#!/usr/bin/env python3
"""Danh gia phat hien ton thuong X-quang nguc theo Bang 3 cua review dot 1 (28/09/2026).

Chi so:
  1. mAP40 (chi so chinh, khop SP-Det va cuoc thi VinBigData), mAP40:95, mAP50, mAP50:95 (COCO)
  2. AP theo kich thuoc (small / medium / large), chia theo COCO hoac theo phan vi cua tap test
  3. AP tung lop kem so hop trong tap test
  4. FROC: do nhay tai 0.5, 1, 2, 4 duong tinh gia tren anh
  5. AUC muc anh (diem anh = diem hop cao nhat), can co anh binh thuong trong --images
  6. D-ECE (sai so hieu chuan cho phat hien)
  7. Khoang tin cay bootstrap theo anh (mac dinh 1,000 lan) va kiem dinh cap giua hai cau hinh
(Chi so 8, tham so / GFLOPs / thoi gian, do bang script rieng.)

Dinh dang dau vao (CSV, co dong tieu de):
  --gt     image_id,class_id,x_min,y_min,x_max,y_max          (hop da gop, vd. bang WBF)
  --pred   image_id,class_id,x_min,y_min,x_max,y_max,score
  --images danh sach image_id cua TOAN BO tap test, mot dong mot anh (co the co cot dau la image_id);
           anh khong co hop trong --gt la anh binh thuong.
Toa do tinh bang pixel cua anh goc de AP theo kich thuoc co nghia.

Vi du:
  python cxr_eval.py --gt gt.csv --pred spdet.csv --images test_ids.txt --out results/spdet
  python cxr_eval.py --gt gt.csv --pred spdet.csv --pred-b yolov8.csv --images test_ids.txt --out results/cmp
"""
import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np

VINDR_CLASSES = [
    "Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly", "Consolidation",
    "ILD", "Infiltration", "Lung Opacity", "Nodule/Mass", "Other lesion", "Pleural effusion",
    "Pleural thickening", "Pneumothorax", "Pulmonary fibrosis",
]
REC_THRS = np.linspace(0.0, 1.0, 101)
COCO_SIZE = {"small": (0, 32 ** 2), "medium": (32 ** 2, 96 ** 2), "large": (96 ** 2, float("inf"))}
FROC_FPS = (0.5, 1.0, 2.0, 4.0)


# ---------------------------------------------------------------- doc du lieu
def read_boxes(path, with_score, ignore_classes=()):
    out = defaultdict(list)
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("x_min", "") in ("", "nan", "NaN"):
                continue
            c = int(float(r["class_id"]))
            if c in ignore_classes:
                continue
            row = [c, float(r["x_min"]), float(r["y_min"]), float(r["x_max"]), float(r["y_max"])]
            if with_score:
                row.append(float(r["score"]))
            out[r["image_id"]].append(row)
    return out


def read_images(path):
    ids = []
    with open(path, newline="") as f:
        for line in f:
            tok = line.strip().split(",")[0]
            if tok and tok != "image_id":
                ids.append(tok)
    return ids


def safe_mean(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return float(x.mean()) if len(x) else np.nan


# ---------------------------------------------------------------- ghep hop
def iou_matrix(a, b):
    """a: (N,4), b: (M,4) dang x1,y1,x2,y2 -> (N,M)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = aa[:, None] + ab[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def match_image(gt, pred, thr, area_rng):
    """Ghep tham lam kieu COCO trong mot anh, mot lop.

    gt: (M,4); pred: (N,4) da sap xep giam dan theo diem.
    Tra ve (tp, ignore_pred, n_gt_valid). GT ngoai area_rng bi bo qua; hop du doan
    ghep vao GT bi bo qua, hoac khong ghep duoc va nam ngoai area_rng, cung bi bo qua.
    """
    garea = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1]) if len(gt) else np.zeros(0)
    g_ign = (garea < area_rng[0]) | (garea >= area_rng[1])
    tp = np.zeros(len(pred), bool)
    ign = np.zeros(len(pred), bool)
    ious = iou_matrix(pred, gt)
    used = np.zeros(len(gt), bool)
    for i in range(len(pred)):
        best, bj = thr - 1e-10, -1
        # uu tien GT hop le, roi moi toi GT bi bo qua (nhu pycocotools)
        for want_ign in (False, True):
            for j in range(len(gt)):
                if used[j] or g_ign[j] != want_ign:
                    continue
                if ious[i, j] >= best:
                    best, bj = ious[i, j], j
            if bj >= 0:
                break
        if bj >= 0:
            used[bj] = True
            if g_ign[bj]:
                ign[i] = True
            else:
                tp[i] = True
        else:
            pa = (pred[i, 2] - pred[i, 0]) * (pred[i, 3] - pred[i, 1])
            ign[i] = pa < area_rng[0] or pa >= area_rng[1]
    return tp, ign, int((~g_ign).sum())


class Evaluator:
    """Ghep truoc mot lan cho moi (nguong IoU, dai kich thuoc); bootstrap chi can trong so theo anh."""

    def __init__(self, gt, pred, images, n_cls, iou_thrs, size_rngs, max_det=100):
        self.images = images
        self.n_img = len(images)
        self.n_cls = n_cls
        self.iou_thrs = list(iou_thrs)
        self.size_rngs = dict(size_rngs)
        img_idx = {im: k for k, im in enumerate(images)}
        missing = [im for im in list(gt) + list(pred) if im not in img_idx]
        if missing:
            raise ValueError(f"{len(set(missing))} image_id co trong gt/pred nhung khong co trong --images, vd. {missing[0]}")

        # dem GT
        self.gt_count = {r: np.zeros((self.n_img, n_cls)) for r in self.size_rngs}
        self.gt_areas = []
        # ket qua ghep: key (thr, rng) -> list mang theo lop: (img, score, tp, ign)
        self.m = {(t, r): [[] for _ in range(n_cls)] for t in self.iou_thrs for r in self.size_rngs}
        # cho AUC muc anh
        self.img_score = np.zeros(self.n_img)
        self.img_pos = np.zeros(self.n_img, bool)

        for im in images:
            k = img_idx[im]
            g = np.array(gt.get(im, []), float).reshape(-1, 5)
            p = np.array(pred.get(im, []), float).reshape(-1, 6)
            if len(p):
                p = p[np.argsort(-p[:, 5], kind="mergesort")][:max_det]
                self.img_score[k] = p[:, 5].max()
            self.img_pos[k] = len(g) > 0
            self.gt_areas.extend(((g[:, 3] - g[:, 1]) * (g[:, 4] - g[:, 2])).tolist())
            for c in range(n_cls):
                gc = g[g[:, 0] == c, 1:5]
                pc = p[p[:, 0] == c]
                for r, rng in self.size_rngs.items():
                    for t in self.iou_thrs:
                        tp, ign, ng = match_image(gc, pc[:, 1:5], t, rng)
                        if t == self.iou_thrs[0]:
                            self.gt_count[r][k, c] = ng
                        if len(pc):
                            self.m[(t, r)][c].append((np.full(len(pc), k), pc[:, 5], tp, ign))
        for key, per_cls in self.m.items():
            for c in range(n_cls):
                if per_cls[c]:
                    a = [np.concatenate(x) for x in zip(*per_cls[c])]
                    o = np.argsort(-a[1], kind="mergesort")
                    per_cls[c] = tuple(x[o] for x in a)
                else:
                    per_cls[c] = (np.zeros(0, int), np.zeros(0), np.zeros(0, bool), np.zeros(0, bool))

    # ---------- AP
    @staticmethod
    def _ap(tp, w, n_pos):
        if n_pos <= 0:
            return np.nan
        if len(tp) == 0:
            return 0.0
        ctp = np.cumsum(w * tp)
        cfp = np.cumsum(w * (~tp))
        rc = ctp / n_pos
        pr = ctp / np.maximum(ctp + cfp, np.finfo(float).eps)
        pr = np.maximum.accumulate(pr[::-1])[::-1]
        idx = np.searchsorted(rc, REC_THRS, side="left")
        q = np.where(idx < len(pr), pr[np.minimum(idx, len(pr) - 1)], 0.0)
        return float(q.mean())

    def ap_table(self, thr, rng="all", weights=None):
        w_img = np.ones(self.n_img) if weights is None else weights
        aps = np.full(self.n_cls, np.nan)
        for c in range(self.n_cls):
            img, _, tp, ign = self.m[(thr, rng)][c]
            keep = ~ign
            n_pos = float((self.gt_count[rng][:, c] * w_img).sum())
            aps[c] = self._ap(tp[keep], w_img[img[keep]], n_pos)
        return aps

    def map_over(self, thrs, rng="all", weights=None):
        return safe_mean([safe_mean(self.ap_table(t, rng, weights)) for t in thrs])

    # ---------- FROC (gop moi lop, ghep theo lop o nguong IoU thr)
    def froc(self, thr, weights=None):
        w_img = np.ones(self.n_img) if weights is None else weights
        img = np.concatenate([self.m[(thr, "all")][c][0] for c in range(self.n_cls)])
        sc = np.concatenate([self.m[(thr, "all")][c][1] for c in range(self.n_cls)])
        tp = np.concatenate([self.m[(thr, "all")][c][2] for c in range(self.n_cls)])
        o = np.argsort(-sc, kind="mergesort")
        w = w_img[img[o]]
        tp = tp[o]
        n_les = float((self.gt_count["all"].sum(1) * w_img).sum())
        n_im = float(w_img.sum())
        sens = np.cumsum(w * tp) / max(n_les, 1e-12)
        fppi = np.cumsum(w * (~tp)) / max(n_im, 1e-12)
        out = {}
        for f in FROC_FPS:
            ok = fppi <= f
            out[f"sens@{f:g}FP"] = float(sens[ok].max()) if ok.any() else 0.0
        return out, (fppi, sens)

    # ---------- AUC muc anh (Mann-Whitney co trong so)
    def image_auc(self, weights=None):
        w = np.ones(self.n_img) if weights is None else weights
        pos, neg = self.img_pos, ~self.img_pos
        if w[pos].sum() == 0 or w[neg].sum() == 0:
            return np.nan
        s = self.img_score
        order = np.argsort(s, kind="mergesort")
        s_o, w_o, p_o = s[order], w[order], pos[order]
        auc_num, cum_neg, i = 0.0, 0.0, 0
        while i < len(s_o):
            j = i
            while j < len(s_o) and s_o[j] == s_o[i]:
                j += 1
            wp = (w_o[i:j] * p_o[i:j]).sum()
            wn = (w_o[i:j] * ~p_o[i:j]).sum()
            auc_num += wp * (cum_neg + 0.5 * wn)
            cum_neg += wn
            i = j
        return float(auc_num / (w[pos].sum() * w[neg].sum()))

    # ---------- D-ECE (Kuppers et al. 2020), hop dung = ghep duoc o nguong thr
    def dece(self, thr, n_bins=10, weights=None):
        w_img = np.ones(self.n_img) if weights is None else weights
        sc = np.concatenate([self.m[(thr, "all")][c][1] for c in range(self.n_cls)])
        tp = np.concatenate([self.m[(thr, "all")][c][2] for c in range(self.n_cls)])
        img = np.concatenate([self.m[(thr, "all")][c][0] for c in range(self.n_cls)])
        w = w_img[img]
        if w.sum() == 0:
            return np.nan
        b = np.minimum((sc * n_bins).astype(int), n_bins - 1)
        e = 0.0
        for k in range(n_bins):
            s = b == k
            ws = w[s].sum()
            if ws > 0:
                e += ws * abs((w[s] * tp[s]).sum() / ws - (w[s] * sc[s]).sum() / ws)
        return float(e / w.sum())


# ---------------------------------------------------------------- tong hop
def headline(ev, weights=None):
    t40_95 = [round(x, 2) for x in np.arange(0.40, 0.951, 0.05)]
    t50_95 = [round(x, 2) for x in np.arange(0.50, 0.951, 0.05)]
    froc, _ = ev.froc(0.4, weights)
    m = {
        "mAP40": ev.map_over([0.4], "all", weights),
        "mAP40:95": ev.map_over(t40_95, "all", weights),
        "mAP50": ev.map_over([0.5], "all", weights),
        "mAP50:95": ev.map_over(t50_95, "all", weights),
        **froc,
        "image_AUC": ev.image_auc(weights),
        "D-ECE@0.4": ev.dece(0.4, weights=weights),
    }
    return m


def bootstrap(evs, n_boot, seed):
    """evs: list Evaluator tren cung --images. Moi lan lay mau dung chung cho moi cau hinh (kiem dinh cap)."""
    rng = np.random.default_rng(seed)
    n = evs[0].n_img
    samples = [[] for _ in evs]
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        for k, ev in enumerate(evs):
            samples[k].append(headline(ev, w))
    return [{key: np.array([s[key] for s in smp]) for key in smp[0]} for smp in samples]


def ci(x, alpha=0.05):
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return [np.nan, np.nan]
    return [float(np.quantile(x, alpha / 2)), float(np.quantile(x, 1 - alpha / 2))]


def fmt(x, pct=True):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return f"{100 * x:.1f}" if pct else f"{x:.3f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--pred-b", help="cau hinh thu hai de kiem dinh cap (A tru B)")
    ap.add_argument("--images", required=True)
    ap.add_argument("--n-classes", type=int, default=14)
    ap.add_argument("--ignore-class", type=int, action="append", default=None,
                    help="lop bo qua, mac dinh 14 (No finding cua VinDr)")
    ap.add_argument("--size-mode", choices=["coco", "quantile"], default="coco",
                    help="coco: 32^2 / 96^2 pixel; quantile: chia ba theo phan vi dien tich hop GT")
    ap.add_argument("--max-det", type=int, default=100)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="tien to file ket qua (.json, .md)")
    a = ap.parse_args()
    ignore = tuple(a.ignore_class) if a.ignore_class is not None else (14,)

    images = read_images(a.images)
    gt = read_boxes(a.gt, False, ignore)
    preds = [read_boxes(a.pred, True, ignore)] + ([read_boxes(a.pred_b, True, ignore)] if a.pred_b else [])

    if a.size_mode == "coco":
        size = COCO_SIZE
    else:
        areas = [(r[3] - r[1]) * (r[4] - r[2]) for v in gt.values() for r in v]
        q1, q2 = np.quantile(areas, [1 / 3, 2 / 3])
        size = {"small": (0, q1), "medium": (q1, q2), "large": (q2, float("inf"))}
    rngs = {"all": (0, float("inf")), **size}
    thrs = sorted({round(x, 2) for x in np.arange(0.40, 0.951, 0.05)})

    evs = [Evaluator(gt, p, images, a.n_classes, thrs, rngs, a.max_det) for p in preds]
    names = ["A"] + (["B"] if a.pred_b else [])
    boots = bootstrap(evs, a.n_boot, a.seed) if a.n_boot > 0 else None

    result = {"n_images": len(images), "n_normal_images": int((~evs[0].img_pos).sum()),
              "size_ranges": {k: list(v) for k, v in size.items()}, "configs": {}}
    md = []
    for k, (name, ev) in enumerate(zip(names, evs)):
        h = headline(ev)
        res = {"metrics": h}
        if boots:
            res["ci95"] = {m: ci(boots[k][m]) for m in h}
        res["ap_by_size@0.4"] = {r: safe_mean(ev.ap_table(0.4, r)) for r in size}
        per_cls = ev.ap_table(0.4)
        n_box = ev.gt_count["all"].sum(0)
        res["per_class@0.4"] = [{"class_id": c, "name": VINDR_CLASSES[c] if c < len(VINDR_CLASSES) else str(c),
                                 "AP40": None if np.isnan(per_cls[c]) else float(per_cls[c]), "n_boxes": int(n_box[c])}
                                for c in range(a.n_classes)]
        _, (fppi, sens) = ev.froc(0.4)
        sel = np.unique(np.searchsorted(fppi, np.linspace(0, min(8, fppi[-1] if len(fppi) else 0), 200)))
        sel = sel[sel < len(fppi)]
        res["froc_curve"] = {"fppi": fppi[sel].tolist(), "sensitivity": sens[sel].tolist()}
        result["configs"][name] = res

        src = a.pred if name == "A" else a.pred_b
        md.append(f"## Cau hinh {name}: `{os.path.basename(src)}`\n")
        md.append("| Chi so | Gia tri | KTC 95% |\n|---|---|---|")
        for m, v in h.items():
            pct = m != "D-ECE@0.4" and True
            c95 = res.get("ci95", {}).get(m)
            c_s = f"[{fmt(c95[0], pct)}, {fmt(c95[1], pct)}]" if c95 else ""
            md.append(f"| {m} | {fmt(v, pct)} | {c_s} |")
        md.append("\n| AP40 theo kich thuoc | " + " | ".join(size) + " |\n|---|" + "---|" * len(size))
        md.append("| | " + " | ".join(fmt(res["ap_by_size@0.4"][r]) for r in size) + " |\n")
        md.append("| Lop | AP40 | So hop test |\n|---|---|---|")
        for r in res["per_class@0.4"]:
            md.append(f"| {r['name']} | {fmt(r['AP40'])} | {r['n_boxes']} |")
        md.append("")

    if a.pred_b and boots:
        md.append("## Kiem dinh cap A tru B (bootstrap theo anh, cung mau)\n")
        md.append("| Chi so | A tru B | KTC 95% | p (hai phia) |\n|---|---|---|---|")
        paired = {}
        for m in boots[0]:
            d = boots[0][m] - boots[1][m]
            d = d[~np.isnan(d)]
            point = result["configs"]["A"]["metrics"][m] - result["configs"]["B"]["metrics"][m]
            p = float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))) if len(d) else np.nan
            paired[m] = {"diff": point, "ci95": ci(d), "p": p}
            pct = m != "D-ECE@0.4"
            md.append(f"| {m} | {fmt(point, pct)} | [{fmt(paired[m]['ci95'][0], pct)}, {fmt(paired[m]['ci95'][1], pct)}] | {'<0.001' if p < 0.001 else f'{p:.3f}'} |")
        result["paired_A_minus_B"] = paired

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out + ".json", "w") as f:
        json.dump(result, f, indent=1, ensure_ascii=False)
    header = (f"# Ket qua danh gia\n\nSo anh test: {result['n_images']} "
              f"(binh thuong: {result['n_normal_images']}). Chi so % tru D-ECE. "
              f"Bootstrap {a.n_boot} lan theo anh, seed {a.seed}.\n")
    with open(a.out + ".md", "w") as f:
        f.write(header + "\n" + "\n".join(md) + "\n")
    print(header + "\n" + "\n".join(md))


if __name__ == "__main__":
    main()
