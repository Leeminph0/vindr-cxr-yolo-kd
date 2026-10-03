"""SP-Det tren nen Ultralytics 8.4.171 (tai lap theo Li, Zou et al., Knowledge-Based Systems 2025).

Bai goc khong cong bo ma, nen module nay dung lai kien truc tu mo ta trong bai:
  - Nen YOLO-World v2 (YOLOv8s + RepVL-PAN), khoi tao tu yolov8s-worldv2.pt (muc 4.3).
  - Bidirectional Feature Enhancer (muc 3.2, Eq. 1-11) dat sau SPPF (feature map cap cao nhat P5),
    do sau 2 (Bang 6). Moi loai goi y co mot nhanh cross-attention rieng (muc 3.1.3):
      SCP (bao cao CheXagent) -> token BERT-base (768 chieu)
      DBP (ten benh trich tu bao cao) -> CLIP ViT-B/32 (512 chieu)
  - Dau phan loai: tuong phan vung-van ban cua YOLO-World voi 14 ten lop qua CLIP (muc 3.3.1),
    loss = BCE tuong phan + CIoU + DFL (Eq. 13-14) theo v8DetectionLoss.

Nhung cho bai khong noi ro va lua chon o day (ghi lai trong bai tai lap):
  1. P5 sau enhancer thay cho P5 goc, roi di vao PAN (PAN noi P5 voi P3/P4, ung voi Eq. 11).
  2. Eq. 9 duoc them residual, va out_proj cua self-attention / lop cuoi FFN khoi tao bang 0, gamma = 0,
     de luc bat dau mo hinh trung voi YOLO-World da hoc truoc.
  3. Moi nhanh van ban co them mot token "null" hoc duoc, de anh khong co cum ten benh nao van chay duoc.
  4. Goi y duoc truyen qua bien toan cuc CTX (dat truoc moi lan forward) vi pipeline Ultralytics
     chi truyen anh vao model. Chi chay mot GPU (khong DDP).
"""
from __future__ import annotations

import csv
import math
from copy import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.models.yolo.world.train import WorldTrainer, on_pretrain_routine_end
from ultralytics.nn.modules import SPPF
from ultralytics.nn.tasks import WorldModel
from ultralytics.utils import LOGGER, RANK

SCP_DIM, DBP_DIM = 768, 512


# ----------------------------------------------------------------------------- goi y theo anh
class PromptStore:
    """Doc goi y da ma hoa san: <root>/<image_id>.pt = {"scp": (L,768) fp16, "dbp": (K,512) fp16}."""

    def __init__(self, root, max_cache=20000):
        self.root = Path(root)
        self.cache = {}
        self.max_cache = max_cache

    def get(self, image_id):
        if image_id not in self.cache:
            f = self.root / f"{image_id}.pt"
            if f.exists():
                d = torch.load(f, map_location="cpu", weights_only=True)
                item = (d["scp"], d["dbp"])  # giu fp16 trong RAM, doi sang fp32 khi collate
            else:
                LOGGER.warning(f"SP-Det: thieu goi y cho {image_id}, dung token null")
                item = (torch.zeros(0, SCP_DIM, dtype=torch.float16), torch.zeros(0, DBP_DIM, dtype=torch.float16))
            if len(self.cache) < self.max_cache:
                self.cache[image_id] = item
            return item
        return self.cache[image_id]

    @staticmethod
    def _pad(seqs, dim):
        n = max(1, max(len(s) for s in seqs))
        out = torch.zeros(len(seqs), n, dim)
        mask = torch.ones(len(seqs), n, dtype=torch.bool)  # True = bo qua
        for i, s in enumerate(seqs):
            out[i, : len(s)] = s
            mask[i, : len(s)] = False
        return out, mask

    def collate(self, im_files, device):
        items = [self.get(Path(f).stem) for f in im_files]
        scp, scp_m = self._pad([a for a, _ in items], SCP_DIM)
        dbp, dbp_m = self._pad([b for _, b in items], DBP_DIM)
        return {k: v.to(device) for k, v in dict(scp=scp, scp_mask=scp_m, dbp=dbp, dbp_mask=dbp_m).items()}


class _Ctx:
    """Goi y cua batch hien tai. store: PromptStore; batch: dict tensor da collate."""

    store: PromptStore | None = None
    batch: dict | None = None

    def set_files(self, im_files, device):
        self.batch = self.store.collate(im_files, device) if self.store is not None else None


CTX = _Ctx()


# ----------------------------------------------------------------------------- Bidirectional Feature Enhancer
def _ffn(c, hidden):
    f = nn.Sequential(nn.Linear(c, hidden), nn.GELU(), nn.Linear(hidden, c))
    nn.init.zeros_(f[2].weight)
    nn.init.zeros_(f[2].bias)
    return f


class TextBranch(nn.Module):
    """Eq. 5-8 cho mot loai van ban: X^ = X Wv, T^ = T Wt, T_guided = MHA(X^, T^, T^), X_cross = MHA(T_guided, X^, X^)."""

    def __init__(self, c, dt, d=256, heads=8):
        super().__init__()
        self.wv = nn.Linear(c, d)
        self.wt = nn.Linear(dt, d)
        self.null = nn.Parameter(torch.zeros(1, 1, dt))
        self.i2t = nn.MultiheadAttention(d, heads, batch_first=True)
        self.t2i = nn.MultiheadAttention(d, heads, batch_first=True)
        self.wp = nn.Linear(d, c)
        self.gamma = nn.Parameter(torch.zeros(1))  # chi gamma = 0 (wp khong = 0, neu khong ca hai khong nhan gradient)

    def forward(self, x, t, t_mask):
        b = x.shape[0]
        t = torch.cat([self.null.expand(b, -1, -1).to(x.dtype), t.to(x.dtype)], 1)
        t_mask = torch.cat([t_mask.new_zeros(b, 1), t_mask], 1)
        xh, th = self.wv(x), self.wt(t)
        t_guided = self.i2t(xh, th, th, key_padding_mask=t_mask, need_weights=False)[0]
        x_cross = self.t2i(t_guided, xh, xh, need_weights=False)[0]
        return self.gamma * self.wp(x_cross)


class EnhancerLayer(nn.Module):
    def __init__(self, c, d=256, heads=8, pe_hw=20):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, c, pe_hw, pe_hw))  # PE hoc duoc, noi suy theo H, W (Eq. 2)
        nn.init.trunc_normal_(self.pe, std=0.02)
        self.ln_sa = nn.LayerNorm(c)
        self.sa = nn.MultiheadAttention(c, heads, batch_first=True)
        nn.init.zeros_(self.sa.out_proj.weight)
        nn.init.zeros_(self.sa.out_proj.bias)
        self.ln1 = nn.LayerNorm(c)
        self.ffn1 = _ffn(c, 2 * c)
        self.scp = TextBranch(c, SCP_DIM, d, heads)
        self.dbp = TextBranch(c, DBP_DIM, d, heads)
        self.ln2 = nn.LayerNorm(c)
        self.ffn2 = _ffn(c, 2 * c)

    def forward(self, x, ctx):
        b, c, h, w = x.shape
        pe = F.interpolate(self.pe, size=(h, w), mode="bilinear", align_corners=False)
        xf = x.flatten(2).transpose(1, 2)  # Eq. 1
        xpos = (x + pe).flatten(2).transpose(1, 2)  # Eq. 2
        q = self.ln_sa(xpos)
        xs = self.sa(q, q, self.ln_sa(xf), need_weights=False)[0] + xf  # Eq. 3
        xr = self.ffn1(self.ln1(xs)) + xs  # Eq. 4
        acc = xr
        if ctx is None:
            ctx = dict(
                scp=xr.new_zeros(b, 0, SCP_DIM), scp_mask=torch.zeros(b, 0, dtype=torch.bool, device=x.device),
                dbp=xr.new_zeros(b, 0, DBP_DIM), dbp_mask=torch.zeros(b, 0, dtype=torch.bool, device=x.device),
            )
        acc = acc + self.scp(xr, ctx["scp"], ctx["scp_mask"])  # Eq. 5-8, nhanh SCP
        acc = acc + self.dbp(xr, ctx["dbp"], ctx["dbp_mask"])  # Eq. 5-8, nhanh DBP
        xe = acc + self.ffn2(self.ln2(acc))  # Eq. 9 (them residual)
        return xe.transpose(1, 2).reshape(b, c, h, w)  # Eq. 10


class BidirectionalFeatureEnhancer(nn.Module):
    def __init__(self, c, depth=2, d=256, heads=8):
        super().__init__()
        self.layers = nn.ModuleList(EnhancerLayer(c, d, heads) for _ in range(depth))

    def forward(self, x, ctx):
        for layer in self.layers:
            x = layer(x, ctx)
        return x


# ----------------------------------------------------------------------------- model
class SPDetModel(WorldModel):
    def __init__(self, cfg="yolov8s-worldv2.yaml", ch=3, nc=None, verbose=True, depth=2):
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=False)
        self.enh_idx = max(i for i, m in enumerate(self.model) if isinstance(m, SPPF))
        c = self.model[self.enh_idx].cv2.conv.out_channels
        self.spdet_enhancer = BidirectionalFeatureEnhancer(c, depth=depth)
        if verbose and RANK in {-1, 0}:
            n = sum(p.numel() for p in self.spdet_enhancer.parameters())
            LOGGER.info(f"SP-Det: enhancer sau layer {self.enh_idx} (SPPF, {c} kenh), do sau {depth}, {n / 1e6:.2f}M tham so")
            self.info()

    def _ctx_for(self, x):
        bt = CTX.batch
        if bt is None or bt["scp"].shape[0] != x.shape[0]:
            return None
        return {k: v.to(x.device) for k, v in bt.items()}

    def predict(self, x, profile=False, txt_feats=None, augment=False, embed=None, **kwargs):
        enh = getattr(self, "spdet_enhancer", None)
        if enh is None:  # trong luc __init__ tinh stride
            return super().predict(x, profile=profile, txt_feats=txt_feats, augment=augment, embed=embed)
        from ultralytics.nn.modules import C2fAttn, ImagePoolingAttn, WorldDetect

        txt_feats = (self.txt_feats if txt_feats is None else txt_feats).type_as(x)
        if txt_feats.shape[0] != x.shape[0] or self.model[-1].export:
            txt_feats = txt_feats.expand(x.shape[0], -1, -1)
        ori_txt_feats = txt_feats.clone()
        ctx = self._ctx_for(x)
        y = []
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            if isinstance(m, C2fAttn):
                x = m(x, txt_feats)
            elif isinstance(m, WorldDetect):
                x = m(x, ori_txt_feats)
            elif isinstance(m, ImagePoolingAttn):
                txt_feats = m(x, txt_feats)
            else:
                x = m(x)
            if m.i == self.enh_idx:
                x = enh(x, ctx)
            y.append(x if m.i in self.save else None)
        return x

    def loss(self, batch, preds=None):
        if preds is None and "im_file" in batch:
            CTX.set_files(batch["im_file"], batch["img"].device)
        return super().loss(batch, preds)


# ----------------------------------------------------------------------------- trainer / validator
class SPDetValidator(DetectionValidator):
    def preprocess(self, batch):
        batch = super().preprocess(batch)
        CTX.set_files(batch["im_file"], batch["img"].device)
        return batch


class SPDetTrainer(WorldTrainer):
    """WorldTrainer + SPDetModel. Dat CTX.store = PromptStore(...) truoc khi train."""

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = SPDetModel(
            cfg["yaml_file"] if isinstance(cfg, dict) else cfg,
            ch=self.data["channels"],
            nc=min(self.data["nc"], 80),
            verbose=verbose and RANK == -1,
        )
        if weights:
            model.load(weights)
        if on_pretrain_routine_end not in self.callbacks["on_pretrain_routine_end"]:
            self.add_callback("on_pretrain_routine_end", on_pretrain_routine_end)
        return model

    def get_validator(self):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return SPDetValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks)

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        CTX.set_files(batch["im_file"], self.device)
        return batch


# ----------------------------------------------------------------------------- xuat du doan cho eval/cxr_eval.py
def export_predictions(model, image_paths, meta, out_csv, use_prompts=False, imgsz=640, conf=0.001, iou=0.7,
                       max_det=100, device=None):
    """Chay du doan tung anh, doi toa do ve pixel anh goc va ghi CSV cho cxr_eval.py.

    model: ultralytics YOLO / YOLOWorld da nap trong so. meta: {image_id: (orig_w, orig_h)}.
    use_prompts=True cho SP-Det (dat CTX theo tung anh truoc khi du doan).
    """
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_csv, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["image_id", "class_id", "x_min", "y_min", "x_max", "y_max", "score"])
        for p in image_paths:
            p = Path(p)
            if use_prompts:
                CTX.set_files([str(p)], torch.device("cpu"))  # SPDetModel tu chuyen sang device cua anh
            r = model.predict(str(p), imgsz=imgsz, conf=conf, iou=iou, max_det=max_det, device=device, verbose=False)[0]
            h0, w0 = r.orig_shape
            ow, oh = meta[p.stem]
            sx, sy = ow / w0, oh / h0
            for (x1, y1, x2, y2), c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                wr.writerow([p.stem, int(c), round(x1 * sx, 1), round(y1 * sy, 1), round(x2 * sx, 1), round(y2 * sy, 1),
                             round(s, 5)])
                n += 1
    LOGGER.info(f"Da ghi {n} hop du doan cua {len(image_paths)} anh vao {out_csv}")
    return out_csv
