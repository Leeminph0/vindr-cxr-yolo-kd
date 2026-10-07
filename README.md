# Knowledge Distillation for Chest X-ray Lesion Detection on VinDr-CXR

> **Status: proposal stage.** The literature survey and experiment design are done. Training has not started yet, so this repository reports **no results**. Numbers below come from published papers and are quoted for comparison only.

## Goal

Detect 14 types of thoracic abnormalities on chest X-rays with a **lightweight detector (≤10M parameters)** that keeps most of the accuracy of a large detector, by distilling knowledge from a high-mAP **teacher** into a small **student**.

The project will:

1. Train several teacher and student candidates under one fixed protocol.
2. Distill the best teacher into the student candidates.
3. Pick the best model on the **official VinDr-CXR test set** (3,000 images).

## Why this project

We surveyed 16 recent papers (2024–2026) on detection and grounding on VinDr-CXR / VinBigData. Two findings shaped the plan:

- **Research gap.** None of them distills a large detector into a small one for 14-class lesion detection on VinDr-CXR. The closest works use a teacher to make pseudo-labels from point annotations (DExTeR), apply KD to classification (Multi-TeSt KD), or transfer text prompts from a vision-language model (SP-Det).
- **Inconsistent evaluation.** Most papers use their own random split. One study shows the same YOLOv11 reaching mAP50 64.5 with an image-level split but 38.7 with a patient-level split, a sign of data leakage. Only three papers report on the official 3,000-image test set, so we evaluate there.

## Dataset

[VinDr-CXR](https://physionet.org/content/vindr-cxr/) (Scientific Data, 2022), also released as the [VinBigData Chest X-ray Abnormalities Detection](https://www.kaggle.com/competitions/vinbigdata-chest-xray-abnormalities-detection) Kaggle competition.

| Split | Images | Annotation |
|---|---|---|
| Train | 15,000 | 3 radiologists per image |
| Test (official) | 3,000 | 5 radiologists per image (consensus) |

**14 classes** (standard Kaggle set; 22 raw labels with 8 rare ones merged into *Other lesion*): Aortic enlargement, Atelectasis, Calcification, Cardiomegaly, Consolidation, ILD, Infiltration, Lung Opacity, Nodule/Mass, Other lesion, Pleural effusion, Pleural thickening, Pneumothorax, Pulmonary fibrosis.

Data is not included in this repository. Access requires a Kaggle account (train images and labels) and a credentialed PhysioNet account (official test labels).

## Evaluation protocol

| Item | Choice |
|---|---|
| Test set | Official 3,000-image VinDr-CXR test set |
| Training labels | Boxes from the 3 radiologists fused with Weighted Boxes Fusion (WBF), IoU 0.4 |
| Primary metric | mAP@0.4 (Kaggle metric) |
| Also reported | mAP@0.5, mAP@0.5:0.95, per-class AP@0.5, parameters, GFLOPs, latency |
| Analysis | Per-class AP for teacher, student and student + KD; confusion between Lung Opacity, Consolidation and Infiltration |

## Model candidates

| Role | Candidates |
|---|---|
| Teacher (high mAP) | DEIM-D-FINE-X, RF-DETR-L, YOLO26x @1024, SP-Det |
| Student (≤10M params) | YOLO26n, YOLO26s, DEIM-N, YOLOv11n-MFF |

YOLOv11-MFF is the only surveyed VinDr-CXR detector with public training code ([guanli-nangong/chest-X-ray-Anomaly-detection](https://github.com/guanli-nangong/chest-X-ray-Anomaly-detection), MIT). Its reported 41.5 mAP50 uses a custom split without normal images, so it will be re-run under our protocol.

## Distillation plan

- **Feature distillation at shallow levels (P2/P3)**, where small lesions are represented.
- **Localization distillation** with a Normalized Wasserstein Distance (NWD) term for small boxes. In CEFEN-AFFN, an NWD-based loss alone added +6.0 mAP50.
- **Logit / classification distillation** from teacher scores.
- **Copy-Paste augmentation** for rare classes.
- Focus on the hardest classes from the survey: Calcification (best reported AP50 0.205), Atelectasis (0.275), Pulmonary fibrosis, ILD and Other lesion. Easy classes such as Cardiomegaly (0.926) serve as a sanity check.

## Benchmarks to beat (official test set, from papers)

| Paper | Venue | mAP50 | Params |
|---|---|---|---|
| NSEC-YOLO | J. Radiation Research and Applied Sciences, 2024 | 41.6 | 76.5M |
| CEFEN-AFFN | Scientific Reports, 2026 | 40.1 | 25.0M |
| YOLO-CXR | IEEE Access, 2024 | 33.8 | n/a |

The target is a student with ≤10M parameters that approaches or exceeds 41.6 mAP50 on the official test set.

## Roadmap

- [x] Literature survey (16 papers) and comparison table
- [x] Teacher and student candidates selected
- [x] KD pipeline and evaluation protocol designed
- [ ] Reproduce YOLOv11-MFF code
- [ ] Training and evaluation configs for each candidate
- [ ] Train teacher baselines (needs GPU and data access)
- [ ] Train students with and without KD, compare, select the best model

## Repository structure (planned)

```
configs/      training configs per candidate
data/         scripts to download VinDr-CXR and build WBF labels (no images)
kd/           distillation losses and training loop
eval/         evaluation on the official test set
docs/         survey table and proposal
```

## License

To be decided. Dataset usage follows the VinDr-CXR / PhysioNet and Kaggle terms.
