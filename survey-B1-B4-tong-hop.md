# Tổng hợp survey B1–B4: KD cho detection trên VinDr-CXR

Cập nhật 07/10/2026. Gộp kết quả 4 bước survey bằng NotebookLM.
Chi tiết từng chủ đề B4: thư mục `b4/` (4 file). Bảng số liệu: `survey_table.xlsx`.

---

## Tóm tắt 5 điểm chính

1. **Khoảng trống nghiên cứu đã xác nhận:** chưa có bài nào chưng cất detector lớn sang detector nhỏ (≤10M tham số) cho 14 lớp tổn thương VinDr-CXR, đánh giá trên test chính thức.
2. **Mốc cần so trên test chính thức 3.000 ảnh:** NSEC-YOLO mAP50 41.6 (76.5M), CEFEN-AFFN 40.1 (25M), YOLO-CXR 33.8.
3. **Student baseline mạnh nhất có code:** YOLOv11-MFF, mAP50 41.5 với 2.7M tham số, nhưng tự chia dữ liệu và bỏ ảnh bình thường, nên phải chạy lại trên test chính thức.
4. **Protocol của đề tài:** test chính thức 3.000 ảnh, 14 lớp chuẩn Kaggle, gộp nhãn train bằng WBF, báo cáo mAP40, mAP50, mAP50:95 và AP từng lớp.
5. **Trọng tâm KD:** distill ở lớp nông P2/P3, dùng loss NWD cho hộp nhỏ, nhắm vào các lớp khó Calcification và Atelectasis.

---

## B1. Gom nguồn (06/10)

**Đã làm:**
- Tải và đặt tên PDF theo mẫu `NN_TenNgan_NoiDang_Nam.pdf`.
- Xuất code 3 repo thành file văn bản trong `notebooklm-code/`: #03 YOLOv11-MFF, #15 RadVLM-GRPO, #16 AnatomiX.
- Lấy dataset card của #18 VinDr-CXR-VQA từ Hugging Face.

**Danh sách bài:** từ 20 bài ban đầu còn **16 bài**, cộng bài dataset VinDr-CXR.
- Bỏ #9–#12 và #19 vì không tải được toàn văn.
- Bỏ #06 RT-DETR-CXR và #13 FuzzyDet (07/10) vì không tải được đúng bài: file tải về là bài khác (RT-DETRv4 trên COCO, và một bài phát hiện lửa).

**Còn mở:** mục B1 chưa tick vì ban đầu tính 5 nguồn code. Hiện có đủ 3 repo và dataset card #18; #13 đã bỏ.

## B2. Dựng notebook (06/10)

- **Notebook A** (bài báo): 16 bài, bài dataset VinDr-CXR, proposal. Lần chụp gần nhất hiển thị 23 nguồn, tức vẫn còn file thừa cần xóa (#06, #13).
- **Notebook B** (code): 3 file code (#03, #15, #16), 4 PDF tương ứng (#03, #15, #16, #18) và dataset card #18.

## B3. Trích số liệu từng bài (06–07/10)

- Trích 12 mục cho từng bài bằng một prompt chuẩn, kiểm bằng trích dẫn, ghi vào `survey_table.xlsx`. Bảng có 3 sheet: Tổng quan, Xếp hạng mAP50, Chi tiết 12 mục.
- B3b sửa các chỗ sai:
  - #03 thay đúng PDF.
  - #15 tải đúng bản RadVLM + RL.
  - #14 DExTeR: số 58.3 là của teacher, 43.5 là của student.
  - #16 AnatomiX: 0.20 là detection, 0.180 là phrase grounding.
  - #02, #21, #22 đã kiểm lại cách chia dữ liệu.

**Bảng 16 bài**

| # | Mô hình | Nơi đăng, năm | Nhóm | Test chính thức | mAP50 | Tham số | Code |
|---|---|---|---|---|---|---|---|
| 01 | CEFEN-AFFN | Scientific Reports 2026 | Detection | Có | 40.1 | 25.0M | Không nêu |
| 02 | SP-Det | Knowledge-Based Systems 2025 | Detection + prompt văn bản | Không | 38.2 (mAP40 42.4) | ~13M + VLM | Không |
| 03 | YOLOv11-MFF | PLOS ONE 2025 | Detection (student) | Không | 41.5 | 2.7M | Có (MIT) |
| 04 | Sensitivity-Oriented YOLOv11 | Applied Computer Systems 2026 | Detection | Không | 38.7 | ~20M | Không nêu |
| 05 | Mamba-YOLOvX | Expert Systems with Applications 2025 | Detection | Không | 36.6 | 10.8M | Không nêu |
| 07 | CD-DETR | PLOS ONE 2025 | Detection | Không | Không báo mAP | 39.3M | Không nêu |
| 08 | CXR-MultiTaskNet | Scientific Reports 2025 | Detection + phân loại | Không | 92.7 (chỉ 6 lớp) | ~25M | Liên hệ |
| 14 | DExTeR | arXiv 2026 | Giám sát yếu, nhãn điểm | Không rõ | 58.3 (teacher, 50% nhãn) | — | Không |
| 15 | RadVLM + RL | arXiv 2025 | VLM grounding | Test grounding VinDr | 45.9 (grounding) | ~7B | Có |
| 16 | AnatomiX | arXiv 2026 | VLM grounding | Theo split gốc | 20.0 (VinDr-Instruct) | — | Có |
| 17 | ChestGPT | ICVGIP 2025 | VLM | Có | Không báo mAP | ~7B | Không |
| 18 | VinDr-CXR-VQA | arXiv 2025 | Bộ dữ liệu VQA | Không | Không báo mAP | ~4B | Có (HF) |
| 20 | Multi-TeSt KD | Discover AI 2025 | KD phân loại (ống thông) | Không dùng VinDr | AUC 0.950 | 63.5M | Không nêu |
| 21 | YOLO-CXR | IEEE Access 2024 | Detection (baseline) | Có | 33.8 | Không nêu | Không nêu |
| 22 | NSEC-YOLO | JRRAS 2024 | Detection (baseline) | Có | 41.6 | 76.5M | Không nêu |
| 23 | Ultralytics YOLO Evolution | arXiv 2025 | Tổng quan | Không dùng VinDr | YOLO26n 40.3 mAP50:95 trên COCO | 2.4M (YOLO26n) | Có |

Kèm bài dataset VinDr-CXR (Scientific Data 2022): 15.000 ảnh train (3 bác sĩ/ảnh), 3.000 ảnh test (5 bác sĩ/ảnh).

## B4. Tổng hợp chéo (07/10)

### Knowledge distillation (`b4/1_knowledge-distillation.md`)

Ba bài gần nhất đều chưa lấp khoảng trống:

| Bài | Cách làm | Vì sao chưa phải "KD detection 14 lớp VinDr-CXR" |
|---|---|---|
| #14 DExTeR | Teacher sinh pseudo-box cho student | Mục tiêu giảm chi phí nhãn, không nén model |
| #20 Multi-TeSt | KD nhiều teacher | Bài toán phân loại, không dùng VinDr-CXR |
| #02 SP-Det | Chuyển tri thức từ VLM qua văn bản | Không có teacher detector |

Bằng chứng ủng hộ:
- #14: pseudo-box giúp student Faster R-CNN tăng +12.1 mAP50 trên VinDr-CXR.
- #20: KD logit kết hợp đặc trưng nâng F1 từ 0.86 lên 0.95.

### Protocol và gộp nhãn (`b4/2_protocol-va-gop-nhan.md`)

- Mọi bài detection dùng **14 lớp chuẩn Kaggle**: 22 nhãn gốc, trong đó 8 lớp hiếm gộp thành "Other lesion".
- **Tập test:** chỉ #01, #21, #22 dùng test chính thức 3.000 ảnh; các bài khác tự chia. #04 cho thấy chia theo ảnh làm mAP50 tăng ảo từ 38.7 lên 64.5.
- **Ảnh bình thường:** #03 bỏ hết; #01 và #22 thêm 500 ảnh; #05 chỉ dùng ảnh có tổn thương.
- **Gộp hộp của 3 bác sĩ:** WBF là phổ biến nhất (#01, #04, #05, #21; IoU 0.5–0.55). Proposal đang dùng IoU 0.4.

### Tổn thương nhỏ (`b4/3_ton-thuong-nho.md`)

| Bài | Module | Mức tăng mAP50 |
|---|---|---|
| #01 | CEFEN, AFFN, loss SAOS (NWD + MSIoU) | 32.7 → 40.1; riêng SAOS +6.0 |
| #03 | MSPLC, FF, FAHG | 38.8 → 41.5 |
| #21 | Đầu tổn thương nhỏ, SFF, ECLA, RefConv | 30.6 → 33.8 |
| #22 | ANS, GPAdetect, AccurEIoU | 39.2 → 41.6 |

Xu hướng chung:
- Đầu phát hiện ở lớp nông.
- Chú ý kênh-không gian.
- Tích chập giãn.
- Loss riêng cho hộp nhỏ.
- Lọc miền tần số.

### Lớp khó (`b4/4_lop-kho.md`)

- **Dễ:** Cardiomegaly (AP50 0.926) và Aortic enlargement (0.885).
- **5 lớp khó nhất:**
  - Calcification (tốt nhất 0.205).
  - Atelectasis (0.275).
  - Pulmonary fibrosis và ILD.
  - Other lesion.
  - Nhóm Lung Opacity / Consolidation / Infiltration, vì hay nhầm lẫn chéo.

---

## Quyết định rút ra cho proposal

| Hạng mục | Quyết định |
|---|---|
| Đánh giá | Test chính thức 3.000 ảnh, 14 lớp; mAP40 (chính), mAP50, mAP50:95, AP từng lớp |
| Nhãn train | WBF; ghi rõ ngưỡng IoU (0.4 theo Kaggle hoặc 0.5 như đa số bài) |
| Student baseline | YOLOv11-MFF (có code) cùng các ứng viên YOLO26n/s và DEIM-N |
| Mốc so sánh | NSEC-YOLO 41.6, CEFEN-AFFN 40.1 trên test chính thức |
| KD | Distill đặc trưng P2/P3, localization distillation có NWD, Copy-Paste cho lớp hiếm |
| Phân tích | AP từng lớp cho teacher, student và student + KD; confusion matrix cho nhóm mờ phế nang |
| Không dùng làm teacher | #15 RadVLM, #16 AnatomiX (cần biết trước tên bệnh trong câu hỏi) |

## Còn mở

1. Số tham số YOLOv11-MFF: bài ghi 2.7M, repo ghi 3.1M. Kiểm ở B5 khi đọc code.
2. Student DExTeR là 42.6 hay 44.7 mAP50.
3. Trích dẫn cho câu "bài dataset VinDr-CXR khuyến nghị WBF".
4. Xóa #06, #13 khỏi Notebook A.
5. Bước tiếp theo:
   - B5: khai thác code trong Notebook B.
   - B6: viết Related Work và cập nhật proposal (đổi 18 bài thành 16 bài, thêm hình tổng quan).
