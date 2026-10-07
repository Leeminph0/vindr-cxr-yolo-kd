# B5. Kết quả khai thác code (07/10/2026)

NotebookLM không làm được B5: repo #03 không có README hay script train, và PDF #03 trong Notebook B là bài khác ("YOLOv11 Demystified"). Vì vậy Claude đọc code và chạy thử trực tiếp trong container (CPU, ultralytics 8.4.174, torch 2.14.1).

## Bảng kết luận

| Nguồn | Kết luận | Lý do |
|---|---|---|
| #03 YOLOv11-MFF | **Dùng** làm student baseline | Code dựng và chạy forward được; 3,14M tham số. Phải tự viết script train. |
| #15 RadVLM + RL | **Loại** làm teacher, tham khảo reward IoU | Câu hỏi grounding bắt buộc có tên bệnh; model 7B, ảnh 512. |
| #16 AnatomiX | **Tham khảo** (phần mở rộng) | APM cần nhãn giải phẫu Chest ImaGenome (MIMIC, cần chứng chỉ); trọng số "coming soon". |
| #18 VinDr-CXR-VQA | **Tham khảo** | Không thêm nhãn hộp mới; cần kiểm ảnh không trùng test chính thức trước khi dùng. |

## 1. #03 YOLOv11-MFF

### Repo có gì
12 file: 3 module (`FAHG module.py`, `FF module.py`, `MS_LK conv.py`), `parse_model.py`, `cxr.yaml` (14 lớp), `yolo11-MFF.yaml` và 6 yaml ablation.
Không có: README, script train, trọng số, file chia dữ liệu, phiên bản Ultralytics.

### Module trong code
| Module trong bài | Class trong code | Vị trí trong yolo11-MFF.yaml |
|---|---|---|
| MSPLC (tích chập giãn đa tỉ lệ) | `MSPLC`, `C3_MSLC2` (file `MS_LK conv.py`) | Thay mọi `C3k2` ở backbone và neck |
| FF (gộp đặc trưng) | `FFM_Concat` (file `FF module.py`), một trọng số học được cho mỗi kênh | Thay mọi `Concat` ở neck |
| FAHG (lọc tần số) | `FAHG` (file `FAHG module.py`), dùng FFT | Trên P3, P4, P5 ngay trước Detect |

### Số tham số đo từ code
| Model (14 lớp, ảnh 640) | Tham số | GFLOPs |
|---|---|---|
| YOLO11n gốc | 2.592.570 | 6,51 |
| YOLOv11-MFF (đo từ code) | **3.139.984** (gồm 1.216 trọng số FF) | **~10,7** (chưa tính FFT trong FAHG) |
| YOLOv11-MFF (bài ghi) | 2,7M | 6,7 |

Kết luận: số của bài gần với YOLO11n gốc, không khớp với code. Bảng survey dùng số đo từ code (3,1M / ~10,7 GFLOPs). Số "2.6M, 6.6 GFLOPs" trong yaml chỉ là chú thích copy từ YOLO11n. Vẫn dưới 10M nên vẫn hợp làm student.

Script đo: `code/mff_count_params.py` và `code/mff_n_alias.yaml`.

### Cần làm để train
- Tự viết script train trên Ultralytics: đăng ký 3 module vào `tasks.py`, dùng `parse_model` của repo.
- `FFM_Concat` tạo trọng số theo số kênh lúc chạy; cần khởi tạo trước (ví dụ trong lần forward tính stride) để optimizer nhận đủ tham số.
- Cấu hình theo bài: 300 epoch, SGD, ảnh 640. Nhưng đánh giá theo protocol đề tài (test chính thức 3.000 ảnh, giữ ảnh bình thường), nên số sẽ khác 41,5.

## 2. #15 RadVLM + RL
- Prompt grounding có dạng "Where is [observation] located…": phải biết trước tên bệnh, không thể quét cả 14 lớp trong một lần như detector.
- Model 7B (LLaVA-OneVision), ảnh 512, cần GPU lớn để suy luận.
- Phần dùng được: reward grounding (soft-F1 theo IoU, ghép Hungarian), có thể nhắc trong Related Work như một cách tối ưu trực tiếp chất lượng hộp.

## 3. #16 AnatomiX
- APM (Anatomy Perception Module) là DETR với ResNet-50 và PubMedBERT, train trên Chest ImaGenome (thuộc MIMIC-CXR, cần chứng chỉ PhysioNet).
- Chưa có trọng số công bố.
- Ý tưởng "prior giải phẫu" (vùng phổi, tim) có thể nêu ở phần hướng mở rộng, không đưa vào phạm vi chính.

## 4. #18 VinDr-CXR-VQA
- Chỉ là hỏi-đáp trên hộp gốc của VinDr-CXR, không thêm nhãn hộp.
- 4.394 ảnh nhiều khả năng lấy từ ảnh có tổn thương của tập train. Chưa có bằng chứng chắc chắn, nên nếu dùng phải đối chiếu image_id với 3.000 ảnh test chính thức.

## Còn mở
1. Thay PDF #03 trong Notebook B bằng đúng bài PLOS ONE (hiện là "YOLOv11 Demystified").
2. Kiểm trùng ảnh #18 với test chính thức khi có dữ liệu.
3. Bước tiếp: B6 (Related Work) và "Chuẩn bị code train/eval", trong đó có script train MFF.
