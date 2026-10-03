# Chạy notebook 1 (tái lập SP-Det) trên Kaggle

Cần hai file trong thư mục dự án:
- `spdet_kaggle/spdet-code.zip`: mã nguồn (prep.py, spdet_model.py, eval/, configs/).
- `spdet/notebooks/01_reproduce_spdet_kaggle.ipynb`: notebook 1 đã chỉnh cho Kaggle.

Notebook này giống notebook 1 gốc, chỉ khác ở bốn điểm:
- Tự tìm đường dẫn input.
- Tự dừng trước giới hạn 12 giờ của Kaggle.
- Đóng gói tiến độ vào `spdet_work.tar` để phiên sau chạy tiếp.
- Chạy CheXagent ở fp16 trên T4.

## 0. Chuẩn bị tài khoản (làm một lần)

1. **Xác minh số điện thoại** ở Settings → Phone verification. Không xác minh thì không bật được GPU và Internet.
2. **Tham gia cuộc thi.** Mở trang cuộc thi *VinBigData Chest X-ray Abnormalities Detection*, bấm Join / Late Submission rồi chấp nhận luật. Không cần tải dữ liệu về máy.
3. **Quota:** khoảng 30 giờ GPU mỗi tuần, mỗi phiên tối đa 12 giờ. Số giờ còn lại hiện ở trang Settings.

## 1. Upload mã nguồn thành Dataset

1. Vào Kaggle → **Datasets → New Dataset**.
2. Kéo thả `spdet-code.zip` vào; Kaggle tự giải nén.
3. Đặt tên `spdet-code`, để **Private**, bấm Create.

Khi mình sửa mã, bạn tải zip mới về rồi bấm **New Version** trên dataset đó.

## 2. Tạo notebook

1. Vào **Code → New Notebook**, chọn **File → Import Notebook**, rồi chọn `01_reproduce_spdet_kaggle.ipynb`.
2. Ở bảng bên phải (Session options), chỉnh hai mục:
   - **Accelerator: GPU T4 x2.** Đừng chọn P100: PyTorch mới trên Kaggle không còn hỗ trợ P100. Notebook chỉ dùng GPU 0 (khối gợi ý không chạy được nhiều GPU), nên GPU thứ hai để trống là bình thường.
   - **Internet: On.** Cần để tải CheXagent, BERT, CLIP và trọng số YOLO.
3. **Add Input** hai mục:
   - Cuộc thi `vinbigdata-chest-xray-abnormalities-detection` (tìm ở tab Competitions).
   - Dataset `spdet-code` (tìm ở tab Your Datasets).

Không cần sửa đường dẫn nào, vì notebook tự tìm `train.csv`, thư mục `train/` và `spdet_model.py` trong `/kaggle/input`.

## 3. Chạy thử nhanh trước khi chạy thật (nên làm, khoảng 30 phút)

1. Bấm **Run All** ở chế độ tương tác. Theo dõi ô cấu hình, ô chuyển ảnh và ô sinh báo cáo.
2. Kiểm tra ô cấu hình in ra đủ ba thứ:
   - `SPDET_ROOT = ...`
   - `KAGGLE_DIR = ...`
   - `Tesla T4`
3. Ở ô sinh báo cáo, ba báo cáo đầu phải là câu tiếng Anh đọc được.
4. Ghi lại con số **`s/ảnh`** in cạnh mỗi báo cáo.
5. Bấm **Stop** (ô vuông) để dừng. Phần đã chạy ở chế độ này không được lưu, đây chỉ là bước kiểm tra.

Tổng thời gian sinh báo cáo xấp xỉ `s/ảnh × 4,394 / 3600` giờ.

## 4. Phiên 1: chạy thật

1. Bấm **Save Version**, chọn **Save & Run All (Commit)**, rồi Save.
2. Notebook chạy nền; bạn tắt trình duyệt được. Xem tiến độ ở trang notebook → Versions → Logs.
3. Khi xong, tab **Output** có:
   - `spdet_work.tar`: toàn bộ tiến độ (ảnh PNG, báo cáo, gợi ý, mô hình đã train).
   - `results_step1/`: các bảng kết quả.

Nếu hết giờ phiên giữa chừng, notebook tự dừng việc đang làm (ở mức 11 giờ), đóng gói, rồi in ra những việc còn lại.

## 5. Các phiên sau: chạy tiếp

1. Mở notebook (**Edit**), bấm **Add Input**, vào tab **Notebooks → Your Work**, chọn **chính notebook này**. Kaggle sẽ gắn `spdet_work.tar` của version vừa chạy.
   - Nếu Kaggle không cho thêm chính notebook: mở version vừa chạy → tab Output → **New Dataset** để tạo dataset từ output, rồi thêm dataset đó làm input.
2. **Chỉ giữ một `spdet_work.tar` trong input**, và đó phải là version mới nhất. Mỗi lần trước khi Save Version:
   - xoá input của version cũ;
   - hoặc bấm ⋮ trên input → cập nhật lên version mới nhất.
3. Bấm **Save Version → Save & Run All (Commit)** như phiên 1.

Notebook tự bỏ qua những phần đã xong: ảnh đã chuyển, báo cáo đã có, mô hình đã train xong. Một mô hình chỉ được bắt đầu khi thời gian còn lại lớn hơn `EST_HOURS` của nó.

## 6. Dự kiến số phiên

Đây là ước tính, chưa đo. Mình lấy theo bảng ước tính cho 3060 trước đây, vì T4 có sức mạnh gần tương đương.

| Phiên | Việc | Thời gian ước tính |
|---|---|---|
| 1 | Chuyển 4,394 DICOM sang PNG, sinh báo cáo CheXagent | 0.5–1 giờ chuyển ảnh, 4–7 giờ báo cáo (có thể cần 2 phiên nếu `s/ảnh` lớn hơn khoảng 8) |
| 2 | Mã hoá gợi ý, train YOLOv8s, train YOLO-World | khoảng 6 giờ |
| 3 | Train SP-Det, chấm điểm cả ba mô hình | khoảng 4–5 giờ |

Tổng khoảng 12–18 giờ GPU, vừa quota một tuần.

Sau phiên đầu tiên có train, log sẽ in dòng kiểu `yolov8s s0: 1.83 giờ`. Hãy sửa `EST_HOURS` trong ô cấu hình theo số thật, cộng thêm khoảng 20%.

## 7. Lỗi thường gặp

| Hiện tượng | Cách sửa |
|---|---|
| `AttributeError: 'NoneType' object has no attribute 'parent'` ở ô cấu hình | Chưa Add Input cuộc thi hoặc dataset `spdet-code` |
| pip báo lỗi mạng | Internet chưa bật (cần xác minh số điện thoại) |
| `Sai phiên bản ultralytics` | Ô cài gói chưa chạy hoặc bị lỗi; xem log của ô đầu tiên |
| `CUDA out of memory` khi train | Đổi `TRAIN_BATCH = 16`. Ultralytics vẫn cộng dồn gradient đủ 64 ảnh mỗi lần cập nhật, nên kết quả gần như không đổi. Ghi lại trong bài. |
| Báo cáo CheXagent rỗng, toàn ký tự `!`, hoặc lặp một từ | Đổi `CHEX_DTYPE = "float32"` (chậm hơn khoảng 2 lần). Xoá input tar cũ nếu báo cáo lỗi đã được lưu. |
| Version bị Kaggle dừng ở 12 giờ, không có output | Chạy lại từ tar của version trước. Tăng `EST_HOURS` hoặc giảm `MAX_HOURS` |
| `Để spdet s0 cho phiên sau` | Bình thường: không đủ giờ trong phiên này, phiên sau sẽ train |

**Ghi vào bài:** bài gốc chạy CheXagent ở bf16 trên 2 × A5000. Trên Kaggle, notebook chạy fp16 trên 1 × T4. Đây là một nguồn chênh lệch có thể có.

## 8. Khi xong

Ô cuối sẽ in `Còn phải train: không, bước 1 đã xong`. Lúc đó:
1. Vào tab Output, tải các file trong `results_step1/`:
   - `table1_reproduced.csv`
   - `table2_reproduced.csv`
   - các file `spdet_vs_*.md`
2. Gửi các file đó vào thread này. Mình sẽ so sánh với bài gốc và tick mục 1.1 đến 1.3.

Giữ lại `spdet_work.tar` của version cuối. Notebook 2 dùng lại báo cáo và gợi ý của 4,394 ảnh này.
