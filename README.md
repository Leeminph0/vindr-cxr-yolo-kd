# SP-Det cai tien: ma nguon theo review dot 1 (28/09/2026)

Thu muc nay chua ma cho task list trong thread du an. Thu tu cong viec theo Bang 7 cua review.

| Thu muc | Noi dung | Task |
|---|---|---|
| `requirements.txt` | Phien ban goi co dinh (ultralytics 8.4.171) | 0.2 |
| `configs/common.yaml` | Cau hinh chung cho moi mo hinh, 5 seed | 0.2 |
| `eval/cxr_eval.py` | Danh gia du chi so Bang 3 (tru chi so chi phi) | 0.3 |
| `eval/test_cxr_eval.py` | Kiem tra: AP khop pycocotools, ca de, bootstrap | 0.3 |

## Danh gia

```bash
pip install -r requirements.txt
python eval/test_cxr_eval.py          # phai in "OK"
python eval/cxr_eval.py --gt gt.csv --pred spdet.csv --images test_ids.txt --out results/spdet
python eval/cxr_eval.py --gt gt.csv --pred spdet.csv --pred-b yolov8s.csv --images test_ids.txt --out results/spdet_vs_v8
```

- `gt.csv`: `image_id,class_id,x_min,y_min,x_max,y_max` (hop da gop bang WBF), toa do pixel anh goc.
- `*.csv` du doan: them cot `score`.
- `test_ids.txt`: TOAN BO anh test, ke ca anh binh thuong (can cho AUC muc anh va FROC).
- Dau ra: `.md` (bang dan vao bai) va `.json` (so lieu day du, ca duong FROC).
- 3,000 anh, 2 cau hinh, 1,000 lan bootstrap: khoang 1 phut tren CPU.

Ghi chu: AP dung noi suy 101 diem kieu COCO, da doi chieu khop pycocotools o IoU 0.4, 0.5, 0.5:0.95 va ca ba nhom kich thuoc.
D-ECE: 10 khoang diem, hop dung = ghep duoc GT cung lop o IoU 0.4.
Kiem dinh cap: bootstrap theo anh dung chung mau cho hai cau hinh, p hai phia.
