# Edge AI Challenge 2026 — gói nộp cuối (submission_v5, LB 98,95)

## Nội dung
| Đường dẫn | Mô tả |
|---|---|
| `train_v4_full_pipeline.ipynb` | Toàn bộ pipeline: intent → teacher → student (KD) → xuất submission. Output trong notebook là của lần chạy tạo ra file này. |
| `train_v4_full_pipeline.py` | Bản .py của các cell code trong notebook |
| `export_submission.py` | Chỉ suy luận: nạp 2 model ≤200k tham số và ghi `output/submission_v5.csv` |
| `submission_v5.csv` | File đã nộp (md5 `90e8913863367198aace39c4ff097aa2`) |
| `output/intent_v4/best.pt` | TextCNN intent dùng để nộp: 199.852 tham số, train trên toàn bộ 6000 câu + augmentation |
| `output/intent_v4/val_run_best.pt`, `history_*.csv` | Lần chạy validation và log train |
| `checkpoints/student_v4/best.pt` | Detector StudentV4 (EMA, epoch 129, val mAP@0.5 0,9829 @640 / 0,9841 @704) |
| `checkpoints/student_v4/deploy.pt` | Cùng model đó, đã fuse: 198.568 tham số |
| `checkpoints/student_v4/last.pt`, `history.csv` | Trạng thái train (để notebook resume, không train lại) và log |
| `checkpoints/teacher_yolo26m_full/` | Teacher YOLO26m: CHỈ dùng khi train để distill, không bao giờ dùng lúc suy luận |
| `data` | Symlink tới `../data` (dữ liệu của BTC, không chép vào gói) |

## Tuân thủ luật
- Mỗi model khi suy luận đều < 200k tham số (intent 199.852, detector 198.568).
- Không TTA, không ensemble: chạy đúng một lượt ở 704×528, NMS 0.55.
- Không dùng model pretrained khi xuất submission (teacher chỉ dùng lúc train).
- Tối đa 100 box mỗi ảnh (có assert trong code).

## Tạo lại submission
```bash
python export_submission.py                 # khoảng 1 phút trên GPU -> output/submission_v5.csv
# hoặc chạy notebook từ đầu đến cuối (nạp lại checkpoint sẵn có, không train lại)
```
Thư mục này phải chứa `data/` (với `intent_test.csv`, `sample_submission.csv`, `test/images`, `train/...`), hoặc đặt biến `EDGE_AI_ROOT`.
Đã kiểm tra trên RTX 3060: cả hai cách đều cho ra file trùng byte với `submission_v5.csv`.
Trên GPU hoặc driver khác, confidence và tọa độ box có thể lệch ở chữ số thập phân cuối do autocast fp16.
