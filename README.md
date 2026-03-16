# HỆ THỐNG HỖ TRỢ CHẨN ĐOÁN BỆNH LÝ TIM MẠCH QUA TÍN HIỆU ECG 12 CHUYỂN ĐẠO (AI)

> **Dự án Nghiên cứu & Phát triển:** Ứng dụng mạng Multi-scale CNN kết hợp Transformer để phân loại đa nhãn 27 bệnh lý tim mạch trên bộ dữ liệu chuẩn lâm sàng PhysioNet Challenge 2020.

![Kaggle Kernel](https://img.shields.io/badge/Platform-Kaggle%20Kernels-blue?style=for-the-badge&logo=kaggle)
![Framework](https://img.shields.io/badge/Deep%20Learning-PyTorch-orange?style=for-the-badge&logo=pytorch)
![Dataset](https://img.shields.io/badge/Dataset-PhysioNet%20Challenge%202020-green?style=for-the-badge&logo=physionet)
![Architecture](https://img.shields.io/badge/Model-Multi--scale%20CNN%20%2B%20Transformer-red?style=for-the-badge&logo=pytorch)

---

## 📑 MỤC LỤC (TABLE OF CONTENTS)
1. Báo cáo Tiến độ & Lộ trình (Progress & Roadmap)
2. Giới thiệu & Tổng quan (Introduction)
3. Phương pháp Nghiên cứu (Research Methodology)
4. Kiến trúc Mô hình V4 (Model Architecture)
5. Kết quả Đánh giá (Evaluation Results)
6. Hướng dẫn Thực hiện (Implementation Guide)

---

## 1. BÁO CÁO TIẾN ĐỘ & LỘ TRÌNH (PROGRESS & ROADMAP)

### Giai Đoạn 1: Tiền Xử Lý Dữ Liệu (Data Preprocessing)
**Trạng thái:** ✅ Đã hoàn thành

- [x] Khởi tạo môi trường Kaggle Notebook với GPU Tesla P100-PCIE-16GB.
- [x] Đọc và parse nhãn SNOMED từ file `.hea` của PhysioNet Challenge 2020.
- [x] Ánh xạ 27 mã SNOMED sang 27 nhãn bệnh tim có chấm điểm.
- [x] Trích xuất tín hiệu thô định dạng numpy từ file `.mat` qua thư viện `wfdb`.
- [x] Chuẩn hóa tín hiệu: normalize mean/std từng chuyển đạo, crop/pad về 4096 mẫu.
- [x] Thiết lập đầu vào Tensor shape `(12, 4096)` — 12 chuyển đạo, 4096 time-steps.
- [x] Chia tập dữ liệu: Train 80% / Val 10% / Test 10% với SEED=42.
  - Train: 30,199 mẫu | Val: 3,775 mẫu | Test: 3,775 mẫu

### Giai Đoạn 2: Xây Dựng & Huấn Luyện Mô Hình (Modeling & Training)
**Trạng thái:** ✅ Đã hoàn thành

- [x] Thiết kế kiến trúc Multi-scale CNN với 4 nhánh song song (kernel=3, 9, 19, AvgPool).
- [x] Xây dựng 4 stages CNN: 12→128→256→512→512 channels với GroupNorm + GELU + Residual.
- [x] Tích hợp Transformer Encoder: 8 layers, 8 heads, d=384, ff=1536, Pre-LayerNorm.
- [x] Classification Head: GlobalAvgPool → Linear(512) → GELU → Linear(27) → Sigmoid.
- [x] Áp dụng Weighted ASL Loss với class weights sqrt-inverse frequency.
- [x] Gradient accumulation (effective batch=64), Cosine LR Scheduler.
- [x] **Giai đoạn Warmup (15 epoch):** Freeze CNN+Transformer, chỉ train Head. Best F1=0.334.
- [x] **Giai đoạn Fine-tune (50 epoch):** Unfreeze toàn bộ, early stop tại ep31, best ep19.
- [x] Lưu checkpoint tốt nhất: `cardiac_v4_best.pth` (epoch=19, val_F1=0.4984, val_AUC=0.9229).

### Giai Đoạn 3: Đánh Giá Hệ Thống (Evaluation)
**Trạng thái:** ✅ Đã hoàn thành

- [x] Predict TTA 5x trên tập Test độc lập (1 clean + 5 augmented rounds, average).
- [x] Tìm ngưỡng tối ưu 3 chiến lược trên Val set (step=0.01):
  - **Chiến lược 1 — Youden Index:** TPR − FPR max → `thr_youden_v4.json`
  - **Chiến lược 2 — F1+Recall:** F1 tối ưu với Recall ≥ 0.75 cho bệnh nguy hiểm → `thr_f1rec_v4.json` ✅ Khuyên dùng
  - **Chiến lược 3 — Screening:** Max Recall với F1 ≥ 0.05 → `thr_screening_v4.json`
- [x] Đánh giá Macro F1, Micro F1, Macro AUC, mAP, Sensitivity, Specificity từng bệnh.
- [x] Vẽ ROC Curve, PR Curve, Confusion Matrix, Radar Chart, Heatmap Recall.
- [x] Xuất báo cáo CSV đầy đủ per-class: F1, Precision, Recall, Specificity, AUC, AP.

### Giai Đoạn 4: Triển Khai Ứng Dụng (Deployment)
**Trạng thái:** ✅ Đã hoàn thành

- [x] Upload model lên Kaggle Dataset: `tuyenngoc12/cardiac-ai-v4-model`.
- [x] Xây dựng notebook inference (`ecg-inference-v4.ipynb`) với giao diện upload file ECG.
- [x] Hiển thị kết quả: 27 nhãn bệnh, xác suất, mức độ nguy hiểm (Bình thường / Thông tin / Cảnh báo / Nguy hiểm).
- [x] Vẽ biểu đồ tín hiệu ECG 12 chuyển đạo và biểu đồ xác suất từng bệnh.

---

## 2. GIỚI THIỆU & TỔNG QUAN (INTRODUCTION)

### 2.1. Bối cảnh
Điện tâm đồ (ECG) 12 chuyển đạo là công cụ tiêu chuẩn vàng trong chẩn đoán tim mạch. Tuy nhiên, việc phân tích đồng thời 12 kênh tín hiệu phụ thuộc nhiều vào kinh nghiệm bác sĩ lâm sàng, dẫn đến thiếu nhất quán ở các tuyến y tế cơ sở.

### 2.2. Mục tiêu Dự án
- **Phân tích 12 chuyển đạo đồng thời:** Tận dụng thông tin đa kênh thay vì chỉ 1 kênh.
- **Chẩn đoán đa nhãn (Multi-label):** Xác định đồng thời 27 bệnh lý tim mạch, phù hợp thực tế bệnh nhân mắc nhiều bệnh cùng lúc.
- **Ưu tiên an toàn lâm sàng:** Áp dụng Recall constraint cho 6 bệnh nguy hiểm (AF, AFL, LBBB, PVC, LQT, VPB) — đảm bảo Recall ≥ 0.75.

---

## 3. PHƯƠNG PHÁP NGHIÊN CỨU (RESEARCH METHODOLOGY)

### 3.1. Dữ liệu (Dataset)
Sử dụng bộ dữ liệu lâm sàng **PhysioNet/Computing in Cardiology Challenge 2020**.

| Thông số | Giá trị |
|----------|---------|
| Tổng bản ghi | 43,101 |
| Bản ghi hợp lệ (có nhãn) | 37,749 |
| Số bệnh nhân | ~37,000 |
| Định dạng | `.mat` + `.hea` (WFDB) |
| Tần số lấy mẫu | 500Hz |
| Số chuyển đạo | 12 |
| Số nhãn (scored) | 27 nhãn SNOMED |

### 3.2. Quy trình Xử lý (Pipeline)
1. **Đọc file:** `wfdb.rdrecord()` đọc file `.mat`, parse mã SNOMED từ file `.hea`.
2. **Normalize:** Chuẩn hóa mean/std từng chuyển đạo riêng biệt.
3. **Crop/Pad:** Crop center 4096 mẫu nếu dài hơn, pad zeros nếu ngắn hơn.
4. **Augmentation (train):** Gaussian noise (σ=0.03) + Amplitude scale (×0.85~1.15).
5. **TTA (inference):** 1 clean predict + 5 augmented rounds → average.

### 3.3. Chia tập dữ liệu
```
37,749 records (SEED=42)
├── Train : 30,199 mẫu (80%) — Huấn luyện model
├── Val   :  3,775 mẫu (10%) — Tìm ngưỡng tối ưu, Early stopping
└── Test  :  3,775 mẫu (10%) — Đánh giá cuối cùng (độc lập)
```

---

## 4. KIẾN TRÚC MÔ HÌNH V4 (MODEL ARCHITECTURE)

### 4.1. Tổng quan
```
Input (12, 4096)
    ↓
MultiScaleConvBlock × 4  [12→128→256→512→512 channels]
    ↓
Linear Projection        [512 → 384] + LayerNorm
    ↓
Positional Encoding      [Sinusoidal, max_len=1024]
    ↓
TransformerEncoder × 8   [8 heads, d=384, ff=1536, Pre-LN, GELU]
    ↓
GlobalAvgPool            [(batch, 384)]
    ↓
Classification Head      [LayerNorm → Dropout → Linear(512) → GELU → Linear(27)]
    ↓
Sigmoid → 27 xác suất
```

### 4.2. MultiScaleConvBlock
Mỗi block gồm 4 nhánh song song:

| Nhánh | Cấu trúc | Mục đích |
|-------|----------|----------|
| branch_small | Conv1d(k=3) + GroupNorm + GELU | Bắt pattern ngắn (QRS spike) |
| branch_mid | Conv1d(k=9) + GroupNorm + GELU | Bắt pattern trung (P-wave, T-wave) |
| branch_large | Conv1d(k=19) + GroupNorm + GELU | Bắt pattern dài (ST segment) |
| branch_pool | AvgPool + Conv1d(k=1) + GroupNorm + GELU | Bắt xu hướng tổng quát |

Sau đó: Concat 4 nhánh → Fusion Conv → MaxPool + Residual shortcut.

### 4.3. Thông số mô hình

| Thông số | Giá trị |
|----------|---------|
| Tổng tham số | ~20M |
| CNN Channels | [128, 256, 512, 512] |
| Transformer Layers | 8 |
| Transformer Heads | 8 |
| Transformer Dim | 384 |
| FeedForward Dim | 1536 |
| Dropout | 0.15 |
| Loss | Weighted ASL |
| Optimizer | AdamW |

### 4.4. Quy trình Huấn luyện

**Giai đoạn 1 — Warmup (15 epoch):**
- Freeze CNN + Transformer, chỉ train Classification Head.
- LR: 1e-3 | Batch: 64

**Giai đoạn 2 — Fine-tune (50 epoch):**
- Unfreeze toàn bộ model.
- LR: 1e-4 | Effective Batch: 64 (gradient accumulation ×4).
- Early stopping patience=12 epoch theo Val F1.
- **Best: Epoch 19 | Val F1=0.4984 | Val AUC=0.9229**
- Early stop: Epoch 31

---

## 5. KẾT QUẢ ĐÁNH GIÁ (EVALUATION RESULTS)

### 5.1. Kết quả tổng quan (Test set, TTA 5x)

| Chiến lược | Macro F1 | Micro F1 | AUC | mAP |
|------------|----------|----------|-----|-----|
| D1 — Youden Index | 0.4893 | 0.6177 | 0.9211 | 0.5173 |
| **D2 — F1+Recall ✅** | **0.4884** | **0.6066** | **0.9211** | **0.5173** |
| D3 — Screening | 0.1528 | 0.1916 | 0.9211 | 0.5173 |

### 5.2. Kết quả từng nhóm bệnh (D2 — F1+Recall)

**Rất tốt (F1 ≥ 0.8):** PR (0.973), SNR (0.895), AF (0.869), LBBB (0.831), RBBB (0.820), STach (0.816)

**Tốt (F1 0.5~0.8):** LAD (0.737), LAnFB (0.732), CRBBB (0.686), SB (0.672), IAVB (0.663), RAD (0.634), IRBBB (0.574), AFL (0.565), TAb (0.534)

**Cần cải thiện (F1 < 0.5):** LQT (0.453), Brady (0.431), LPR (0.400), PAC (0.329), QAb (0.311), VPB (0.290), SVPB (0.269), NSIVCB (0.265), TInv (0.238), LQRSV (0.222), PVC (0.182), SA (0.127)

### 5.3. Files đầu ra sau đánh giá

```
outputs/
├── thresholds/
│   ├── thr_youden_v4.json       ← Ngưỡng Youden Index
│   ├── thr_f1rec_v4.json        ← Ngưỡng F1+Recall (KHUYÊN DÙNG)
│   └── thr_screening_v4.json    ← Ngưỡng Screening
├── eval_d1_youden_v4.csv        ← Báo cáo D1 per-class
├── eval_d2_f1rec_v4.csv         ← Báo cáo D2 per-class
├── eval_d3_screen_v4.csv        ← Báo cáo D3 per-class
├── roc_pr_v4.png                ← ROC & PR Curves
├── radar_heatmap_v4.png         ← Radar Chart & Heatmap Recall
├── confusion_D1_Youden.png      ← Confusion Matrix D1
└── confusion_D2_F1+Recall.png   ← Confusion Matrix D2
```

---

## 6. HƯỚNG DẪN THỰC HIỆN TRÊN KAGGLE (IMPLEMENTATION GUIDE)

### Bước 1: Datasets cần thêm vào notebook
```
gamalasran/physionet-challenge-2020     ← Dữ liệu ECG (5GB)
bjoernjostein/physionet-snomed-mappings ← Bảng ánh xạ nhãn
tuyenngoc12/cardiac-ai-v4-model         ← Model weights + thresholds
```

### Bước 2: Load Model V4
```python
import torch
import torch.nn as nn
import numpy as np

MODEL_PATH     = '/kaggle/input/datasets/tuyenngoc12/cardiac-ai-v4-model/cardiac_v4_best.pth'
THRESHOLD_PATH = '/kaggle/input/datasets/tuyenngoc12/cardiac-ai-v4-model/thr_f1rec_v4.json'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load checkpoint
ckpt = torch.load(MODEL_PATH, map_location=device)
model = ECGModelV4().to(device)
model.load_state_dict(ckpt['state'])
model.eval()
print(f'Epoch={ckpt["epoch"]} | Val F1={ckpt["score"]:.4f}')

# Load thresholds
import json
with open(THRESHOLD_PATH) as f:
    thresh_dict = json.load(f)
THRESHOLDS = np.array([thresh_dict[str(i)] for i in range(27)])
```

### Bước 3: Tiền xử lý tín hiệu ECG
```python
import wfdb

def preprocess_ecg(mat_path, seq_len=4096, num_leads=12):
    rec  = wfdb.rdrecord(mat_path.replace('.mat', ''))
    sig  = rec.p_signal.astype(np.float32)

    # Normalize từng chuyển đạo
    mu  = sig.mean(0, keepdims=True)
    std = sig.std(0,  keepdims=True) + 1e-8
    sig = (sig - mu) / std

    # Crop center hoặc Pad zeros
    T = sig.shape[0]
    if T >= seq_len:
        start = (T - seq_len) // 2
        sig   = sig[start:start+seq_len]
    else:
        sig = np.vstack([sig, np.zeros((seq_len-T, num_leads), dtype=np.float32)])

    # Chuyển sang tensor shape (12, 4096)
    return torch.from_numpy(sig.T).unsqueeze(0)  # (1, 12, 4096)
```

### Bước 4: Inference TTA 5x
```python
def predict_tta(mat_path, n_tta=5):
    x0 = preprocess_ecg(mat_path).to(device)

    with torch.no_grad():
        probs = torch.sigmoid(model(x0)).squeeze().cpu().numpy()

    for _ in range(n_tta):
        # Augment nhẹ
        x_aug  = x0.clone()
        x_aug += torch.randn_like(x_aug) * 0.03
        x_aug *= (0.85 + torch.rand(1,1,1) * 0.30)
        with torch.no_grad():
            probs += torch.sigmoid(model(x_aug)).squeeze().cpu().numpy()

    probs /= (n_tta + 1)
    preds  = (probs >= THRESHOLDS).astype(int)
    return probs, preds
```

### Bước 5: Chạy Notebook
| Notebook | Mục đích |
|----------|---------|
| `ecg-huong4-v4.ipynb` | Train model V4 từ đầu |
| `ecg-inference-v4.ipynb` | Inference — upload file .mat+.hea |
| `ecg-eval-v4-pro.ipynb` | Đánh giá toàn diện + tìm ngưỡng |

---

## 27 NHÃN BỆNH TIM (27 CARDIAC CONDITIONS)

| Idx | Viết tắt | Tên đầy đủ | Tiếng Việt | F1 |
|-----|----------|------------|------------|-----|
| 0 | IAVB | 1st Degree AV Block | Blốc nhĩ thất độ 1 | 0.663 |
| 1 | AF | Atrial Fibrillation | Rung nhĩ ⚠️ | 0.869 |
| 2 | AFL | Atrial Flutter | Cuồng nhĩ ⚠️ | 0.565 |
| 3 | Brady | Bradycardia | Nhịp tim chậm | 0.431 |
| 4 | CRBBB | Complete RBBB | Blốc nhánh phải hoàn toàn | 0.686 |
| 5 | IRBBB | Incomplete RBBB | Blốc nhánh phải không hoàn toàn | 0.574 |
| 6 | LAnFB | Left Anterior Fascicular Block | Blốc phân nhánh trái trước | 0.732 |
| 7 | LAD | Left Axis Deviation | Lệch trục trái | 0.737 |
| 8 | LBBB | Left Bundle Branch Block | Blốc nhánh trái ⚠️ | 0.831 |
| 9 | LQRSV | Low QRS Voltages | Điện thế QRS thấp | 0.222 |
| 10 | NSIVCB | Nonspecific IVCB | Rối loạn dẫn truyền trong thất | 0.265 |
| 11 | PR | PR Interval | Khoảng PR | 0.973 |
| 12 | PAC | Premature Atrial Contraction | Ngoại tâm thu nhĩ | 0.329 |
| 13 | PVC | Premature Ventricular Contraction | Ngoại tâm thu thất ⚠️ | 0.182 |
| 14 | LPR | Prolonged PR Interval | Khoảng PR kéo dài | 0.400 |
| 15 | LQT | Prolonged QT Interval | Khoảng QT kéo dài ⚠️ | 0.453 |
| 16 | QAb | Q Wave Abnormal | Sóng Q bất thường | 0.311 |
| 17 | RAD | Right Axis Deviation | Lệch trục phải | 0.634 |
| 18 | RBBB | Right Bundle Branch Block | Blốc nhánh phải | 0.820 |
| 19 | SA | Sinus Arrhythmia | Rối loạn nhịp xoang | 0.127 |
| 20 | SB | Sinus Bradycardia | Nhịp chậm xoang | 0.672 |
| 21 | SNR | Sinus Normal Rhythm | Nhịp xoang bình thường | 0.895 |
| 22 | STach | Sinus Tachycardia | Nhịp nhanh xoang | 0.816 |
| 23 | SVPB | Supraventricular Premature Beats | Nhịp sớm trên thất | 0.269 |
| 24 | TAb | T-wave Abnormal | Sóng T bất thường | 0.534 |
| 25 | TInv | T-wave Inversion | Sóng T đảo ngược | 0.238 |
| 26 | VPB | Ventricular Premature Beats | Nhịp sớm thất ⚠️ | 0.290 |

> ⚠️ = Bệnh nguy hiểm, áp dụng Recall constraint ≥ 0.75

---

**Thông tin liên hệ:**
- **Thực hiện bởi:**Huỳnh Trọng Ngữ
- **Phiên bản:** V4.0 — Multi-scale CNN + Transformer (20M params)
- **Ngày hoàn thành:** Tháng 3, 2026
