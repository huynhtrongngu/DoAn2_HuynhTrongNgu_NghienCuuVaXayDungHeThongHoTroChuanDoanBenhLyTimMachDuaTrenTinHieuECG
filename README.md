# HỆ THỐNG HỖ TRỢ CHẨN ĐOÁN BỆNH LÝ TIM MẠCH QUA TÍN HIỆU ECG 12 ĐẠO TRÌNH (AI)

> **Dự án Nghiên cứu & Phát triển:** Ứng dụng mạng Nơ-ron Tích chập 1 Chiều (1D-CNN / 1D-ResNet) để phân loại đa nhãn 5 nhóm bệnh lý tim mạch chính trên bộ dữ liệu chuẩn lâm sàng PTB-XL.

![Kaggle Kernel](https://img.shields.io/badge/Platform-Kaggle%20Kernels-blue?style=for-the-badge&logo=kaggle)
![Framework](https://img.shields.io/badge/Deep%20Learning-TensorFlow%20%2F%20Keras-orange?style=for-the-badge&logo=tensorflow)
![Dataset](https://img.shields.io/badge/Dataset-PTB--XL%2012--Lead%20ECG-green?style=for-the-badge&logo=physionet)
![Architecture](https://img.shields.io/badge/Model-1D--ResNet%20%2B%20Multi--label-red?style=for-the-badge&logo=keras)

---

## 📑 MỤC LỤC (TABLE OF CONTENTS)
1.  Báo cáo Tiến độ & Lộ trình (Progress & Roadmap)
2.  Giới thiệu & Tổng quan (Introduction)
3.  Phương pháp Nghiên cứu (Research Methodology)
4.  Kiến trúc Mô hình 12 Kênh Đề xuất (Proposed 12-Channel Architecture)
5.  So sánh Hiệu năng & Các dự án trước (Performance & Baseline Comparison)
6.  Hướng dẫn Thực hiện (Implementation Guide)

---

## 1. BÁO CÁO TIẾN ĐỘ & LỘ TRÌNH (PROGRESS & ROADMAP)

Bảng theo dõi trạng thái thực hiện dự án chi tiết từ khâu xử lý dữ liệu đến đóng gói hệ thống.

### Giai Đoạn 1: Tiền Xử Lý Dữ Liệu (Data Preprocessing)
**Trạng thái:** ✅ Đã hoàn thành (Done)
- [x] Khởi tạo môi trường Kaggle Notebook.
- [x] Đọc và hiểu cấu trúc nhãn chuẩn SCP-ECG của PTB-XL.
- [x] Chuyển đổi nhãn từ đa lớp (Multi-class) sang đa nhãn (Multi-label) cho 5 Superclass.
- [x] Trích xuất tín hiệu thô định dạng numpy array từ file `.dat`/`.hea`.
- [x] Thiết lập đầu vào dạng ma trận 12 kênh `(1000, 12)` với tần số lấy mẫu 100Hz.

### Giai Đoạn 2: Xây Dựng & Huấn Luyện Mô Hình (Modeling & Training)
**Trạng thái:** ⏳ Đang thực hiện (In Progress)
- [x] "Độ" lại kiến trúc Custom 1D-CNN để nhận Input 12 kênh đồng thời.
- [ ] Triển khai kiến trúc 1D-ResNet để tăng chiều sâu mạng mà không bị suy biến đạo hàm.
- [ ] Sử dụng hàm kích hoạt `Sigmoid` ở lớp Output (5 nơ-ron).
- [ ] Cấu hình hàm Loss: `BinaryCrossentropy` để xử lý bài toán đa nhãn.
- [ ] Áp dụng Callbacks: `ModelCheckpoint`, `ReduceLROnPlateau`, `EarlyStopping`.

### Giai Đoạn 3: Đánh Giá Hệ Thống (Evaluation)
**Trạng thái:** 🔴 Chưa bắt đầu (Pending)
- [ ] Dự đoán trên tập Test độc lập (Fold 10).
- [ ] Đánh giá bằng các chỉ số: Macro-AUC, Độ nhạy (Sensitivity/Recall), Độ đặc hiệu (Specificity).
- [ ] Vẽ biểu đồ ROC-AUC Curves cho từng nhóm bệnh lý.

### Giai Đoạn 4: Đóng Gói Ứng Dụng (Deployment)
**Trạng thái:** 🔴 Chưa bắt đầu (Pending)
- [ ] Export mô hình đã huấn luyện ra định dạng `.h5` hoặc `.onnx`.
- [ ] Xây dựng Backend API (FastAPI/Flask) để nhận file ECG từ người dùng.
- [ ] Tích hợp giao diện Frontend hiển thị cảnh báo lâm sàng.

---

## 2. GIỚI THIỆU & TỔNG QUAN (INTRODUCTION)

### 2.1. Bối cảnh
Điện tâm đồ (ECG) 12 đạo trình là công cụ tiêu chuẩn vàng, chi phí thấp trong chẩn đoán tim mạch. Tuy nhiên, việc phân tích sự thay đổi hình thái vi mô trên 12 kênh tín hiệu cùng lúc phụ thuộc rất nhiều vào kinh nghiệm của bác sĩ lâm sàng, dẫn đến sự thiếu nhất quán trong chẩn đoán ở các tuyến y tế cơ sở.

### 2.2. Mục tiêu Dự án
Dự án hướng tới việc xây dựng một hệ thống AI end-to-end có khả năng:
* **Phân tích 12 kênh đồng thời:** Tận dụng thông tin không gian từ 12 đạo trình thay vì chỉ 1 kênh duy nhất.
* **Chẩn đoán đa nhãn (Multi-label):** Xác định đồng thời 5 nhóm bệnh lý chính (Bình thường, Nhồi máu cơ tim, Thay đổi ST/T, Rối loạn dẫn truyền, Phì đại tâm thất), phù hợp với thực tế bệnh nhân có thể mắc nhiều hội chứng cùng lúc.

---

## 3. PHƯƠNG PHÁP NGHIÊN CỨU (RESEARCH METHODOLOGY)

### 3.1. Dữ liệu (Dataset)
Sử dụng bộ dữ liệu lâm sàng quy mô lớn **PTB-XL**.
* **Quy mô:** 21,837 bản ghi ECG kéo dài 10 giây từ 18,885 bệnh nhân.
* **Đặc điểm:** Tín hiệu 12 đạo trình (I, II, III, aVL, aVR, aVF, V1-V6).
* **Định dạng sử dụng:** Tần số lấy mẫu 100Hz (1000 điểm dữ liệu / 10 giây).

### 3.2. Quy trình Xử lý (Pipeline)
1.  **Lọc nhiễu (Denoising):** Áp dụng bộ lọc thông dải (Bandpass Filter) để loại bỏ nhiễu đường nền (baseline wander) và nhiễu điện lưới (powerline interference).
2.  **Chuẩn hóa:** Đưa các giá trị biên độ điện thế về cùng một thang đo bằng `StandardScaler`.
3.  **Chia tập dữ liệu:** Sử dụng kỹ thuật Stratified K-Fold (10 Folds) được cung cấp sẵn bởi PTB-XL. Dùng Folds 1-8 để Train, Fold 9 để Validate, và Fold 10 để Test.

---

## 4. KIẾN TRÚC MÔ HÌNH 12 KÊNH ĐỀ XUẤT (PROPOSED ARCHITECTURE)

Mô hình được chuyển đổi từ cấu trúc 1D-CNN cơ bản sang mạng **1D-ResNet** để xử lý bài toán 12 kênh.

### 4.1. Lớp Đầu Vào (Input Layer)
* **Input Shape:** `(1000, 12)` - Tương đương 1000 bước thời gian và 12 kênh đặc trưng tại mỗi bước.

### 4.2. Khối Trích Xuất Đặc Trưng (Feature Extraction Blocks)
* Thay vì dùng `Conv2D` như xử lý ảnh, dự án dùng các lớp `Conv1D` trượt dọc theo trục thời gian (Time-steps).
* **Residual Blocks:** Sử dụng các kết nối tắt (Skip Connections) cộng tín hiệu đầu vào của block với đầu ra của block để chống mất mát thông tin.
* **Batch Normalization & ReLU:** Chuẩn hóa dữ liệu sau mỗi lớp tích chập và áp dụng hàm kích hoạt phi tuyến.

### 4.3. Khối Phân Loại Đa Nhãn (Multi-label Classifier)
* **Global Average Pooling 1D:** Giảm ma trận về một vector đặc trưng duy nhất.
* **Output Layer:** Lớp Dense chứa **5 nơ-ron** (cho 5 Superclass).
* **Activation:** Bắt buộc sử dụng `Sigmoid` xuất ra xác suất độc lập (từ 0 đến 1) cho từng loại bệnh.

---

## 5. SO SÁNH HIỆU NĂNG & CÁC DỰ ÁN TRƯỚC (BASELINE COMPARISON)

Để đánh giá tính hiệu quả, dự án sẽ so sánh trực tiếp kết quả (dự kiến) với 2 công bố nghiên cứu quốc tế đã thực hiện trên cùng tập dữ liệu PTB-XL (sử dụng tín hiệu 100Hz, bài toán 5 Superclass).

| Tiêu chí | Dự án 1: Strodthoff et al. (2020) | Dự án 2: Smigiel et al. (2021) | **DỰ ÁN CỦA TÔI (TBD)** |
| :--- | :--- | :--- | :--- |
| **Phương pháp nghiên cứu** | Deep Learning thuần (xresnet1d101) <br> *(Nhận diện End-to-End)* | Học sâu lai (Inception1D + Entropy) <br> *(Trích xuất đặc trưng đa độ phân giải)* | **Custom 1D-ResNet** <br> *(Tinh chỉnh bộ lọc cho 12 kênh)* |
| **Độ chính xác (Accuracy)** | Không báo cáo chi tiết | ~ 86.40% | **TBD** |
| **Macro-AUC (Trung bình)** | **0.933** | 0.928 | **TBD** |
| **Độ nhạy (Sensitivity/Recall)** | ~ 79.5% | ~ 78.2% | **TBD** |
| **Độ đặc hiệu (Specificity)** | ~ 91.2% | ~ 90.5% | **TBD** |

> **Nhận xét:** Các nghiên cứu trước cho thấy mạng ResNet 1 chiều (Dự án 1) đang giữ mức AUC rất cao. Mục tiêu của dự án này là đạt ngưỡng AUC > 0.90 trên tập Test và tối ưu hóa độ nhạy (khả năng không bỏ sót bệnh) tiệm cận 80%.

---

## 6. HƯỚNG DẪN THỰC HIỆN TRÊN KAGGLE (IMPLEMENTATION GUIDE)

### Bước 1: Khởi tạo Đầu vào 12 Kênh (Model Architecture)
```python
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation, GlobalAveragePooling1D, Dense
from tensorflow.keras.models import Model

# Khai báo Input: 1000 time-steps, 12 channels
input_layer = Input(shape=(1000, 12))

# Block 1: Quét dọc 12 kênh đồng thời
x = Conv1D(filters=64, kernel_size=15, strides=1, padding='same')(input_layer)
x = BatchNormalization()(x)
x = Activation('relu')(x)

# ... (Thêm các Residual Blocks tại đây) ...

# Output Head cho Multi-label Classification
x = GlobalAveragePooling1D()(x)
output_layer = Dense(5, activation='sigmoid')(x)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
model.summary()
```
### Bước 2: Load Dữ Liệu Tín Hiệu Thô (X_train)
Lưu ý: Tập PTB-XL lưu tín hiệu gốc dưới dạng .dat và .hea. Cần thư viện wfdb để đọc và chuyển thành ma trận numpy 3 chiều.
```python
import pandas as pd
import numpy as np
import wfdb
import ast

# 1. Cài đặt thư viện (Chạy lệnh này trong cell riêng: !pip install wfdb)

# 2. Đọc file CSV chứa metadata
path = '/kaggle/input/ptb-xl-dataset/'
df = pd.read_csv(path + 'ptbxl_database.csv', index_col='ecg_id')
df.scp_codes = df.scp_codes.apply(lambda x: ast.literal_eval(x)) # Chuyển string thành dictionary

# 3. Hàm đọc tín hiệu ECG thô
def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path + f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path + f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data

# 4. Load dữ liệu tín hiệu 100Hz
X = load_raw_data(df, 100, path)

print(f"✅ Kích thước tập X: {X.shape}") 
# KẾT QUẢ KỲ VỌNG: (21837, 1000, 12)
```
### Bước 3: Xử Lý Nhãn Đa Lớp thành Đa Nhãn (Y_train)
Giải thích: Chuyển đổi mã bệnh chi tiết (SCP_codes) thành 5 nhóm bệnh chính (Superclass) và áp dụng One-Hot Encoding.
```python
from sklearn.preprocessing import MultiLabelBinarizer

# 1. Load bảng ánh xạ từ Subclass sang Superclass
agg_df = pd.read_csv(path + 'scp_statements.csv', index_col=0)
agg_df = agg_df[agg_df.diagnostic == 1]

# 2. Hàm gom nhóm nhãn
def aggregate_diagnostic(y_dic):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))

# 3. Áp dụng gom nhóm
df['diagnostic_superclass'] = df.scp_codes.apply(aggregate_diagnostic)

# 4. Chuyển đổi thành ma trận nhị phân (Multi-Hot Encoding) cho 5 Superclass
mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(df['diagnostic_superclass'])

print(f"✅ Kích thước tập Y: {Y.shape}") 
print(f"✅ Các lớp bệnh lý: {mlb.classes_}")
# KẾT QUẢ KỲ VỌNG: Kích thước (21837, 5). Classes: ['CD' 'HYP' 'MI' 'NORM' 'STTC']
```
---
**Thông tin Liên hệ:**
*   **Thực hiện bởi:** Huỳnh Trọng Ngữ
*   **Giáo Viên Hướng Dẫn:** (Thầy) Trần Văn Thiện
*   **Phiên bản:** 1.0.0
