"""
CardiacAI V4 — Flask Backend
Phân tích ECG 12 chuyển đạo, 27 bệnh tim mạch
Model: Multi-scale CNN + Transformer (20M params)
"""

import os, json, tempfile, io, base64, warnings, shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import wfdb
import torch
import torch.nn as nn
from flask import Flask, request, jsonify, send_from_directory

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════

NUM_LEADS    = 12
SEQ_LEN      = 4096
DROPOUT      = 0.15
MS_CHANNELS  = [128, 256, 512, 512]
TRANS_DIM    = 384
TRANS_HEADS  = 8
TRANS_LAYERS = 8
TRANS_FF_DIM = 1536
NUM_CLASSES  = 27

LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
              'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

IDX_TO_NAME = {
    0:'IAVB', 1:'AF', 2:'AFL', 3:'Brady', 4:'CRBBB', 5:'IRBBB',
    6:'LAnFB', 7:'LAD', 8:'LBBB', 9:'LQRSV', 10:'NSIVCB', 11:'PR',
    12:'PAC', 13:'PVC', 14:'LPR', 15:'LQT', 16:'QAb', 17:'RAD',
    18:'RBBB', 19:'SA', 20:'SB', 21:'SNR', 22:'STach', 23:'SVPB',
    24:'TAb', 25:'TInv', 26:'VPB'
}

IDX_TO_FULLNAME = {
    0:'1st Degree AV Block', 1:'Atrial Fibrillation', 2:'Atrial Flutter',
    3:'Bradycardia', 4:'Complete Right Bundle Branch Block',
    5:'Incomplete Right Bundle Branch Block', 6:'Left Anterior Fascicular Block',
    7:'Left Axis Deviation', 8:'Left Bundle Branch Block',
    9:'Low QRS Voltages', 10:'Nonspecific IV Conduction Disturbance',
    11:'PR Interval', 12:'Premature Atrial Contraction',
    13:'Premature Ventricular Contraction', 14:'Prolonged PR Interval',
    15:'Prolonged QT Interval', 16:'Q Wave Abnormal', 17:'Right Axis Deviation',
    18:'Right Bundle Branch Block', 19:'Sinus Arrhythmia', 20:'Sinus Bradycardia',
    21:'Sinus Rhythm', 22:'Sinus Tachycardia', 23:'Supraventricular Premature Beats',
    24:'T-wave Abnormal', 25:'T-wave Inversion', 26:'Ventricular Premature Beats'
}

IDX_TO_VIET = {
    0:'Bloc nhĩ thất độ 1', 1:'Rung nhĩ', 2:'Cuồng nhĩ',
    3:'Nhịp chậm', 4:'Bloc nhánh phải hoàn toàn',
    5:'Bloc nhánh phải không hoàn toàn', 6:'Bloc phân nhánh trái trước',
    7:'Lệch trục trái', 8:'Bloc nhánh trái', 9:'Điện thế QRS thấp',
    10:'Rối loạn dẫn truyền trong thất', 11:'Khoảng PR',
    12:'Ngoại tâm thu nhĩ', 13:'Ngoại tâm thu thất',
    14:'Khoảng PR kéo dài', 15:'Khoảng QT kéo dài',
    16:'Sóng Q bất thường', 17:'Lệch trục phải',
    18:'Bloc nhánh phải', 19:'Rối loạn nhịp xoang',
    20:'Nhịp chậm xoang', 21:'Nhịp xoang bình thường',
    22:'Nhịp nhanh xoang', 23:'Nhịp sớm trên thất',
    24:'Sóng T bất thường', 25:'Sóng T đảo ngược', 26:'Nhịp sớm thất'
}

IDX_TO_SEVERITY = {
    0:'warning', 1:'danger', 2:'danger', 3:'warning', 4:'warning', 5:'info',
    6:'info', 7:'info', 8:'danger', 9:'info', 10:'info', 11:'info',
    12:'warning', 13:'warning', 14:'warning', 15:'warning', 16:'warning',
    17:'info', 18:'warning', 19:'info', 20:'info', 21:'normal',
    22:'warning', 23:'warning', 24:'warning', 25:'warning', 26:'warning'
}

SENS_MAP = {
    'precise':   'thresholds/thr_youden_v4.json',
    'balanced':  'thresholds/thr_f1rec_v4.json',
    'sensitive': 'thresholds/thr_screening_v4.json',
}

SENS_DISPLAY = {
    'precise':  '🎯 Chính xác (Youden)',
    'balanced': '⚖️ Cân bằng (F1+Recall)',
    'sensitive':'🔍 Nhạy cao (Sàng lọc)',
}

# ══════════════════════════════════════════════════════════════════════
#  MODEL DEFINITION (copy from training notebook)
# ══════════════════════════════════════════════════════════════════════

class MultiScaleConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool=2):
        super().__init__()
        assert out_ch % 4 == 0
        b = out_ch // 4
        self.branch_small = nn.Sequential(
            nn.Conv1d(in_ch, b, 3,  padding=1,  bias=False),
            nn.GroupNorm(min(8, b), b), nn.GELU())
        self.branch_mid = nn.Sequential(
            nn.Conv1d(in_ch, b, 9,  padding=4,  bias=False),
            nn.GroupNorm(min(8, b), b), nn.GELU())
        self.branch_large = nn.Sequential(
            nn.Conv1d(in_ch, b, 19, padding=9,  bias=False),
            nn.GroupNorm(min(8, b), b), nn.GELU())
        self.branch_pool = nn.Sequential(
            nn.AvgPool1d(3, stride=1, padding=1),
            nn.Conv1d(in_ch, b, 1, bias=False),
            nn.GroupNorm(min(8, b), b), nn.GELU())
        self.fusion = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch), nn.GELU())
        self.pool = nn.MaxPool1d(pool)
        self.shortcut = (nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 1, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch))
            if in_ch != out_ch else nn.Identity())

    def forward(self, x):
        out = self.fusion(torch.cat([
            self.branch_small(x), self.branch_mid(x),
            self.branch_large(x), self.branch_pool(x)], dim=1))
        return self.pool(out + self.shortcut(x))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=DROPOUT):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class ECGModelV4(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        channels = [NUM_LEADS] + MS_CHANNELS
        self.cnn = nn.Sequential(*[
            MultiScaleConvBlock(channels[i], channels[i + 1])
            for i in range(len(MS_CHANNELS))])
        self.proj = nn.Sequential(
            nn.Linear(MS_CHANNELS[-1], TRANS_DIM),
            nn.LayerNorm(TRANS_DIM))
        self.pos_enc = PositionalEncoding(TRANS_DIM, max_len=1024)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=TRANS_DIM, nhead=TRANS_HEADS,
            dim_feedforward=TRANS_FF_DIM, dropout=DROPOUT,
            activation='gelu', batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(
            enc_layer, num_layers=TRANS_LAYERS)
        self.head = nn.Sequential(
            nn.LayerNorm(TRANS_DIM), nn.Dropout(DROPOUT),
            nn.Linear(TRANS_DIM, 512), nn.GELU(),
            nn.Dropout(DROPOUT / 2), nn.Linear(512, num_classes))

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        x = self.proj(x)
        x = self.pos_enc(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.head(x)


# ══════════════════════════════════════════════════════════════════════
#  LOAD MODEL & THRESHOLDS
# ══════════════════════════════════════════════════════════════════════

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'cardiac_v4_best.pth')

device = torch.device('cpu')

print('[*] Loading ECGModelV4...')
model = ECGModelV4().to(device)
ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model.load_state_dict(ckpt['state'])
model.eval()
n_params = sum(p.numel() for p in model.parameters())
print(f'    Model loaded: {n_params:,} params | epoch={ckpt["epoch"]} | F1={ckpt["score"]:.4f}')

print('[*] Loading thresholds...')
thresholds = {}
for mode, rel_path in SENS_MAP.items():
    full_path = os.path.join(BASE_DIR, rel_path)
    with open(full_path) as f:
        thr_dict = json.load(f)
    thresholds[mode] = np.array([thr_dict[str(i)] for i in range(NUM_CLASSES)],
                                dtype=np.float32)
    print(f'    {mode}: loaded ({full_path})')

print('[*] Ready!')

# ══════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════

def save_uploaded_pair(mat_file, hea_file):
    """Save uploaded .mat + .hea to a temp directory. Returns (tmp_dir, record_path)."""
    tmp_dir = tempfile.mkdtemp(prefix='ecg_')
    mat_name = mat_file.filename
    hea_name = hea_file.filename
    record_base = os.path.splitext(mat_name)[0]

    mat_path = os.path.join(tmp_dir, mat_name)
    hea_path = os.path.join(tmp_dir, hea_name)
    mat_file.save(mat_path)
    hea_file.save(hea_path)

    record_path = os.path.join(tmp_dir, record_base)
    return tmp_dir, record_path


def load_ecg(record_path):
    """Load and preprocess ECG signal. Returns (sig_raw, sig_crop, fs)."""
    rec = wfdb.rdrecord(record_path)
    sig = rec.p_signal.astype(np.float32)
    fs = rec.fs

    # Z-score normalization per lead
    mu = sig.mean(0, keepdims=True)
    std = sig.std(0, keepdims=True) + 1e-8
    sig_norm = (sig - mu) / std

    # Crop/pad to SEQ_LEN
    T = sig_norm.shape[0]
    if T >= SEQ_LEN:
        start = (T - SEQ_LEN) // 2
        sig_crop = sig_norm[start:start + SEQ_LEN]
    else:
        sig_crop = np.vstack([
            sig_norm,
            np.zeros((SEQ_LEN - T, NUM_LEADS), dtype=np.float32)
        ])

    return sig, sig_crop, fs


def predict_tta(sig_crop, n_tta=3):
    """Run inference with test-time augmentation. Returns probabilities array (27,)."""
    x0 = torch.from_numpy(sig_crop.T).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.sigmoid(model(x0)).squeeze().cpu().numpy()

    for _ in range(n_tta):
        aug = sig_crop.copy()
        aug += np.random.normal(0, 0.03, aug.shape).astype(np.float32)
        aug *= np.random.uniform(0.85, 1.15, (1, NUM_LEADS)).astype(np.float32)
        x_aug = torch.from_numpy(aug.T).unsqueeze(0).to(device)
        with torch.no_grad():
            probs += torch.sigmoid(model(x_aug)).squeeze().cpu().numpy()

    probs /= (n_tta + 1)
    return probs


def generate_ecg_plot(sig_raw, fs=500, title='ECG 12 chuyển đạo'):
    """Generate 12-lead ECG plot, returns base64-encoded PNG."""
    n_leads = min(sig_raw.shape[1], 12)
    max_samples = min(sig_raw.shape[0], int(fs * 10))
    t = np.arange(max_samples) / fs
    sig_plot = sig_raw[:max_samples]

    fig, axes = plt.subplots(n_leads, 1, figsize=(18, n_leads * 1.2),
                             facecolor='#0f172a')
    fig.suptitle(title, color='white', fontsize=13, fontweight='bold')

    for i in range(n_leads):
        ax = axes[i]
        ax.set_facecolor('#0f172a')
        ax.plot(t, sig_plot[:, i], color='#22d3ee', linewidth=0.7, alpha=0.9)
        ax.set_ylabel(LEAD_NAMES[i], color='#94a3b8', fontsize=8,
                      rotation=0, labelpad=22, va='center')
        ax.set_xlim(t[0], t[-1])
        ax.tick_params(colors='#475569', labelsize=7)
        ax.grid(True, alpha=0.12, color='#334155', linewidth=0.4)
        for sp in ax.spines.values():
            sp.set_color('#1e293b')

    axes[-1].set_xlabel('Thời gian (s)', color='#94a3b8', fontsize=9)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                facecolor='#0f172a')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


# ══════════════════════════════════════════════════════════════════════
#  FLASK APP
# ══════════════════════════════════════════════════════════════════════

app = Flask(__name__, static_folder='.', static_url_path='')


@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Main inference endpoint. Accepts .mat + .hea + sensitivity mode."""
    mat_file = request.files.get('mat_file')
    hea_file = request.files.get('hea_file')

    if not mat_file or not hea_file:
        return jsonify({'error': 'Cần upload cả 2 file .mat và .hea'}), 400

    mode = request.form.get('sensitivity', 'balanced')
    if mode not in thresholds:
        mode = 'balanced'

    thr = thresholds[mode]
    tmp_dir = None

    try:
        tmp_dir, record_path = save_uploaded_pair(mat_file, hea_file)
        sig_raw, sig_crop, fs = load_ecg(record_path)
        probs = predict_tta(sig_crop, n_tta=3)

        # Build ECG plot
        record_name = os.path.splitext(mat_file.filename)[0]
        ecg_plot_b64 = generate_ecg_plot(sig_raw, fs, title=f'ECG — {record_name}')

        results = []
        for i in range(NUM_CLASSES):
            results.append({
                'index': i,
                'abbreviation': IDX_TO_NAME[i],
                'label_en': IDX_TO_FULLNAME[i],
                'label_vi': IDX_TO_VIET[i],
                'prob': round(float(probs[i]), 6),
                'threshold': round(float(thr[i]), 4),
                'positive': bool(probs[i] >= thr[i]),
                'severity': IDX_TO_SEVERITY[i],
            })

        return jsonify({
            'success': True,
            'sensitivity_mode': mode,
            'sensitivity_name': SENS_DISPLAY[mode],
            'record_name': record_name,
            'fs': int(fs),
            'n_tta': 3,
            'ecg_plot': f'data:image/png;base64,{ecg_plot_b64}',
            'results': results,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    finally:
        if tmp_dir and os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)


@app.route('/ecg-plot', methods=['POST'])
def ecg_plot():
    """Generate ECG 12-lead plot from uploaded files."""
    mat_file = request.files.get('mat_file')
    hea_file = request.files.get('hea_file')

    if not mat_file or not hea_file:
        return jsonify({'error': 'Cần upload cả 2 file .mat và .hea'}), 400

    tmp_dir = None
    try:
        tmp_dir, record_path = save_uploaded_pair(mat_file, hea_file)
        sig_raw, sig_crop, fs = load_ecg(record_path)
        record_name = os.path.splitext(mat_file.filename)[0]
        ecg_plot_b64 = generate_ecg_plot(sig_raw, fs, title=f'ECG — {record_name}')

        return jsonify({
            'success': True,
            'ecg_plot': f'data:image/png;base64,{ecg_plot_b64}',
            'fs': int(fs),
            'duration_sec': round(sig_raw.shape[0] / fs, 2),
            'n_samples': sig_raw.shape[0],
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    finally:
        if tmp_dir and os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print(f'\n  CardiacAI V4 Backend')
    print(f'  Model: {n_params:,} params')
    print(f'  Thresholds: {list(thresholds.keys())}')
    print(f'  http://localhost:7860\n')
    app.run(host='0.0.0.0', port=7860)
