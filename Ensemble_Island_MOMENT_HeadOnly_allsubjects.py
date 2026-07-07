"""
島研モデル（②KDアンサンブル、ensemble教師）× MOMENT（HeadOnly教師）の予測アンサンブル
それぞれ独立に学習済みの教師モデルをロードし、Arousal/Valenceの予測値を平均する。
新規学習は行わない（基盤モデルの構築・動作確認フェーズ）。
"""
import sys
sys.path.insert(0, '/mnt/learn/usr/hayashi/引継ぎ/プログラム/EmotionRecognition')

import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import torch.nn as nn
from scipy.signal import resample
from momentfm import MOMENTPipeline

from Algorithms.Models.EnsembleFeaturesModel import SingleModel
from Conf.Settings import N_CLASS, ECG_N, PPG_N

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MOMENT_LEN = 512

# ---- パス ----
FT_RAW_PATH        = '/mnt/learn/usr/hayashi/引継ぎ/実験データ/2025飲食実験/計測結果/raw_waveform'
FT_FEATURES_PATH   = '/mnt/learn/usr/hayashi/引継ぎ/実験データ/2025飲食実験/計測結果/features'
FT_CSV             = os.path.join(FT_FEATURES_PATH, 'dataset', 'all_data_filtered_step30s.csv')

KD_RESULT_PATH     = '/mnt/learn/usr/hayashi/引継ぎ/result/KD_ECG_PPG/'
TEACHER2_STAT_PATH = '/mnt/learn/usr/hayashi/引継ぎ/result/FT_KD_ensemble_allsubjects/'
TEACHER2_CKPT      = '/mnt/learn/usr/hayashi/引継ぎ/result/FT_KD_ensemble_allsubjects/model_FT_KD_ensemble'
MOMENT_TEACHER_PATH = '/mnt/learn/usr/hayashi/引継ぎ/result/teacher_models/MOMENT_HeadOnly_allsubjects/final_model.pt'


# =====================================================================
# 島研側（TensorFlow）：5fold分のKDバックボーン + 教師reg_head
# =====================================================================
class IslandRegressionHead(tf.keras.Model):
    def __init__(self, hidden_units=64, **kwargs):
        super().__init__(**kwargs)
        self.hidden  = tf.keras.layers.Dense(hidden_units, activation='elu', name='mlp_layer')
        self.out_ar  = tf.keras.layers.Dense(1, name='out_ar')
        self.out_val = tf.keras.layers.Dense(1, name='out_val')

    def call(self, z, training=None):
        h = self.hidden(z)
        return self.out_ar(h), self.out_val(h)


mean_ecg = np.load(os.path.join(TEACHER2_STAT_PATH, 'mean_ecg.npy'))
std_ecg  = np.load(os.path.join(TEACHER2_STAT_PATH, 'std_ecg.npy'))
mean_ppg = np.load(os.path.join(TEACHER2_STAT_PATH, 'mean_ppg.npy'))
std_ppg  = np.load(os.path.join(TEACHER2_STAT_PATH, 'std_ppg.npy'))

island_backbones = []
for fold in range(1, 6):
    ckpt_prefix = os.path.join(KD_RESULT_PATH, f'fold_{fold}', 'model_student_ECG_PPG_KD')
    m = SingleModel(num_output=N_CLASS).loadBaseModel(ckpt_prefix)
    m.trainable = False
    island_backbones.append(m)

island_reg_head = IslandRegressionHead(hidden_units=64, name='island_reg_head')
_ = island_reg_head(tf.zeros([1, 32]))
island_ckpt = tf.train.Checkpoint(step=tf.Variable(1), reg_head=island_reg_head)
island_ckpt.restore(tf.train.latest_checkpoint(TEACHER2_CKPT)).expect_partial()
island_reg_head.trainable = False
print('島研モデル（②KDアンサンブル教師）をロードしました')


def predict_island(x_tea_ecg_ppg):
    """x_tea_ecg_ppg: (B, ECG_N+PPG_N) の正規化済みECG+PPG特徴量"""
    embeddings = []
    for m in island_backbones:
        _, _, _, z = m(x_tea_ecg_ppg, training=False)
        embeddings.append(z)
    avg_z = tf.reduce_mean(tf.stack(embeddings, axis=0), axis=0)
    ar, val = island_reg_head(avg_z, training=False)
    return ar.numpy().flatten(), val.numpy().flatten()


# =====================================================================
# MOMENT側（PyTorch）：HeadOnly教師
# =====================================================================
class TeacherRegressionHead(nn.Module):
    def __init__(self, in_dim=1024, hidden1=256, hidden2=64):
        super().__init__()
        self.fc1  = nn.Linear(in_dim, hidden1)
        self.act1 = nn.ELU()
        self.drop = nn.Dropout(0.3)
        self.fc2  = nn.Linear(hidden1, hidden2)
        self.act2 = nn.ELU()
        self.out_ar  = nn.Linear(hidden2, 1)
        self.out_val = nn.Linear(hidden2, 1)

    def forward(self, z):
        h = self.act1(self.fc1(z))
        h = self.drop(h)
        h = self.act2(self.fc2(h))
        return self.out_ar(h).squeeze(-1), self.out_val(h).squeeze(-1)


moment_teacher = MOMENTPipeline.from_pretrained(
    'AutonLab/MOMENT-1-large',
    model_kwargs={'task_name': 'embedding'}
)
moment_teacher.init()
moment_teacher = moment_teacher.to(DEVICE)

moment_reg_head = TeacherRegressionHead().to(DEVICE)
ckpt = torch.load(MOMENT_TEACHER_PATH, map_location=DEVICE)
if hasattr(moment_teacher, 'model'):
    moment_teacher.model.load_state_dict(ckpt['moment_state_dict'])
else:
    moment_teacher.load_state_dict(ckpt['moment_state_dict'])
moment_reg_head.load_state_dict(ckpt['reg_head_state_dict'])
moment_teacher.eval()
moment_reg_head.eval()
for p in moment_teacher.parameters():
    p.requires_grad = False
for p in moment_reg_head.parameters():
    p.requires_grad = False
print('MOMENT（HeadOnly教師）をロードしました')


def predict_moment(x_tea_waveform):
    """x_tea_waveform: (B, 2, 512) のECG+PPG生波形（z-score済み）"""
    with torch.no_grad():
        out = moment_teacher(x_enc=x_tea_waveform.to(DEVICE))
        emb = out.embeddings
        if emb.dim() == 3:
            emb = emb.mean(dim=1)
        ar, val = moment_reg_head(emb)
    return ar.cpu().numpy(), val.cpu().numpy()


# =====================================================================
# データ読み込み（島研用の特徴量 + MOMENT用の生波形を両方用意）
# =====================================================================
def load_waveform(subject, modality, data_path):
    pattern = os.path.join(data_path, str(subject), modality,
                           f"filtered_{subject}_*_{modality}.csv")
    files = glob.glob(pattern)
    if not files:
        return None
    df = pd.read_csv(files[0])
    ts = pd.to_datetime(df['Timestamp'])
    t0 = ts.iloc[0].replace(hour=0, minute=0, second=0, microsecond=0)
    elapsed_sec = (ts - t0).dt.total_seconds().values.astype(np.float64)
    signal = df[modality].values.astype(np.float32)
    return elapsed_sec, signal


def extract_segment(elapsed_sec, signal, start_sec, end_sec):
    mask = (elapsed_sec >= start_sec) & (elapsed_sec < end_sec)
    seg = signal[mask]
    return seg if len(seg) >= 100 else None


def zscore(seg):
    mu, sigma = seg.mean(), seg.std()
    return (seg - mu) / (sigma + 1e-8)


print("全サンプルのデータを読み込み中...")
all_df = pd.read_csv(FT_CSV)
wf_cache = {}
samples = []

for i in range(len(all_df)):
    row = all_df.iloc[i]
    subject = str(int(row['Subject']))
    idx     = int(row['Idx'])
    start   = float(row['Start'])
    end     = float(row['End'])
    y_ar    = np.float32(row['Arousal'])
    y_val   = np.float32(row['Valence'])

    ecg_path = os.path.join(FT_FEATURES_PATH, subject, 'ECG', '30s', f'ecg_{idx}.npy')
    ppg_path = os.path.join(FT_FEATURES_PATH, subject, 'PPG', '30s', f'ppg_{idx}.npy')
    if not os.path.exists(ecg_path) or not os.path.exists(ppg_path):
        continue
    ecg_feat = (np.load(ecg_path).astype(np.float32) - mean_ecg) / std_ecg
    ppg_feat = (np.load(ppg_path).astype(np.float32) - mean_ppg) / std_ppg
    island_x = np.concatenate([ecg_feat, ppg_feat])

    if subject not in wf_cache:
        wf_cache[subject] = (
            load_waveform(subject, 'ECG', FT_RAW_PATH),
            load_waveform(subject, 'PPG', FT_RAW_PATH),
        )
    ecg_wf, ppg_wf = wf_cache[subject]
    if ecg_wf is None or ppg_wf is None:
        continue

    ecg_seg = extract_segment(ecg_wf[0], ecg_wf[1], start, end)
    ppg_seg = extract_segment(ppg_wf[0], ppg_wf[1], start, end)
    if ecg_seg is None or ppg_seg is None:
        continue

    ecg_res = zscore(resample(ecg_seg, MOMENT_LEN).astype(np.float32))
    ppg_res = zscore(resample(ppg_seg, MOMENT_LEN).astype(np.float32))
    moment_x = np.stack([ecg_res, ppg_res], axis=0)

    if (np.isnan(island_x).any() or np.isnan(moment_x).any()
            or np.isnan(y_ar) or np.isnan(y_val)):
        continue

    samples.append({
        'island_x': island_x,
        'moment_x': moment_x,
        'y_ar': y_ar,
        'y_val': y_val,
    })

print(f"有効サンプル数: {len(samples)}")


# =====================================================================
# 推論・アンサンブル評価（学習なし。参考値であり厳密な汎化性能ではない点に注意）
# =====================================================================
BATCH = 64
island_ar_all, island_val_all = [], []
moment_ar_all, moment_val_all = [], []
y_ar_all, y_val_all = [], []

for i in range(0, len(samples), BATCH):
    batch = samples[i:i + BATCH]
    island_x = tf.constant(np.array([s['island_x'] for s in batch]))
    moment_x = torch.tensor(np.array([s['moment_x'] for s in batch]))

    ar_i, val_i = predict_island(island_x)
    ar_m, val_m = predict_moment(moment_x)

    island_ar_all.append(ar_i);   island_val_all.append(val_i)
    moment_ar_all.append(ar_m);   moment_val_all.append(val_m)
    y_ar_all.extend([s['y_ar'] for s in batch])
    y_val_all.extend([s['y_val'] for s in batch])

island_ar  = np.concatenate(island_ar_all)
island_val = np.concatenate(island_val_all)
moment_ar  = np.concatenate(moment_ar_all)
moment_val = np.concatenate(moment_val_all)
y_ar  = np.array(y_ar_all)
y_val = np.array(y_val_all)

ensemble_ar  = (island_ar + moment_ar) / 2.0
ensemble_val = (island_val + moment_val) / 2.0


def rmse(pred, true):
    return float(np.sqrt(np.mean((pred - true) ** 2)))


print("\n=== 参考値（学習データそのものに対する予測。汎化性能ではない） ===")
print(f"島研単体      RMSE_ar={rmse(island_ar, y_ar):.4f}  RMSE_val={rmse(island_val, y_val):.4f}")
print(f"MOMENT単体    RMSE_ar={rmse(moment_ar, y_ar):.4f}  RMSE_val={rmse(moment_val, y_val):.4f}")
print(f"アンサンブル  RMSE_ar={rmse(ensemble_ar, y_ar):.4f}  RMSE_val={rmse(ensemble_val, y_val):.4f}")

# ---- 誤差の相関チェック：2モデルが同じ間違い方をしていないかを確認 ----
err_island_ar  = island_ar  - y_ar
err_moment_ar  = moment_ar  - y_ar
err_island_val = island_val - y_val
err_moment_val = moment_val - y_val

corr_ar  = float(np.corrcoef(err_island_ar,  err_moment_ar)[0, 1])
corr_val = float(np.corrcoef(err_island_val, err_moment_val)[0, 1])

print("\n=== 誤差の相関（低い/負の方がアンサンブルに向いている） ===")
print(f"Arousal誤差の相関:  {corr_ar:.4f}")
print(f"Valence誤差の相関:  {corr_val:.4f}")
print("(目安: 0.7以上は誤差がほぼ連動＝アンサンブルの効果は薄い可能性 / 0.4以下は違う間違い方をしている可能性が高い)")
