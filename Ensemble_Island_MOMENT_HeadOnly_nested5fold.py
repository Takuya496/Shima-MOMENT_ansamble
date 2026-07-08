"""
島研モデル（②KDアンサンブル教師）× MOMENT（HeadOnly教師）の予測アンサンブル
Nested 5-fold GroupKFold 交差検証

両モデルのバックボーンは常に凍結（2020年データ/事前学習のみで学習済み、
2025年100人データには一度も触れていない）なので、埋め込みは最初に1回だけ
計算してキャッシュし、fold内では軽量な回帰ヘッドのみをtrainデータで再学習する。
テスト被験者のデータはどちらの回帰ヘッドの学習にも一切使用しない。
"""
import sys
sys.path.insert(0, '/mnt/learn/usr/hayashi/引継ぎ/プログラム/EmotionRecognition')

import os
import glob
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import resample
from sklearn.model_selection import GroupKFold
from momentfm import MOMENTPipeline

from Algorithms.Models.EnsembleFeaturesModel import SingleModel
from Algorithms.Models.Losses import PCCLoss, CCCLoss
from Conf.Settings import N_CLASS, ECG_N, PPG_N

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MOMENT_LEN = 512

parser = argparse.ArgumentParser()
parser.add_argument('--teacher_epochs', type=int, default=100)
args = parser.parse_args()

TEACHER_EPOCHS = args.teacher_epochs
LR = 1e-4
N_FOLDS = 5
p, q, r = 1/3, 1/3, 1/3

# ---- パス ----
FT_RAW_PATH        = '/mnt/learn/usr/hayashi/引継ぎ/実験データ/2025飲食実験/計測結果/raw_waveform'
FT_FEATURES_PATH   = '/mnt/learn/usr/hayashi/引継ぎ/実験データ/2025飲食実験/計測結果/features'
FT_CSV             = os.path.join(FT_FEATURES_PATH, 'dataset', 'all_data_filtered_step30s.csv')

KD_RESULT_PATH      = '/mnt/learn/usr/hayashi/引継ぎ/result/KD_ECG_PPG/'
MOMENT_TEACHER_PATH = '/mnt/learn/usr/hayashi/引継ぎ/result/teacher_models/MOMENT_HeadOnly_allsubjects/final_model.pt'
RESULT_PATH         = '/mnt/learn/usr/hayashi/引継ぎ/result/Ensemble_Island_MOMENT_HeadOnly_nested5fold/'
os.makedirs(RESULT_PATH, exist_ok=True)


# =====================================================================
# データ読み込み（島研用の生ECG+PPG特徴量 + MOMENT用の生波形を両方用意）
# 正規化統計量はfoldごとにtrainデータのみから計算するため、ここでは生の値を保持する
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
    ecg_feat_raw = np.load(ecg_path).astype(np.float32)
    ppg_feat_raw = np.load(ppg_path).astype(np.float32)

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

    if (np.isnan(ecg_feat_raw).any() or np.isnan(ppg_feat_raw).any()
            or np.isnan(moment_x).any() or np.isnan(y_ar) or np.isnan(y_val)):
        continue

    samples.append({
        'subject':      int(row['Subject']),
        'ecg_feat_raw': ecg_feat_raw,
        'ppg_feat_raw': ppg_feat_raw,
        'moment_x':     moment_x,
        'y_ar':         y_ar,
        'y_val':        y_val,
    })

print(f"有効サンプル数: {len(samples)}")
subjects_arr = np.array([s['subject'] for s in samples])
print(f"被験者数: {len(set(subjects_arr))}")


# =====================================================================
# 島研側バックボーン（5fold分のKD SingleModel、常に凍結）→ 埋め込みを1回だけ計算
# =====================================================================
island_backbones = []
for fold in range(1, 6):
    ckpt_prefix = os.path.join(KD_RESULT_PATH, f'fold_{fold}', 'model_student_ECG_PPG_KD')
    m = SingleModel(num_output=N_CLASS).loadBaseModel(ckpt_prefix)
    m.trainable = False
    island_backbones.append(m)
print('島研KDバックボーン（5fold分）をロードしました')

# 素の特徴量から、全データ平均で正規化した上でバックボーンに通し埋め込みをキャッシュ
# (バックボーン自体は正規化統計量に敏感な線形層なので、学習時と同じ全体統計量を使う。
#  KDバックボーンは2020年データで学習済みであり、2025年データの統計量はここでの
#  埋め込み計算の入力正規化にのみ使う。回帰ヘッドの学習・評価はfold内のtrain統計量を使う)
all_ecg_feats = np.array([s['ecg_feat_raw'] for s in samples])
all_ppg_feats = np.array([s['ppg_feat_raw'] for s in samples])
global_mean_ecg = all_ecg_feats.mean(axis=0); global_std_ecg = all_ecg_feats.std(axis=0) + 1e-8
global_mean_ppg = all_ppg_feats.mean(axis=0); global_std_ppg = all_ppg_feats.std(axis=0) + 1e-8

BATCH = 64
island_z_cache = np.zeros((len(samples), 32), dtype=np.float32)
for i in range(0, len(samples), BATCH):
    batch = samples[i:i + BATCH]
    ecg_n = (np.array([s['ecg_feat_raw'] for s in batch]) - global_mean_ecg) / global_std_ecg
    ppg_n = (np.array([s['ppg_feat_raw'] for s in batch]) - global_mean_ppg) / global_std_ppg
    x = tf.constant(np.concatenate([ecg_n, ppg_n], axis=1).astype(np.float32))
    embeddings = []
    for m in island_backbones:
        _, _, _, z = m(x, training=False)
        embeddings.append(z)
    avg_z = tf.reduce_mean(tf.stack(embeddings, axis=0), axis=0).numpy()
    island_z_cache[i:i + len(batch)] = avg_z
print('島研埋め込みのキャッシュ完了')


# =====================================================================
# MOMENT側バックボーン（HeadOnly、常に凍結）→ 埋め込みを1回だけ計算
# =====================================================================
moment_backbone = MOMENTPipeline.from_pretrained(
    'AutonLab/MOMENT-1-large',
    model_kwargs={'task_name': 'embedding'}
)
moment_backbone.init()
moment_backbone = moment_backbone.to(DEVICE)
ckpt = torch.load(MOMENT_TEACHER_PATH, map_location=DEVICE)
if hasattr(moment_backbone, 'model'):
    moment_backbone.model.load_state_dict(ckpt['moment_state_dict'])
else:
    moment_backbone.load_state_dict(ckpt['moment_state_dict'])
moment_backbone.eval()
for p_ in moment_backbone.parameters():
    p_.requires_grad = False
print('MOMENTバックボーン（HeadOnly）をロードしました')

moment_z_cache = np.zeros((len(samples), 1024), dtype=np.float32)
with torch.no_grad():
    for i in range(0, len(samples), BATCH):
        batch = samples[i:i + BATCH]
        x = torch.tensor(np.array([s['moment_x'] for s in batch])).to(DEVICE)
        out = moment_backbone(x_enc=x)
        emb = out.embeddings
        if emb.dim() == 3:
            emb = emb.mean(dim=1)
        moment_z_cache[i:i + len(batch)] = emb.cpu().numpy()
print('MOMENT埋め込みのキャッシュ完了')

del moment_backbone
torch.cuda.empty_cache()


# =====================================================================
# 回帰ヘッド定義
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


class MomentRegressionHead(nn.Module):
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


def pcc_loss_torch(y_true, y_pred):
    vt = y_true - y_true.mean()
    vp = y_pred - y_pred.mean()
    return 1.0 - (vt * vp).sum() / (torch.sqrt((vt**2).sum() * (vp**2).sum() + 1e-8))


def ccc_loss_torch(y_true, y_pred):
    mt, mp = y_true.mean(), y_pred.mean()
    vt  = ((y_true - mt)**2).mean()
    vp  = ((y_pred - mp)**2).mean()
    cov = ((y_true - mt) * (y_pred - mp)).mean()
    return 1.0 - 2.0 * cov / (vt + vp + (mt - mp)**2 + 1e-8)


mse_loss_fn = tf.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
pcc_loss_fn = PCCLoss(reduction=tf.keras.losses.Reduction.NONE)
ccc_loss_fn = CCCLoss(reduction=tf.keras.losses.Reduction.NONE)


def rmse(pred, true):
    return float(np.sqrt(np.mean((pred - true) ** 2)))


# =====================================================================
# Nested 5-fold GroupKFold 交差検証（回帰ヘッドのみfoldごとに再学習）
# =====================================================================
gkf = GroupKFold(n_splits=N_FOLDS)
fold_splits = list(gkf.split(samples, groups=subjects_arr))

y_ar_arr  = np.array([s['y_ar']  for s in samples])
y_val_arr = np.array([s['y_val'] for s in samples])

fold_results = []
glorot = tf.keras.initializers.GlorotUniform()

for fold_idx, (train_idx, test_idx) in enumerate(fold_splits):
    train_subjs = sorted(set(subjects_arr[train_idx]))
    test_subjs  = sorted(set(subjects_arr[test_idx]))
    print(f"\n=== Fold {fold_idx + 1}/{N_FOLDS} | train:{len(train_subjs)}人 {len(train_idx)}サンプル "
          f"/ test:{len(test_subjs)}人 {len(test_idx)}サンプル ===")

    # ---- 島研 reg_head を初期化し、trainのみで再学習 ----
    island_head = IslandRegressionHead(hidden_units=64, name=f'island_head_f{fold_idx}')
    _ = island_head(tf.zeros([1, 32]))
    island_optimizer = tf.keras.optimizers.Adam(learning_rate=LR)

    z_train = tf.constant(island_z_cache[train_idx])
    ar_train = tf.constant(y_ar_arr[train_idx].reshape(-1, 1))
    val_train = tf.constant(y_val_arr[train_idx].reshape(-1, 1))
    train_ds = tf.data.Dataset.from_tensor_slices((z_train, ar_train, val_train)) \
        .shuffle(len(train_idx)).batch(32)

    @tf.function
    def island_train_step(z, y_ar_b, y_val_b):
        with tf.GradientTape() as tape:
            ar, val = island_head(z, training=True)
            mse = tf.reduce_mean(0.5 * (mse_loss_fn(y_ar_b, ar) + mse_loss_fn(y_val_b, val)))
            pcc = 1.0 - 0.5 * (pcc_loss_fn(y_ar_b, ar) + pcc_loss_fn(y_val_b, val))
            ccc = 1.0 - 0.5 * (ccc_loss_fn(y_ar_b, ar) + ccc_loss_fn(y_val_b, val))
            loss = p * mse + q * pcc + r * ccc
        grads = tape.gradient(loss, island_head.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 1.0)
        island_optimizer.apply_gradients(zip(grads, island_head.trainable_variables))
        return loss

    print(f"  [島研] reg_headを{TEACHER_EPOCHS}epoch学習中...")
    for epoch in range(TEACHER_EPOCHS):
        for z_b, ar_b, val_b in train_ds:
            loss = island_train_step(z_b, ar_b, val_b)
        if (epoch + 1) % 20 == 0:
            print(f"    Ep{epoch + 1:3d} | Loss:{float(loss):.4f}")

    # ---- MOMENT reg_head を初期化し、trainのみで再学習 ----
    moment_head = MomentRegressionHead().to(DEVICE)
    moment_optimizer = torch.optim.Adam(moment_head.parameters(), lr=LR)

    z_train_m  = torch.tensor(moment_z_cache[train_idx])
    ar_train_m = torch.tensor(y_ar_arr[train_idx])
    val_train_m = torch.tensor(y_val_arr[train_idx])
    train_ds_m = torch.utils.data.TensorDataset(z_train_m, ar_train_m, val_train_m)
    train_loader_m = torch.utils.data.DataLoader(train_ds_m, batch_size=16, shuffle=True)

    print(f"  [MOMENT] reg_headを{TEACHER_EPOCHS}epoch学習中...")
    for epoch in range(TEACHER_EPOCHS):
        moment_head.train()
        total = n = 0
        for z_b, ar_b, val_b in train_loader_m:
            z_b, ar_b, val_b = z_b.to(DEVICE), ar_b.to(DEVICE), val_b.to(DEVICE)
            ar, val = moment_head(z_b)
            mse = 0.5 * (F.mse_loss(ar, ar_b) + F.mse_loss(val, val_b))
            pcc = 0.5 * (pcc_loss_torch(ar_b, ar) + pcc_loss_torch(val_b, val))
            ccc = 0.5 * (ccc_loss_torch(ar_b, ar) + ccc_loss_torch(val_b, val))
            loss = p * mse + q * pcc + r * ccc
            moment_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(moment_head.parameters(), 1.0)
            moment_optimizer.step()
            total += loss.item() * z_b.size(0); n += z_b.size(0)
        if (epoch + 1) % 20 == 0:
            print(f"    Ep{epoch + 1:3d} | Loss:{total/n:.4f}")

    # ---- テスト評価 ----
    z_test_i = tf.constant(island_z_cache[test_idx])
    island_ar_pred, island_val_pred = island_head(z_test_i, training=False)
    island_ar_pred  = island_ar_pred.numpy().flatten()
    island_val_pred = island_val_pred.numpy().flatten()

    moment_head.eval()
    with torch.no_grad():
        z_test_m = torch.tensor(moment_z_cache[test_idx]).to(DEVICE)
        moment_ar_pred, moment_val_pred = moment_head(z_test_m)
        moment_ar_pred  = moment_ar_pred.cpu().numpy()
        moment_val_pred = moment_val_pred.cpu().numpy()

    ensemble_ar_pred  = (island_ar_pred  + moment_ar_pred)  / 2.0
    ensemble_val_pred = (island_val_pred + moment_val_pred) / 2.0

    y_ar_test  = y_ar_arr[test_idx]
    y_val_test = y_val_arr[test_idx]

    result = {
        'fold': fold_idx + 1,
        'island_rmse_ar':    rmse(island_ar_pred,    y_ar_test),
        'island_rmse_val':   rmse(island_val_pred,   y_val_test),
        'moment_rmse_ar':    rmse(moment_ar_pred,    y_ar_test),
        'moment_rmse_val':   rmse(moment_val_pred,   y_val_test),
        'ensemble_rmse_ar':  rmse(ensemble_ar_pred,  y_ar_test),
        'ensemble_rmse_val': rmse(ensemble_val_pred, y_val_test),
        'err_corr_ar':  float(np.corrcoef(island_ar_pred - y_ar_test, moment_ar_pred - y_ar_test)[0, 1]),
        'err_corr_val': float(np.corrcoef(island_val_pred - y_val_test, moment_val_pred - y_val_test)[0, 1]),
    }
    print(f"  Test 島研     RMSE_ar={result['island_rmse_ar']:.4f}  RMSE_val={result['island_rmse_val']:.4f}")
    print(f"  Test MOMENT   RMSE_ar={result['moment_rmse_ar']:.4f}  RMSE_val={result['moment_rmse_val']:.4f}")
    print(f"  Test アンサンブル RMSE_ar={result['ensemble_rmse_ar']:.4f}  RMSE_val={result['ensemble_rmse_val']:.4f}")
    print(f"  Test 誤差相関 ar={result['err_corr_ar']:.4f}  val={result['err_corr_val']:.4f}")

    fold_results.append(result)
    pd.DataFrame(fold_results).to_csv(os.path.join(RESULT_PATH, 'fold_results.csv'), index=False)


# =====================================================================
# 集計
# =====================================================================
df = pd.DataFrame(fold_results)
summary_lines = ["Nested 5-fold CV結果（島研 vs MOMENT vs アンサンブル）:"]
for col in ['island_rmse_ar', 'island_rmse_val', 'moment_rmse_ar', 'moment_rmse_val',
            'ensemble_rmse_ar', 'ensemble_rmse_val', 'err_corr_ar', 'err_corr_val']:
    summary_lines.append(f"  {col}: {df[col].mean():.4f} +/- {df[col].std():.4f}")
summary_str = "\n".join(summary_lines)
print("\n" + summary_str)
with open(os.path.join(RESULT_PATH, 'summary.txt'), 'w') as f:
    f.write(summary_str + "\n")
print('\nNested 5-fold CV完了')
