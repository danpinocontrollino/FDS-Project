"""
Quick retrain script (safe, fast): builds features, trains a RandomForest
and saves metrics + train/val/test arrays and feature list under results/.
Run: python scripts/retrain_quick.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier

RES = Path('results')
RES.mkdir(exist_ok=True)
LOG_PATH = RES / 'retrain_quick_log.txt'

CSV = Path('data/raw/daily_all.csv')
if not CSV.exists():
    LOG_PATH.write_text('Missing data/raw/daily_all.csv\n')
    raise SystemExit('Missing data/raw/daily_all.csv')

try:
    df = pd.read_csv(CSV)
    LOG_PATH.write_text(f'Loaded CSV rows={len(df)} cols={df.shape[1]}\n')
except Exception as e:
    LOG_PATH.write_text('Failed to read CSV:\n' + str(e) + '\n')
    raise
if 'ID' in df.columns:
    df = df.rename(columns={'ID':'user_id'})
if 'date' not in df.columns and 'Date' in df.columns:
    df = df.rename(columns={'Date':'date'})
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])

NUMERIC_COLS = [c for c in df.columns if df[c].dtype != 'object' and c not in ['user_id','date']][:50]
for col in NUMERIC_COLS:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.5).astype(float)
if 'user_id' not in df.columns:
    df['user_id'] = np.arange(len(df)) // 90
users = sorted(df['user_id'].unique())[:300]

df = df[df['user_id'].isin(users)].copy()

CATEGORICAL_COLS = [c for c in ['work_pressure','BMI Category','smoking','alcohol','profession'] if c in df.columns]
if CATEGORICAL_COLS:
    enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    enc_cols = [c for c in CATEGORICAL_COLS if c in df.columns]
    if enc_cols:
        enc_vals = enc.fit_transform(df[enc_cols])
        enc_df = pd.DataFrame(enc_vals, columns=enc.get_feature_names_out(enc_cols))
        df = pd.concat([df.reset_index(drop=True).drop(columns=enc_cols), enc_df.reset_index(drop=True)], axis=1)

avail_num = [c for c in NUMERIC_COLS if c in df.columns]
if avail_num:
    df[avail_num] = MinMaxScaler().fit_transform(df[avail_num])

WINDOW_LONG = 14

def calc(group):
    for col in avail_num:
        group[f'Avg_{col}_{WINDOW_LONG}d'] = group[col].rolling(window=WINDOW_LONG, min_periods=1).mean().shift(1)
    group['Stress_High'] = (group[avail_num[0]] > 0.8).astype(int) if avail_num else 0
    group[f'N_HighStress_{WINDOW_LONG}d'] = group['Stress_High'].rolling(window=WINDOW_LONG, min_periods=1).sum().shift(1)
    return group

df = df.groupby('user_id', group_keys=False).apply(calc)
RISK_THRESHOLD = 6
df['Burnout_Risk'] = (df.get(f'N_HighStress_{WINDOW_LONG}d', 0) >= RISK_THRESHOLD).astype(int)
obj = df.select_dtypes(include=['object']).columns.tolist()
if obj:
    df = df.drop(columns=obj)

FINAL_FEATURES = [c for c in df.columns if c not in ['date','user_id','Burnout_Risk','Stress_High']]

X=[]; y=[]
for uid,g in df.groupby('user_id'):
    gr=g.reset_index(drop=True)
    for i in range(WINDOW_LONG, len(gr)):
        last_row = gr[FINAL_FEATURES].iloc[i].values
        X.append(last_row)
        y.append(gr['Burnout_Risk'].iloc[i])
X = np.array(X)
y = np.array(y)
LOG_PATH.write_text(LOG_PATH.read_text() + f'Dataset shape: {X.shape} Positives: {int(y.sum())}\n')

if len(np.unique(y)) < 2:
    # write a metrics file indicating problem and exit gracefully
    (RES / 'metrics_retrained_rf.txt').write_text('Error: No positive labels to train on\n')
    LOG_PATH.write_text(LOG_PATH.read_text() + 'No positive labels to train on. Exiting.\n')
    raise SystemExit('No positive labels to train on')

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
rel = 0.15/0.85
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=rel, random_state=42, stratify=y_temp)

clf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)
LOG_PATH.write_text(LOG_PATH.read_text() + f'Training RF on {X_train.shape[0]} samples\n')
try:
    clf.fit(X_train, y_train)
except Exception as e:
    LOG_PATH.write_text(LOG_PATH.read_text() + 'RF fit failed:\n' + str(e) + '\n')
    (RES / 'metrics_retrained_rf.txt').write_text('Error: RF fit failed\n' + str(e) + '\n')
    raise

pv = clf.predict_proba(X_val)[:,1]
best_thr=0.5; best_f1_val=0
for thr in [i/100 for i in range(0,101)]:
    p=(pv>thr).astype(int)
    f=f1_score(y_val,p,zero_division=0)
    if f>best_f1_val:
        best_f1_val=f; best_thr=thr

pt = clf.predict_proba(X_test)[:,1]
preds = (pt>best_thr).astype(int)
acc = accuracy_score(y_test,preds)
prec = precision_score(y_test,preds,zero_division=0)
rec = recall_score(y_test,preds,zero_division=0)
f1 = f1_score(y_test,preds,zero_division=0)

with open(RES/'metrics_retrained_rf.txt','w') as f:
    f.write(f'Accuracy: {acc:.4f}\nPrecision: {prec:.4f}\nRecall: {rec:.4f}\nF1: {f1:.4f}\nBestValThreshold: {best_thr:.4f}\n')
LOG_PATH.write_text(LOG_PATH.read_text() + 'Wrote metrics_retrained_rf.txt\n')

np.save(RES/'X_train_rf.npy', X_train, allow_pickle=True)
np.save(RES/'y_train_rf.npy', y_train, allow_pickle=True)
np.save(RES/'X_val_rf.npy', X_val, allow_pickle=True)
np.save(RES/'y_val_rf.npy', y_val, allow_pickle=True)
np.save(RES/'X_test_rf.npy', X_test, allow_pickle=True)
np.save(RES/'y_test_rf.npy', y_test, allow_pickle=True)
with open(RES/'final_features_rf.txt','w') as f:
    f.write('\n'.join(FINAL_FEATURES))

print('Saved metrics to', RES/'metrics_retrained_rf.txt')
LOG_PATH.write_text(LOG_PATH.read_text() + 'Completed script successfully\n')
