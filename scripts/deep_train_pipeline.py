"""
End-to-end deep learning pipeline for Burnout_Risk.
- Preprocessing: robust numeric coercion, one-hot categorical, rolling features
- Target: multi-variable Burnout_Risk (14-day rolling counts)
- Sequence creation (WINDOW_LONG=14)
- Train a small LSTM with class_weight and EarlyStopping
- Saves: results/metrics_dl.txt, final_features_dl.txt, X/y arrays, models/burnout_dl_model.h5, results/deep_pipeline_log.txt

To run quickly for testing, this script uses a small user subset and few epochs by default.
Run: python scripts/deep_train_pipeline.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import traceback
import json

RES = Path('results')
MODELS = Path('models')
RES.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)
LOG = RES / 'deep_pipeline_log.txt'
METRICS = RES / 'metrics_dl.txt'
FEATURES_FILE = RES / 'final_features_dl.txt'

# Parameters (tune as needed)
CSV = Path('data/raw/daily_all.csv')
USERS_LIMIT = 120        # limit users for quick runs; set to None to use all
WINDOW_LONG = 14
EPOCHS = 4               # small for smoke test; increase for production
BATCH_SIZE = 64
RISK_THRESHOLD = 6
SEED = 42

def log(msg):
    with open(LOG, 'a') as f:
        f.write(str(msg) + '\n')

try:
    log('Starting deep pipeline')
    if not CSV.exists():
        log('Missing CSV: ' + str(CSV))
        raise SystemExit('Missing data file')

    df = pd.read_csv(CSV)
    log(f'Loaded CSV rows={len(df)} cols={df.shape[1]}')

    # normalize column names
    if 'ID' in df.columns and 'user_id' not in df.columns:
        df = df.rename(columns={'ID': 'user_id'})
    if 'date' not in df.columns and 'Date' in df.columns:
        df = df.rename(columns={'Date': 'date'})
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])

    # numeric coercion heuristic: try to coerce many columns
    candidate_cols = [c for c in df.columns if c not in ('user_id', 'date')]
    numeric_cols = []
    for c in candidate_cols:
        # attempt coercion
        coerced = pd.to_numeric(df[c], errors='coerce')
        nan_frac = coerced.isna().mean()
        # accept as numeric if <=50% NaNs after coercion
        if nan_frac <= 0.5:
            df[c] = coerced.fillna(coerced.median() if coerced.notna().any() else 0.5).astype(float)
            numeric_cols.append(c)
    log(f'Identified numeric cols: {len(numeric_cols)}')

    # categorical: force object dtype columns (small set)
    categorical_cols = [c for c in df.select_dtypes(include=['object']).columns if c not in ('date',)]
    if 'work_pressure' in df.columns and 'work_pressure' not in categorical_cols:
        categorical_cols.append('work_pressure')
    log(f'Categorical cols: {categorical_cols}')

    # limit users for quick test
    if 'user_id' not in df.columns:
        df['user_id'] = np.arange(len(df)) // 90
    users = sorted(df['user_id'].unique())
    if USERS_LIMIT:
        users = users[:USERS_LIMIT]
    df = df[df['user_id'].isin(users)].copy()
    log(f'Filtered to users count={len(users)}; rows now={len(df)}')

    # one-hot categorical
    if categorical_cols:
        from sklearn.preprocessing import OneHotEncoder
        enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        enc_cols = [c for c in categorical_cols if c in df.columns]
        if enc_cols:
            enc_vals = enc.fit_transform(df[enc_cols])
            enc_df = pd.DataFrame(enc_vals, columns=enc.get_feature_names_out(enc_cols))
            df = pd.concat([df.reset_index(drop=True).drop(columns=enc_cols), enc_df.reset_index(drop=True)], axis=1)
            log('Applied one-hot to categoricals')

    # MinMax scale numerics
    from sklearn.preprocessing import MinMaxScaler
    avail_num = [c for c in numeric_cols if c in df.columns]
    if avail_num:
        df[avail_num] = MinMaxScaler().fit_transform(df[avail_num])
        log(f'Scaled {len(avail_num)} numeric features')

    # rolling features + multi-variable events
    def add_rolling(g):
        for col in avail_num:
            g[f'Avg_{col}_{WINDOW_LONG}d'] = g[col].rolling(window=WINDOW_LONG, min_periods=1).mean().shift(1)
        # define event flags on some heuristics (first numeric as stress proxy)
        if avail_num:
            g['Stress_High'] = (g[avail_num[0]] > 0.8).astype(int)
            # sleep proxy if exists
            if len(avail_num) > 1:
                g['Sleep_Low'] = (g[avail_num[1]] < 0.2).astype(int)
        else:
            g['Stress_High'] = 0
            g['Sleep_Low'] = 0
        # rolling counts
        g[f'N_Stress_{WINDOW_LONG}d'] = g['Stress_High'].rolling(window=WINDOW_LONG, min_periods=1).sum().shift(1)
        g[f'N_SleepLow_{WINDOW_LONG}d'] = g['Sleep_Low'].rolling(window=WINDOW_LONG, min_periods=1).sum().shift(1)
        return g

    df = df.groupby('user_id', group_keys=False).apply(add_rolling)
    log('Added rolling features and event flags')

    # Target: Burnout_Risk if any rolling count exceeds thresholds
    df[f'N_Stress_{WINDOW_LONG}d'] = df[f'N_Stress_{WINDOW_LONG}d'].fillna(0).astype(int)
    df[f'N_SleepLow_{WINDOW_LONG}d'] = df[f'N_SleepLow_{WINDOW_LONG}d'].fillna(0).astype(int)
    df['Burnout_Risk'] = ((df[f'N_Stress_{WINDOW_LONG}d'] >= RISK_THRESHOLD) | (df[f'N_SleepLow_{WINDOW_LONG}d'] >= (RISK_THRESHOLD // 2))).astype(int)
    log(f'Target distribution: {df["Burnout_Risk"].value_counts().to_dict()}')

    # Drop object dtypes
    obj_cols = df.select_dtypes(include=['object']).columns.tolist()
    if obj_cols:
        df = df.drop(columns=obj_cols)
        log(f'Dropped object cols: {obj_cols}')

    # Final feature ordering
    final_features = [c for c in df.columns if c not in ('date', 'user_id', 'Burnout_Risk') and not c.startswith('N_')]
    # include rolling count features too
    final_features += [c for c in df.columns if c.startswith('Avg_') or c.startswith('N_')]
    # deduplicate
    final_features = list(dict.fromkeys(final_features))
    FEATURES_FILE.write_text('\n'.join(final_features))
    log(f'Final features count: {len(final_features)}')

    # Create sequences
    X = []
    y = []
    for uid, g in df.groupby('user_id'):
        gr = g.sort_values('date').reset_index(drop=True)
        for i in range(WINDOW_LONG, len(gr)):
            X.append(gr[final_features].iloc[i-WINDOW_LONG:i].values)
            y.append(int(gr['Burnout_Risk'].iloc[i]))
    X = np.array(X)
    y = np.array(y)
    log(f'Created sequences X.shape={X.shape} y.shape={y.shape} positives={int(y.sum())}')

    # save arrays for reproducibility
    np.save(RES / 'X_all.npy', X, allow_pickle=True)
    np.save(RES / 'y_all.npy', y, allow_pickle=True)

    # if too small positives, warn
    if y.sum() < 10:
        log('WARNING: very few positive samples, DL training may fail or overfit')

    # split by stratified sample
    from sklearn.model_selection import train_test_split
    if len(np.unique(y)) < 2:
        log('Not enough classes to continue')
        METRICS.write_text('Error: not enough classes in target\n')
        raise SystemExit('Not enough classes')

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=SEED, stratify=y)
    rel = 0.15 / 0.85
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=rel, random_state=SEED, stratify=y_temp)
    log(f'Splits: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}')

    # Build LSTM model
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from sklearn.utils import class_weight

    tf.random.set_seed(SEED)
    input_shape = X_train.shape[1:]
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(1e-3), loss='binary_crossentropy')
    log('Model compiled')

    # class weights
    cw = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = {i: float(cw[i]) for i in range(len(cw))}
    log(f'Class weights: {class_weights}')

    # callbacks
    ckp = MODELS / 'burnout_dl_model.h5'
    cb_ckp = tf.keras.callbacks.ModelCheckpoint(str(ckp), save_best_only=True)
    cb_es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # train (small epochs for test)
    model.fit(X_train.astype('float32'), y_train, validation_data=(X_val.astype('float32'), y_val),
              epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[cb_es, cb_ckp], class_weight=class_weights, verbose=2)
    log('Training finished')

    # load best
    m = tf.keras.models.load_model(str(ckp))
    pv = m.predict(X_val.astype('float32')).flatten()
    # find best threshold on val (maximize F1)
    from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, precision_recall_curve, average_precision_score
    best_thr = 0.5
    best_f1 = 0.0
    for thr in np.linspace(0,1,101):
        f = f1_score(y_val, (pv > thr).astype(int), zero_division=0)
        if f > best_f1:
            best_f1 = f; best_thr = thr

    # Precision-Recall curve on validation
    try:
        precision, recall, pr_thresholds = precision_recall_curve(y_val, pv)
        ap = average_precision_score(y_val, pv)
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'AP={ap:.3f}', color='#2C7BB6')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve (Validation)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        pr_png = RES / 'precision_recall_val.png'
        plt.savefig(pr_png, dpi=200)
        plt.close()
        # save arrays for inspection
        np.save(RES / 'pr_precision.npy', precision, allow_pickle=True)
        np.save(RES / 'pr_recall.npy', recall, allow_pickle=True)
        np.save(RES / 'pr_thresholds.npy', pr_thresholds, allow_pickle=True)
        log(f'Saved Precision-Recall curve to {pr_png} and arrays')
    except Exception as e:
        log('Failed to compute Precision-Recall curve: ' + str(e))
    # evaluate on test
    pt = m.predict(X_test.astype('float32')).flatten()
    preds = (pt > best_thr).astype(int)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)

    METRICS.write_text(f'Accuracy: {acc:.4f}\nPrecision: {prec:.4f}\nRecall: {rec:.4f}\nF1: {f1:.4f}\nBestValThreshold: {best_thr:.4f}\n')
    log(f'Wrote metrics to {METRICS}')

    # save arrays
    np.save(RES / 'X_train_dl.npy', X_train, allow_pickle=True)
    np.save(RES / 'y_train_dl.npy', y_train, allow_pickle=True)
    np.save(RES / 'X_val_dl.npy', X_val, allow_pickle=True)
    np.save(RES / 'y_val_dl.npy', y_val, allow_pickle=True)
    np.save(RES / 'X_test_dl.npy', X_test, allow_pickle=True)
    np.save(RES / 'y_test_dl.npy', y_test, allow_pickle=True)
    log('Saved arrays and model')

except Exception as exc:
    tb = traceback.format_exc()
    log('Exception: ' + str(exc))
    log(tb)
    METRICS.write_text('Error during pipeline:\n' + str(exc) + '\n' + tb)
    raise

else:
    log('Pipeline completed successfully')
