# mlp_new.py
"""
MLP Model for Apple Stock Prediction (Simplified)
Lightweight architecture — consistent output format with CNN-LSTM and Linear.
"""

import os
import warnings
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from scipy import stats

# ──────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# ──────────────────────────────────────
# Config
# ──────────────────────────────────────
DATA_PATH = 'apple_5yr_one1.csv'
OUTPUT_DIR = 'stock_prediction_mlp_only'
MODEL_FILE = 'mlp_model_final.h5'
WINDOW = 30           # Reduced from 60
N_FUTURE = 30

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ──────────────────────────────────────
# 1. Load data
# ──────────────────────────────────────
print('=' * 70)
print('MLP Model (Simplified)')
print('=' * 70)

print('\n1. Loading data ...')
df = pd.read_csv(DATA_PATH)
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
print(f'   Shape: {df.shape}')
print(f'   Range: {df.index.min().date()} -> {df.index.max().date()}')

# ──────────────────────────────────────
# 2. Feature engineering (streamlined)
# ──────────────────────────────────────
print('\n2. Feature engineering ...')


def create_features(data):
    d = data.copy()

    d['Price_Range'] = d['High'] - d['Low']
    d['Body_Size']   = abs(d['Close'] - d['Open'])

    d['MA_5']  = d['Close'].rolling(5).mean()
    d['MA_20'] = d['Close'].rolling(20).mean()
    d['MA_50'] = d['Close'].rolling(50).mean()

    d['MA5_MA20_Cross'] = d['MA_5'] - d['MA_20']

    ema12 = d['Close'].ewm(span=12, adjust=False).mean()
    ema26 = d['Close'].ewm(span=26, adjust=False).mean()
    d['MACD']        = ema12 - ema26
    d['MACD_Signal'] = d['MACD'].ewm(span=9, adjust=False).mean()

    delta = d['Close'].diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    d['RSI'] = 100 - (100 / (1 + gain / loss))

    bb_mid = d['Close'].rolling(20).mean()
    bb_std = d['Close'].rolling(20).std()
    d['BB_Width']    = bb_std * 4
    d['BB_Position'] = (d['Close'] - (bb_mid - bb_std * 2)) / (bb_std * 4)

    d['Momentum_5'] = d['Close'].diff(5)
    d['Return_5']   = d['Close'].pct_change(5)
    d['Return_1']   = d['Close'].pct_change()
    d['Volatility_20'] = d['Return_1'].rolling(20).std()

    d['Volume_MA20'] = d['Volume'].rolling(20).mean()
    d['Volume_Ratio'] = d['Volume'] / d['Volume_MA20']

    d['High_Low_Ratio']   = d['High'] / d['Low']
    d['Close_Open_Ratio'] = d['Close'] / d['Open']

    return d.dropna()


df_feat = create_features(df)
print(f'   After features: {df_feat.shape}')

# ──────────────────────────────────────
# 3. Simple approach: predict Close from
#    a flat window of Close prices only.
#    MLP sees: [close_t-30, ..., close_t-1]
#    and predicts close_t.
# ──────────────────────────────────────
print('\n3. Scaling & sequencing ...')

close_values = df_feat[['Close']].values
scaler_y = MinMaxScaler(feature_range=(0, 1))
close_scaled = scaler_y.fit_transform(close_values).flatten()

X, y = [], []
for i in range(len(close_scaled) - WINDOW):
    X.append(close_scaled[i:i + WINDOW])
    y.append(close_scaled[i + WINDOW])

X = np.array(X)
y = np.array(y)

# Corresponding dates
seq_dates = df_feat.index[WINDOW:]
print(f'   Sequences: X={X.shape}  y={y.shape}')

# ──────────────────────────────────────
# 4. Split: val = 2024-05 to 2024-10
# ──────────────────────────────────────
print('\n4. Splitting ...')

val_start = pd.Timestamp('2024-05-01')
val_end   = pd.Timestamp('2024-10-31')

val_mask = (seq_dates >= val_start) & (seq_dates <= val_end)
val_idx  = np.where(val_mask)[0]

if len(val_idx) == 0:
    print('   Validation period not found, using ratio split.')
    train_end = int(len(X) * 0.70)
    val_end_i = int(len(X) * 0.85)
    X_train, y_train = X[:train_end], y[:train_end]
    X_val,   y_val   = X[train_end:val_end_i], y[train_end:val_end_i]
    X_test,  y_test  = X[val_end_i:], y[val_end_i:]
    val_dates  = seq_dates[train_end:val_end_i]
    test_dates = seq_dates[val_end_i:]
else:
    vs, ve = val_idx[0], val_idx[-1]
    X_train, y_train = X[:vs], y[:vs]
    X_val,   y_val   = X[vs:ve+1], y[vs:ve+1]
    X_test,  y_test  = X[ve+1:], y[ve+1:]
    val_dates  = seq_dates[vs:ve+1]
    test_dates = seq_dates[ve+1:]

print(f'   Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}')

# ──────────────────────────────────────
# 5. Build lightweight MLP
# ──────────────────────────────────────
print('\n5. Building MLP ...')

mlp_model = Sequential([
    Dense(64, activation='relu', input_shape=(WINDOW,)),
    BatchNormalization(),
    Dropout(0.1),

    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.1),

    Dense(16, activation='relu'),

    Dense(1),
])

mlp_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae'],
)

mlp_model.summary()

# ──────────────────────────────────────
# 6. Train
# ──────────────────────────────────────
print('\n6. Training ...')

callbacks = [
    EarlyStopping(monitor='val_loss', patience=20,
                  restore_best_weights=True, verbose=1, min_delta=1e-5),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                      patience=8, min_lr=1e-6, verbose=1),
    ModelCheckpoint(filepath=f'{OUTPUT_DIR}/best_mlp_model.h5',
                    monitor='val_loss', save_best_only=True, verbose=1),
]

mlp_history = mlp_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=200,
    batch_size=32,
    callbacks=callbacks,
    shuffle=False,
    verbose=1,
)

try:
    mlp_model = load_model(f'{OUTPUT_DIR}/best_mlp_model.h5')
    print('Loaded best checkpoint.')
except Exception:
    print('Using final weights.')

# ──────────────────────────────────────
# 7. Evaluate — Validation Set
# ──────────────────────────────────────
print('\n7. Evaluating ...')

# -- Validation --
mlp_val_pred_s = mlp_model.predict(X_val, verbose=0).flatten()
mlp_val_pred_flat = scaler_y.inverse_transform(mlp_val_pred_s.reshape(-1, 1)).flatten()
y_val_flat = scaler_y.inverse_transform(y_val.reshape(-1, 1)).flatten()

mlp_val_rmse = np.sqrt(mean_squared_error(y_val_flat, mlp_val_pred_flat))
mlp_val_mae  = mean_absolute_error(y_val_flat, mlp_val_pred_flat)
mlp_val_r2   = r2_score(y_val_flat, mlp_val_pred_flat)
mlp_val_dir  = np.mean(np.sign(np.diff(y_val_flat)) == np.sign(np.diff(mlp_val_pred_flat))) * 100

mlp_val_residuals = y_val_flat - mlp_val_pred_flat
mlp_val_pct_error = (mlp_val_residuals / y_val_flat) * 100

print(f'   Validation (2024-05 to 2024-10):')
print(f'     RMSE: {mlp_val_rmse:.4f}')
print(f'     MAE:  {mlp_val_mae:.4f}')
print(f'     R2:   {mlp_val_r2:.4f}')
print(f'     Dir:  {mlp_val_dir:.2f}%')

# -- Test --
mlp_test_pred_s = mlp_model.predict(X_test, verbose=0).flatten()
mlp_test_pred_flat = scaler_y.inverse_transform(mlp_test_pred_s.reshape(-1, 1)).flatten()
y_test_flat = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

mlp_test_rmse = np.sqrt(mean_squared_error(y_test_flat, mlp_test_pred_flat))
mlp_test_mae  = mean_absolute_error(y_test_flat, mlp_test_pred_flat)
mlp_test_r2   = r2_score(y_test_flat, mlp_test_pred_flat)
mlp_test_dir  = np.mean(np.sign(np.diff(y_test_flat)) == np.sign(np.diff(mlp_test_pred_flat))) * 100

test_residuals = y_test_flat - mlp_test_pred_flat
test_pct_error = np.abs(test_residuals / y_test_flat) * 100

print(f'   Test Set:')
print(f'     RMSE: {mlp_test_rmse:.4f}')
print(f'     MAE:  {mlp_test_mae:.4f}')
print(f'     R2:   {mlp_test_r2:.4f}')
print(f'     Dir:  {mlp_test_dir:.2f}%')

# ──────────────────────────────────────
# 8. Plots
# ──────────────────────────────────────
print('\n8. Generating plots ...')

# -- Plot 1: Training history --
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].plot(mlp_history.history['loss'], label='Train Loss', color='blue', linewidth=2)
axes[0].plot(mlp_history.history['val_loss'], label='Val Loss', color='red', linewidth=2)
axes[0].set_title('MLP Training Loss', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('MSE Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(mlp_history.history['mae'], label='Train MAE', color='blue', linewidth=2)
axes[1].plot(mlp_history.history['val_mae'], label='Val MAE', color='red', linewidth=2)
axes[1].set_title('MLP Training MAE', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('MAE')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/1_mlp_training_history.png', dpi=300, bbox_inches='tight')
plt.close()
print('   Saved: 1_mlp_training_history.png')

# -- Plot 2: Validation prediction comparison --
plt.figure(figsize=(18, 8))
plt.plot(val_dates, y_val_flat, label='Actual Stock Price', color='blue',
         linewidth=3, alpha=0.9, marker='o', markersize=4)
plt.plot(val_dates, mlp_val_pred_flat, label='MLP Prediction', color='red',
         linewidth=2, linestyle='--', alpha=0.8, marker='s', markersize=3)
plt.fill_between(val_dates,
                 mlp_val_pred_flat - mlp_val_rmse,
                 mlp_val_pred_flat + mlp_val_rmse,
                 color='orange', alpha=0.15, label=f'MLP +/-RMSE ({mlp_val_rmse:.2f})')

txt = (f'Validation (2024-05 to 2024-10):\n'
       f'RMSE = {mlp_val_rmse:.3f}\n'
       f'R2 = {mlp_val_r2:.3f}\n'
       f'Dir Acc = {mlp_val_dir:.2f}%')
plt.text(0.02, 0.98, txt, transform=plt.gca().transAxes,
         fontsize=10, verticalalignment='top', color='red',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.title('Apple Stock Price Prediction Results - Validation Set', fontsize=16, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Stock Price (USD)')
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/2_prediction_comparison_validation_2024_05_10.png', dpi=300, bbox_inches='tight')
plt.close()
print('   Saved: 2_prediction_comparison_validation_2024_05_10.png')

# -- Plot 3: Residual analysis (validation) --
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

axes[0, 0].hist(mlp_val_residuals, bins=40, color='red', alpha=0.6, edgecolor='black')
axes[0, 0].axvline(x=0, color='black', linestyle='--')
axes[0, 0].set_title('Residual Distribution (Val)', fontsize=12)
axes[0, 0].set_xlabel('Residual')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].scatter(mlp_val_pred_flat, mlp_val_residuals, alpha=0.5, color='red', s=15)
axes[0, 1].axhline(y=0, color='black', linestyle='--')
axes[0, 1].set_title('Residuals vs Predictions (Val)', fontsize=12)
axes[0, 1].set_xlabel('Predicted')
axes[0, 1].set_ylabel('Residual')
axes[0, 1].grid(True, alpha=0.3)

axes[0, 2].scatter(y_val_flat, mlp_val_pred_flat, alpha=0.5, color='red', s=15)
axes[0, 2].plot([y_val_flat.min(), y_val_flat.max()],
                [y_val_flat.min(), y_val_flat.max()], 'k--', linewidth=2)
axes[0, 2].set_title('Actual vs Predicted (Val)', fontsize=12)
axes[0, 2].set_xlabel('Actual')
axes[0, 2].set_ylabel('Predicted')
axes[0, 2].grid(True, alpha=0.3)

axes[1, 0].plot(val_dates, mlp_val_residuals, color='red', alpha=0.7, linewidth=1)
axes[1, 0].axhline(y=0, color='black', linestyle='--')
axes[1, 0].set_title('Residuals Over Time (Val)', fontsize=12)
axes[1, 0].set_xlabel('Date')
plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45)
axes[1, 0].grid(True, alpha=0.3)

cum_err = np.cumsum(mlp_val_residuals)
axes[1, 1].plot(val_dates, cum_err, color='red', linewidth=2)
axes[1, 1].axhline(y=0, color='black', linestyle='--')
axes[1, 1].set_title('Cumulative Error (Val)', fontsize=12)
axes[1, 1].set_xlabel('Date')
plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45)
axes[1, 1].grid(True, alpha=0.3)

axes[1, 2].boxplot(mlp_val_residuals, labels=['MLP Val'], patch_artist=True,
                   boxprops=dict(facecolor='lightblue', color='black'),
                   medianprops=dict(color='red'))
axes[1, 2].set_title('Residuals Boxplot (Val)', fontsize=12)
axes[1, 2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/3_residual_analysis_validation.png', dpi=300, bbox_inches='tight')
plt.close()
print('   Saved: 3_residual_analysis_validation.png')

# -- Plot 4: Error analysis (validation) --
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].plot(val_dates, mlp_val_pct_error, color='red', alpha=0.7, linewidth=1)
axes[0].axhline(y=0, color='black', linestyle='--')
axes[0].set_title('Percentage Error (Val)', fontsize=14)
axes[0].set_xlabel('Date')
axes[0].set_ylabel('Error (%)')
plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45)
axes[0].grid(True, alpha=0.3)

mae_rolling = pd.Series(np.abs(mlp_val_residuals)).rolling(10).mean()
axes[1].plot(val_dates, mae_rolling, color='red', linewidth=2)
axes[1].axhline(y=mlp_val_mae, color='black', linestyle='--', label=f'Avg MAE: {mlp_val_mae:.2f}')
axes[1].set_title('Rolling MAE (10-day)', fontsize=14)
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Abs Error')
axes[1].legend()
plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/4_error_analysis_validation.png', dpi=300, bbox_inches='tight')
plt.close()
print('   Saved: 4_error_analysis_validation.png')

# -- Plot 5: Performance metrics --
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

metrics = ['RMSE', 'MAE', 'R2', 'Dir Acc']
val_values  = [mlp_val_rmse, mlp_val_mae, mlp_val_r2, mlp_val_dir]
test_values = [mlp_test_rmse, mlp_test_mae, mlp_test_r2, mlp_test_dir]
x = np.arange(len(metrics))
w = 0.35

axes[0, 0].bar(x - w/2, val_values, w, label='Validation', color='#ff6b6b', alpha=0.8, edgecolor='black')
axes[0, 0].bar(x + w/2, test_values, w, label='Test', color='#ff9999', alpha=0.8, edgecolor='black')
axes[0, 0].set_title('Performance Comparison', fontsize=12, fontweight='bold')
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(metrics)
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3, axis='y')

# Radar chart
ax_radar = fig.add_subplot(2, 2, 2, polar=True)
cats = ['RMSE\n(Lower)', 'MAE\n(Lower)', 'R2\n(Higher)', 'Dir Acc\n(Higher)']
r2_safe = max(mlp_val_r2, 0)
vals = [1 - min(mlp_val_rmse / 50, 1), 1 - min(mlp_val_mae / 50, 1),
        r2_safe, mlp_val_dir / 100]
angles = np.linspace(0, 2 * np.pi, len(cats), endpoint=False).tolist()
vals += vals[:1]
angles += angles[:1]
ax_radar.plot(angles, vals, 'o-', linewidth=2, color='red')
ax_radar.fill(angles, vals, alpha=0.25, color='red')
ax_radar.set_ylim(0, 1)
ax_radar.set_xticks(angles[:-1])
ax_radar.set_xticklabels(cats, size=9)
ax_radar.set_title('Radar (Val)', fontsize=12, fontweight='bold', pad=20)

epochs = range(1, len(mlp_history.history['loss']) + 1)
axes[1, 0].plot(epochs, mlp_history.history['loss'], label='Train', color='blue', alpha=0.7)
axes[1, 0].plot(epochs, mlp_history.history['val_loss'], label='Val', color='red', alpha=0.7)
axes[1, 0].set_title('Loss Curve', fontsize=12)
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

if 'lr' in mlp_history.history:
    axes[1, 1].plot(epochs, mlp_history.history['lr'], color='green', linewidth=2)
    axes[1, 1].set_yscale('log')
else:
    axes[1, 1].text(0.5, 0.5, 'LR History\nN/A', ha='center', va='center',
                    transform=axes[1, 1].transAxes)
axes[1, 1].set_title('Learning Rate', fontsize=12)
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/5_model_performance_metrics.png', dpi=300, bbox_inches='tight')
plt.close()
print('   Saved: 5_model_performance_metrics.png')

# -- Plot 6: Feature importance (weights) --
weights = mlp_model.layers[0].get_weights()[0]  # shape (WINDOW, 64)
importance = np.mean(np.abs(weights), axis=1)
day_labels = [f'Day-{WINDOW - i}' for i in range(WINDOW)]
sorted_idx = np.argsort(importance)

plt.figure(figsize=(14, 8))
plt.barh(range(WINDOW), importance[sorted_idx], color='steelblue', alpha=0.7, edgecolor='black')
plt.yticks(range(WINDOW), [day_labels[i] for i in sorted_idx], fontsize=9)
plt.xlabel('Mean Absolute Weight')
plt.title('MLP Feature Importance (First Layer Weights)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/6_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print('   Saved: 6_feature_importance.png')

# -- Plot 7: Future prediction (FIXED rolling) --
print('\n9. Predicting future ...')

current_seq = close_scaled[-WINDOW:].tolist()
mlp_future = []

for _ in range(N_FUTURE):
    inp = np.array(current_seq[-WINDOW:]).reshape(1, WINDOW)
    nxt = mlp_model.predict(inp, verbose=0)[0, 0]
    mlp_future.append(nxt)
    current_seq.append(nxt)

mlp_future_prices = scaler_y.inverse_transform(
    np.array(mlp_future).reshape(-1, 1)
).flatten()

last_date    = df_feat.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                             periods=N_FUTURE, freq='B')

plt.figure(figsize=(16, 8))
hist_days = min(180, len(df_feat))
plt.plot(df_feat.index[-hist_days:], df_feat['Close'].values[-hist_days:],
         label='Historical Price', color='blue', linewidth=2.5)

# Connected line
connect_dates  = [df_feat.index[-1]] + list(future_dates)
connect_prices = [df_feat['Close'].values[-1]] + list(mlp_future_prices)

plt.plot(connect_dates, connect_prices, label='MLP Future Prediction',
         color='red', linewidth=3, marker='o', markersize=6, linestyle='--')

# Confidence band
std_res = np.std(test_residuals)
connect_arr = np.array(connect_prices)
plt.fill_between(connect_dates,
                 connect_arr - std_res, connect_arr + std_res,
                 color='red', alpha=0.12, label='+/-1 std')

for i, (d, pr) in enumerate(zip(future_dates, mlp_future_prices)):
    if i % 5 == 0:
        plt.annotate(f'{pr:.2f}', xy=(d, pr), xytext=(0, 10),
                     textcoords='offset points', ha='center', fontsize=9,
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

plt.title(f'Apple Stock Price Forecast for Next {N_FUTURE} Days', fontsize=16, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Stock Price (USD)')
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/7_future_prediction.png', dpi=300, bbox_inches='tight')
plt.close()
print('   Saved: 7_future_prediction.png')

# ──────────────────────────────────────
# 9. Save
# ──────────────────────────────────────
print('\n10. Saving ...')

mlp_model.save(f'{OUTPUT_DIR}/{MODEL_FILE}')
mlp_model.save(MODEL_FILE)
print(f'   Model: ./{MODEL_FILE}')
print(f'   Model: {OUTPUT_DIR}/{MODEL_FILE}')

joblib.dump(scaler_y, f'{OUTPUT_DIR}/scaler_y.pkl')

pd.DataFrame({
    'Model': ['MLP Validation (2024-05 to 2024-10)', 'MLP Test'],
    'RMSE': [mlp_val_rmse, mlp_test_rmse],
    'MAE': [mlp_val_mae, mlp_test_mae],
    'R2': [mlp_val_r2, mlp_test_r2],
    'Direction_Accuracy': [mlp_val_dir, mlp_test_dir],
}).to_csv(f'{OUTPUT_DIR}/evaluation_metrics.csv', index=False)

pd.DataFrame({
    'Date': val_dates,
    'Actual_Price': y_val_flat,
    'MLP_Predicted': mlp_val_pred_flat,
    'Residual': mlp_val_residuals,
    'Percentage_Error': mlp_val_pct_error,
}).to_csv(f'{OUTPUT_DIR}/validation_prediction_results.csv', index=False)

pd.DataFrame({
    'Date': test_dates,
    'Actual_Price': y_test_flat,
    'MLP_Predicted': mlp_test_pred_flat,
    'Residual': test_residuals,
    'Percentage_Error': test_pct_error,
}).to_csv(f'{OUTPUT_DIR}/test_prediction_results.csv', index=False)

pd.DataFrame(mlp_history.history).to_csv(
    f'{OUTPUT_DIR}/mlp_training_history.csv', index=False
)

pd.DataFrame({
    'Date': future_dates,
    'Predicted_Price': mlp_future_prices,
}).to_csv(f'{OUTPUT_DIR}/future_predictions.csv', index=False)

# ──────────────────────────────────────
# Summary
# ──────────────────────────────────────
print(f'\n{"=" * 70}')
print('MLP Model Training Complete!')
print(f'{"=" * 70}')
print(f'Data:       {DATA_PATH}')
print(f'Range:      {df.index.min().date()} -> {df.index.max().date()}')
print(f'Window:     {WINDOW} days (close price only)')
print(f'')
print(f'Architecture:')
print(f'  Dense(64) -> BN -> Drop(0.1)')
print(f'  Dense(32) -> BN -> Drop(0.1)')
print(f'  Dense(16) -> Dense(1)')
print(f'  Params: {mlp_model.count_params()}')
print(f'')
print(f'Training:')
print(f'  Optimizer: Adam (lr=0.001)')
print(f'  Loss:      MSE')
print(f'  Epochs:    {len(mlp_history.history["loss"])}')
print(f'  Best val:  {min(mlp_history.history["val_loss"]):.6f}')
print(f'')
print(f'Validation (2024-05 to 2024-10):')
print(f'  RMSE: ${mlp_val_rmse:.2f}')
print(f'  MAE:  ${mlp_val_mae:.2f}')
print(f'  R2:   {mlp_val_r2:.4f}')
print(f'  Dir:  {mlp_val_dir:.1f}%')
print(f'')
print(f'Test:')
print(f'  RMSE: ${mlp_test_rmse:.2f}')
print(f'  MAE:  ${mlp_test_mae:.2f}')
print(f'  R2:   {mlp_test_r2:.4f}')
print(f'  Dir:  {mlp_test_dir:.1f}%')
print(f'')
print(f'Output: ./{MODEL_FILE} + {OUTPUT_DIR}/')
print(f'{"=" * 70}')
