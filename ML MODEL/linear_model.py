# linear_model.py
"""
Linear Regression Baseline (Keras Dense(1)) for Apple Stock Prediction
Outputs .h5 model file — unified format with MLP and CNN-LSTM.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

warnings.filterwarnings('ignore')

# ──────────────────────────────────────
# Config
# ──────────────────────────────────────
DATA_PATH = 'apple_5yr_one1.csv'
WINDOW = 30
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
OUTPUT_DIR = 'stock_prediction_linear_plots'
MODEL_NAME = 'linear_model_final.h5'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ──────────────────────────────────────
# 1. Load data
# ──────────────────────────────────────
print('=' * 60)
print('Linear Regression Baseline (Keras)')
print('=' * 60)

print('\n1. Loading data …')
df = pd.read_csv(DATA_PATH)
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
print(f'   Shape: {df.shape}')
print(f'   Range: {df.index.min().date()} → {df.index.max().date()}')

# ──────────────────────────────────────
# 2. Feature engineering
# ──────────────────────────────────────
print('\n2. Creating features …')


def create_features(data):
    d = data.copy()
    d['MA5']  = d['Close'].rolling(5).mean()
    d['MA10'] = d['Close'].rolling(10).mean()
    d['MA20'] = d['Close'].rolling(20).mean()
    d['Return'] = d['Close'].pct_change()
    d['Volatility'] = d['Return'].rolling(20).std()

    delta = d['Close'].diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    d['RSI'] = 100 - (100 / (1 + gain / loss))

    exp1 = d['Close'].ewm(span=12, adjust=False).mean()
    exp2 = d['Close'].ewm(span=26, adjust=False).mean()
    d['MACD'] = exp1 - exp2

    d['High_Low_Spread'] = d['High'] - d['Low']
    d['Volume_Change']   = d['Volume'].pct_change()
    return d.dropna()


df_feat = create_features(df)
print(f'   After features: {df_feat.shape}')

# ──────────────────────────────────────
# 3. Scaling & sequences
# ──────────────────────────────────────
print('\n3. Creating sequences …')

close_values = df_feat[['Close']].values
scaler = MinMaxScaler(feature_range=(0, 1))
close_scaled = scaler.fit_transform(close_values).flatten()

X, y = [], []
for i in range(len(close_scaled) - WINDOW):
    X.append(close_scaled[i:i + WINDOW])
    y.append(close_scaled[i + WINDOW])

X = np.array(X)
y = np.array(y)
print(f'   X={X.shape}, y={y.shape}')

# ──────────────────────────────────────
# 4. Split
# ──────────────────────────────────────
print('\n4. Splitting …')
n = len(X)
train_end = int(n * TRAIN_RATIO)
val_end   = int(n * (TRAIN_RATIO + VAL_RATIO))

X_train, y_train = X[:train_end], y[:train_end]
X_val,   y_val   = X[train_end:val_end], y[train_end:val_end]
X_test,  y_test  = X[val_end:], y[val_end:]

print(f'   Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}')

# ──────────────────────────────────────
# 5. Build & train Keras linear model
# ──────────────────────────────────────
print('\n5. Building Keras Linear model …')

model = Sequential([
    Dense(1, input_shape=(WINDOW,), name='linear_output')
    # No activation = pure linear regression
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae'],
)

model.summary()

callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
    ModelCheckpoint(
        filepath=f'{OUTPUT_DIR}/best_linear.h5',
        monitor='val_loss', save_best_only=True, verbose=1,
    ),
]

print('\n   Training …')
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=200,
    batch_size=32,
    callbacks=callbacks,
    verbose=1,
)

# Load best weights
try:
    model = load_model(f'{OUTPUT_DIR}/best_linear.h5')
    print('   Loaded best checkpoint.')
except Exception:
    print('   Using final weights.')

# ──────────────────────────────────────
# 6. Evaluate
# ──────────────────────────────────────
print('\n6. Evaluating …')

y_pred_scaled = model.predict(X_test, verbose=0).flatten()

y_pred_real = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
mae  = mean_absolute_error(y_test_real, y_pred_real)
r2   = r2_score(y_test_real, y_pred_real)

actual_dir = np.diff(y_test_real) > 0
pred_dir   = np.diff(y_pred_real) > 0
dir_acc    = np.mean(actual_dir == pred_dir) * 100

residuals  = y_test_real - y_pred_real
pct_error  = np.abs(residuals / y_test_real) * 100

print(f'   RMSE:               {rmse:.4f}')
print(f'   MAE:                {mae:.4f}')
print(f'   R²:                 {r2:.4f}')
print(f'   Direction Accuracy: {dir_acc:.2f}%')
print(f'   Mean % Error:       {np.mean(pct_error):.2f}%')

# ──────────────────────────────────────
# 7. Plots
# ──────────────────────────────────────
print('\n7. Generating plots …')

test_dates = df_feat.index[-(len(y_test_real)):]

# ── Plot 0: Training curve ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
ax1.plot(history.history['loss'], label='Train Loss', color='#1f77b4')
ax1.plot(history.history['val_loss'], label='Val Loss', color='#ff7f0e')
ax1.set_title('Loss Curve', fontsize=12, fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('MSE Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(history.history['mae'], label='Train MAE', color='#1f77b4')
ax2.plot(history.history['val_mae'], label='Val MAE', color='#ff7f0e')
ax2.set_title('MAE Curve', fontsize=12, fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('MAE')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/0_training_history.png', dpi=300, bbox_inches='tight')
plt.close()
print('   Saved: 0_training_history.png')

# ── Plot 1: Actual vs Predicted ──
plt.figure(figsize=(16, 6))
plt.plot(test_dates, y_test_real, label='Actual', color='#1f77b4', linewidth=2)
plt.plot(test_dates, y_pred_real, label='Linear (Keras) Predicted', color='#ff7f0e', linewidth=2, alpha=0.85)
plt.title('Linear Regression (Keras) — Actual vs Predicted', fontsize=14, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/1_actual_vs_predicted.png', dpi=300, bbox_inches='tight')
plt.close()
print('   Saved: 1_actual_vs_predicted.png')

# ── Plot 2: Residuals ──
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
axes[0].plot(test_dates, residuals, color='#2ca02c', alpha=0.7, linewidth=0.8)
axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
axes[0].set_title('Residuals Over Time', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Date')
axes[0].set_ylabel('Residual (USD)')
axes[0].grid(True, alpha=0.3)

axes[1].hist(residuals, bins=40, color='#9467bd', alpha=0.75, edgecolor='white')
axes[1].axvline(x=0, color='red', linestyle='--', alpha=0.7)
axes[1].set_title('Residual Distribution', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Residual (USD)')
axes[1].set_ylabel('Count')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/2_residuals.png', dpi=300, bbox_inches='tight')
plt.close()
print('   Saved: 2_residuals.png')

# ── Plot 3: Scatter ──
plt.figure(figsize=(8, 8))
plt.scatter(y_test_real, y_pred_real, alpha=0.4, s=15, color='#17becf')
min_val = min(y_test_real.min(), y_pred_real.min())
max_val = max(y_test_real.max(), y_pred_real.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
plt.title(f'Scatter — R² = {r2:.4f}', fontsize=14, fontweight='bold')
plt.xlabel('Actual Price (USD)')
plt.ylabel('Predicted Price (USD)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/3_scatter.png', dpi=300, bbox_inches='tight')
plt.close()
print('   Saved: 3_scatter.png')

# ── Plot 4: Percentage error ──
plt.figure(figsize=(16, 5))
plt.plot(test_dates, pct_error, color='#d62728', alpha=0.6, linewidth=0.8)
plt.axhline(y=np.mean(pct_error), color='blue', linestyle='--', alpha=0.7,
            label=f'Mean: {np.mean(pct_error):.2f}%')
plt.title('Percentage Error Over Time', fontsize=14, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Error (%)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/4_percentage_error.png', dpi=300, bbox_inches='tight')
plt.close()
print('   Saved: 4_percentage_error.png')

# ── Plot 5: Weights ──
weights = model.get_weights()[0].flatten()
plt.figure(figsize=(12, 5))
day_labels = [f'Day-{WINDOW - i}' for i in range(WINDOW)]
colors = ['#2ca02c' if w > 0 else '#d62728' for w in weights]
plt.bar(range(WINDOW), weights, color=colors, alpha=0.8)
plt.title('Keras Linear Layer Weights (≈ Regression Coefficients)', fontsize=14, fontweight='bold')
plt.xlabel('Lag Day')
plt.ylabel('Weight')
plt.xticks(range(0, WINDOW, 5), [day_labels[i] for i in range(0, WINDOW, 5)], rotation=45)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/5_coefficients.png', dpi=300, bbox_inches='tight')
plt.close()
print('   Saved: 5_coefficients.png')

# ── Plot 6: Future forecast (connected) ──
print('\n8. Predicting future …')
n_future = 30
current_seq = close_scaled[-WINDOW:].tolist()
future_preds = []

for _ in range(n_future):
    inp = np.array(current_seq[-WINDOW:]).reshape(1, WINDOW)
    nxt = model.predict(inp, verbose=0)[0, 0]
    future_preds.append(nxt)
    current_seq.append(nxt)

future_prices = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1)).flatten()
last_date    = df_feat.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_future, freq='B')

plt.figure(figsize=(16, 7))
hist_days = min(180, len(df_feat))
plt.plot(df_feat.index[-hist_days:], df_feat['Close'].values[-hist_days:],
         label='Historical', color='#1f77b4', linewidth=2)

# Connect last historical point → forecast
connect_dates  = [df_feat.index[-1]] + list(future_dates)
connect_prices = [df_feat['Close'].values[-1]] + list(future_prices)

plt.plot(connect_dates, connect_prices, label='Linear Forecast (Keras)',
         color='#ff7f0e', linewidth=2.5, marker='o', markersize=5, linestyle='--')

std_res = np.std(residuals)
connect_band = np.array([df_feat['Close'].values[-1]] + list(future_prices))
plt.fill_between(connect_dates,
                 connect_band - std_res, connect_band + std_res,
                 color='#ff7f0e', alpha=0.15, label='±1σ band')

plt.title(f'Apple Stock — {n_future} Day Forecast (Keras Linear)', fontsize=14, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/6_future_forecast.png', dpi=300, bbox_inches='tight')
plt.close()
print('   Saved: 6_future_forecast.png')

# ──────────────────────────────────────
# 8. Save model (.h5)
# ──────────────────────────────────────
print('\n9. Saving …')

model.save(f'{OUTPUT_DIR}/{MODEL_NAME}')
model.save(MODEL_NAME)
print(f'   Model: ./{MODEL_NAME}')
print(f'   Model: {OUTPUT_DIR}/{MODEL_NAME}')

metrics_df = pd.DataFrame({
    'Metric': ['RMSE', 'MAE', 'R2', 'Direction_Accuracy', 'Mean_Pct_Error'],
    'Value':  [rmse, mae, r2, dir_acc, np.mean(pct_error)],
})
metrics_df.to_csv(f'{OUTPUT_DIR}/evaluation_metrics_linear.csv', index=False)

pred_df = pd.DataFrame({
    'Date': test_dates,
    'Actual_Price': y_test_real,
    'Predicted_Price': y_pred_real,
    'Residual': residuals,
    'Percentage_Error': pct_error,
})
pred_df.to_csv(f'{OUTPUT_DIR}/prediction_results_linear.csv', index=False)

hist_df = pd.DataFrame(history.history)
hist_df.to_csv(f'{OUTPUT_DIR}/training_history_linear.csv', index=False)

# ──────────────────────────────────────
# Summary
# ──────────────────────────────────────
print(f'\n{"=" * 60}')
print('Keras Linear Model — Summary')
print(f'{"=" * 60}')
print(f'Architecture:  Dense(1) — no activation (pure linear)')
print(f'Parameters:    {model.count_params()}')
print(f'Window:        {WINDOW} days')
print(f'Optimizer:     Adam (lr=0.001)')
print(f'Loss:          MSE')
print(f'Epochs run:    {len(history.history["loss"])}')
print(f'Best val loss: {min(history.history["val_loss"]):.6f}')
print(f'')
print(f'Test Performance:')
print(f'  RMSE:               ${rmse:.2f}')
print(f'  MAE:                ${mae:.2f}')
print(f'  R²:                 {r2:.4f}')
print(f'  Direction Accuracy: {dir_acc:.1f}%')
print(f'  Mean % Error:       {np.mean(pct_error):.2f}%')
print(f'')
print(f'Output:')
print(f'  ./{MODEL_NAME}  ← prediction_service.py 直接加载')
print(f'  {OUTPUT_DIR}/ 下 7 张图 + 3 个 CSV')
print(f'{"=" * 60}')
