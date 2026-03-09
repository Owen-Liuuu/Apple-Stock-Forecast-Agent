# cnn_lstm_hybrid.py
"""
CNN-LSTM混合模型训练脚本
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建保存文件夹
output_dir = "cnn_lstm_results"
os.makedirs(output_dir, exist_ok=True)

print("="*70)
print("CNN-LSTM混合模型训练 - 股票价格预测")
print("="*70)

# ─────────────────────────────────────────────────────────────
# 1. 加载数据
# ─────────────────────────────────────────────────────────────
print("\n[1/7] 加载数据...")
df = pd.read_csv('apple_5yr_one1.csv')
print(f"✓ 数据形状: {df.shape}")

df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# ─────────────────────────────────────────────────────────────
# 2. 特征工程
# ─────────────────────────────────────────────────────────────
print("\n[2/7] 特征工程...")

def create_features(data):
    df_featured = data.copy()
    
    # 移动平均线
    df_featured['MA5'] = df_featured['Close'].rolling(window=5).mean()
    df_featured['MA10'] = df_featured['Close'].rolling(window=10).mean()
    df_featured['MA20'] = df_featured['Close'].rolling(window=20).mean()
    df_featured['MA50'] = df_featured['Close'].rolling(window=50).mean()
    
    # 移动平均线交叉
    df_featured['MA5_MA20_cross'] = df_featured['MA5'] - df_featured['MA20']
    
    # 价格动量
    df_featured['Momentum'] = df_featured['Close'] - df_featured['Close'].shift(5)
    
    # 价格变化率
    df_featured['Return'] = df_featured['Close'].pct_change()
    
    # 波动率
    df_featured['Volatility'] = df_featured['Return'].rolling(window=20).std()
    
    # 高低价差
    df_featured['High_Low_Spread'] = df_featured['High'] - df_featured['Low']
    
    # 收盘开盘价差
    df_featured['Close_Open_Spread'] = df_featured['Close'] - df_featured['Open']
    
    # 成交量变化
    df_featured['Volume_Change'] = df_featured['Volume'].pct_change()
    
    # RSI
    delta = df_featured['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df_featured['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df_featured['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df_featured['Close'].ewm(span=26, adjust=False).mean()
    df_featured['MACD'] = exp1 - exp2
    df_featured['MACD_Signal'] = df_featured['MACD'].ewm(span=9, adjust=False).mean()
    
    # 布林带
    df_featured['BB_Middle'] = df_featured['Close'].rolling(window=20).mean()
    bb_std = df_featured['Close'].rolling(window=20).std()
    df_featured['BB_Upper'] = df_featured['BB_Middle'] + (bb_std * 2)
    df_featured['BB_Lower'] = df_featured['BB_Middle'] - (bb_std * 2)
    df_featured['BB_Width'] = df_featured['BB_Upper'] - df_featured['BB_Lower']
    
    # 价格位置
    df_featured['Price_Position'] = (df_featured['Close'] - df_featured['Low']) / (
                df_featured['High'] - df_featured['Low'])
    
    return df_featured.dropna()

df_featured = create_features(df)
print(f"✓ 特征数量: {df_featured.shape[1]}列")

# ─────────────────────────────────────────────────────────────
# 3. 数据预处理
# ─────────────────────────────────────────────────────────────
print("\n[3/7] 数据预处理...")

# 18个特征列（和原CNN.py完全一样）
feature_columns = ['Open', 'High', 'Low', 'Volume', 'MA5', 'MA10', 'MA20',
                   'MA5_MA20_cross', 'Momentum', 'Volatility', 'High_Low_Spread',
                   'Close_Open_Spread', 'Volume_Change', 'RSI', 'MACD',
                   'MACD_Signal', 'BB_Width', 'Price_Position']

target_column = 'Close'

print(f"✓ 输入特征: {len(feature_columns)}个")

# 数据标准化
scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))

X_scaled = scaler_X.fit_transform(df_featured[feature_columns])
y_scaled = scaler_y.fit_transform(df_featured[[target_column]])

# 创建时间序列
def create_sequences(X, y, time_steps=60):
    X_seq, y_seq = [], []
    for i in range(len(X) - time_steps):
        X_seq.append(X[i:(i + time_steps)])
        y_seq.append(y[i + time_steps])
    return np.array(X_seq), np.array(y_seq)

time_steps = 60
X_seq, y_seq = create_sequences(X_scaled, y_scaled, time_steps)
print(f"✓ 序列形状: X={X_seq.shape}, y={y_seq.shape}")

# 划分训练集和测试集
train_size = int(len(X_seq) * 0.8)
X_train, X_test = X_seq[:train_size], X_seq[train_size:]
y_train, y_test = y_seq[:train_size], y_seq[train_size:]
print(f"✓ 训练集: {X_train.shape[0]}样本, 测试集: {X_test.shape[0]}样本")

# ─────────────────────────────────────────────────────────────
# 4. 构建CNN-LSTM混合模型 
# ─────────────────────────────────────────────────────────────
print("\n[4/7] 构建CNN-LSTM混合模型...")

model = Sequential([
    # ─── CNN部分：空间特征提取 ───
    Conv1D(filters=128, kernel_size=3, activation='relu',
           input_shape=(time_steps, len(feature_columns)),
           padding='same', name='conv1d_1'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),

    Conv1D(filters=64, kernel_size=3, activation='relu', 
           padding='same', name='conv1d_2'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),

    Conv1D(filters=32, kernel_size=3, activation='relu', 
           padding='same', name='conv1d_3'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),

    # ─── LSTM部分：时序依赖建模  ───
    LSTM(128, return_sequences=True, name='lstm_1'),
    Dropout(0.3),
    
    LSTM(64, return_sequences=True, name='lstm_2'),
    Dropout(0.3),
    
    LSTM(32, return_sequences=False, name='lstm_3'),  # 最后一层不return_sequences
    Dropout(0.2),

    # ─── Dense部分：回归预测 ───
    Dense(64, activation='relu', name='dense_1'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(32, activation='relu', name='dense_2'),
    Dropout(0.2),

    # 输出层
    Dense(1, name='output')
])

# 编译模型
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae', 'mse']
)

print("\n模型架构:")
print("-"*70)
model.summary()
print("-"*70)

# 统计参数量
total_params = model.count_params()
print(f"\n✓ 总参数量: {total_params:,}")

# ─────────────────────────────────────────────────────────────
# 5. 训练模型
# ─────────────────────────────────────────────────────────────
print("\n[5/7] 训练模型...")

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=0.00001,
    verbose=1
)

print("\n开始训练...")
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

print("\n✓ 训练完成!")

# ─────────────────────────────────────────────────────────────
# 6. 模型评估
# ─────────────────────────────────────────────────────────────
print("\n[6/7] 模型评估...")

y_pred_scaled = model.predict(X_test, verbose=0)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_original = scaler_y.inverse_transform(y_test)

y_pred_flat = y_pred.flatten()
y_test_flat = y_test_original.flatten()

rmse = np.sqrt(mean_squared_error(y_test_flat, y_pred_flat))
mae = mean_absolute_error(y_test_flat, y_pred_flat)
r2 = r2_score(y_test_flat, y_pred_flat)

print("\n" + "="*70)
print("测试集性能指标:")
print("="*70)
print(f"RMSE (均方根误差):  {rmse:.4f}")
print(f"MAE  (平均绝对误差): {mae:.4f}")
print(f"R²   (决定系数):    {r2:.4f}")
print("="*70)

# ─────────────────────────────────────────────────────────────
# 7. 保存模型
# ─────────────────────────────────────────────────────────────
print("\n[7/7] 保存模型...")

model.save('cnn_lstm_model.h5')
print(f"✓ 模型已保存: cnn_lstm_model.h5")

# 保存scaler
import joblib
joblib.dump(scaler_X, f'{output_dir}/scaler_X.pkl')
joblib.dump(scaler_y, f'{output_dir}/scaler_y.pkl')
print(f"✓ Scaler已保存: {output_dir}/scaler_X.pkl, scaler_y.pkl")

# ─────────────────────────────────────────────────────────────
# 8. 可视化结果
# ─────────────────────────────────────────────────────────────
print("\n生成可视化图表...")

# 训练历史
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='训练损失', linewidth=2)
plt.plot(history.history['val_loss'], label='验证损失', linewidth=2)
plt.title('CNN-LSTM模型训练损失', fontsize=14, fontweight='bold')
plt.xlabel('训练轮次')
plt.ylabel('损失 (MSE)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='训练MAE', linewidth=2)
plt.plot(history.history['val_mae'], label='验证MAE', linewidth=2)
plt.title('CNN-LSTM模型MAE', fontsize=14, fontweight='bold')
plt.xlabel('训练轮次')
plt.ylabel('MAE')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/training_history.png', dpi=300, bbox_inches='tight')
print(f"✓ 训练历史图已保存: {output_dir}/training_history.png")
plt.show()

# 预测结果对比
plt.figure(figsize=(14, 6))
plt.plot(y_test_flat[:200], label='实际价格', linewidth=2, alpha=0.7)
plt.plot(y_pred_flat[:200], label='预测价格', linewidth=2, alpha=0.7)
plt.title('CNN-LSTM预测结果 vs 实际价格', fontsize=14, fontweight='bold')
plt.xlabel('样本')
plt.ylabel('价格 (USD)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/prediction_comparison.png', dpi=300, bbox_inches='tight')
print(f"✓ 预测对比图已保存: {output_dir}/prediction_comparison.png")
plt.show()

# ─────────────────────────────────────────────────────────────
# 完成
# ─────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("✓ 全部完成!")
print("="*70)
print(f"\n模型架构: 3层Conv1D + 3层LSTM + 2层Dense")
print(f"输入特征: 18个技术指标")
print(f"时序窗口: {time_steps}步")
print(f"\n性能指标:")
print(f"  RMSE = {rmse:.4f}")
print(f"  MAE  = {mae:.4f}")
print(f"  R²   = {r2:.4f}")
print(f"\n保存文件:")
print(f"  - cnn_lstm_model.h5  (模型文件)")
print(f"  - {output_dir}/scaler_X.pkl  (特征scaler)")
print(f"  - {output_dir}/scaler_y.pkl  (目标scaler)")
print(f"  - {output_dir}/training_history.png")
print(f"  - {output_dir}/prediction_comparison.png")
print("="*70)
