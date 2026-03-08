import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体，确保图表中的中文正常显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建保存图片的文件夹
output_dir = "stock_prediction_cnn_plots"
os.makedirs(output_dir, exist_ok=True)
# ──────────────────────────────────────
# 1. 加载数据集
# ──────────────────────────────────────
print("正在加载数据集...")
df = pd.read_csv('apple_5yr_one1.csv')
print(f"数据集形状: {df.shape}")
print(f"列名: {df.columns.tolist()}")
print("\n前5行数据:")
print(df.head())

# 转换日期列
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# ──────────────────────────────────────
# 2. 数据探索和可视化
# ──────────────────────────────────────
print("\n数据基本信息:")
print(df.info())
print("\n数据描述统计:")
print(df.describe())

# 检查缺失值
print(f"\n缺失值数量:\n{df.isnull().sum()}")

# ──────────────────────────────────────
# 3. 数据可视化
# ──────────────────────────────────────
print("\n创建数据可视化图表...")

# 3.1 价格走势图
plt.figure(figsize=(14, 8))
plt.plot(df.index, df['Close'], label='收盘价', color='blue', linewidth=1.5)
plt.plot(df.index, df['High'], label='最高价', color='red', alpha=0.5, linewidth=0.8)
plt.plot(df.index, df['Low'], label='最低价', color='green', alpha=0.5, linewidth=0.8)
plt.title('Apple股票价格走势 (2020-2025)', fontsize=16, fontweight='bold')
plt.xlabel('日期', fontsize=12)
plt.ylabel('价格 (USD)', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/1_price_trend.png', dpi=300, bbox_inches='tight')
plt.show()

# 3.2 成交量图
plt.figure(figsize=(14, 6))
plt.bar(df.index, df['Volume'], color='orange', alpha=0.7, width=0.8)
plt.title('Apple股票成交量', fontsize=16, fontweight='bold')
plt.xlabel('日期', fontsize=12)
plt.ylabel('成交量', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/2_volume.png', dpi=300, bbox_inches='tight')
plt.show()

# 3.3 价格分布图
plt.figure(figsize=(12, 8))
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 收盘价分布
axes[0, 0].hist(df['Close'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
axes[0, 0].set_title('收盘价分布', fontsize=14)
axes[0, 0].set_xlabel('价格')
axes[0, 0].set_ylabel('频率')

# 开盘价分布
axes[0, 1].hist(df['Open'], bins=50, color='lightcoral', edgecolor='black', alpha=0.7)
axes[0, 1].set_title('开盘价分布', fontsize=14)
axes[0, 1].set_xlabel('价格')
axes[0, 1].set_ylabel('频率')

# 高低价差
price_range = df['High'] - df['Low']
axes[1, 0].hist(price_range, bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
axes[1, 0].set_title('每日价格波动范围', fontsize=14)
axes[1, 0].set_xlabel('价格差')
axes[1, 0].set_ylabel('频率')

# 成交量分布
axes[1, 1].hist(df['Volume'], bins=50, color='gold', edgecolor='black', alpha=0.7)
axes[1, 1].set_title('成交量分布', fontsize=14)
axes[1, 1].set_xlabel('成交量')
axes[1, 1].set_ylabel('频率')

plt.tight_layout()
plt.savefig(f'{output_dir}/3_price_distributions.png', dpi=300, bbox_inches='tight')
plt.show()

# ──────────────────────────────────────
# 4. 特征工程
# ──────────────────────────────────────
print("\n进行特征工程...")


# 创建技术指标
def create_features(data):
    # 复制数据避免修改原始数据
    df_featured = data.copy()

    # 1. 移动平均线
    df_featured['MA5'] = df_featured['Close'].rolling(window=5).mean()
    df_featured['MA10'] = df_featured['Close'].rolling(window=10).mean()
    df_featured['MA20'] = df_featured['Close'].rolling(window=20).mean()
    df_featured['MA50'] = df_featured['Close'].rolling(window=50).mean()

    # 2. 移动平均线交叉
    df_featured['MA5_MA20_cross'] = df_featured['MA5'] - df_featured['MA20']

    # 3. 价格动量
    df_featured['Momentum'] = df_featured['Close'] - df_featured['Close'].shift(5)

    # 4. 价格变化率
    df_featured['Return'] = df_featured['Close'].pct_change()

    # 5. 波动率
    df_featured['Volatility'] = df_featured['Return'].rolling(window=20).std()

    # 6. 高低价差
    df_featured['High_Low_Spread'] = df_featured['High'] - df_featured['Low']

    # 7. 收盘开盘价差
    df_featured['Close_Open_Spread'] = df_featured['Close'] - df_featured['Open']

    # 8. 成交量变化
    df_featured['Volume_Change'] = df_featured['Volume'].pct_change()

    # 9. RSI相对强弱指数 (简化版)
    delta = df_featured['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df_featured['RSI'] = 100 - (100 / (1 + rs))

    # 10. MACD指标
    exp1 = df_featured['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df_featured['Close'].ewm(span=26, adjust=False).mean()
    df_featured['MACD'] = exp1 - exp2
    df_featured['MACD_Signal'] = df_featured['MACD'].ewm(span=9, adjust=False).mean()

    # 11. 布林带
    df_featured['BB_Middle'] = df_featured['Close'].rolling(window=20).mean()
    bb_std = df_featured['Close'].rolling(window=20).std()
    df_featured['BB_Upper'] = df_featured['BB_Middle'] + (bb_std * 2)
    df_featured['BB_Lower'] = df_featured['BB_Middle'] - (bb_std * 2)
    df_featured['BB_Width'] = df_featured['BB_Upper'] - df_featured['BB_Lower']

    # 12. 价格位置（相对于高低点的位置）
    df_featured['Price_Position'] = (df_featured['Close'] - df_featured['Low']) / (
                df_featured['High'] - df_featured['Low'])

    # 删除NaN值
    df_featured = df_featured.dropna()

    return df_featured


# 创建特征
df_featured = create_features(df)
print(f"特征工程后数据形状: {df_featured.shape}")
print(f"特征列: {df_featured.columns.tolist()}")

# ──────────────────────────────────────
# 5. 数据预处理
# ──────────────────────────────────────
print("\n进行数据预处理...")

# 选择特征列和目标列
feature_columns = ['Open', 'High', 'Low', 'Volume', 'MA5', 'MA10', 'MA20',
                   'MA5_MA20_cross', 'Momentum', 'Volatility', 'High_Low_Spread',
                   'Close_Open_Spread', 'Volume_Change', 'RSI', 'MACD',
                   'MACD_Signal', 'BB_Width', 'Price_Position']

target_column = 'Close'

# 数据标准化
scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))

X_scaled = scaler_X.fit_transform(df_featured[feature_columns])
y_scaled = scaler_y.fit_transform(df_featured[[target_column]])


# 创建时间序列数据
def create_sequences(X, y, time_steps=60):
    X_seq, y_seq = [], []
    for i in range(len(X) - time_steps):
        X_seq.append(X[i:(i + time_steps)])
        y_seq.append(y[i + time_steps])
    return np.array(X_seq), np.array(y_seq)


# 设置时间步长
time_steps = 60
X_seq, y_seq = create_sequences(X_scaled, y_scaled, time_steps)
print(f"序列数据形状: X_seq={X_seq.shape}, y_seq={y_seq.shape}")

# 划分训练集和测试集
train_size = int(len(X_seq) * 0.8)
X_train, X_test = X_seq[:train_size], X_seq[train_size:]
y_train, y_test = y_seq[:train_size], y_seq[train_size:]
print(f"训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")

# ──────────────────────────────────────
# 6. 构建CNN神经网络模型
# ──────────────────────────────────────
print("\n构建CNN神经网络模型...")

model = Sequential([
    # 第一层卷积层
    Conv1D(filters=128, kernel_size=3, activation='relu',
           input_shape=(time_steps, len(feature_columns)),
           padding='same'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),

    # 第二层卷积层
    Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),

    # 第三层卷积层
    Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),

    # 展平层
    Flatten(),

    # 全连接层
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),

    # 输出层
    Dense(1)
])

# 编译模型
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae', 'mse']
)

model.summary()

# ──────────────────────────────────────
# 7. 训练模型
# ──────────────────────────────────────
print("\n训练模型...")

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

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# ──────────────────────────────────────
# 8. 模型评估
# ──────────────────────────────────────
print("\n模型评估...")

# 预测
y_pred_scaled = model.predict(X_test)

# 反标准化
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_original = scaler_y.inverse_transform(y_test)

# 确保是一维数组，解决残差可视化问题
y_pred_flat = y_pred.flatten()
y_test_flat = y_test_original.flatten()

# 计算指标
rmse = np.sqrt(mean_squared_error(y_test_flat, y_pred_flat))
mae = mean_absolute_error(y_test_flat, y_pred_flat)
r2 = r2_score(y_test_flat, y_pred_flat)

print(f"测试集评估指标:")
print(f"RMSE (均方根误差): {rmse:.4f}")
print(f"MAE (平均绝对误差): {mae:.4f}")
print(f"R² (决定系数): {r2:.4f}")

# ──────────────────────────────────────
# 9. 可视化结果
# ──────────────────────────────────────
print("\n创建结果可视化图表...")

# 9.1 训练历史图
plt.figure(figsize=(14, 10))

# 损失图
plt.subplot(2, 2, 1)
plt.plot(history.history['loss'], label='训练损失', color='blue', linewidth=2)
plt.plot(history.history['val_loss'], label='验证损失', color='red', linewidth=2)
plt.title('CNN模型训练损失', fontsize=14, fontweight='bold')
plt.xlabel('训练轮次', fontsize=12)
plt.ylabel('损失 (MSE)', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

# MAE图
plt.subplot(2, 2, 2)
plt.plot(history.history['mae'], label='训练MAE', color='blue', linewidth=2)
plt.plot(history.history['val_mae'], label='验证MAE', color='red', linewidth=2)
plt.title('CNN模型训练MAE', fontsize=14, fontweight='bold')
plt.xlabel('训练轮次', fontsize=12)
plt.ylabel('MAE', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

# 学习率变化图
if 'lr' in history.history:
    plt.subplot(2, 2, 3)
    plt.plot(history.history['lr'], color='green', linewidth=2)
    plt.title('学习率变化', fontsize=14, fontweight='bold')
    plt.xlabel('训练轮次', fontsize=12)
    plt.ylabel('学习率', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

# 预测误差分布
plt.subplot(2, 2, 4)
errors = y_test_flat - y_pred_flat
plt.hist(errors, bins=50, color='purple', alpha=0.7, edgecolor='black')
plt.title('预测误差分布', fontsize=14, fontweight='bold')
plt.xlabel('预测误差', fontsize=12)
plt.ylabel('频率', fontsize=12)
plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/4_training_history_cnn.png', dpi=300, bbox_inches='tight')
plt.show()

# 9.2 预测结果与真实值对比图
plt.figure(figsize=(16, 8))

# 获取对应的日期
test_dates = df_featured.index[-len(y_test_flat):]

plt.plot(test_dates, y_test_flat, label='实际股价', color='blue', linewidth=2.5, alpha=0.9)
plt.plot(test_dates, y_pred_flat, label='预测股价 (CNN)', color='red', linewidth=2, linestyle='--', alpha=0.9)

# 添加误差带
plt.fill_between(test_dates,
                 y_pred_flat - rmse,
                 y_pred_flat + rmse,
                 color='orange', alpha=0.2, label=f'±RMSE ({rmse:.2f})')

# 在图上添加指标文本
text_str = f'RMSE = {rmse:.4f}\nMAE = {mae:.4f}\nR² = {r2:.4f}'
plt.text(0.02, 0.98, text_str, transform=plt.gca().transAxes,
         fontsize=12, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.title('Apple股票价格预测结果 (CNN模型)', fontsize=16, fontweight='bold')
plt.xlabel('日期', fontsize=12)
plt.ylabel('股价 (USD)', fontsize=12)
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{output_dir}/5_prediction_vs_actual_cnn.png', dpi=300, bbox_inches='tight')
plt.show()

# 9.3 残差图 (修复后的版本)
residuals = y_test_flat - y_pred_flat

plt.figure(figsize=(14, 10))

# 残差分布 - 确保使用一维数组
plt.subplot(2, 2, 1)
# 修复：确保residuals是一维数组
residuals_flat = residuals.flatten() if hasattr(residuals, 'flatten') else residuals
plt.hist(residuals_flat, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
plt.title('预测残差分布', fontsize=14)
plt.xlabel('残差', fontsize=12)
plt.ylabel('频率', fontsize=12)
plt.axvline(x=0, color='red', linestyle='--', linewidth=1.5)

# 残差散点图
plt.subplot(2, 2, 2)
plt.scatter(y_pred_flat, residuals_flat, alpha=0.5, color='green')
plt.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
plt.title('预测值与残差关系', fontsize=14)
plt.xlabel('预测值', fontsize=12)
plt.ylabel('残差', fontsize=12)

# Q-Q图
plt.subplot(2, 2, 3)
from scipy import stats

stats.probplot(residuals_flat, dist="norm", plot=plt)
plt.title('残差Q-Q图', fontsize=14)

# 预测值与实际值散点图
plt.subplot(2, 2, 4)
plt.scatter(y_test_flat, y_pred_flat, alpha=0.5, color='purple')
plt.plot([y_test_flat.min(), y_test_flat.max()],
         [y_test_flat.min(), y_test_flat.max()],
         'r--', linewidth=2)
plt.title('预测值 vs 实际值', fontsize=14)
plt.xlabel('实际值', fontsize=12)
plt.ylabel('预测值', fontsize=12)

# 在散点图上添加R²值
plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes,
         fontsize=12, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(f'{output_dir}/6_residual_analysis_cnn.png', dpi=300, bbox_inches='tight')
plt.show()

# 9.4 预测误差图
plt.figure(figsize=(14, 6))

# 计算百分比误差
percentage_error = (residuals_flat / y_test_flat) * 100

plt.subplot(1, 2, 1)
plt.plot(test_dates, percentage_error, color='orange', alpha=0.7, linewidth=1.5)
plt.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
plt.fill_between(test_dates, 0, percentage_error,
                 where=(percentage_error > 0),
                 color='green', alpha=0.3, label='高估')
plt.fill_between(test_dates, 0, percentage_error,
                 where=(percentage_error < 0),
                 color='red', alpha=0.3, label='低估')
plt.title('预测百分比误差', fontsize=14)
plt.xlabel('日期', fontsize=12)
plt.ylabel('百分比误差 (%)', fontsize=12)
plt.legend()
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
error_bins = [-10, -5, -2, 0, 2, 5, 10]
error_counts = []
for i in range(len(error_bins) - 1):
    count = np.sum((percentage_error >= error_bins[i]) & (percentage_error < error_bins[i + 1]))
    error_counts.append(count)

colors = ['red', 'orange', 'yellow', 'lightgreen', 'green', 'blue']
bars = plt.bar(range(len(error_counts)), error_counts, color=colors, alpha=0.7, edgecolor='black')
plt.title('预测误差分布', fontsize=14)
plt.xlabel('误差范围 (%)', fontsize=12)
plt.ylabel('样本数量', fontsize=12)
plt.xticks(range(len(error_counts)), [f'{error_bins[i]}-{error_bins[i + 1]}' for i in range(len(error_bins) - 1)],
           rotation=45)

# 在柱状图上添加数值
for bar, count in zip(bars, error_counts):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
             str(count), ha='center', va='bottom')

plt.tight_layout()
plt.savefig(f'{output_dir}/7_prediction_error_analysis_cnn.png', dpi=300, bbox_inches='tight')
plt.show()

# 9.5 滚动预测图 (展示滚动预测效果)
plt.figure(figsize=(16, 8))

# 计算滚动预测
rolling_predictions = []
for i in range(len(X_test)):
    # 使用前i+1个测试样本进行预测
    if i < 10:  # 只取前10个预测点，避免计算量过大
        pred = model.predict(X_test[i:i + 1])
        pred_original = scaler_y.inverse_transform(pred)
        rolling_predictions.append(pred_original[0, 0])

# 绘制滚动预测
if len(rolling_predictions) > 0:
    plt.plot(test_dates[:len(rolling_predictions)], y_test_flat[:len(rolling_predictions)],
             label='实际股价', color='blue', linewidth=2, alpha=0.8)
    plt.plot(test_dates[:len(rolling_predictions)], rolling_predictions,
             label='滚动预测', color='green', linewidth=2, marker='o', markersize=4, alpha=0.8)

    plt.title('CNN模型滚动预测效果', fontsize=16, fontweight='bold')
    plt.xlabel('日期', fontsize=12)
    plt.ylabel('股价 (USD)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/8_rolling_prediction_cnn.png', dpi=300, bbox_inches='tight')
    plt.show()

# ──────────────────────────────────────
# 10. 特征重要性分析
# ──────────────────────────────────────
print("\n进行特征重要性分析...")

# 创建特征重要性图（基于模型权重）
try:
    # 获取第一层卷积层的权重
    conv1_weights = model.layers[0].get_weights()[0]

    # 计算每个特征的权重绝对值之和
    feature_importance = np.sum(np.abs(conv1_weights), axis=(0, 1))

    # 归一化
    feature_importance = feature_importance / np.sum(feature_importance)

    # 创建特征重要性图
    plt.figure(figsize=(12, 8))
    indices = np.argsort(feature_importance)[::-1]

    plt.barh(range(len(indices)), feature_importance[indices], align='center',
             color='steelblue', alpha=0.7, edgecolor='black')
    plt.yticks(range(len(indices)), [feature_columns[i] for i in indices], fontsize=10)
    plt.xlabel('特征重要性 (基于第一层卷积权重)', fontsize=12)
    plt.title('CNN特征重要性分析', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')

    # 添加重要性值
    for i, v in enumerate(feature_importance[indices]):
        plt.text(v + 0.001, i, f'{v:.4f}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/9_feature_importance_cnn.png', dpi=300, bbox_inches='tight')
    plt.show()
except Exception as e:
    print(f"特征重要性分析出错: {e}")

# 11. 未来预测（示例）
print("\n进行未来预测示例...")

# 使用最后的时间步数据预测未来几天
last_sequence = X_seq[-1:]  # 最后一个序列
future_predictions = []
n_future = 10  # 预测未来10天

current_sequence = last_sequence.copy()
for i in range(n_future):
    # 预测下一个值
    next_pred_scaled = model.predict(current_sequence)
    future_predictions.append(next_pred_scaled[0, 0])

    # 更新序列（这里简化处理，实际应该更新所有特征）
    # 在实际应用中，需要更新序列中的所有特征值
    current_sequence = np.roll(current_sequence, -1, axis=1)
    # 只更新第一个特征（收盘价），其他特征在实际应用中需要根据规则更新
    current_sequence[0, -1, 0] = next_pred_scaled[0, 0]

# 反标准化
future_predictions_original = scaler_y.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# 创建未来预测图
plt.figure(figsize=(14, 7))

# 历史数据（最后100天）
history_dates = df_featured.index[-100:]
history_prices = df_featured['Close'].values[-100:]

# 未来日期（简单假设每个交易日）
last_date = df_featured.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_future, freq='B')

plt.plot(history_dates, history_prices, label='历史股价', color='blue', linewidth=2)
plt.plot(future_dates, future_predictions_original, label='未来预测 (CNN)',
         color='red', linewidth=2.5, marker='o', markersize=8, linestyle='--')

# 添加置信区间
future_std = np.std(residuals_flat)
plt.fill_between(future_dates,
                 future_predictions_original.flatten() - future_std,
                 future_predictions_original.flatten() + future_std,
                 color='red', alpha=0.2, label='预测区间')

plt.title('Apple股票未来价格预测 (CNN模型)', fontsize=16, fontweight='bold')
plt.xlabel('日期', fontsize=12)
plt.ylabel('股价 (USD)', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{output_dir}/10_future_prediction_cnn.png', dpi=300, bbox_inches='tight')
plt.show()

# 12. 模型架构可视化
print("\n创建模型架构图...")

try:
    from tensorflow.keras.utils import plot_model

    plot_model(model, to_file=f'{output_dir}/11_cnn_model_architecture.png',
               show_shapes=True, show_layer_names=True, dpi=100)
    print("模型架构图已保存")
except:
    print("无法生成模型架构图，需要安装pydot和graphviz")

# 13. 保存模型和结果
print("\n保存模型和结果...")

# 保存模型
model.save(f'{output_dir}/apple_stock_cnn_model.h5')
print(f"模型已保存到: {output_dir}/apple_stock_cnn_model.h5")

# 保存评估指标
results_df = pd.DataFrame({
    'Metric': ['RMSE', 'MAE', 'R2'],
    'Value': [rmse, mae, r2],
    'Description': ['均方根误差', '平均绝对误差', '决定系数']
})

results_df.to_csv(f'{output_dir}/evaluation_metrics_cnn.csv', index=False)
print(f"评估指标已保存到: {output_dir}/evaluation_metrics_cnn.csv")

# 保存预测结果
prediction_results = pd.DataFrame({
    'Date': test_dates,
    'Actual_Price': y_test_flat,
    'Predicted_Price': y_pred_flat,
    'Residual': residuals_flat,
    'Percentage_Error': percentage_error
})

prediction_results.to_csv(f'{output_dir}/prediction_results_cnn.csv', index=False)
print(f"预测结果已保存到: {output_dir}/prediction_results_cnn.csv")

# 14. 生成总结报告
print(f"\n{'=' * 60}")
print("CNN模型训练和评估完成!")
print(f"{'=' * 60}")
print(f"数据集: apple_5yr_one1.csv")
print(f"数据时间范围: {df.index.min()} 到 {df.index.max()}")
print(f"总数据点: {len(df)}")
print(f"特征工程后数据点: {len(df_featured)}")
print(f"使用的特征数量: {len(feature_columns)}")
print(f"时间步长 (Time Steps): {time_steps}")
print(f"训练集大小: {X_train.shape[0]}")
print(f"测试集大小: {X_test.shape[0]}")
print(f"CNN神经网络结构:")
print(f"  - Conv1D(128, kernel=3) → BatchNorm → MaxPooling(2)")
print(f"  - Conv1D(64, kernel=3) → BatchNorm → MaxPooling(2)")
print(f"  - Conv1D(32, kernel=3) → BatchNorm → MaxPooling(2)")
print(f"  - Flatten → Dense(128) → Dense(64) → Dense(32) → Dense(1)")
print(f"最佳验证损失: {min(history.history['val_loss']):.6f}")
print(f"最终评估指标:")
print(f"  - RMSE: {rmse:.4f}")
print(f"  - MAE: {mae:.4f}")
print(f"  - R²: {r2:.4f}")
print(f"所有图表和结果已保存到文件夹: '{output_dir}/'")
print(f"{'=' * 60}")

# 显示保存的文件
print("\n生成的图表文件:")
png_files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
for i, file in enumerate(sorted(png_files), 1):
    print(f"  {i:2d}. {file}")

# 显示保存的模型文件
print("\n保存的模型文件:")
model_files = [f for f in os.listdir(output_dir) if f.endswith(('.h5', '.csv'))]
for i, file in enumerate(sorted(model_files), 1):
    print(f"  {i:2d}. {file}")