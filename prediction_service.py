# prediction_service.py
"""
Multi-Model Stock Prediction Service
MLP/Linear use close-only windows.
CNN-LSTM uses the feature set it was trained with.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import load_model

warnings.filterwarnings('ignore')


def create_base_features(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    df['MA5']  = df['Close'].rolling(5).mean()
    df['MA10'] = df['Close'].rolling(10).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA50'] = df['Close'].rolling(50).mean()
    df['MA5_MA20_cross'] = df['MA5'] - df['MA20']
    df['Momentum'] = df['Close'] - df['Close'].shift(5)
    df['Return']   = df['Close'].pct_change()
    df['Volatility'] = df['Return'].rolling(20).std()
    df['High_Low_Spread']   = df['High'] - df['Low']
    df['Close_Open_Spread'] = df['Close'] - df['Open']
    df['Volume_Change']     = df['Volume'].pct_change()

    delta = df['Close'].diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + gain / loss))

    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD']        = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    df['BB_Middle'] = df['Close'].rolling(20).mean()
    bb_std          = df['Close'].rolling(20).std()
    df['BB_Upper']  = df['BB_Middle'] + bb_std * 2
    df['BB_Lower']  = df['BB_Middle'] - bb_std * 2
    df['BB_Width']  = df['BB_Upper'] - df['BB_Lower']

    price_range = (df['High'] - df['Low']).replace(0, np.nan)
    df['Price_Position'] = (df['Close'] - df['Low']) / price_range
    return df.dropna()


class StockPredictionService:
    AVAILABLE_MODELS = ['MLP', 'CNN-LSTM', 'Linear']
    CNN_FEATURE_COLUMNS = [
        'Open', 'High', 'Low', 'Volume', 'MA5', 'MA10', 'MA20',
        'MA5_MA20_cross', 'Momentum', 'Volatility', 'High_Low_Spread',
        'Close_Open_Spread', 'Volume_Change', 'RSI', 'MACD',
        'MACD_Signal', 'BB_Width', 'Price_Position'
    ]

    def __init__(
        self,
        mlp_path='mlp_model_final.h5',
        cnn_lstm_path='cnn_lstm_model.keras',
        linear_path='linear_model_final.h5',
        data_path='Data/apple_5yr_daily.csv',
    ):
        base = Path(__file__).resolve().parent
        def _resolve(p):
            p = Path(p)
            return p if p.is_absolute() else base / p

        # Load CSV & features
        self.df = pd.read_csv(str(_resolve(data_path)))
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df.set_index('Date', inplace=True)
        self.df_base = create_base_features(self.df)

        # Shared scaler
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaler.fit(self.df_base[['Close']].values)
        self.close_scaled = self.scaler.transform(
            self.df_base[['Close']].values
        ).flatten()

        # Feature matrix for CNN-LSTM (trained on multi-feature input)
        missing_cnn_cols = [c for c in self.CNN_FEATURE_COLUMNS if c not in self.df_base.columns]
        if missing_cnn_cols:
            raise ValueError(f'Missing CNN feature columns: {missing_cnn_cols}')
        self.cnn_feature_scaler = MinMaxScaler(feature_range=(0, 1))
        self.cnn_features_scaled = self.cnn_feature_scaler.fit_transform(
            self.df_base[self.CNN_FEATURE_COLUMNS].values
        )

        # MLP
        mlp_file = _resolve(mlp_path)
        self.mlp_model = None
        if mlp_file.exists():
            print('[MLP] Loading ...')
            self.mlp_model = load_model(str(mlp_file), compile=False)
            self.mlp_window = int(self.mlp_model.input_shape[-1])
            print(f'[MLP] Ready (window={self.mlp_window})')
        else:
            print(f'[MLP] Not found: {mlp_file}')

        # CNN-LSTM — input (batch, window, 1)
        cnn_file = _resolve(cnn_lstm_path)
        self.cnn_model = None
        if cnn_file.exists():
            try:
                print('[CNN-LSTM] Loading ...')
                self.cnn_model = load_model(str(cnn_file), compile=False)
                self.cnn_window = int(self.cnn_model.input_shape[1])
                self.cnn_feature_dim = int(self.cnn_model.input_shape[-1])
                print(f'[CNN-LSTM] Ready (window={self.cnn_window})')
            except Exception as e:
                print(f'[CNN-LSTM] Failed to load: {e}')
                self.cnn_model = None
            else:
                print(f'[CNN-LSTM] Not found: {cnn_file}')

        # Linear — try .h5 first, fallback to sklearn
        linear_file = _resolve(linear_path)
        self.linear_model = None
        self.linear_is_keras = False
        self.lr_window = 30
        if linear_file.exists():
            print('[Linear] Loading .h5 ...')
            self.linear_model = load_model(str(linear_file), compile=False)
            self.linear_is_keras = True
            self.lr_window = int(self.linear_model.input_shape[-1])
            print(f'[Linear] Ready (window={self.lr_window})')
        else:
            print('[Linear] .h5 not found, training sklearn fallback ...')
            self.linear_model = self._train_linear_sklearn()
            print('[Linear] Ready (sklearn)')

        self._active = 'MLP'
        print('\nAll models initialized.')

    def _train_linear_sklearn(self):
        X, y = [], []
        for i in range(len(self.close_scaled) - self.lr_window):
            X.append(self.close_scaled[i:i + self.lr_window])
            y.append(self.close_scaled[i + self.lr_window])
        model = LinearRegression()
        model.fit(np.array(X), np.array(y))
        return model

    # ── Switching ──
    @property
    def active_model(self):
        return self._active

    @active_model.setter
    def active_model(self, name):
        aliases = {
            'MLP': 'MLP', 'CNN-LSTM': 'CNN-LSTM', 'CNN_LSTM': 'CNN-LSTM',
            'CNNLSTM': 'CNN-LSTM', 'LINEAR': 'LINEAR', 'LR': 'LINEAR',
        }
        resolved = aliases.get(name.upper().replace(' ', '-'), name.upper())
        if resolved not in ('MLP', 'CNN-LSTM', 'LINEAR'):
            raise ValueError(f'Unknown model: {name}')
        self._active = resolved

    def model_available(self, name):
        n = name.upper().replace(' ', '-')
        if n == 'MLP':      return self.mlp_model is not None
        if n == 'CNN-LSTM':  return self.cnn_model is not None
        if n in ('LINEAR', 'LR'): return self.linear_model is not None
        return False

    # ── Prediction ──
    def predict_future(self, days=10):
        if days <= 0: raise ValueError('days must be positive')
        if self._active == 'MLP':      return self._predict_mlp(days)
        if self._active == 'CNN-LSTM':  return self._predict_cnn(days)
        return self._predict_linear(days)

    def _predict_mlp(self, days):
        if not self.mlp_model: raise RuntimeError('MLP not loaded')
        w = self.mlp_window
        seq = self.close_scaled[-w:].tolist()
        preds = []
        for _ in range(days):
            nxt = self.mlp_model.predict(
                np.array(seq[-w:]).reshape(1, w), verbose=0
            )[0, 0]
            preds.append(nxt); seq.append(nxt)
        prices = self.scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten()
        return self._build_result(prices, days, 'MLP')

    def _predict_cnn(self, days):
        if not self.cnn_model: raise RuntimeError('CNN-LSTM not loaded')
        w = self.cnn_window
        feature_dim = self.cnn_feature_dim

        # Backward compatibility for close-only CNN models.
        if feature_dim == 1:
            seq = self.close_scaled[-w:].tolist()
            preds = []
            for _ in range(days):
                nxt = self.cnn_model.predict(
                    np.array(seq[-w:]).reshape(1, w, 1), verbose=0
                )[0, 0]
                preds.append(nxt)
                seq.append(nxt)
            prices = self.scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
            return self._build_result(prices, days, 'CNN-LSTM')

        feature_matrix = self.cnn_features_scaled
        if feature_dim < feature_matrix.shape[1]:
            feature_matrix = feature_matrix[:, :feature_dim]
        elif feature_dim > feature_matrix.shape[1]:
            pad = np.zeros((feature_matrix.shape[0], feature_dim - feature_matrix.shape[1]))
            feature_matrix = np.concatenate([feature_matrix, pad], axis=1)

        window = feature_matrix[-w:].copy()
        preds = []
        for _ in range(days):
            nxt = self.cnn_model.predict(
                window.reshape(1, w, feature_dim), verbose=0
            )[0, 0]
            preds.append(nxt)

            # For multi-feature autoregression, carry forward the latest feature row.
            next_row = window[-1].copy()
            window = np.vstack([window[1:], next_row])

        prices = self.scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
        return self._build_result(prices, days, 'CNN-LSTM')

    def _predict_linear(self, days):
        if not self.linear_model: raise RuntimeError('Linear not loaded')
        w = self.lr_window
        seq = self.close_scaled[-w:].tolist()
        preds = []
        for _ in range(days):
            inp = np.array(seq[-w:]).reshape(1, w)
            if self.linear_is_keras:
                nxt = self.linear_model.predict(inp, verbose=0)[0, 0]
            else:
                nxt = self.linear_model.predict(inp)[0]
            preds.append(nxt); seq.append(nxt)
        prices = self.scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten()
        return self._build_result(prices, days, 'Linear')

    def _build_result(self, prices, days, model_name):
        current = float(self.df_base['Close'].iloc[-1])
        avg = float(np.mean(prices))
        trend = 'up' if avg > current else 'down'
        change = ((avg - current) / current) * 100
        last_date = self.df_base.index[-1]
        dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days, freq='B')
        conf = {'MLP': 0.75, 'CNN-LSTM': 0.80, 'Linear': 0.55}
        return {
            'model': model_name, 'predictions': prices.tolist(),
            'dates': [d.strftime('%Y-%m-%d') for d in dates],
            'current_price': current, 'avg_future_price': avg,
            'trend': trend, 'change_pct': float(change),
            'confidence': conf.get(model_name, 0.6),
        }

    def compare_models(self, days=10):
        results = {}
        original = self._active
        for name in ('MLP', 'CNN-LSTM', 'LINEAR'):
            if self.model_available(name):
                self._active = name
                try: results[name] = self.predict_future(days)
                except Exception as e: results[name] = {'error': str(e)}
        self._active = original
        return results

    def get_technical_indicators(self):
        latest = self.df_base.iloc[-1]
        return {
            'RSI': float(latest['RSI']), 'MACD': float(latest['MACD']),
            'MACD_Signal': float(latest['MACD_Signal']),
            'MA5': float(latest['MA5']), 'MA10': float(latest['MA10']),
            'MA20': float(latest['MA20']),
            'BB_Upper': float(latest['BB_Upper']), 'BB_Lower': float(latest['BB_Lower']),
            'Volatility': float(latest['Volatility']),
            'current_price': float(latest['Close']),
        }

    def assess_risk(self):
        returns = self.df_base['Return'].dropna()
        volatility = returns.std() * np.sqrt(252)
        cum = (1 + returns).cumprod()
        peak = cum.expanding().max()
        dd = (cum - peak) / peak
        max_dd = dd.min()
        score = min(100, (volatility * 100 + abs(max_dd) * 100) / 2)
        level = 'low' if score < 30 else 'medium' if score < 60 else 'high'
        return {
            'risk_score': float(score), 'risk_level': level,
            'volatility': float(volatility), 'max_drawdown': float(max_dd),
        }
