# ◈ AI Investment Agent

An intelligent investment analysis system for Apple (AAPL) stock that combines multiple deep learning models with LLM-powered reasoning to deliver actionable trading recommendations.

## Architecture

```
                         ┌─────────────────────────────┐
                         │       agent_system.py        │
  ┌──────────────┐       │                             │       ┌──────────────┐
  │              │       │  Tool 1: Price Predictor    │       │              │
  │   app.py     │──────▶│  Tool 2: Model Comparison   │──────▶│   Gemini     │
  │  (Streamlit) │       │  Tool 3: Technical Analysis │       │  2.0 Flash   │
  │              │◀──────│  Tool 4: Risk Assessment    │◀──────│              │
  └──────────────┘       └──────────────┬──────────────┘       └──────────────┘
                                        │
                         ┌──────────────▼──────────────┐
                         │    prediction_service.py     │
                         │                             │
                         │  ┌───────┐ ┌────────┐ ┌──┐ │
                         │  │  MLP  │ │CNN-LSTM│ │LR│ │
                         │  └───────┘ └────────┘ └──┘ │
                         │  + Technical Indicators     │
                         │  + Risk Engine              │
                         └─────────────────────────────┘
```

**Data flow:** User selects a model and asks a question → Agent runs all four tools (single-model prediction, cross-model comparison, technical analysis, risk assessment) → Gemini synthesizes a BUY / HOLD / SELL recommendation grounded in multi-model consensus.

## Models

| Model | Architecture | Input | Scaler | Strengths |
|---|---|---|---|---|
| **MLP** | Dense feed-forward | Close price window (flat) | MinMaxScaler | Fast, stable on blue-chip patterns |
| **CNN-LSTM** | Conv1D → LSTM stack | 60-step × 38 features (3D) | StandardScaler | Highest capacity, captures complex temporal + spatial patterns |
| **Linear Regression** | OLS | 30-day close window | MinMaxScaler | Interpretable baseline for comparison |

The frontend overlays all three forecasts on a single chart and displays a consensus badge (ALL BULLISH / ALL BEARISH / MIXED) so users can gauge model agreement at a glance.

## Core Components

| File | Role |
|---|---|
| `prediction_service.py` | Loads all models, runs predictions, computes indicators and risk metrics. Exposes `active_model` switching and `compare_models()`. |
| `agent_system.py` | Defines four analysis tools, runs them, and sends combined output to Gemini for structured reasoning. |
| `app.py` | Streamlit dashboard — model selector, metric cards, comparison table, multi-line forecast chart, and natural-language Q&A. |

## Quick Start

### 1. Prerequisites

- Python 3.9+
- `mlp_model_final.h5` (trained MLP weights)
- `stock_prediction_cnn_lstm_plots/apple_stock_cnn_lstm_model.h5` (trained CNN-LSTM weights)
- `apple_5yr_one1.csv` (historical AAPL data)
- Google API key for Gemini

### 2. Install

```bash
pip install -r requirements.txt
```

### 3. Configure

Create `.env` in the project root:

```env
GOOGLE_API_KEY=your_key_here
```

### 4. Run

```bash
streamlit run app.py
```

Open `http://localhost:8501`.

## Project Structure

```
ai_investment_agent/
├── app.py                                          # Streamlit frontend
├── agent_system.py                                 # LLM agent + tools
├── prediction_service.py                           # Multi-model service
├── mlp_model_final.h5                              # MLP weights
├── stock_prediction_cnn_lstm_plots/
│   └── apple_stock_cnn_lstm_model.h5               # CNN-LSTM weights
├── apple_5yr_one1.csv                              # Price data
├── cnn_lstm.py                                     # CNN-LSTM training script
├── .env                                            # API keys (git-ignored)
├── requirements.txt
└── README.md
```

## Analysis Pipeline

For every user query the agent runs four tools:

1. **Price Predictor** — forecasts the next N days using the currently selected model (MLP, CNN-LSTM, or Linear).
2. **Model Comparison** — runs all available models and reports trend, change %, and confidence for each, plus a consensus signal.
3. **Technical Analysis** — RSI, MACD crossover, moving averages, and Bollinger Band positioning.
4. **Risk Assessment** — annualised volatility, max drawdown, composite risk score, and position-sizing advice.

Gemini receives all four outputs and returns a structured recommendation with reasoning, position size, and stop-loss rules.

## Tech Stack

| Layer | Technology |
|---|---|
| Deep Learning | MLP + CNN-LSTM (Keras / TensorFlow) |
| Baseline Model | Linear Regression (scikit-learn) |
| Feature Engineering | pandas, NumPy, scikit-learn |
| LLM Reasoning | Gemini 2.0 Flash (Google) |
| Frontend | Streamlit + Plotly |
| Config | python-dotenv |

## Requirements

```
streamlit>=1.30.0
plotly>=5.18.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
tensorflow>=2.14.0
python-dotenv>=1.0.0
google-genai>=1.0.0
```

## Disclaimer

This project is for **educational and demonstration purposes only**. It does not constitute financial advice. Stock predictions are inherently uncertain — never make investment decisions based solely on model outputs.

## License

MIT
