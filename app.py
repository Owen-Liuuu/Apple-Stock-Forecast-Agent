# app.py
"""
AI Investment Agent — Multi-Model Streamlit Frontend
Supports: MLP · CNN-LSTM · Linear Regression
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from prediction_service import StockPredictionService
from agent_system import ask_investment_agent

# ──────────────────────────────────────
# Page config
# ──────────────────────────────────────
st.set_page_config(
    page_title="AI Investment Agent",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────
# CSS — dark tech-minimal theme
# ──────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600&family=Outfit:wght@300;400;500;600;700&display=swap');

    :root {
        --bg: #0a0a0f;
        --bg2: #111118;
        --card: #16161f;
        --border: #1e1e2e;
        --t1: #e8e8ed;
        --t2: #8888a0;
        --accent: #00d4aa;
        --accent-d: #00d4aa22;
        --red: #ff4466;
        --red-d: #ff446622;
        --yellow: #ffbb33;
        --blue: #5b8cff;
    }

    .stApp,
    [data-testid="stAppViewContainer"] {
        background: var(--bg) !important;
        color: var(--t1) !important;
        font-family: 'Outfit', sans-serif !important;
    }
    .main .block-container { max-width: 1100px; padding-top: 2rem; padding-bottom: 4rem; }

    h1,h2,h3 { font-family: 'Outfit', sans-serif !important; font-weight: 600 !important; }
    h1 { color: var(--t1) !important; font-size: 2rem !important; letter-spacing: -0.03em; }
    .stApp p, .stApp li, .stApp span { color: var(--t1); }

    /* Header */
    .hdr { display:flex; align-items:center; justify-content:space-between; padding:1rem 0 1.5rem; border-bottom:1px solid var(--border); margin-bottom:2rem; }
    .hdr-title { font-family:'JetBrains Mono',monospace; font-size:1.1rem; font-weight:500; color:var(--t1); }
    .hdr-tag { font-family:'JetBrains Mono',monospace; font-size:0.7rem; color:var(--accent); background:var(--accent-d); padding:4px 10px; border-radius:4px; letter-spacing:0.05em; }

    /* Metric cards */
    .mrow { display:flex; gap:12px; margin-bottom:1.5rem; }
    .mcard { flex:1; background:var(--card); border:1px solid var(--border); border-radius:8px; padding:1.1rem 1.2rem; }
    .mlbl { font-family:'JetBrains Mono',monospace; font-size:0.65rem; text-transform:uppercase; letter-spacing:0.08em; color:var(--t2); margin-bottom:6px; }
    .mval { font-family:'JetBrains Mono',monospace; font-size:1.5rem; font-weight:600; color:var(--t1); }
    .msub { font-family:'JetBrains Mono',monospace; font-size:0.75rem; margin-top:4px; }
    .up   { color:var(--accent) !important; }
    .down { color:var(--red) !important; }

    /* Chips */
    .chips { display:flex; gap:8px; flex-wrap:wrap; margin-bottom:1rem; }
    .chip { font-family:'JetBrains Mono',monospace; font-size:0.72rem; background:var(--card); border:1px solid var(--border); border-radius:6px; padding:6px 12px; color:var(--t2); }
    .chip b { color:var(--t1); font-weight:500; }

    /* Section */
    .sec { font-family:'JetBrains Mono',monospace; font-size:1rem; font-weight:600; text-transform:uppercase; letter-spacing:0.1em; color:var(--t2) !important; margin:2rem 0 0.8rem; padding-bottom:0.5rem; border-bottom:1px solid var(--border); }

    /* Agent output */
    .agent-out { background:var(--card); border:1px solid var(--border); border-radius:8px; padding:1.5rem; font-size:0.9rem; line-height:1.7; white-space:pre-wrap; font-family:'Outfit',sans-serif; }

    /* Risk badge */
    .rbadge { display:inline-block; font-family:'JetBrains Mono',monospace; font-size:0.7rem; font-weight:600; letter-spacing:0.06em; padding:4px 12px; border-radius:4px; }
    .risk-low    { background:var(--accent-d); color:var(--accent); }
    .risk-medium { background:#ffbb3322; color:var(--yellow); }
    .risk-high   { background:var(--red-d); color:var(--red); }

    /* Comparison table */
    .cmp-table { width:100%; border-collapse:collapse; font-family:'JetBrains Mono',monospace; font-size:0.8rem; }
    .cmp-table th { text-align:left; color:var(--t2); font-weight:400; font-size:0.65rem; text-transform:uppercase; letter-spacing:0.08em; padding:8px 12px; border-bottom:1px solid var(--border); }
    .cmp-table td { padding:10px 12px; border-bottom:1px solid var(--border); color:var(--t1); }
    .cmp-table tr:last-child td { border-bottom:none; }

    /* Streamlit overrides */
    .stTextInput > div > div > input { background:var(--card) !important; border:1px solid var(--border) !important; border-radius:8px !important; color:var(--t1) !important; font-family:'Outfit',sans-serif !important; padding:0.7rem 1rem !important; }
    .stTextInput > div > div > input:focus { border-color:var(--accent) !important; box-shadow:0 0 0 1px var(--accent-d) !important; }
    .stButton > button { background:var(--accent) !important; color:var(--bg) !important; border:none !important; border-radius:8px !important; font-family:'JetBrains Mono',monospace !important; font-weight:600 !important; font-size:0.8rem !important; letter-spacing:0.04em !important; padding:0.6rem 2rem !important; }
    .stButton > button:hover { opacity:0.85 !important; }
    .stSelectbox > div > div { background:var(--card) !important; border:1px solid var(--border) !important; border-radius:8px !important; color:var(--t1) !important; }
    .stSelectbox [data-baseweb="select"] > div { background:var(--card) !important; color:var(--t1) !important; border-color:var(--border) !important; }
    .stSelectbox [data-baseweb="select"] input,
    .stSelectbox [data-baseweb="select"] span,
    .stSelectbox [data-baseweb="select"] div { color:var(--t1) !important; }
    body [data-baseweb="popover"] [role="listbox"] { background:#ffffff !important; }
    body [data-baseweb="popover"] [role="option"] { color:#000000 !important; background:#ffffff !important; }
    body [data-baseweb="popover"] [role="option"] * {
        color:#000000 !important;
        -webkit-text-fill-color:#000000 !important;
        opacity:1 !important;
    }
    body [data-baseweb="popover"] * {
        color:#000000 !important;
        -webkit-text-fill-color:#000000 !important;
    }
    body [data-baseweb="popover"] [role="option"]:hover { background:#f2f2f2 !important; }
    body [data-baseweb="popover"] [role="option"][aria-selected="true"] { background:#e9f7f2 !important; color:#000000 !important; }
    .stSelectbox label { color:var(--t2) !important; font-family:'JetBrains Mono',monospace !important; font-size:1rem !important; text-transform:uppercase !important; letter-spacing:0.08em !important; }
    .stSpinner > div { color:var(--accent) !important; }
    div[data-testid="stExpander"] { background:var(--card) !important; border:1px solid var(--border) !important; border-radius:8px !important; }
    header[data-testid="stHeader"] { background:transparent !important; }
    #MainMenu, footer, .stDeployButton { display:none !important; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────
# Load service (cached)
# ──────────────────────────────────────
@st.cache_resource
def load_service():
    return StockPredictionService()
st.cache_resource.clear()
service = load_service()
st.write("MLP available:", service.model_available("MLP"))
st.write("CNN-LSTM available:", service.model_available("CNN-LSTM"))
st.write("LINEAR available:", service.model_available("LINEAR"))
st.write("CNN debug:", service.cnn_debug)


# ──────────────────────────────────────
# Header
# ──────────────────────────────────────
st.markdown("""
<div class="hdr">
    <div class="hdr-title">◈&ensp;AI Investment Agent</div>
    <div class="hdr-tag">AAPL · Multi-Model</div>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────
# Model selector
# ──────────────────────────────────────
st.markdown('<div class="sec">Model Selection</div>', unsafe_allow_html=True)

model_labels = {
    'MLP': 'MLP  —  Multilayer Perceptron',
    'CNN-LSTM': 'CNN-LSTM  —  Hybrid Deep Learning',
    'Linear': 'Linear Regression  —  Baseline',
}

model_options = [k for k in model_labels if service.model_available(k)]
if not model_options:
    st.error('No models available. Check model .h5 files.')
    st.stop()

sel_col, info_col = st.columns([2, 3])
with sel_col:
    selected_model = st.selectbox(
        'Active model',
        model_options,
        format_func=lambda k: model_labels.get(k, k),
        label_visibility='collapsed',
    )
    service.active_model = selected_model

with info_col:
    desc = {
        'MLP': 'Feed-forward network on close-price sequences. Fast inference, moderate accuracy.',
        'CNN-LSTM': 'Conv1D feature extraction + LSTM temporal modelling. Close-price input, highest capacity.',
        'Linear': 'OLS on a 30-day window. Statistical baseline for comparison.',
    }
    st.markdown(
        f'<div style="font-size:0.82rem; color:#8888a0; padding-top:8px;">{desc.get(selected_model, "")}</div>',
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────
# Dashboard — metrics
# ──────────────────────────────────────
indicators = service.get_technical_indicators()
risk       = service.assess_risk()
prediction = service.predict_future(days=10)

current    = indicators['current_price']
change_pct = prediction['change_pct']
tc         = 'up' if change_pct >= 0 else 'down'
sign       = '+' if change_pct >= 0 else ''
rl         = risk['risk_level'].lower()

st.markdown(f"""
<div class="mrow">
    <div class="mcard">
        <div class="mlbl">Current Price</div>
        <div class="mval">${current:,.2f}</div>
    </div>
    <div class="mcard">
        <div class="mlbl">10-Day Forecast · {selected_model}</div>
        <div class="mval {tc}">{sign}{change_pct:.2f}%</div>
        <div class="msub {tc}">→ ${prediction['avg_future_price']:,.2f} avg</div>
    </div>
    <div class="mcard">
        <div class="mlbl">Risk Level</div>
        <div style="margin-top:6px;"><span class="rbadge risk-{rl}">{rl.upper()}</span></div>
        <div class="msub" style="color:var(--t2);">Score {risk['risk_score']:.0f}/100</div>
    </div>
    <div class="mcard">
        <div class="mlbl">Model Confidence</div>
        <div class="mval" style="font-size:1.3rem;">{prediction['confidence']*100:.0f}%</div>
        <div class="msub" style="color:var(--t2);">Vol {risk['volatility']*100:.1f}% · DD {risk['max_drawdown']*100:.1f}%</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Indicator chips
st.markdown(f"""
<div class="chips">
    <div class="chip">RSI <b>{indicators['RSI']:.1f}</b></div>
    <div class="chip">MACD <b>{indicators['MACD']:.2f}</b></div>
    <div class="chip">Signal <b>{indicators['MACD_Signal']:.2f}</b></div>
    <div class="chip">MA5 <b>${indicators['MA5']:.2f}</b></div>
    <div class="chip">MA10 <b>${indicators['MA10']:.2f}</b></div>
    <div class="chip">MA20 <b>${indicators['MA20']:.2f}</b></div>
    <div class="chip">BB ↑ <b>${indicators['BB_Upper']:.2f}</b></div>
    <div class="chip">BB ↓ <b>${indicators['BB_Lower']:.2f}</b></div>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────
# Multi-model comparison table
# ──────────────────────────────────────
st.markdown('<div class="sec">Model Comparison — 10 Day Forecast</div>', unsafe_allow_html=True)

comparison = service.compare_models(days=10)

rows_html = ''
for name, r in comparison.items():
    if 'error' in r:
        rows_html += f'<tr><td>{name}</td><td colspan="4" style="color:var(--red);">Error: {r["error"]}</td></tr>'
    else:
        c = 'up' if r['change_pct'] >= 0 else 'down'
        s = '+' if r['change_pct'] >= 0 else ''
        rows_html += (
            f'<tr>'
            f'<td style="font-weight:500;">{name}</td>'
            f'<td>{r["trend"].upper()}</td>'
            f'<td class="{c}">{s}{r["change_pct"]:.2f}%</td>'
            f'<td>${r["avg_future_price"]:,.2f}</td>'
            f'<td>{r["confidence"]*100:.0f}%</td>'
            f'</tr>'
        )

st.markdown(f"""
<table class="cmp-table">
    <thead><tr><th>Model</th><th>Trend</th><th>Change</th><th>Avg Price</th><th>Confidence</th></tr></thead>
    <tbody>{rows_html}</tbody>
</table>
""", unsafe_allow_html=True)

# Consensus badge
trends = [r['trend'] for r in comparison.values() if 'error' not in r]
up_n = trends.count('up')
if trends:
    if up_n == len(trends):
        badge = '<span class="rbadge risk-low">CONSENSUS: ALL BULLISH</span>'
    elif up_n == 0:
        badge = '<span class="rbadge risk-high">CONSENSUS: ALL BEARISH</span>'
    else:
        badge = f'<span class="rbadge risk-medium">MIXED: {up_n}/{len(trends)} BULLISH</span>'
    st.markdown(f'<div style="margin-top:12px;">{badge}</div>', unsafe_allow_html=True)


# ──────────────────────────────────────
# Multi-model forecast chart
# ──────────────────────────────────────
st.markdown('<div class="sec">Forecast Overlay</div>', unsafe_allow_html=True)

hist_df    = service.df_base[['Close']].tail(120).copy()
hist_dates = hist_df.index.tolist()
hist_prices= hist_df['Close'].tolist()

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=hist_dates, y=hist_prices,
    mode='lines', name='Historical',
    line=dict(color='#555570', width=1.5),
    hovertemplate='%{x|%Y-%m-%d}<br>$%{y:,.2f}<extra></extra>',
))

palette = {'MLP': '#00d4aa', 'CNN-LSTM': '#5b8cff', 'LINEAR': '#ffbb33'}
style   = {'MLP': 'solid',  'CNN-LSTM': 'solid',   'LINEAR': 'solid'}

for name, r in comparison.items():
    if 'error' in r:
        continue
    fd = pd.to_datetime(r['dates'])
    fp = r['predictions']
    cl = palette.get(name, '#888')
    dl = style.get(name, 'solid')

    fig.add_trace(go.Scatter(
        x=[hist_dates[-1], fd[0]], y=[hist_prices[-1], fp[0]],
        mode='lines', line=dict(color=cl, width=1, dash='dot'),
        showlegend=False, hoverinfo='skip',
    ))
    fig.add_trace(go.Scatter(
        x=fd, y=fp, mode='lines+markers', name=name,
        line=dict(color=cl, width=1.2, dash=dl),
        marker=dict(size=4, color=cl),
        hovertemplate='%{x|%Y-%m-%d}<br>$%{y:,.2f}<extra></extra>',
    ))

fig.update_layout(
    template='plotly_dark',
    paper_bgcolor='#16161f', plot_bgcolor='#16161f',
    height=380, margin=dict(l=50, r=20, t=20, b=40),
    font=dict(family='JetBrains Mono, monospace', size=11, color="#ffffff"),
    xaxis=dict(gridcolor='#1e1e2e', zeroline=False),
    yaxis=dict(gridcolor='#1e1e2e', zeroline=False, tickprefix='$'),
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, font=dict(size=10)),
    hovermode='x unified',
)

st.plotly_chart(fig, use_container_width=True, theme=None)


# ──────────────────────────────────────
# Agent Q&A
# ──────────────────────────────────────
st.markdown('<div class="sec">Ask the Agent</div>', unsafe_allow_html=True)

col_in, col_btn = st.columns([5, 1])
with col_in:
    question = st.text_input(
        'q', placeholder='e.g. Should I buy Apple stock right now?',
        label_visibility='collapsed',
    )
with col_btn:
    run = st.button('Analyze')

presets = [
    "Should I buy Apple stock now?",
    "Which model is most bullish?",
    "How risky is Apple currently?",
]
pcols = st.columns(len(presets))
for i, pq in enumerate(presets):
    with pcols[i]:
        if st.button(pq, key=f'p{i}', use_container_width=True):
            question = pq
            run = True

if run and question:
    with st.spinner('Running multi-model analysis …'):
        answer = ask_investment_agent(question, model=selected_model)
    st.markdown(f'<div class="agent-out">{answer}</div>', unsafe_allow_html=True)

    with st.expander("Raw tool outputs"):
        from agent_system import (
            predict_price_tool, compare_models_tool,
            technical_analysis_tool, risk_assessment_tool,
        )
        st.code(predict_price_tool('10'), language=None)
        st.code(compare_models_tool('10'), language=None)
        st.code(technical_analysis_tool(), language=None)
        st.code(risk_assessment_tool(), language=None)


# ──────────────────────────────────────
# Footer
# ──────────────────────────────────────
st.markdown("""
<div style="text-align:center; margin-top:4rem; padding:1.5rem 0; border-top:1px solid #1e1e2e;">
    <span style="font-family:'JetBrains Mono',monospace; font-size:0.65rem; color:#555570; letter-spacing:0.06em;">
        AI INVESTMENT AGENT &ensp;·&ensp; MLP · CNN-LSTM · LINEAR &ensp;·&ensp; FOR EDUCATIONAL USE ONLY
    </span>
</div>
""", unsafe_allow_html=True)
