# agent_system.py
"""
AI Investment Agent System — Multi-Model Edition
Supports MLP, CNN-LSTM, and Linear Regression with model comparison.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

try:
    from google import genai as google_genai
except Exception:
    google_genai = None

try:
    import google.generativeai as legacy_genai
except Exception:
    legacy_genai = None

from prediction_service import StockPredictionService

env_path = Path(__file__).resolve().parent / '.env'
if env_path.exists():
    loaded = False
    for enc in ('utf-8', 'utf-8-sig', 'utf-16', 'utf-16-le'):
        try:
            # Some editors save .env as UTF-16 on Windows.
            if load_dotenv(dotenv_path=env_path, encoding=enc):
                loaded = True
                break
        except UnicodeDecodeError:
            continue
        except TypeError:
            # Fallback for older python-dotenv versions without encoding argument.
            if load_dotenv(dotenv_path=env_path):
                loaded = True
                break
    if not loaded:
        print(f'Warning: failed to decode .env at {env_path}')

print('Initializing Multi-Model Prediction Service ...')
stock_service = StockPredictionService()
print('Service ready!\n')


# ============= Local Analysis Tools =============

def predict_price_tool(days='10', model=None):
    try:
        days_int = int(days)
        if model:
            stock_service.active_model = model
        result = stock_service.predict_future(days=days_int)
        name = result['model']
        lines = [
            f'{name} Model Prediction Results:', '',
            f"Current Price: ${result['current_price']:.2f}",
            f"Predicted Trend: {result['trend'].upper()}",
            f"Expected Change: {result['change_pct']:.2f}%",
            f"Average Predicted Price: ${result['avg_future_price']:.2f}",
            f"Confidence: {result['confidence'] * 100:.0f}%",
            '', f'Future {days_int} Business Days:',
        ]
        for i, (date, price) in enumerate(
            zip(result['dates'][:5], result['predictions'][:5]), start=1
        ):
            lines.append(f'  Day {i} ({date}): ${price:.2f}')
        if days_int > 5 and result['dates']:
            lines.append('  ...')
            lines.append(f"  Day {days_int} ({result['dates'][-1]}): ${result['predictions'][-1]:.2f}")
        return '\n'.join(lines)
    except Exception as e:
        return f'Prediction failed: {e}'


def compare_models_tool(days='10'):
    try:
        days_int = int(days)
        results = stock_service.compare_models(days=days_int)
        lines = ['Multi-Model Comparison:', '']
        for name, r in results.items():
            if 'error' in r:
                lines.append(f'  {name}: ERROR - {r["error"]}')
            else:
                sign = '+' if r['change_pct'] >= 0 else ''
                lines.append(
                    f"  {name:10s} | Trend: {r['trend'].upper():4s} "
                    f"| Change: {sign}{r['change_pct']:.2f}% "
                    f"| Avg Price: ${r['avg_future_price']:.2f} "
                    f"| Confidence: {r['confidence']*100:.0f}%"
                )
        trends = [r['trend'] for r in results.values() if 'error' not in r]
        up_count = trends.count('up')
        if trends:
            lines.append('')
            if up_count == len(trends):
                lines.append('Consensus: ALL models predict UP.')
            elif up_count == 0:
                lines.append('Consensus: ALL models predict DOWN.')
            else:
                lines.append(f'Consensus: MIXED - {up_count}/{len(trends)} models predict UP.')
        return '\n'.join(lines)
    except Exception as e:
        return f'Model comparison failed: {e}'


def technical_analysis_tool(_=''):
    try:
        ind = stock_service.get_technical_indicators()
        rsi = ind['RSI']
        rsi_sig = ('Oversold' if rsi < 30 else 'Overbought' if rsi > 70 else 'Neutral')
        macd_trend = ('Bullish' if ind['MACD'] > ind['MACD_Signal'] else 'Bearish')
        price = ind['current_price']
        if price > ind['BB_Upper'] * 0.95: bb = 'Near upper band'
        elif price < ind['BB_Lower'] * 1.05: bb = 'Near lower band'
        else: bb = 'Within normal range'
        return '\n'.join([
            'Technical Indicator Analysis:', '',
            f'RSI: {rsi:.2f} | Signal: {rsi_sig}',
            f"MACD: {ind['MACD']:.2f} | Signal: {ind['MACD_Signal']:.2f} | Trend: {macd_trend}",
            f"MA5: ${ind['MA5']:.2f} | MA10: ${ind['MA10']:.2f} | MA20: ${ind['MA20']:.2f}",
            f"BB Upper: ${ind['BB_Upper']:.2f} | Lower: ${ind['BB_Lower']:.2f} | Current: ${price:.2f}",
            f'Bollinger: {bb}',
        ])
    except Exception as e:
        return f'Technical analysis failed: {e}'


def risk_assessment_tool(_=''):
    try:
        risk = stock_service.assess_risk()
        level = risk['risk_level'].lower()
        advice = {'high': 'Position <= 10%, strict stop-loss.',
                  'medium': 'Position <= 20%, build in batches.'
                  }.get(level, 'Position up to 30% with risk controls.')
        return '\n'.join([
            'Risk Assessment:', '',
            f"Risk Level: {level.upper()}",
            f"Risk Score: {risk['risk_score']:.2f}/100",
            f"Volatility: {risk['volatility'] * 100:.2f}%",
            f"Max Drawdown: {risk['max_drawdown'] * 100:.2f}%",
            f'Advice: {advice}',
        ])
    except Exception as e:
        return f'Risk assessment failed: {e}'

def _is_stock_question(question: str) -> bool:
    q = (question or '').strip().lower()
    stock_keywords = [
        'stock', 'aapl', 'apple', 'price', 'trend', 'forecast', 'predict',
        'prediction', 'risk', 'buy', 'sell', 'hold', 'bullish', 'bearish',
        'market', 'model', 'signal', 'technical', 'rsi', 'macd', 'volatility'
    ]
    return any(k in q for k in stock_keywords)

def _call_gemini(prompt):
    api_key = os.getenv('GOOGLE_API_KEY', '').strip()
    if not api_key:
        raise RuntimeError('GOOGLE_API_KEY is not set in .env')
    last_error = None
    if google_genai is not None:
        try:
            client = google_genai.Client(api_key=api_key)
            resp = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
            text = getattr(resp, 'text', None)
            return (text or str(resp)).strip()
        except Exception as e:
            last_error = e
    if legacy_genai is not None:
        try:
            legacy_genai.configure(api_key=api_key)
            model = legacy_genai.GenerativeModel('gemini-2.5-flash')
            resp = model.generate_content(prompt)
            return (getattr(resp, 'text', '') or '').strip()
        except Exception as e:
            last_error = e
    if last_error:
        raise RuntimeError(f'Gemini call failed: {last_error}')
    raise RuntimeError('No usable Gemini client installed.')


def ask_investment_agent(question, model=None):
    try:
        q = (question or '').strip()
        if not q:
            return "Please enter a question."

        if model:
            stock_service.active_model = model

        # Mode 1: casual chat
        if not _is_stock_question(q):
            prompt = f"""You are a friendly assistant inside an Apple stock analysis app.

User message: {q}

Reply naturally and briefly.
Rules:
- If this is a greeting, greet back warmly.
- If this is casual small talk, respond like a normal assistant.
- Mention that you can also help with Apple stock analysis if relevant.
- Do NOT give investment recommendations unless the user asks about stocks.
- Keep the reply within 2-4 sentences.
"""
            answer = _call_gemini(prompt)
            return answer if answer else "Hi! I can chat, and I can also help analyze Apple stock, model forecasts, and risk."

        # Mode 2: stock analysis
        price_report   = predict_price_tool('10')
        compare_report = compare_models_tool('10')
        ta_report      = technical_analysis_tool()
        risk_report    = risk_assessment_tool()

        prompt = f"""You are an investment analysis assistant focused on Apple (AAPL).

User question: {q}

Base your answer ONLY on these tool outputs:

[Tool 1: PricePredictor - {stock_service.active_model}]
{price_report}

[Tool 2: Multi-Model Comparison]
{compare_report}

[Tool 3: TechnicalAnalysis]
{ta_report}

[Tool 4: RiskAssessment]
{risk_report}

Instructions:
- Answer the user's exact stock-related question.
- Be concise, clear, and practical.
- If the user asks for a recommendation, include BUY / HOLD / SELL.
- If the user asks about risk, focus on risk.
- If the user asks about trend or models, focus on trend/model comparison.
- Do not invent facts beyond the tool outputs.

If appropriate, structure the answer as:
1) Recommendation
2) Model consensus summary
3) Key reasoning
4) Risk note
"""
        answer = _call_gemini(prompt)
        return answer if answer else '\n\n'.join([
            'Gemini returned empty. Raw outputs:',
            price_report, compare_report, ta_report, risk_report,
        ])

    except Exception as e:
        msg = str(e)
        if 'API_KEY' in msg or 'api key' in msg.lower():
            return 'API Key Error: Set GOOGLE_API_KEY in .env and rerun.'
        return f'Agent error: {msg}'