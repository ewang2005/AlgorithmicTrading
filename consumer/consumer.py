import json
import os
import pickle
import requests
import numpy as np
import sys
from kafka import KafkaConsumer
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

BOOTSTRAP_SERVERS = os.getenv('BOOTSTRAP_SERVERS', 'localhost:9092').split(',')
DISCORD_WEBHOOK_URL = os.getenv('DISCORD_WEBHOOK_URL')
TOPIC = 'ohlcv-1min'
MODEL_PATH = os.getenv('MODEL_PATH', 'signal_model.pkl')
CONFIDENCE_THRESHOLD = 0.1   # low threshold due to class imbalance in training data (precision ~0.29)

print("Script starting...", flush=True)
print(f"MODEL_PATH: {MODEL_PATH}", flush=True)
print(f"BOOTSTRAP_SERVERS: {BOOTSTRAP_SERVERS}", flush=True)

portfolio = {
    'cash': 10000.0,
    'positions': {},
    'trades': []
}

price_history  = defaultdict(list)
volume_history = defaultdict(list)
high_history   = defaultdict(list)
low_history    = defaultdict(list)

def send_discord(message):
    if not DISCORD_WEBHOOK_URL:
        print(f"[Discord] {message}", flush=True)
        return
    requests.post(DISCORD_WEBHOOK_URL, json={"content": message})

def compute_sma(values, period):
    if len(values) < period:
        return None
    return sum(values[-period:]) / period

def compute_ema(values, period):
    if len(values) < period:
        return None
    k = 2 / (period + 1)
    ema = sum(values[:period]) / period
    for v in values[period:]:
        ema = v * k + ema * (1 - k)
    return ema

def compute_rsi(values, period=14):
    if len(values) < period + 1:
        return None
    deltas = [values[i+1] - values[i] for i in range(len(values)-1)]
    recent = deltas[-period:]
    gains  = [d for d in recent if d > 0]
    losses = [-d for d in recent if d < 0]
    avg_gain = sum(gains) / period if gains else 0
    avg_loss = sum(losses) / period if losses else 0
    if avg_loss == 0:
        return 100
    return 100 - (100 / (1 + avg_gain / avg_loss))

def get_features(symbol, close, high, low, volume):
    closes  = price_history[symbol]
    volumes = volume_history[symbol]

    sma_5   = compute_sma(closes, 5)
    sma_20  = compute_sma(closes, 20)
    ema_5   = compute_ema(closes, 5)
    ema_20  = compute_ema(closes, 20)
    rsi_14  = compute_rsi(closes, 14)

    if any(v is None for v in [sma_5, sma_20, ema_5, ema_20, rsi_14]):
        return None

    if len(closes) < 6 or len(volumes) < 2:
        return None

    momentum     = (closes[-1] - closes[-6]) / closes[-6]
    vol_change   = (volumes[-1] - volumes[-2]) / volumes[-2]
    vol_sma      = compute_sma(volumes, 20)
    rel_volume   = volumes[-1] / vol_sma if vol_sma else None

    if rel_volume is None:
        return None

    high_low_range = (high - low) / close
    close_vs_high  = (close - high) / high
    close_vs_low   = (close - low) / low
    price_vs_sma5  = (close - sma_5) / sma_5
    price_vs_sma20 = (close - sma_20) / sma_20
    sma_cross      = sma_5 - sma_20

    return [
        sma_5, sma_20, ema_5, ema_20,
        rsi_14, momentum, vol_change,
        price_vs_sma5, price_vs_sma20, sma_cross,
        high_low_range, close_vs_high, close_vs_low, rel_volume
    ]

def simulate_trade(candle):
    symbol     = candle['symbol']
    close      = float(candle['close'])
    high       = float(candle['high'])
    low        = float(candle['low'])
    volume     = float(candle['volume'])
    window_end = candle['window_end']

    price_history[symbol].append(close)
    volume_history[symbol].append(volume)
    high_history[symbol].append(high)
    low_history[symbol].append(low)

    features = get_features(symbol, close, high, low, volume)
    if features is None:
        print(f"[{symbol}] Building history... ({len(price_history[symbol])}/20)", flush=True)
        return

    prob = model.predict_proba([features])[0][1]
    position = portfolio['positions'].get(symbol)

    sma_5  = compute_sma(price_history[symbol], 5)
    ema_5  = compute_ema(price_history[symbol], 5)
    rsi_14 = compute_rsi(price_history[symbol], 14)

    buy_signal  = close > sma_5 and close > ema_5 and rsi_14 < 75
    sell_signal = close < sma_5 and close < ema_5 and rsi_14 > 25

    print(f"[{symbol}] Close: ${close:,.2f} | RSI: {rsi_14:.1f} | ML Confidence: {prob:.2f}", flush=True)

    if buy_signal and prob >= CONFIDENCE_THRESHOLD and position is None:
        qty  = round((portfolio['cash'] * 0.1) / close, 6)
        cost = qty * close
        if cost > portfolio['cash']:
            return
        portfolio['cash'] -= cost
        portfolio['positions'][symbol] = {'qty': qty, 'entry_price': close}
        msg = (
            f"🟢 **BUY** `{symbol}` *(ML confidence: {prob:.0%})*\n"
            f"Price: `${close:,.2f}` | SMA: `${sma_5:,.2f}` | RSI: `{rsi_14:.1f}`\n"
            f"Qty: `{qty}` | Cost: `${cost:,.2f}`\n"
            f"Cash remaining: `${portfolio['cash']:,.2f}`\n"
            f"Time: `{window_end}`"
        )
        print(msg, flush=True)
        send_discord(msg)

    elif sell_signal and position is not None:
        qty      = position['qty']
        entry    = position['entry_price']
        proceeds = qty * close
        pnl      = proceeds - (qty * entry)
        portfolio['cash'] += proceeds
        del portfolio['positions'][symbol]
        msg = (
            f"🔴 **SELL** `{symbol}` *(ML confidence: {prob:.0%})*\n"
            f"Price: `${close:,.2f}` | SMA: `${sma_5:,.2f}` | RSI: `{rsi_14:.1f}`\n"
            f"Qty: `{qty}` | Proceeds: `${proceeds:,.2f}`\n"
            f"P&L: `${pnl:+,.2f}`\n"
            f"Cash: `${portfolio['cash']:,.2f}`\n"
            f"Time: `{window_end}`"
        )
        print(msg, flush=True)
        send_discord(msg)

    else:
        print(f"[{symbol}] HOLD | Conf: {prob:.2f} | Buy: {buy_signal} | Sell: {sell_signal}", flush=True)

# load model first
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully", flush=True)
except Exception as e:
    print(f"Failed to load model: {e}", flush=True)
    raise

# then create kafka consumer
try:
    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        auto_offset_reset='earliest',
        group_id='paper-trader-ml-v2'
    )
    print("Kafka consumer created successfully", flush=True)
except Exception as e:
    print(f"Failed to create Kafka consumer: {e}", flush=True)
    raise

print(f"Paper trader started. Listening to '{TOPIC}'...", flush=True)

for message in consumer:
    candle = message.value
    simulate_trade(candle)