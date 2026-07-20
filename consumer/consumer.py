import json
import os
import requests
from kafka import KafkaConsumer
from collections import defaultdict

BOOTSTRAP_SERVERS = os.getenv('BOOTSTRAP_SERVERS', 'localhost:9092').split(',')
DISCORD_WEBHOOK_URL = os.getenv('DISCORD_WEBHOOK_URL')
TOPIC = 'ohlcv-1min'

# virtual portfolio
portfolio = {
    'cash': 10000.0,
    'positions': {},  # symbol -> {qty, entry_price}
    'trades': []
}

# store last N closes per symbol for moving average
price_history = defaultdict(list)
MA_PERIOD = 5  # 5 candle moving average

def send_discord(message):
    if not DISCORD_WEBHOOK_URL:
        print(f"[Discord] {message}")
        return
    requests.post(DISCORD_WEBHOOK_URL, json={"content": message})

def get_moving_average(symbol):
    prices = price_history[symbol]
    if len(prices) < MA_PERIOD:
        return None
    return sum(prices[-MA_PERIOD:]) / MA_PERIOD

def simulate_trade(candle):
    symbol = candle['symbol']
    close = candle['close']
    window_end = candle['window_end']

    price_history[symbol].append(close)
    ma = get_moving_average(symbol)

    if ma is None:
        print(f"[{symbol}] Building history... ({len(price_history[symbol])}/{MA_PERIOD})")
        return

    position = portfolio['positions'].get(symbol)

    # BUY signal: close crosses above MA and no position
    if close > ma and position is None:
        qty = round((portfolio['cash'] * 0.1) / close, 6)  # use 10% of cash
        cost = qty * close
        if cost > portfolio['cash']:
            return
        portfolio['cash'] -= cost
        portfolio['positions'][symbol] = {'qty': qty, 'entry_price': close}
        msg = (
            f"**BUY** `{symbol}`\n"
            f"Price: `${close:,.2f}` | MA: `${ma:,.2f}`\n"
            f"Qty: `{qty}` | Cost: `${cost:,.2f}`\n"
            f"Cash remaining: `${portfolio['cash']:,.2f}`\n"
            f"Time: `{window_end}`"
        )
        print(msg)
        send_discord(msg)

    # SELL signal: close crosses below MA and have position
    elif close < ma and position is not None:
        qty = position['qty']
        entry = position['entry_price']
        proceeds = qty * close
        pnl = proceeds - (qty * entry)
        portfolio['cash'] += proceeds
        del portfolio['positions'][symbol]
        msg = (
            f"**SELL** `{symbol}`\n"
            f"Price: `${close:,.2f}` | MA: `${ma:,.2f}`\n"
            f"Qty: `{qty}` | Proceeds: `${proceeds:,.2f}`\n"
            f"P&L: `${pnl:+,.2f}`\n"
            f"Cash: `${portfolio['cash']:,.2f}`\n"
            f"Time: `{window_end}`"
        )
        print(msg)
        send_discord(msg)

consumer = KafkaConsumer(
    TOPIC,
    bootstrap_servers=BOOTSTRAP_SERVERS,
    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
    auto_offset_reset='latest',
    group_id='paper-trader'
)

print(f"Paper trader started. Listening to '{TOPIC}'...")

for message in consumer:
    candle = message.value
    simulate_trade(candle)