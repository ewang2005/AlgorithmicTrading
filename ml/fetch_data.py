import os
import time
import pandas as pd
from dotenv import load_dotenv
from coinbase.rest import RESTClient

load_dotenv()

client = RESTClient(
    api_key=os.getenv("COINBASE_API_KEY"),
    api_secret=os.getenv("COINBASE_API_SECRET")
)

def fetch_candles(product_id, start_ts, end_ts, granularity="ONE_HOUR"):
    all_candles = []
    chunk_size = 300 * 3600  # 300 hours in seconds

    current_start = start_ts
    while current_start < end_ts:
        current_end = min(current_start + chunk_size, end_ts)
        print(f"Fetching {product_id} from {current_start} to {current_end}...")

        response = client.get_candles(
            product_id=product_id,
            start=str(current_start),
            end=str(current_end),
            granularity=granularity
        )
        all_candles.extend(response.candles)
        current_start = current_end
        time.sleep(0.5)  # avoid rate limiting

    return all_candles

# fetch 6 months of hourly data
end_ts = int(time.time())
start_ts = end_ts - (180 * 24 * 3600)  # 180 days ago

for symbol in ["BTC-USD", "ETH-USD", "SOL-USD"]:
    print(f"\nFetching {symbol}...")
    candles = fetch_candles(symbol, start_ts, end_ts)

    df = pd.DataFrame([{
        'start': c.start,
        'open': float(c.open),
        'high': float(c.high),
        'low': float(c.low),
        'close': float(c.close),
        'volume': float(c.volume)
    } for c in candles])

    df['timestamp'] = pd.to_datetime(df['start'].astype(int), unit='s')
    df = df.sort_values('timestamp').reset_index(drop=True)
    df.to_csv(f"ml/{symbol.replace('-', '_')}_1h.csv", index=False)
    print(f"Saved {len(df)} candles for {symbol}")