import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle
import os

def compute_sma(prices, period):
    return prices.rolling(window=period).mean()

def compute_ema(prices, period):
    return prices.ewm(span=period, adjust=False).mean()

def compute_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = -delta.clip(upper=0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def engineer_features(df):
    df = df.copy()
    df['sma_5']         = compute_sma(df['close'], 5)
    df['sma_20']        = compute_sma(df['close'], 20)
    df['ema_5']         = compute_ema(df['close'], 5)
    df['ema_20']        = compute_ema(df['close'], 20)
    df['rsi_14']        = compute_rsi(df['close'], 14)
    df['momentum']      = df['close'].pct_change(periods=5)
    df['vol_change']    = df['volume'].pct_change()
    df['price_vs_sma5'] = (df['close'] - df['sma_5']) / df['sma_5']
    df['price_vs_sma20'] = (df['close'] - df['sma_20']) / df['sma_20']
    df['sma_cross']     = df['sma_5'] - df['sma_20']
    # new features
    df['high_low_range'] = (df['high'] - df['low']) / df['close']  # volatility
    df['close_vs_high']  = (df['close'] - df['high']) / df['high']  # where close is in range
    df['close_vs_low']   = (df['close'] - df['low']) / df['low']
    df['volume_sma']     = compute_sma(df['volume'], 20)
    df['rel_volume']     = df['volume'] / df['volume_sma']  # relative volume spike
    return df

def label_signals(df, forward_periods=3, threshold=0.005):
    # 0.5% move in next 3 hours instead of 1% in 5 hours
    df['future_return'] = df['close'].shift(-forward_periods) / df['close'] - 1
    df['label'] = (df['future_return'] > threshold).astype(int)
    return df

FEATURES = [
    'sma_5', 'sma_20', 'ema_5', 'ema_20',
    'rsi_14', 'momentum', 'vol_change',
    'price_vs_sma5', 'price_vs_sma20', 'sma_cross',
    'high_low_range', 'close_vs_high', 'close_vs_low', 'rel_volume'
]

all_dfs = []
for symbol in ['BTC_USD', 'ETH_USD', 'SOL_USD']:
    df = pd.read_csv(f'ml/{symbol}_1h.csv')
    df = engineer_features(df)
    df = label_signals(df)
    df['symbol'] = symbol
    all_dfs.append(df)

data = pd.concat(all_dfs).dropna().reset_index(drop=True)

print(f"Total samples: {len(data)}")
print(f"Label distribution:\n{data['label'].value_counts()}")

X = data[FEATURES]
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False  # time-based split, no shuffling
)

model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\nModel Performance:")
print(classification_report(y_test, y_pred))

# feature importance
importance = pd.Series(model.feature_importances_, index=FEATURES)
print("\nFeature Importance:")
print(importance.sort_values(ascending=False))

# save model
with open('ml/signal_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("\nModel saved to ml/signal_model.pkl")