# AlgoTrader - Real-Time Crypto Paper Trading System

A real-time algorithmic paper trading system that streams live cryptocurrency data from Coinbase, processes it with Apache Flink, generates trade signals using a Random Forest ML model trained on historical OHLCV data, and delivers alerts to Discord.

Built as a full end-to-end data + AI engineering project to demonstrate real-time stream processing, ML-powered signal generation, containerized microservices, and algorithmic trading concepts.

---

## Architecture

```
Coinbase WebSocket
        │
        ▼
┌───────────────┐
│   Producer    │  Python — streams live trades (BTC, ETH, SOL)
└───────┬───────┘
        │
        ▼
┌───────────────┐
│   Redpanda    │  Kafka-compatible broker (2-node cluster)
│  (raw trades) │  Topic: real-market-trades
└───────┬───────┘
        │
        ▼
┌───────────────┐
│  Apache Flink │  Tumbling window aggregation (1-min OHLCV candles)
└───────┬───────┘
        │
        ▼
┌───────────────┐
│   Redpanda    │  Topic: ohlcv-1min
│   (OHLCV)     │
└───────┬───────┘
        │
        ▼
┌───────────────────┐
│   Consumer / ML   │  Random Forest signal engine — SMA + EMA + RSI features
│   Signal Engine   │  + ML confidence gating via signal_model.pkl
└───────┬───────────┘
        │
        ▼
┌───────────────┐
│    Discord    │  Trade alerts and P&L notifications
└───────────────┘
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Stream Broker | Apache Redpanda (Kafka-compatible) |
| Stream Processor | Apache Flink 1.18 |
| Producer | Python, websocket-client, kafka-python |
| Consumer / Signal Engine | Python, kafka-python, scikit-learn, pandas, requests |
| ML Model | Random Forest Classifier (scikit-learn) |
| Model Training | pandas, numpy, scikit-learn |
| Historical Data | Coinbase REST API (6 months, 1h candles) |
| Testing | pytest |
| Containerization | Docker, Docker Compose |
| Alerts | Discord Webhooks |

---

## Features

- **Live market data** - connects to Coinbase WebSocket feed and streams real completed trades for BTC-USD, ETH-USD, and SOL-USD
- **Real-time OHLCV aggregation** - Apache Flink tumbling window queries aggregate raw ticks into 1-minute candles
- **ML-powered signal generation** - a Random Forest classifier trained on 14 engineered features scores each candle with a buy probability; trades are only executed when both technical conditions and ML confidence align
- **Multi-indicator confluence strategy** - SMA, EMA, and RSI must all confirm the signal direction before a trade fires
- **Paper portfolio simulation** - simulates trade execution against a virtual $10,000 portfolio with P&L tracking
- **Discord alerts** - real-time BUY/SELL notifications with price, indicators, ML confidence score, quantity, and P&L
- **Unit-tested signal engine** - 28 pytest tests cover indicator math, feature engineering, trade logic, and model/feature consistency

---

## ML Model

The signal engine uses a **Random Forest Classifier** trained on 180 days of hourly OHLCV data fetched from the Coinbase REST API across BTC, ETH, and SOL.

### Feature Engineering (14 features)

| Feature | Description |
|---|---|
| `sma_5`, `sma_20` | Simple moving averages over 5 and 20 candles |
| `ema_5`, `ema_20` | Exponential moving averages over 5 and 20 candles |
| `rsi_14` | Relative Strength Index (14-period) |
| `momentum` | 5-period price percent change |
| `vol_change` | Volume percent change vs previous candle |
| `price_vs_sma5` | Close price deviation from SMA-5 |
| `price_vs_sma20` | Close price deviation from SMA-20 |
| `sma_cross` | SMA-5 minus SMA-20 (crossover signal) |
| `high_low_range` | Candle range normalized by close (volatility) |
| `close_vs_high` | Where close landed relative to candle high |
| `close_vs_low` | Where close landed relative to candle low |
| `rel_volume` | Volume relative to 20-period volume SMA |

### Label Definition

A candle is labeled a **buy opportunity (1)** if the close price rises more than **0.5%** within the next **3 candles**. Otherwise it is labeled **0**.

### Training Details

- **Model:** `RandomForestClassifier(n_estimators=100, class_weight='balanced')`
- **Split:** Time-based, across all three coins at a single cutoff date: the model trains on everything before the cutoff and is tested on the final ~20% of the timeline (from Jun 15, 2026). A 3-hour gap separates the two sets because each label looks 3 candles ahead. Splitting by row position instead would leak information, since BTC, ETH, and SOL move together.
- **Class imbalance:** Handled via `class_weight='balanced'`; confidence threshold set low (~0.1) at inference time due to natural class imbalance in financial data

### Model Performance

Evaluated on the held-out time period (2,577 candles; 10,293 used for training). About 22% of candles are buy opportunities, so that is the baseline a random guess would hit.

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Buy (1) | 0.26 | 0.14 | 0.19 |
| No buy (0) | 0.78 | 0.88 | 0.83 |

Overall accuracy is 0.72. **This is a weak signal**: buy-class precision is only modestly above the 22% base rate, which is typical for short-horizon crypto price prediction. For that reason the model is used as a confidence filter alongside the SMA/EMA/RSI rules, not as a standalone predictor. The value of this project is the real-time streaming, serving, and testing infrastructure around the model. See Future Improvements for better modeling and a proper backtest.

### Trade Execution Logic

A **BUY** fires when all of the following are true:
- Close is above SMA-5 and EMA-5
- RSI-14 is below 75 (not overbought)
- ML confidence (`predict_proba`) ≥ threshold
- No existing position in that symbol

A **SELL** fires when:
- Close is below SMA-5 and EMA-5
- RSI-14 is above 25 (not oversold)
- An open position exists in that symbol

---

## Project Structure

```
AlgorithmicTrading/
├── consumer/
│   ├── consumer.py           # ML signal engine → Discord alerts
│   ├── Dockerfile
│   ├── requirements.txt
│   └── signal_model.pkl      # Trained Random Forest model
├── flink/
│   ├── Dockerfile            # Flink image with Kafka connector JAR
│   └── init.sql              # Auto-initializes Flink tables on startup
├── ml/
│   ├── BTC_USD_1h.csv        # Historical hourly candles (BTC)
│   ├── ETH_USD_1h.csv        # Historical hourly candles (ETH)
│   ├── SOL_USD_1h.csv        # Historical hourly candles (SOL)
│   ├── fetch_data.py         # Coinbase REST API data fetcher
│   ├── signal_model.pkl      # Trained model artifact
│   └── train_model.py        # Feature engineering + model training
├── producer/
│   ├── main.py               # Coinbase WebSocket → Redpanda producer
│   ├── Dockerfile
│   └── requirements.txt
├── tests/
│   └── test_consumer.py      # Unit tests for the signal engine
├── .env                      # Local secrets (gitignored)
├── .env.example              # Environment variable template
├── .gitignore
├── docker-compose.yml
├── pytest.ini
└── README.md
```

---

## Getting Started

### Prerequisites
- Docker Desktop
- Discord server with a webhook URL
- Coinbase API key and secret (for historical data fetching only)
- Internet connection (for Coinbase WebSocket)

### 1. Clone the repository
```bash
git clone https://github.com/ewang2005/AlgorithmicTrading.git
cd AlgorithmicTrading
```

### 2. Set up environment variables
```bash
cp .env.example .env
```
Open `.env` and fill in your credentials:
```
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/your/webhook/url
COINBASE_API_KEY=your_coinbase_api_key_here
COINBASE_API_SECRET=your_coinbase_api_secret_here
```

### 3. Train the ML model (optional — pre-trained model included)

If you want to retrain on fresh data, run these from the repo root:
```bash
pip install -r consumer/requirements.txt
python ml/fetch_data.py       # downloads 180 days of hourly OHLCV data
python ml/train_model.py      # trains the Random Forest and saves ml/signal_model.pkl
cp ml/signal_model.pkl consumer/signal_model.pkl   # Windows: copy ml\signal_model.pkl consumer\signal_model.pkl
```

### 4. Start the stack
```bash
docker compose up -d --build
```

That's it. The Flink tables are created and the INSERT job is started automatically via `flink/init.sql` on startup — no manual SQL entry required.

### 5. Verify the pipeline
```bash
# Check producer is streaming
docker logs producer -f

# Check OHLCV candles are being written
docker exec -it redpanda-1 rpk topic consume ohlcv-1min --num 5

# Check consumer signals and ML output
docker logs consumer -f
```

After ~20 candles of history are built, you should see ML confidence scores alongside BUY/SELL signals in the logs and Discord.

---

## Running the Tests

```bash
pip install pytest
pytest
```

The 28 tests cover:
- Indicator math (SMA, EMA, RSI) and feature engineering
- The BUY/SELL logic in `simulate_trade`, run against a stub model so no Kafka or model file is needed
- Discord alert delivery
- A check that the trained model's feature names and order match the consumer's `FEATURES`

---

## Monitoring

| Dashboard | URL |
|---|---|
| Redpanda Console | http://localhost:8080 |
| Flink Dashboard | http://localhost:8081 |

---

## Future Improvements

- [ ] Replace Random Forest with a gradient boosting model (XGBoost/LightGBM) for improved signal precision
- [ ] Add LSTM or Transformer-based model for sequence-aware price prediction
- [ ] Persist portfolio state to a database so it survives container restarts
- [ ] Automated model retraining pipeline on a rolling data window
- [ ] Daily P&L summary Discord message
- [ ] Grafana dashboard for real-time candle, indicator, and portfolio visualization
- [ ] Backtesting framework to evaluate model performance against historical data
- [ ] Support for additional trading pairs

---

## Disclaimer

This project is for educational purposes only. It does not execute real trades. All portfolio simulation is paper trading with virtual funds.
