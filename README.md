# AlgoTrader — Real-Time Crypto Paper Trading System

A real-time algorithmic paper trading system that streams live cryptocurrency data from Coinbase, processes it with Apache Flink, generates trade signals using technical indicators, and delivers alerts to Discord.

Built as a full end-to-end data engineering project to demonstrate real-time stream processing, containerized microservices, and algorithmic trading concepts.

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
┌───────────────┐
│   Consumer    │  Multi-indicator signal engine (SMA + EMA + RSI)
└───────┬───────┘
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
| Consumer | Python, kafka-python, requests |
| Containerization | Docker, Docker Compose |
| Alerts | Discord Webhooks |

---

## Features

- **Live market data** — connects to Coinbase WebSocket feed and streams real completed trades for BTC-USD, ETH-USD, and SOL-USD
- **Real-time OHLCV aggregation** — Apache Flink tumbling window queries aggregate raw ticks into 1-minute candles
- **Multi-indicator confluence strategy** — trading signals require SMA, EMA, and RSI to confirm before firing, reducing false positives
- **Paper portfolio simulation** — simulates trade execution against a virtual $10,000 portfolio with P&L tracking
- **Discord alerts** — real-time BUY/SELL notifications with price, indicators, quantity, and P&L

---

## Trading Strategy

The consumer implements a **multi-indicator confluence strategy** combining three technical indicators:

| Indicator | Period | Role |
|---|---|---|
| SMA (Simple Moving Average) | 5 candles | Baseline trend direction |
| EMA (Exponential Moving Average) | 5 candles | Trend confirmation, weights recent prices more heavily |
| RSI (Relative Strength Index) | 5 candles | Momentum filter, avoids overbought/oversold entries |

**BUY signal** — fires when:
- Close price is above SMA
- Close price is above EMA
- RSI is below 70 (not overbought)
- No existing position in that symbol

**SELL signal** — fires when:
- Close price is below SMA
- Close price is below EMA
- RSI is above 30 (not oversold)
- An open position exists in that symbol

All three conditions must be met simultaneously before a trade executes — this is known as **signal confirmation** or **confluence trading**.

---

## Project Structure

```
AlgorithmicTrading/
├── producer/
│   ├── Dockerfile
│   ├── main.py           # Coinbase WebSocket → Redpanda producer
│   └── requirements.txt
├── consumer/
│   ├── Dockerfile
│   ├── consumer.py       # Signal engine → Discord alerts
│   └── requirements.txt
├── flink/
│   └── Dockerfile        # Flink image with Kafka connector JAR
├── .env.example          # Environment variable template
├── .gitignore
└── docker-compose.yml
```

---

## Getting Started

### Prerequisites
- Docker Desktop
- Discord server with a webhook URL
- Internet connection (for Coinbase WebSocket)

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/AlgorithmicTrading.git
cd AlgorithmicTrading
```

### 2. Set up environment variables
```bash
cp .env.example .env
```
Open `.env` and add your Discord webhook URL:
```
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/your/webhook/url
```

### 3. Start the stack
```bash
docker compose up -d --build
```

### 4. Set up Flink tables
Attach to the SQL client:
```bash
docker attach sql-client
```

Run the following SQL in order:

```sql
SET 'table.local-time-zone' = 'UTC';

CREATE TABLE real_market_trades (
    trade_id        BIGINT,
    symbol          STRING,
    price           DOUBLE,
    volume          DOUBLE,
    side            STRING,
    trade_time      STRING,
    proc_time       AS PROCTIME()
) WITH (
    'connector' = 'kafka',
    'topic' = 'real-market-trades',
    'properties.bootstrap.servers' = 'redpanda-1:29092,redpanda-2:29093',
    'scan.startup.mode' = 'latest-offset',
    'format' = 'json'
);

CREATE TABLE ohlcv_1min (
    symbol          STRING,
    window_start    TIMESTAMP(3),
    window_end      TIMESTAMP(3),
    `open`          DOUBLE,
    high            DOUBLE,
    low             DOUBLE,
    `close`         DOUBLE,
    volume          DOUBLE
) WITH (
    'connector' = 'kafka',
    'topic' = 'ohlcv-1min',
    'properties.bootstrap.servers' = 'redpanda-1:29092,redpanda-2:29093',
    'format' = 'json'
);

INSERT INTO ohlcv_1min
SELECT
    symbol,
    TUMBLE_START(proc_time, INTERVAL '1' MINUTE) AS window_start,
    TUMBLE_END(proc_time, INTERVAL '1' MINUTE)   AS window_end,
    FIRST_VALUE(price)  AS `open`,
    MAX(price)          AS high,
    MIN(price)          AS low,
    LAST_VALUE(price)   AS `close`,
    SUM(volume)         AS volume
FROM real_market_trades
GROUP BY
    symbol,
    TUMBLE(proc_time, INTERVAL '1' MINUTE);
```

### 5. Verify the pipeline
Check producer is streaming:
```bash
docker logs producer -f
```

Check OHLCV candles are being written:
```bash
docker exec -it redpanda-1 rpk topic consume ohlcv-1min --num 5
```

Check consumer signals:
```bash
docker logs consumer -f
```

After ~6 minutes (5 candles to build history + 1 to trigger) you should see BUY/SELL signals in the logs and Discord.

---

## Monitoring

| Dashboard | URL |
|---|---|
| Redpanda Console | http://localhost:8080 |
| Flink Dashboard | http://localhost:8081 |

---

## Future Improvements

- [ ] ML-based signal scoring using logistic regression trained on historical indicator values
- [ ] Persist portfolio state to a database so it survives container restarts
- [ ] Daily P&L summary Discord message
- [ ] Grafana dashboard for real-time candle and portfolio visualization
- [ ] Support for additional trading pairs
- [ ] Backtesting framework against historical data

---

## Disclaimer

This project is for educational purposes only. It does not execute real trades. All portfolio simulation is paper trading with virtual funds.
