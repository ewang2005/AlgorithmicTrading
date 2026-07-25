SET 'table.local-time-zone' = 'UTC';

CREATE TABLE IF NOT EXISTS real_market_trades (
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

CREATE TABLE IF NOT EXISTS ohlcv_1min (
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