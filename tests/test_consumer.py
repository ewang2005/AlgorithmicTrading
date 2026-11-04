"""Tests for consumer/consumer.py.

These import the real functions from consumer.py (pytest.ini puts consumer/ on
the import path), so a bug in the real code fails the tests.
"""
import os
import pickle
from collections import defaultdict
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import consumer
from consumer import (
    CONFIDENCE_THRESHOLD,
    FEATURES,
    compute_ema,
    compute_rsi,
    compute_sma,
    get_features,
)

# grab the real function before the fixture below replaces it with a recorder
real_send_discord = consumer.send_discord

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_CANDIDATES = [
    os.path.join(HERE, "..", "consumer", "signal_model.pkl"),
    os.path.join(HERE, "..", "ml", "signal_model.pkl"),
]


# ── helpers ──────────────────────────────────────────────────────────────────

class StubModel:
    """Stands in for the Random Forest; always returns the same buy probability."""

    def __init__(self, prob):
        self.prob = prob

    def predict_proba(self, X):
        return np.array([[1 - self.prob, self.prob]])


def make_candle(symbol, close, volume=1000.0, spread=0.001):
    return {
        "symbol": symbol,
        "close": close,
        "high": close * (1 + spread),
        "low": close * (1 - spread),
        "volume": volume,
        "window_end": "2026-01-01 00:00:00",
    }


def feed(symbol, closes):
    for close in closes:
        consumer.simulate_trade(make_candle(symbol, close))


def zigzag(start, up, down, n):
    """Closes that alternate +up / -down, ending on an up move when n is odd."""
    out, price = [start], start
    for i in range(n - 1):
        price += up if i % 2 == 0 else -down
        out.append(price)
    return out


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    """Reset module state before every test and capture Discord messages.

    The history dicts are cleared in place (not replaced) because get_features
    binds them as default arguments.
    """
    def reset():
        consumer.portfolio.clear()
        consumer.portfolio.update({"cash": 10000.0, "positions": {}, "trades": []})
        for h in (consumer.price_history, consumer.volume_history,
                  consumer.high_history, consumer.low_history):
            h.clear()

    reset()
    sent = []
    monkeypatch.setattr(consumer, "send_discord", sent.append)
    monkeypatch.setattr(consumer, "model", StubModel(0.5))
    yield sent
    reset()


# ── indicator math ───────────────────────────────────────────────────────────

class TestComputeSMA:
    def test_returns_none_when_insufficient_data(self):
        assert compute_sma([1, 2, 3], 5) is None

    def test_exact_period(self):
        assert compute_sma([1, 2, 3, 4, 5], 5) == 3.0

    def test_uses_last_n_values(self):
        assert compute_sma([100, 200, 1, 2, 3], 3) == 2.0

    def test_period_one_returns_last_value(self):
        assert compute_sma([5, 10, 15], 1) == 15.0


class TestComputeEMA:
    def test_returns_none_when_insufficient_data(self):
        assert compute_ema([1, 2], 5) is None

    def test_constant_series_equals_constant(self):
        assert compute_ema([10] * 5, 5) == pytest.approx(10.0)

    def test_reacts_more_to_a_recent_spike_than_sma(self):
        values = [10.0] * 9 + [20.0]
        assert compute_ema(values, 5) > compute_sma(values, 5)


class TestComputeRSI:
    def test_returns_none_when_insufficient_data(self):
        assert compute_rsi([1, 2, 3], 14) is None

    def test_all_gains_returns_100(self):
        assert compute_rsi(list(range(1, 17)), 14) == 100

    def test_all_losses_returns_zero(self):
        assert compute_rsi(list(range(16, 0, -1)), 14) == 0

    def test_known_value(self):
        # 7 gains of 2 and 7 losses of 1 over 14 deltas: RS = 2, RSI = 66.67
        values = [100]
        for i in range(14):
            values.append(values[-1] + (2 if i % 2 == 0 else -1))
        assert compute_rsi(values, 14) == pytest.approx(100 - 100 / 3)


# ── feature engineering ──────────────────────────────────────────────────────

class TestGetFeatures:
    def _histories(self, n=25):
        ph, vh = defaultdict(list), defaultdict(list)
        for i in range(n):
            ph["BTC"].append(100.0 + i * 0.1)
            vh["BTC"].append(1000.0 + i * 10)
        return ph, vh

    def test_returns_none_with_insufficient_history(self):
        ph = defaultdict(list, {"BTC": [100.0] * 10})
        vh = defaultdict(list, {"BTC": [1000.0] * 10})
        assert get_features("BTC", 100.0, 101.0, 99.0, 1000.0, ph, vh) is None

    def test_returns_none_when_volume_history_too_short(self):
        ph = defaultdict(list, {"BTC": [100.0 + i for i in range(25)]})
        vh = defaultdict(list, {"BTC": [1000.0] * 5})
        assert get_features("BTC", 102.0, 103.0, 101.0, 1100.0, ph, vh) is None

    def test_returns_one_value_per_feature(self):
        ph, vh = self._histories()
        result = get_features("BTC", 102.0, 103.0, 101.0, 1100.0, ph, vh)
        assert len(result) == len(FEATURES) == 14

    def test_values_land_in_the_right_feature_slots(self):
        ph, vh = self._histories()
        closes = ph["BTC"]
        result = dict(zip(FEATURES,
                          get_features("BTC", 102.0, 105.0, 99.0, 1100.0, ph, vh)))
        assert result["sma_5"] == pytest.approx(compute_sma(closes, 5))
        assert result["sma_20"] == pytest.approx(compute_sma(closes, 20))
        assert result["sma_cross"] == pytest.approx(result["sma_5"] - result["sma_20"])
        assert result["rsi_14"] == pytest.approx(compute_rsi(closes, 14))
        assert result["high_low_range"] == pytest.approx((105.0 - 99.0) / 102.0)
        assert result["momentum"] == pytest.approx((closes[-1] - closes[-6]) / closes[-6])


# ── trading logic (runs the real simulate_trade) ─────────────────────────────

class TestSimulateTrade:
    def test_no_trade_while_history_is_building(self, clean_state):
        feed("BTC", zigzag(100, 2, 1, 15))
        assert consumer.portfolio["positions"] == {}
        assert clean_state == []

    def test_buys_on_confirmed_uptrend(self, clean_state):
        feed("BTC", zigzag(100, 2, 1, 25))
        position = consumer.portfolio["positions"]["BTC"]
        cost = position["qty"] * position["entry_price"]
        assert cost == pytest.approx(1000.0, rel=1e-3)  # 10% of $10,000
        assert consumer.portfolio["cash"] == pytest.approx(10000.0 - cost)
        assert sum("BUY" in m for m in clean_state) == 1

    def test_low_ml_confidence_blocks_buy(self, clean_state, monkeypatch):
        monkeypatch.setattr(consumer, "model", StubModel(CONFIDENCE_THRESHOLD / 2))
        feed("BTC", zigzag(100, 2, 1, 25))
        assert consumer.portfolio["positions"] == {}
        assert consumer.portfolio["cash"] == 10000.0

    def test_confidence_exactly_at_threshold_allows_buy(self, clean_state, monkeypatch):
        monkeypatch.setattr(consumer, "model", StubModel(CONFIDENCE_THRESHOLD))
        feed("BTC", zigzag(100, 2, 1, 25))
        assert "BTC" in consumer.portfolio["positions"]

    def test_does_not_buy_same_symbol_twice(self, clean_state):
        feed("BTC", zigzag(100, 2, 1, 25))
        feed("BTC", zigzag(130, 2, 1, 9))  # more uptrend, still holding
        assert sum("BUY" in m for m in clean_state) == 1

    def test_sells_when_price_drops_below_averages_and_books_a_loss(self, clean_state):
        feed("BTC", zigzag(100, 2, 1, 25))  # bought somewhere around 110
        assert "BTC" in consumer.portfolio["positions"]
        feed("BTC", [100.0])                # sharp drop below SMA-5 and EMA-5
        assert "BTC" not in consumer.portfolio["positions"]
        assert any("SELL" in m for m in clean_state)
        assert consumer.portfolio["cash"] < 10000.0

    def test_flat_market_holds(self, clean_state):
        feed("BTC", [100.0] * 30)
        assert consumer.portfolio["positions"] == {}
        assert clean_state == []

    def test_symbols_are_tracked_independently(self, clean_state):
        feed("BTC", zigzag(100, 2, 1, 25))
        feed("ETH", [50.0] * 25)
        assert "BTC" in consumer.portfolio["positions"]
        assert "ETH" not in consumer.portfolio["positions"]


# ── Discord ──────────────────────────────────────────────────────────────────

class TestSendDiscord:
    def test_posts_payload_to_webhook(self, monkeypatch):
        monkeypatch.setattr(consumer, "DISCORD_WEBHOOK_URL", "https://example.com/hook")
        with patch.object(consumer.requests, "post") as post:
            real_send_discord("hello")
        post.assert_called_once_with("https://example.com/hook", json={"content": "hello"})

    def test_without_webhook_prints_and_does_not_post(self, monkeypatch, capsys):
        monkeypatch.setattr(consumer, "DISCORD_WEBHOOK_URL", None)
        with patch.object(consumer.requests, "post") as post:
            real_send_discord("hello")
        post.assert_not_called()
        assert "hello" in capsys.readouterr().out


# ── trained model artifact ───────────────────────────────────────────────────

def _load_model():
    for path in MODEL_CANDIDATES:
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
    return None


class TestModelArtifact:
    def test_model_file_exists(self):
        assert any(os.path.exists(p) for p in MODEL_CANDIDATES), \
            "signal_model.pkl not found in consumer/ or ml/"

    def test_model_probability_is_between_0_and_1(self):
        model = _load_model()
        if model is None:
            pytest.skip("model file not found")
        dummy = pd.DataFrame([[100.0, 100.0, 100.0, 100.0, 50.0, 0.01, 0.01,
                               0.01, 0.01, 0.0, 0.02, -0.01, 0.01, 1.0]],
                             columns=FEATURES)
        assert 0.0 <= model.predict_proba(dummy)[0][1] <= 1.0

    def test_model_was_trained_on_the_same_features_in_the_same_order(self):
        model = _load_model()
        if model is None:
            pytest.skip("model file not found")
        assert model.n_features_in_ == len(FEATURES)
        if hasattr(model, "feature_names_in_"):
            assert list(model.feature_names_in_) == FEATURES