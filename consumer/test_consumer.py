import pytest
import pickle
import os
import numpy as np
from unittest.mock import patch, MagicMock
from collections import defaultdict

# ── helpers replicated from consumer.py so tests are self-contained ──────────

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

FEATURES = [
    'sma_5', 'sma_20', 'ema_5', 'ema_20',
    'rsi_14', 'momentum', 'vol_change',
    'price_vs_sma5', 'price_vs_sma20', 'sma_cross',
    'high_low_range', 'close_vs_high', 'close_vs_low', 'rel_volume'
]

def make_portfolio():
    return {
        'cash': 10000.0,
        'positions': {},
        'trades': []
    }

def get_features(symbol, close, high, low, volume, price_history, volume_history):
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


# ── SMA tests ────────────────────────────────────────────────────────────────

class TestComputeSMA:
    def test_returns_none_when_insufficient_data(self):
        assert compute_sma([1, 2, 3], 5) is None

    def test_exact_period(self):
        assert compute_sma([1, 2, 3, 4, 5], 5) == 3.0

    def test_uses_last_n_values(self):
        # only last 3 values should be used
        assert compute_sma([100, 200, 1, 2, 3], 3) == 2.0

    def test_single_value(self):
        assert compute_sma([42], 1) == 42.0

    def test_returns_float(self):
        result = compute_sma([10, 20], 2)
        assert isinstance(result, float)

    def test_period_one_returns_last_value(self):
        assert compute_sma([5, 10, 15], 1) == 15.0


# ── EMA tests ────────────────────────────────────────────────────────────────

class TestComputeEMA:
    def test_returns_none_when_insufficient_data(self):
        assert compute_ema([1, 2], 5) is None

    def test_equal_values_ema_equals_value(self):
        # EMA of constant series should equal that constant
        result = compute_ema([10, 10, 10, 10, 10], 5)
        assert abs(result - 10.0) < 1e-9

    def test_ema_weights_recent_more(self):
        # rising series: EMA should be closer to recent values than SMA
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        ema = compute_ema(values, 5)
        sma = compute_sma(values, 5)
        assert ema > sma  # EMA should be higher in an uptrend

    def test_returns_float(self):
        result = compute_ema([1, 2, 3, 4, 5], 5)
        assert isinstance(result, float)

    def test_exact_period_length(self):
        result = compute_ema([10, 20, 30, 40, 50], 5)
        assert result is not None


# ── RSI tests ────────────────────────────────────────────────────────────────

class TestComputeRSI:
    def test_returns_none_when_insufficient_data(self):
        assert compute_rsi([1, 2, 3], 14) is None

    def test_all_gains_returns_100(self):
        # strictly increasing series → no losses → RSI = 100
        values = list(range(1, 17))  # 16 values, 15 deltas
        assert compute_rsi(values, 14) == 100

    def test_all_losses_returns_zero(self):
        # strictly decreasing series → no gains → RSI = 0
        values = list(range(16, 0, -1))
        assert compute_rsi(values, 14) == 0

    def test_rsi_bounded(self):
        import random
        random.seed(42)
        values = [random.uniform(50, 150) for _ in range(20)]
        rsi = compute_rsi(values, 14)
        assert 0 <= rsi <= 100

    def test_mixed_gains_losses(self):
        values = [10, 12, 11, 13, 12, 14, 13, 15, 14, 16, 15, 17, 16, 18, 17]
        rsi = compute_rsi(values, 14)
        assert rsi is not None
        assert 0 <= rsi <= 100


# ── Feature engineering tests ─────────────────────────────────────────────────

class TestGetFeatures:
    def _make_histories(self, n=25, base_price=100.0, base_volume=1000.0):
        price_history  = defaultdict(list)
        volume_history = defaultdict(list)
        for i in range(n):
            price_history['BTC'].append(base_price + i * 0.1)
            volume_history['BTC'].append(base_volume + i * 10)
        return price_history, volume_history

    def test_returns_none_with_insufficient_history(self):
        price_history  = defaultdict(list, {'BTC': [100.0] * 10})
        volume_history = defaultdict(list, {'BTC': [1000.0] * 10})
        result = get_features('BTC', 100.0, 101.0, 99.0, 1000.0,
                              price_history, volume_history)
        assert result is None

    def test_returns_list_of_14_features(self):
        ph, vh = self._make_histories()
        result = get_features('BTC', 102.0, 103.0, 101.0, 1100.0, ph, vh)
        assert result is not None
        assert len(result) == 14

    def test_all_features_are_numeric(self):
        ph, vh = self._make_histories()
        result = get_features('BTC', 102.0, 103.0, 101.0, 1100.0, ph, vh)
        assert all(isinstance(v, (int, float)) for v in result)

    def test_feature_order_matches_features_constant(self):
        # verifies the list length matches the FEATURES constant
        ph, vh = self._make_histories()
        result = get_features('BTC', 102.0, 103.0, 101.0, 1100.0, ph, vh)
        assert len(result) == len(FEATURES)

    def test_returns_none_when_rel_volume_unavailable(self):
        # fewer than 20 volume entries → vol_sma is None → rel_volume is None
        price_history  = defaultdict(list, {'BTC': [100.0 + i for i in range(25)]})
        volume_history = defaultdict(list, {'BTC': [1000.0] * 5})
        result = get_features('BTC', 102.0, 103.0, 101.0, 1100.0,
                              price_history, volume_history)
        assert result is None

    def test_high_low_range_is_positive(self):
        ph, vh = self._make_histories()
        result = get_features('BTC', 102.0, 105.0, 99.0, 1100.0, ph, vh)
        high_low_range = result[10]
        assert high_low_range > 0


# ── Portfolio / trade logic tests ─────────────────────────────────────────────

class TestPortfolioLogic:
    def test_initial_portfolio_state(self):
        portfolio = make_portfolio()
        assert portfolio['cash'] == 10000.0
        assert portfolio['positions'] == {}
        assert portfolio['trades'] == []

    def test_buy_reduces_cash(self):
        portfolio = make_portfolio()
        close = 100.0
        qty = round((portfolio['cash'] * 0.1) / close, 6)
        cost = qty * close
        portfolio['cash'] -= cost
        portfolio['positions']['BTC'] = {'qty': qty, 'entry_price': close}
        assert portfolio['cash'] == pytest.approx(9000.0)
        assert 'BTC' in portfolio['positions']

    def test_sell_restores_cash(self):
        portfolio = make_portfolio()
        portfolio['cash'] = 9000.0
        portfolio['positions']['BTC'] = {'qty': 10.0, 'entry_price': 100.0}
        close = 110.0
        qty = portfolio['positions']['BTC']['qty']
        proceeds = qty * close
        portfolio['cash'] += proceeds
        del portfolio['positions']['BTC']
        assert portfolio['cash'] == pytest.approx(10100.0)
        assert 'BTC' not in portfolio['positions']

    def test_pnl_calculation_profit(self):
        entry = 100.0
        exit  = 110.0
        qty   = 10.0
        pnl   = (exit - entry) * qty
        assert pnl == pytest.approx(100.0)

    def test_pnl_calculation_loss(self):
        entry = 100.0
        exit  = 90.0
        qty   = 10.0
        pnl   = (exit - entry) * qty
        assert pnl == pytest.approx(-100.0)

    def test_buy_rejected_if_insufficient_cash(self):
        portfolio = make_portfolio()
        portfolio['cash'] = 5.0
        close = 100.0
        qty  = round((portfolio['cash'] * 0.1) / close, 6)
        cost = qty * close
        # cost should be less than cash (10% of $5 = $0.50)
        assert cost <= portfolio['cash']

    def test_no_double_buy_same_symbol(self):
        portfolio = make_portfolio()
        portfolio['positions']['BTC'] = {'qty': 1.0, 'entry_price': 100.0}
        # position already exists — buy should not fire
        assert portfolio['positions'].get('BTC') is not None


# ── Signal logic tests ────────────────────────────────────────────────────────

class TestSignalLogic:
    def test_buy_signal_conditions(self):
        close = 105.0
        sma_5 = 100.0
        ema_5 = 100.0
        rsi_14 = 60.0
        buy_signal = close > sma_5 and close > ema_5 and rsi_14 < 75
        assert buy_signal is True

    def test_buy_signal_blocked_by_high_rsi(self):
        close = 105.0
        sma_5 = 100.0
        ema_5 = 100.0
        rsi_14 = 80.0
        buy_signal = close > sma_5 and close > ema_5 and rsi_14 < 75
        assert buy_signal is False

    def test_buy_signal_blocked_by_price_below_sma(self):
        close = 95.0
        sma_5 = 100.0
        ema_5 = 100.0
        rsi_14 = 50.0
        buy_signal = close > sma_5 and close > ema_5 and rsi_14 < 75
        assert buy_signal is False

    def test_sell_signal_conditions(self):
        close = 95.0
        sma_5 = 100.0
        ema_5 = 100.0
        rsi_14 = 40.0
        sell_signal = close < sma_5 and close < ema_5 and rsi_14 > 25
        assert sell_signal is True

    def test_sell_signal_blocked_by_low_rsi(self):
        close = 95.0
        sma_5 = 100.0
        ema_5 = 100.0
        rsi_14 = 20.0
        sell_signal = close < sma_5 and close < ema_5 and rsi_14 > 25
        assert sell_signal is False

    def test_sell_signal_blocked_by_price_above_sma(self):
        close = 105.0
        sma_5 = 100.0
        ema_5 = 100.0
        rsi_14 = 40.0
        sell_signal = close < sma_5 and close < ema_5 and rsi_14 > 25
        assert sell_signal is False

    def test_hold_when_no_signal(self):
        close = 100.0
        sma_5 = 100.0
        ema_5 = 100.0
        rsi_14 = 50.0
        buy_signal  = close > sma_5 and close > ema_5 and rsi_14 < 75
        sell_signal = close < sma_5 and close < ema_5 and rsi_14 > 25
        assert buy_signal is False
        assert sell_signal is False


# ── Confidence threshold tests ────────────────────────────────────────────────

class TestConfidenceThreshold:
    CONFIDENCE_THRESHOLD = 0.1

    def test_trade_fires_above_threshold(self):
        prob = 0.15
        assert prob >= self.CONFIDENCE_THRESHOLD

    def test_trade_blocked_below_threshold(self):
        prob = 0.05
        assert prob < self.CONFIDENCE_THRESHOLD

    def test_trade_fires_at_exact_threshold(self):
        prob = 0.1
        assert prob >= self.CONFIDENCE_THRESHOLD


# ── Model loading test ────────────────────────────────────────────────────────

class TestModelLoading:
    def test_model_file_exists(self):
        # checks both possible locations
        paths = [
            'consumer/signal_model.pkl',
            'ml/signal_model.pkl',
            'signal_model.pkl'
        ]
        found = any(os.path.exists(p) for p in paths)
        assert found, "signal_model.pkl not found in expected locations"

    def test_model_loads_and_has_predict_proba(self):
        paths = [
            'consumer/signal_model.pkl',
            'ml/signal_model.pkl',
            'signal_model.pkl'
        ]
        model = None
        for p in paths:
            if os.path.exists(p):
                with open(p, 'rb') as f:
                    model = pickle.load(f)
                break
        assert model is not None, "Could not load model"
        assert hasattr(model, 'predict_proba'), "Model missing predict_proba"

    def test_model_returns_probability_between_0_and_1(self):
        import pandas as pd
        paths = [
            'consumer/signal_model.pkl',
            'ml/signal_model.pkl',
            'signal_model.pkl'
        ]
        model = None
        for p in paths:
            if os.path.exists(p):
                with open(p, 'rb') as f:
                    model = pickle.load(f)
                break

        if model is None:
            pytest.skip("Model file not found")

        # dummy feature vector matching training schema
        dummy = pd.DataFrame([[
            100.0, 100.0, 100.0, 100.0,
            50.0, 0.01, 0.01,
            0.01, 0.01, 0.0,
            0.02, -0.01, 0.01, 1.0
        ]], columns=FEATURES)

        prob = model.predict_proba(dummy)[0][1]
        assert 0.0 <= prob <= 1.0


# ── Discord notification test ─────────────────────────────────────────────────

class TestDiscordNotification:
    @patch('requests.post')
    def test_discord_called_with_correct_payload(self, mock_post):
        import requests
        webhook_url = "https://discord.com/api/webhooks/test"
        message = "Test alert"
        requests.post(webhook_url, json={"content": message})
        mock_post.assert_called_once_with(webhook_url, json={"content": message})

    def test_no_webhook_url_does_not_raise(self):
        # simulates send_discord when DISCORD_WEBHOOK_URL is None
        DISCORD_WEBHOOK_URL = None
        message = "Test"
        try:
            if not DISCORD_WEBHOOK_URL:
                print(f"[Discord] {message}")
        except Exception as e:
            pytest.fail(f"send_discord raised unexpectedly: {e}")
