"""Tests for metric calculations"""
import pytest
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from src.api.data_client import Trade, Position
from src.core.analyzer.groups.base_group import WalletData, BaseMetricGroup
from src.core.analyzer.groups.group_a_profitability import GroupAProfitability
from src.core.analyzer.groups.group_b_risk import GroupBRisk
from src.core.analyzer.groups.group_c_winloss import GroupCWinLoss
from src.core.analyzer.groups.group_d_activity import GroupDActivity
from src.core.analyzer.groups.group_n_bot_quant import GroupNBotQuant
from src.core.analyzer.metric_engine import MetricEngine


def create_test_trade(
    price: float = 0.5,
    size: float = 100.0,
    side: str = "BUY",
    days_ago: int = 0,
    market_id: str = "market_001",
    condition_id: str = "cond_001",
    seconds_ago: int = None,
    fee: float = 0.01,
) -> Trade:
    """Helper to create test trades"""
    if seconds_ago is not None:
        timestamp = datetime.utcnow() - timedelta(seconds=seconds_ago)
    else:
        timestamp = datetime.utcnow() - timedelta(days=days_ago)
    return Trade(
        id=f"trade_{np.random.randint(100000)}",
        transaction_hash=f"0x{np.random.randint(100000000):08x}",
        timestamp=timestamp,
        market_id=market_id,
        condition_id=condition_id,
        wallet_address="0xtest_wallet",
        side=side,
        outcome="Yes",
        price=price,
        size=size,
        value=price * size,
        fee=fee,
    )


def create_test_position(
    size: float = 100.0,
    avg_price: float = 0.4,
    current_price: float = 0.6,
) -> Position:
    """Helper to create test positions"""
    return Position(
        market_id="market_001",
        condition_id="cond_001",
        wallet_address="0xtest_wallet",
        outcome="Yes",
        outcome_index=0,
        size=size,
        avg_price=avg_price,
        current_price=current_price,
        value=size * current_price,
        unrealized_pnl=(current_price - avg_price) * size,
        realized_pnl=0.0,
    )


class TestWalletData:
    """Tests for WalletData container"""

    def test_empty_wallet_data(self):
        """Test wallet data with no trades"""
        data = WalletData(address="0xtest", trades=[], positions=[])

        assert data.has_trades == False
        assert data.total_trades == 0
        assert data.total_volume == 0.0
        assert data.trades_df.empty

    def test_wallet_data_with_trades(self):
        """Test wallet data with trades"""
        trades = [create_test_trade() for _ in range(5)]
        data = WalletData(address="0xtest", trades=trades, positions=[])

        assert data.has_trades == True
        assert data.total_trades == 5
        assert data.total_volume > 0
        assert len(data.trades_df) == 5

    def test_trades_df_columns(self):
        """Test trades DataFrame has expected columns"""
        trades = [create_test_trade()]
        data = WalletData(address="0xtest", trades=trades, positions=[])

        expected_cols = ['id', 'timestamp', 'market_id', 'side', 'price', 'size', 'value']
        for col in expected_cols:
            assert col in data.trades_df.columns

    def test_trades_sorted_by_timestamp(self):
        """Test trades are sorted by timestamp"""
        trades = [
            create_test_trade(days_ago=0),
            create_test_trade(days_ago=5),
            create_test_trade(days_ago=2),
        ]
        data = WalletData(address="0xtest", trades=trades, positions=[])

        timestamps = data.trades_df['timestamp'].tolist()
        assert timestamps == sorted(timestamps)


class TestGroupAProfitability:
    """Tests for Group A: Profitability metrics"""

    def test_metric_ids(self):
        """Test Group A has correct metric IDs"""
        group = GroupAProfitability()
        assert group.GROUP_ID == "A"
        assert group.METRIC_IDS == list(range(1, 11))

    def test_empty_trades(self):
        """Test metrics with no trades"""
        group = GroupAProfitability()
        data = WalletData(address="0xtest", trades=[], positions=[])

        results = group.calculate(data)

        assert len(results) == 10
        assert all(v == 0.0 for v in results.values())

    def test_total_pnl_calculation(self):
        """Test total PnL metric (ID 1)"""
        group = GroupAProfitability()

        # Create trades: BUY at 0.3, outcome resolves to 1.0
        trades = [create_test_trade(price=0.3, size=100, side="BUY")]
        data = WalletData(address="0xtest", trades=trades, positions=[])

        results = group.calculate(data)

        # PnL should be positive (bought at 0.3, worth 1.0 if wins)
        assert 1 in results  # Total PnL metric

    def test_profit_factor(self):
        """Test profit factor metric (ID 4)"""
        group = GroupAProfitability()

        # Mix of winning and losing trades
        trades = [
            create_test_trade(price=0.2, size=100, side="BUY"),  # Good buy
            create_test_trade(price=0.8, size=100, side="BUY"),  # Bad buy
        ]
        data = WalletData(address="0xtest", trades=trades, positions=[])

        results = group.calculate(data)

        assert 4 in results  # Profit factor metric


class TestGroupDActivity:
    """Tests for Group D: Activity metrics"""

    def test_metric_ids(self):
        """Test Group D has correct metric IDs"""
        group = GroupDActivity()
        assert group.GROUP_ID == "D"
        assert group.METRIC_IDS == list(range(33, 41))

    def test_total_trades_count(self):
        """Test total trades metric (ID 33)"""
        group = GroupDActivity()

        trades = [create_test_trade() for _ in range(25)]
        data = WalletData(address="0xtest", trades=trades, positions=[])

        results = group.calculate(data)

        assert results[33] == 25  # Total trades

    def test_total_volume(self):
        """Test total volume metric (ID 34)"""
        group = GroupDActivity()

        trades = [create_test_trade(price=0.5, size=100) for _ in range(10)]
        data = WalletData(address="0xtest", trades=trades, positions=[])

        results = group.calculate(data)

        assert results[34] == 500.0  # 10 * (0.5 * 100)


class TestMetricEngine:
    """Tests for MetricEngine orchestration"""

    def test_engine_initialization(self):
        """Test engine initializes all 15 groups"""
        engine = MetricEngine()

        assert len(engine.groups) == 15
        assert len(engine.group_map) == 15

    def test_all_metric_ids(self):
        """Test engine returns all 117 metric IDs"""
        ids = MetricEngine.get_all_metric_ids()

        assert len(ids) == 117
        assert ids == list(range(1, 118))

    def test_group_ids(self):
        """Test engine returns all 15 group IDs"""
        groups = MetricEngine.get_group_ids()

        assert len(groups) == 15
        assert groups == list("ABCDEFGHIJKLMNO")

    def test_calculate_all_returns_result(self):
        """Test calculate_all returns MetricResult"""
        engine = MetricEngine()
        trades = [create_test_trade() for _ in range(10)]

        result = engine.calculate_all("0xtest", trades)

        assert result.wallet_address == "0xtest"
        assert isinstance(result.metrics, dict)
        assert isinstance(result.quant_score, float)
        assert result.classification in ["bot", "quant", "human", "unknown"]

    def test_calculate_all_with_no_trades(self):
        """Test calculate_all with empty trades"""
        engine = MetricEngine()

        result = engine.calculate_all("0xtest", [])

        assert any("No trades found" in e for e in result.errors)

    def test_classification_thresholds(self):
        """Test classification based on quant score"""
        engine = MetricEngine()

        assert engine._classify_wallet(85) == "bot"
        assert engine._classify_wallet(60) == "quant"
        assert engine._classify_wallet(15) == "human"
        assert engine._classify_wallet(35) == "unknown"

    def test_get_group_metrics(self):
        """Test getting metrics for a specific group"""
        engine = MetricEngine()

        # Group A should have metrics 1-10
        group_a_metrics = engine.get_group_metrics("A")
        assert group_a_metrics == list(range(1, 11))

        # Group O should have metrics 114-117
        group_o_metrics = engine.get_group_metrics("O")
        assert group_o_metrics == list(range(114, 118))

    def test_invalid_group(self):
        """Test getting metrics for invalid group"""
        engine = MetricEngine()

        assert engine.get_group_metrics("Z") == []


class TestMetricCalculations:
    """Integration tests for metric calculations"""

    def test_all_groups_calculate_without_error(self):
        """Test all 15 groups calculate without raising exceptions"""
        engine = MetricEngine()

        # Create realistic trade data
        trades = []
        for i in range(50):
            trades.append(create_test_trade(
                price=0.3 + np.random.random() * 0.4,
                size=50 + np.random.random() * 200,
                side="BUY" if np.random.random() > 0.5 else "SELL",
                days_ago=np.random.randint(0, 30),
            ))

        positions = [create_test_position() for _ in range(5)]
        data = WalletData(address="0xtest", trades=trades, positions=positions)

        # Calculate all metrics
        result = engine.calculate_all("0xtest", trades, positions)

        # Should have calculated metrics without critical errors
        assert result.total_calculated > 0
        # Some errors may occur for edge cases, but shouldn't be all errors
        assert len(result.errors) < 15  # Not all groups failed

    def test_metrics_have_reasonable_values(self):
        """Test that calculated metrics have reasonable values"""
        engine = MetricEngine()

        trades = [create_test_trade(price=0.5, size=100) for _ in range(20)]
        result = engine.calculate_all("0xtest", trades)

        # Volume should be positive
        assert result.metrics.get(34, 0) > 0  # Total volume

        # Trade count should match
        assert result.metrics.get(33, 0) == 20  # Total trades

        # Win rate should be 0-100
        win_rate = result.metrics.get(22, 0)
        assert 0 <= win_rate <= 100


class TestPnLCalculation:
    """Tests for corrected PnL cost-basis calculation"""

    def test_pnl_buy_and_sell_same_market(self):
        """Buy at 0.40, sell at 0.70 -> positive PnL"""
        group = GroupAProfitability()

        trades = [
            create_test_trade(price=0.40, size=100, side="BUY", days_ago=5, fee=0.0),
            create_test_trade(price=0.70, size=100, side="SELL", days_ago=1, fee=0.0),
        ]
        data = WalletData(address="0xtest", trades=trades, positions=[])
        results = group.calculate(data)

        # PnL = (0.70 - 0.40) * 100 = 30.0
        assert results[1] == pytest.approx(30.0, abs=0.01)

    def test_pnl_open_position_only(self):
        """Buy only, no sell -> PnL from positions_df"""
        group = GroupAProfitability()

        trades = [
            create_test_trade(price=0.40, size=100, side="BUY", fee=0.0),
        ]
        positions = [create_test_position(size=100, avg_price=0.40, current_price=0.60)]
        data = WalletData(address="0xtest", trades=trades, positions=positions)
        results = group.calculate(data)

        # Unrealized PnL from position = (0.60 - 0.40) * 100 = 20.0
        assert results[1] >= 20.0 - 0.1

    def test_pnl_multiple_markets(self):
        """Mixed trades across two markets"""
        group = GroupAProfitability()

        trades = [
            # Market A: buy at 0.30, sell at 0.60 -> profit 30
            create_test_trade(price=0.30, size=100, side="BUY", days_ago=5,
                              market_id="m_a", condition_id="c_a", fee=0.0),
            create_test_trade(price=0.60, size=100, side="SELL", days_ago=1,
                              market_id="m_a", condition_id="c_a", fee=0.0),
            # Market B: buy at 0.80, sell at 0.50 -> loss -30
            create_test_trade(price=0.80, size=100, side="BUY", days_ago=5,
                              market_id="m_b", condition_id="c_b", fee=0.0),
            create_test_trade(price=0.50, size=100, side="SELL", days_ago=1,
                              market_id="m_b", condition_id="c_b", fee=0.0),
        ]
        data = WalletData(address="0xtest", trades=trades, positions=[])
        results = group.calculate(data)

        # Total PnL = 30 + (-30) = 0
        assert results[1] == pytest.approx(0.0, abs=0.01)

    def test_pnl_partial_exit(self):
        """Buy 100, sell 50 -> PnL on closed portion only"""
        group = GroupAProfitability()

        trades = [
            create_test_trade(price=0.40, size=100, side="BUY", days_ago=5, fee=0.0),
            create_test_trade(price=0.70, size=50, side="SELL", days_ago=1, fee=0.0),
        ]
        data = WalletData(address="0xtest", trades=trades, positions=[])
        results = group.calculate(data)

        # PnL = (0.70 - 0.40) * 50 = 15.0
        assert results[1] == pytest.approx(15.0, abs=0.01)

    def test_pnl_loss_scenario(self):
        """Buy at 0.70, sell at 0.30 -> negative PnL"""
        group = GroupAProfitability()

        trades = [
            create_test_trade(price=0.70, size=100, side="BUY", days_ago=5, fee=0.0),
            create_test_trade(price=0.30, size=100, side="SELL", days_ago=1, fee=0.0),
        ]
        data = WalletData(address="0xtest", trades=trades, positions=[])
        results = group.calculate(data)

        # PnL = (0.30 - 0.70) * 100 = -40.0
        assert results[1] == pytest.approx(-40.0, abs=0.01)

    def test_winning_losing_trades_accuracy(self):
        """Verify get_winning/losing use real prices"""
        group = GroupAProfitability()

        trades = [
            # Winning market
            create_test_trade(price=0.30, size=100, side="BUY", days_ago=5,
                              market_id="m_w", condition_id="c_w", fee=0.0),
            create_test_trade(price=0.70, size=100, side="SELL", days_ago=1,
                              market_id="m_w", condition_id="c_w", fee=0.0),
            # Losing market
            create_test_trade(price=0.70, size=100, side="BUY", days_ago=5,
                              market_id="m_l", condition_id="c_l", fee=0.0),
            create_test_trade(price=0.30, size=100, side="SELL", days_ago=1,
                              market_id="m_l", condition_id="c_l", fee=0.0),
        ]
        data = WalletData(address="0xtest", trades=trades, positions=[])
        df = data.trades_df

        winners = group.get_winning_trades(df)
        losers = group.get_losing_trades(df)

        assert len(winners) >= 1  # At least the sell at profit
        assert len(losers) >= 1   # At least the sell at loss

    def test_market_pnl_breakdown_matches_total(self):
        """Sum of per-market breakdown == total PnL"""
        group = GroupAProfitability()

        trades = [
            create_test_trade(price=0.30, size=100, side="BUY", days_ago=5,
                              market_id="m_a", condition_id="c_a", fee=0.0),
            create_test_trade(price=0.60, size=100, side="SELL", days_ago=1,
                              market_id="m_a", condition_id="c_a", fee=0.0),
            create_test_trade(price=0.50, size=200, side="BUY", days_ago=5,
                              market_id="m_b", condition_id="c_b", fee=0.0),
            create_test_trade(price=0.80, size=200, side="SELL", days_ago=1,
                              market_id="m_b", condition_id="c_b", fee=0.0),
        ]
        data = WalletData(address="0xtest", trades=trades, positions=[])
        df = data.trades_df

        total_pnl = group.calculate_pnl_by_market(df)
        market_map = group._calculate_market_pnl_map(df)
        breakdown_sum = sum(market_map.values())

        assert total_pnl == pytest.approx(breakdown_sum, abs=0.01)


class TestBotDetection:
    """Tests for Group N bot detection fixes"""

    def test_simultaneity_performance(self):
        """500+ trades completes in reasonable time"""
        group = GroupNBotQuant()

        trades = []
        for i in range(500):
            trades.append(create_test_trade(
                price=0.5,
                size=10,
                side="BUY",
                seconds_ago=i * 2,
                market_id=f"market_{i % 10}",
                condition_id=f"cond_{i % 10}",
            ))
        data = WalletData(address="0xtest", trades=trades, positions=[])

        start = time.time()
        results = group.calculate(data)
        elapsed = time.time() - start

        assert elapsed < 5.0  # Should complete in under 5 seconds
        assert 112 in results

    def test_simultaneity_correctness(self):
        """Known simultaneous trades counted correctly"""
        group = GroupNBotQuant()

        now = datetime.utcnow()
        # 3 trades within 5 seconds across 3 different markets
        trades = [
            Trade(id="t1", transaction_hash="0x1", timestamp=now,
                  market_id="m1", condition_id="c1", wallet_address="0x",
                  side="BUY", outcome="Yes", price=0.5, size=10, value=5, fee=0),
            Trade(id="t2", transaction_hash="0x2", timestamp=now + timedelta(seconds=2),
                  market_id="m2", condition_id="c2", wallet_address="0x",
                  side="BUY", outcome="Yes", price=0.5, size=10, value=5, fee=0),
            Trade(id="t3", transaction_hash="0x3", timestamp=now + timedelta(seconds=4),
                  market_id="m3", condition_id="c3", wallet_address="0x",
                  side="BUY", outcome="Yes", price=0.5, size=10, value=5, fee=0),
        ]
        data = WalletData(address="0xtest", trades=trades, positions=[])
        results = group.calculate(data)

        # All 3 trades in different markets within 10s window
        # t2 sees t1 (different market) = 1
        # t3 sees t1, t2 (both different) = 2
        # Total = 3
        assert results[112] == 3

    def test_entropy_perfectly_regular(self):
        """Trades at exact 60s intervals -> low entropy (not 50)"""
        group = GroupNBotQuant()

        base_time = datetime.utcnow()
        trades = []
        for i in range(20):
            t = Trade(
                id=f"t_{i}", transaction_hash=f"0x{i:08x}",
                timestamp=base_time + timedelta(seconds=i * 60),
                market_id="m1", condition_id="c1", wallet_address="0x",
                side="BUY", outcome="Yes", price=0.5, size=10, value=5, fee=0,
            )
            trades.append(t)
        data = WalletData(address="0xtest", trades=trades, positions=[])
        results = group.calculate(data)

        # Perfectly regular intervals should give low entropy (bot-like), well below 50
        assert results[109] < 40.0

    def test_round_number_detection(self):
        """Verify round vs non-round prices"""
        group = GroupNBotQuant()

        # Full 0-1 range: round prices at exact fractions -> low avoidance %
        round_prices = [0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0]
        round_trades = [
            create_test_trade(price=p, size=10, side="BUY")
            for p in round_prices
        ]
        round_data = WalletData(address="0xtest", trades=round_trades, positions=[])
        round_results = group.calculate(round_data)

        # Prices far from any round fraction -> high avoidance %
        nonround_prices = [0.037, 0.143, 0.267, 0.383, 0.437, 0.567, 0.633, 0.817, 0.883, 0.937, 0.013, 0.957, 0.067]
        nonround_trades = [
            create_test_trade(price=p, size=10, side="BUY")
            for p in nonround_prices
        ]
        nonround_data = WalletData(address="0xtest", trades=nonround_trades, positions=[])
        nonround_results = group.calculate(nonround_data)

        assert nonround_results[110] > round_results[110]
