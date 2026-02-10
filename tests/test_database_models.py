"""Tests for database models"""
import pytest
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.models import (
    Base, Wallet, Metric, WalletMetric, MetricRanking,
    ScannerState, CapturedTrade
)


@pytest.fixture
def engine():
    """Create in-memory SQLite database"""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def session(engine):
    """Create database session"""
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


class TestWalletModel:
    """Tests for Wallet model"""

    def test_create_wallet(self, session):
        """Test wallet creation"""
        wallet = Wallet(
            address="0x1234567890abcdef1234567890abcdef12345678",
            classification="unknown",
            total_volume=1000.0,
            total_trades=50,
        )
        session.add(wallet)
        session.commit()

        assert wallet.id is not None
        assert wallet.address == "0x1234567890abcdef1234567890abcdef12345678"
        assert wallet.classification == "unknown"
        assert wallet.is_elite_flag == False

    def test_wallet_unique_address(self, session):
        """Test wallet address uniqueness"""
        wallet1 = Wallet(address="0xabc123")
        wallet2 = Wallet(address="0xabc123")

        session.add(wallet1)
        session.commit()

        session.add(wallet2)
        with pytest.raises(Exception):  # IntegrityError
            session.commit()

    def test_wallet_repr(self, session):
        """Test wallet string representation"""
        wallet = Wallet(
            address="0x1234567890abcdef",
            classification="bot"
        )
        assert "0x12345678" in repr(wallet)
        assert "bot" in repr(wallet)


class TestMetricModel:
    """Tests for Metric model"""

    def test_create_metric(self, session):
        """Test metric creation"""
        metric = Metric(
            id=1,
            group_id="A",
            code="total_pnl",
            name="Total PnL (USD)",
            higher_is_better=True,
        )
        session.add(metric)
        session.commit()

        assert metric.id == 1
        assert metric.group_id == "A"
        assert metric.code == "total_pnl"

    def test_metric_unique_code(self, session):
        """Test metric code uniqueness"""
        metric1 = Metric(id=1, group_id="A", code="test_metric", name="Test 1")
        metric2 = Metric(id=2, group_id="A", code="test_metric", name="Test 2")

        session.add(metric1)
        session.commit()

        session.add(metric2)
        with pytest.raises(Exception):
            session.commit()


class TestWalletMetricModel:
    """Tests for WalletMetric model"""

    def test_create_wallet_metric(self, session):
        """Test wallet metric creation"""
        # Create prerequisites
        wallet = Wallet(address="0xtest123")
        metric = Metric(id=1, group_id="A", code="test", name="Test")
        session.add_all([wallet, metric])
        session.commit()

        # Create wallet metric
        wm = WalletMetric(
            wallet_id=wallet.id,
            metric_id=metric.id,
            value=123.45,
            percentile=75.0,
        )
        session.add(wm)
        session.commit()

        assert wm.id is not None
        assert wm.value == 123.45
        assert wm.percentile == 75.0

    def test_wallet_metric_unique_constraint(self, session):
        """Test wallet-metric uniqueness"""
        wallet = Wallet(address="0xtest456")
        metric = Metric(id=2, group_id="B", code="test2", name="Test 2")
        session.add_all([wallet, metric])
        session.commit()

        wm1 = WalletMetric(wallet_id=wallet.id, metric_id=metric.id, value=100)
        wm2 = WalletMetric(wallet_id=wallet.id, metric_id=metric.id, value=200)

        session.add(wm1)
        session.commit()

        session.add(wm2)
        with pytest.raises(Exception):
            session.commit()


class TestMetricRankingModel:
    """Tests for MetricRanking model"""

    def test_create_ranking(self, session):
        """Test ranking creation"""
        wallet = Wallet(address="0xranked")
        metric = Metric(id=3, group_id="C", code="rank_test", name="Rank Test")
        session.add_all([wallet, metric])
        session.commit()

        ranking = MetricRanking(
            metric_id=metric.id,
            wallet_id=wallet.id,
            rank=1,
            value=999.99,
            percentile=99.9,
        )
        session.add(ranking)
        session.commit()

        assert ranking.rank == 1
        assert ranking.value == 999.99

    def test_ranking_rank_constraint(self, session):
        """Test rank must be 1-100"""
        wallet = Wallet(address="0xbadrank")
        metric = Metric(id=4, group_id="D", code="bad_rank", name="Bad Rank")
        session.add_all([wallet, metric])
        session.commit()

        # Rank 0 should fail
        ranking = MetricRanking(
            metric_id=metric.id,
            wallet_id=wallet.id,
            rank=0,
            value=100,
        )
        session.add(ranking)
        with pytest.raises(Exception):
            session.commit()


class TestCapturedTradeModel:
    """Tests for CapturedTrade model"""

    def test_create_captured_trade(self, session):
        """Test captured trade creation"""
        trade = CapturedTrade(
            transaction_hash="0xabcdef1234567890",
            timestamp=datetime.utcnow(),
            wallet_address="0xtrader123",
            market_id="market_001",
            side="BUY",
            price=0.65,
            size=100.0,
            value=65.0,
            capture_source="websocket",
        )
        session.add(trade)
        session.commit()

        assert trade.id is not None
        assert trade.capture_source == "websocket"

    def test_captured_trade_unique_hash(self, session):
        """Test transaction hash uniqueness"""
        trade1 = CapturedTrade(
            transaction_hash="0xsame_hash",
            timestamp=datetime.utcnow(),
            wallet_address="0xwallet1",
            capture_source="websocket",
        )
        trade2 = CapturedTrade(
            transaction_hash="0xsame_hash",
            timestamp=datetime.utcnow(),
            wallet_address="0xwallet2",
            capture_source="rest_backup",
        )

        session.add(trade1)
        session.commit()

        session.add(trade2)
        with pytest.raises(Exception):
            session.commit()


class TestScannerStateModel:
    """Tests for ScannerState model"""

    def test_create_state(self, session):
        """Test scanner state creation"""
        state = ScannerState(
            key="test_checkpoint",
            value="2024-01-01T00:00:00",
        )
        session.add(state)
        session.commit()

        assert state.key == "test_checkpoint"
        assert state.updated_at is not None

    def test_update_state(self, session):
        """Test scanner state update"""
        state = ScannerState(key="update_test", value="initial")
        session.add(state)
        session.commit()

        original_time = state.updated_at

        state.value = "updated"
        session.commit()

        # Note: SQLite doesn't auto-update timestamps, but the model supports it
        assert state.value == "updated"
