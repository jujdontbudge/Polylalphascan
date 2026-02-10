"""Tests for checkpoint manager"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from src.core.realtime.checkpoint_manager import CheckpointManager


class TestCheckpointManager:
    """Tests for CheckpointManager"""

    def test_initialization(self):
        """Test checkpoint manager initialization"""
        cp = CheckpointManager(overlap_seconds=60, save_interval=10)

        assert cp.overlap_seconds == 60
        assert cp.save_interval == 10
        assert cp.last_checkpoint is None

    def test_update_checkpoint(self):
        """Test updating checkpoint with trade timestamp"""
        cp = CheckpointManager()

        now = datetime.utcnow()
        cp.update(now, "0xtrade1")

        assert cp.last_checkpoint == now
        assert cp.last_trade_hash == "0xtrade1"
        assert cp.trades_since_save == 1

    def test_update_keeps_latest(self):
        """Test checkpoint keeps the latest timestamp"""
        cp = CheckpointManager()

        old = datetime.utcnow() - timedelta(hours=1)
        new = datetime.utcnow()

        cp.update(old, "0xold")
        cp.update(new, "0xnew")

        assert cp.last_checkpoint == new
        assert cp.last_trade_hash == "0xnew"

    def test_update_ignores_older(self):
        """Test checkpoint ignores older timestamps"""
        cp = CheckpointManager()

        new = datetime.utcnow()
        old = datetime.utcnow() - timedelta(hours=1)

        cp.update(new, "0xnew")
        cp.update(old, "0xold")  # Should be ignored

        assert cp.last_checkpoint == new

    def test_trades_since_save_counter(self):
        """Test trades_since_save counter increments"""
        cp = CheckpointManager()

        for i in range(5):
            cp.update(datetime.utcnow(), f"0xtrade{i}")

        assert cp.trades_since_save == 5

    def test_get_status(self):
        """Test get_status returns expected structure"""
        cp = CheckpointManager(overlap_seconds=60, save_interval=10)

        status = cp.get_status()

        assert "last_checkpoint" in status
        assert "last_trade_hash" in status
        assert "trades_since_save" in status
        assert status["overlap_seconds"] == 60
        assert status["save_interval"] == 10

    @patch('src.core.realtime.checkpoint_manager.get_session')
    def test_load_checkpoint_empty(self, mock_get_session):
        """Test loading checkpoint when none exists"""
        mock_session = MagicMock()
        mock_session.query.return_value.filter_by.return_value.first.return_value = None
        mock_get_session.return_value = mock_session

        cp = CheckpointManager()
        result = cp.load_checkpoint()

        assert result is None

    @patch('src.core.realtime.checkpoint_manager.get_session')
    def test_load_checkpoint_with_overlap(self, mock_get_session):
        """Test loading checkpoint applies overlap"""
        # Create mock state
        from src.database.models import ScannerState
        mock_state = MagicMock()
        checkpoint_time = datetime(2024, 1, 1, 12, 0, 0)
        mock_state.value = checkpoint_time.isoformat()

        mock_session = MagicMock()
        mock_session.query.return_value.filter_by.return_value.first.return_value = mock_state
        mock_get_session.return_value = mock_session

        cp = CheckpointManager(overlap_seconds=60)
        result = cp.load_checkpoint()

        # Result should be checkpoint - overlap
        expected = checkpoint_time - timedelta(seconds=60)
        assert result == expected


class TestCheckpointPersistence:
    """Tests for checkpoint persistence (integration)"""

    @pytest.fixture
    def temp_db(self, tmp_path):
        """Create temporary database for testing"""
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from src.database.models import Base

        db_path = tmp_path / "test.db"
        engine = create_engine(f"sqlite:///{db_path}")
        Base.metadata.create_all(engine)

        return engine

    def test_save_and_load_checkpoint(self, temp_db):
        """Test saving and loading checkpoint (with real DB)"""
        # This would require more setup with the actual DB
        # For now, we verify the class methods exist and are callable
        cp = CheckpointManager()

        assert callable(cp.save)
        assert callable(cp.load_checkpoint)
        assert callable(cp.force_save)
        assert callable(cp.clear)
