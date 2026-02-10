"""
Checkpoint Manager - Timestamp-based checkpoints for restart recovery.

Features:
- Saves checkpoint every N seconds
- On restart, re-fetches trades from checkpoint - overlap_seconds
- Prevents data loss during crashes/restarts
"""
import logging
from datetime import datetime, timedelta
from typing import Optional

from src.database.models import get_session, ScannerState
from config.settings import realtime_config

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages checkpoint persistence for restart recovery"""

    CHECKPOINT_KEY = "realtime_checkpoint_timestamp"
    LAST_TRADE_KEY = "realtime_last_trade_hash"

    def __init__(
        self,
        overlap_seconds: int = None,
        save_interval: float = None,
    ):
        """
        Initialize checkpoint manager.

        Args:
            overlap_seconds: Seconds to re-fetch on restart (default from config)
            save_interval: Seconds between checkpoint saves (default from config)
        """
        self.overlap_seconds = overlap_seconds or realtime_config.CHECKPOINT_OVERLAP
        self.save_interval = save_interval or realtime_config.CHECKPOINT_INTERVAL

        # In-memory state
        self.last_checkpoint: Optional[datetime] = None
        self.last_trade_hash: Optional[str] = None
        self.last_save_time: datetime = datetime.utcnow()
        self.trades_since_save: int = 0

    def load_checkpoint(self) -> Optional[datetime]:
        """
        Load checkpoint from database.

        Returns:
            Checkpoint timestamp with overlap applied, or None if no checkpoint
        """
        session = get_session()
        try:
            state = session.query(ScannerState).filter_by(
                key=self.CHECKPOINT_KEY
            ).first()

            if state and state.value:
                checkpoint = datetime.fromisoformat(state.value)
                # Apply overlap for safety
                safe_checkpoint = checkpoint - timedelta(seconds=self.overlap_seconds)
                self.last_checkpoint = checkpoint

                logger.info(
                    f"Loaded checkpoint: {checkpoint.isoformat()}, "
                    f"fetching from: {safe_checkpoint.isoformat()} "
                    f"({self.overlap_seconds}s overlap)"
                )
                return safe_checkpoint

            logger.info("No checkpoint found, starting fresh")
            return None

        finally:
            session.close()

    def update(self, trade_timestamp: datetime, trade_hash: str = None) -> None:
        """
        Update checkpoint with new trade timestamp.

        Args:
            trade_timestamp: Timestamp of processed trade
            trade_hash: Optional trade hash for tracking
        """
        # Update in-memory state
        if self.last_checkpoint is None or trade_timestamp > self.last_checkpoint:
            self.last_checkpoint = trade_timestamp
        self.last_trade_hash = trade_hash
        self.trades_since_save += 1

        # Check if we should save to database
        now = datetime.utcnow()
        if (now - self.last_save_time).total_seconds() >= self.save_interval:
            self.save()

    def save(self) -> None:
        """Save current checkpoint to database"""
        if self.last_checkpoint is None:
            return

        session = get_session()
        try:
            # Update or create checkpoint
            state = session.query(ScannerState).filter_by(
                key=self.CHECKPOINT_KEY
            ).first()

            if state:
                state.value = self.last_checkpoint.isoformat()
                state.updated_at = datetime.utcnow()
            else:
                state = ScannerState(
                    key=self.CHECKPOINT_KEY,
                    value=self.last_checkpoint.isoformat(),
                )
                session.add(state)

            # Also save last trade hash
            if self.last_trade_hash:
                hash_state = session.query(ScannerState).filter_by(
                    key=self.LAST_TRADE_KEY
                ).first()

                if hash_state:
                    hash_state.value = self.last_trade_hash
                    hash_state.updated_at = datetime.utcnow()
                else:
                    hash_state = ScannerState(
                        key=self.LAST_TRADE_KEY,
                        value=self.last_trade_hash,
                    )
                    session.add(hash_state)

            session.commit()

            logger.debug(
                f"Saved checkpoint: {self.last_checkpoint.isoformat()}, "
                f"{self.trades_since_save} trades since last save"
            )
            self.last_save_time = datetime.utcnow()
            self.trades_since_save = 0

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            session.rollback()
        finally:
            session.close()

    def force_save(self) -> None:
        """Force immediate checkpoint save"""
        self.save()

    def get_restart_timestamp(self) -> Optional[datetime]:
        """
        Get timestamp to restart from.

        Returns:
            Timestamp with overlap, or None for fresh start
        """
        return self.load_checkpoint()

    def get_status(self) -> dict:
        """Get checkpoint status"""
        return {
            "last_checkpoint": (
                self.last_checkpoint.isoformat()
                if self.last_checkpoint else None
            ),
            "last_trade_hash": self.last_trade_hash,
            "last_save_time": self.last_save_time.isoformat(),
            "trades_since_save": self.trades_since_save,
            "overlap_seconds": self.overlap_seconds,
            "save_interval": self.save_interval,
        }

    def clear(self) -> None:
        """Clear checkpoint (for testing)"""
        session = get_session()
        try:
            session.query(ScannerState).filter(
                ScannerState.key.in_([self.CHECKPOINT_KEY, self.LAST_TRADE_KEY])
            ).delete(synchronize_session=False)
            session.commit()

            self.last_checkpoint = None
            self.last_trade_hash = None
            logger.info("Checkpoint cleared")

        finally:
            session.close()
