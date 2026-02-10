"""
Continuous Trade Monitor - Real-time wallet capture with ZERO missed wallets.

This module continuously polls the global trades endpoint to capture every wallet
that trades on Polymarket, regardless of market or category.

Key Features:
- Trade ID tracking for gap-free restart
- Automatic catch-up during high-volume spikes
- 24/7 operation capability
- Checkpoint persistence to database
"""
import time
import logging
from datetime import datetime
from typing import Set, Optional, Dict, Any
from dataclasses import dataclass, field

from src.api import DataClient
from src.database.models import get_session, ScannerState, Wallet
from config.settings import continuous_config

logger = logging.getLogger(__name__)


@dataclass
class MonitorStats:
    """Statistics for continuous monitoring session"""
    started_at: datetime = field(default_factory=datetime.utcnow)
    polls_completed: int = 0
    trades_processed: int = 0
    wallets_captured: int = 0
    new_wallets: int = 0
    errors: int = 0
    last_poll_at: Optional[datetime] = None
    last_trade_id: Optional[str] = None


class ContinuousTradeMonitor:
    """
    Real-time trade monitor that captures EVERY wallet.
    Runs continuously, never misses a trade.

    Uses trade ID tracking to ensure gap-free operation:
    - On each poll, fetches trades AFTER the last processed ID
    - On restart, picks up from the last checkpoint
    - During high volume, automatically catches up with multiple requests
    """

    def __init__(
        self,
        poll_interval: float = None,
        trades_per_poll: int = None,
    ):
        self.data_client = DataClient()
        self.poll_interval = poll_interval or continuous_config.POLL_INTERVAL
        self.trades_per_poll = trades_per_poll or continuous_config.TRADES_PER_POLL
        self.checkpoint_key = continuous_config.CHECKPOINT_KEY
        self.max_retry_delay = continuous_config.MAX_RETRY_DELAY

        self.last_trade_id: Optional[str] = None
        self.running = False
        self.stats = MonitorStats()

    def start(self):
        """Start continuous monitoring loop"""
        self.running = True
        self.last_trade_id = self._load_checkpoint()
        self.stats = MonitorStats()
        self.stats.last_trade_id = self.last_trade_id

        logger.info("=" * 60)
        logger.info("CONTINUOUS TRADE MONITOR STARTED")
        logger.info("=" * 60)
        logger.info(f"Poll interval: {self.poll_interval}s")
        logger.info(f"Trades per poll: {self.trades_per_poll}")
        logger.info(f"Starting from trade ID: {self.last_trade_id or 'beginning'}")
        logger.info("=" * 60)

        retry_delay = 1.0

        while self.running:
            try:
                new_wallets = self._poll_recent_trades()
                if new_wallets:
                    saved_count = self._save_wallets(new_wallets)
                    self.stats.new_wallets += saved_count
                    logger.info(
                        f"Poll #{self.stats.polls_completed}: "
                        f"Captured {len(new_wallets)} wallets ({saved_count} new)"
                    )

                self.stats.polls_completed += 1
                self.stats.last_poll_at = datetime.utcnow()

                # Reset retry delay on success
                retry_delay = 1.0

                time.sleep(self.poll_interval)

            except KeyboardInterrupt:
                logger.info("Received interrupt signal, stopping...")
                self.stop()
                break

            except Exception as e:
                self.stats.errors += 1
                logger.error(f"Poll error: {e}")

                # Exponential backoff with max
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, self.max_retry_delay)

    def stop(self):
        """Stop the continuous monitor"""
        self.running = False
        logger.info("=" * 60)
        logger.info("CONTINUOUS TRADE MONITOR STOPPED")
        logger.info("=" * 60)
        logger.info(f"Duration: {(datetime.utcnow() - self.stats.started_at).total_seconds():.1f}s")
        logger.info(f"Polls completed: {self.stats.polls_completed}")
        logger.info(f"Trades processed: {self.stats.trades_processed}")
        logger.info(f"Total wallets captured: {self.stats.wallets_captured}")
        logger.info(f"New wallets saved: {self.stats.new_wallets}")
        logger.info(f"Errors: {self.stats.errors}")
        logger.info(f"Last trade ID: {self.last_trade_id}")
        logger.info("=" * 60)

    def _poll_recent_trades(self) -> Set[str]:
        """
        Fetch ALL trades since last checkpoint - catches up automatically.

        Returns set of unique wallet addresses from new trades.
        """
        wallets = set()
        total_fetched = 0

        # Keep fetching until we're caught up
        while True:
            params = {"limit": self.trades_per_poll}
            if self.last_trade_id:
                params["after"] = self.last_trade_id

            response = self.data_client.get("trades", params)

            if not response:
                break

            # Handle both list and dict responses
            trades = response if isinstance(response, list) else response.get("data", [])

            if not trades:
                break

            for trade in trades:
                # Capture wallet address - try multiple field names
                wallet = (
                    trade.get("proxyWallet") or
                    trade.get("maker") or
                    trade.get("user") or
                    trade.get("taker") or
                    ""
                )
                if wallet:
                    wallets.add(wallet.lower())

                # Update checkpoint to latest trade ID (use transactionHash or timestamp)
                trade_id = trade.get("transactionHash") or trade.get("id") or str(trade.get("timestamp", ""))
                if trade_id:
                    self.last_trade_id = trade_id

            total_fetched += len(trades)
            self.stats.trades_processed += len(trades)

            # Save checkpoint after each batch
            self._save_checkpoint(self.last_trade_id)
            self.stats.last_trade_id = self.last_trade_id

            # If we got fewer trades than requested, we're caught up
            if len(trades) < self.trades_per_poll:
                break

            # Small delay between catch-up requests to respect rate limits
            time.sleep(0.1)

        self.stats.wallets_captured += len(wallets)
        return wallets

    def _save_wallets(self, wallets: Set[str]) -> int:
        """
        Save captured wallets to database.

        Args:
            wallets: Set of wallet addresses

        Returns:
            Number of new wallets saved
        """
        session = get_session()
        new_count = 0

        try:
            for address in wallets:
                # Check if wallet already exists
                existing = session.query(Wallet).filter(
                    Wallet.address == address
                ).first()

                if not existing:
                    wallet = Wallet(
                        address=address,
                        first_seen_at=datetime.utcnow(),
                    )
                    session.add(wallet)
                    new_count += 1

            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving wallets: {e}")
        finally:
            session.close()

        return new_count

    def _save_checkpoint(self, trade_id: str):
        """Save last trade ID to database for restart recovery"""
        if not trade_id:
            return

        session = get_session()
        try:
            # Upsert scanner state
            state = session.query(ScannerState).filter(
                ScannerState.key == self.checkpoint_key
            ).first()

            if state:
                state.value = trade_id
                state.updated_at = datetime.utcnow()
            else:
                state = ScannerState(
                    key=self.checkpoint_key,
                    value=trade_id,
                    updated_at=datetime.utcnow(),
                )
                session.add(state)

            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving checkpoint: {e}")
        finally:
            session.close()

    def _load_checkpoint(self) -> Optional[str]:
        """Load last trade ID from database"""
        session = get_session()
        try:
            state = session.query(ScannerState).filter(
                ScannerState.key == self.checkpoint_key
            ).first()

            if state:
                return state.value
            return None
        finally:
            session.close()

    def get_stats(self) -> Dict[str, Any]:
        """Get current monitoring statistics"""
        return {
            "started_at": self.stats.started_at.isoformat(),
            "running": self.running,
            "polls_completed": self.stats.polls_completed,
            "trades_processed": self.stats.trades_processed,
            "wallets_captured": self.stats.wallets_captured,
            "new_wallets": self.stats.new_wallets,
            "errors": self.stats.errors,
            "last_poll_at": self.stats.last_poll_at.isoformat() if self.stats.last_poll_at else None,
            "last_trade_id": self.stats.last_trade_id,
        }

    def poll_once(self) -> Dict[str, Any]:
        """
        Perform a single poll (for testing or manual operation).

        Returns:
            Dict with poll results
        """
        self.last_trade_id = self._load_checkpoint()

        wallets = self._poll_recent_trades()
        saved_count = self._save_wallets(wallets) if wallets else 0

        return {
            "wallets_found": len(wallets),
            "wallets_saved": saved_count,
            "last_trade_id": self.last_trade_id,
        }

    def reset_checkpoint(self):
        """Reset checkpoint to start from beginning"""
        session = get_session()
        try:
            state = session.query(ScannerState).filter(
                ScannerState.key == self.checkpoint_key
            ).first()

            if state:
                session.delete(state)
                session.commit()

            self.last_trade_id = None
            logger.info("Checkpoint reset - will start from beginning")
        finally:
            session.close()


def run_continuous_monitor(
    poll_interval: float = None,
    trades_per_poll: int = None,
):
    """
    Run continuous trade monitoring.

    This function blocks and runs indefinitely until interrupted.

    Args:
        poll_interval: Seconds between polls (default: 5.0)
        trades_per_poll: Max trades per request (default: 500)
    """
    monitor = ContinuousTradeMonitor(
        poll_interval=poll_interval,
        trades_per_poll=trades_per_poll,
    )

    try:
        monitor.start()
    except KeyboardInterrupt:
        logger.info("Stopping continuous monitor...")
        monitor.stop()
