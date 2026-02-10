"""
Trade Capture Orchestrator - Coordinates WebSocket + REST + Gap Detection.

Main coordinator for the bulletproof trade capture system that guarantees
zero missed trades through multiple redundant capture mechanisms.
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, Callable, Any

from src.database.models import get_session, CapturedTrade, Wallet
from src.database.repositories import WalletRepository

from .websocket_manager import WebSocketManager, WebSocketTrade
from .rest_backup import RESTBackupPoller, RESTTrade
from .gap_detector import GapDetector, GapInfo
from .checkpoint_manager import CheckpointManager
from .deduplicator import TradeDeduplicator

logger = logging.getLogger(__name__)


class TradeCaptureOrchestrator:
    """Coordinates WebSocket + REST backup + Gap Detection for zero-missed trades"""

    def __init__(self, on_new_wallet: Callable[[str], Any] = None):
        """
        Initialize the orchestrator.

        Args:
            on_new_wallet: Optional callback when a new wallet is discovered
        """
        self.on_new_wallet = on_new_wallet

        # Initialize components
        self.ws_manager = WebSocketManager(on_trade=self._on_ws_trade)
        self.rest_poller = RESTBackupPoller()
        self.gap_detector = GapDetector()
        self.checkpoint_manager = CheckpointManager()
        self.deduplicator = TradeDeduplicator()

        # State
        self.running = False
        self.start_time: Optional[datetime] = None

        # Stats
        self.total_trades_captured = 0
        self.total_wallets_discovered = 0
        self.trades_from_ws = 0
        self.trades_from_rest = 0
        self.trades_from_gap_fill = 0

    async def start(self) -> None:
        """Start all capture components concurrently"""
        self.running = True
        self.start_time = datetime.utcnow()

        logger.info("Starting Trade Capture Orchestrator...")

        # Fill any gaps from restart
        await self._fill_restart_gap()

        # Run all components concurrently
        try:
            await asyncio.gather(
                self._ws_capture_loop(),
                self._rest_backup_loop(),
                self._gap_detection_loop(),
                self._checkpoint_loop(),
                self._stats_loop(),
            )
        except asyncio.CancelledError:
            logger.info("Orchestrator cancelled")
        except Exception as e:
            logger.error(f"Orchestrator error: {e}")
            raise

    async def _fill_restart_gap(self) -> None:
        """Fill any gap from the last checkpoint on restart"""
        restart_timestamp = self.checkpoint_manager.get_restart_timestamp()

        if restart_timestamp:
            logger.info(f"Filling restart gap from {restart_timestamp.isoformat()}")

            trades = await self.rest_poller.poll_trades_async(
                since_timestamp=restart_timestamp
            )

            for trade in trades:
                await self._process_trade(trade, source="restart_fill")

            logger.info(f"Filled {len(trades)} trades from restart gap")

    async def _ws_capture_loop(self) -> None:
        """Main WebSocket capture with reconnection"""
        logger.info("Starting WebSocket capture loop...")
        await self.ws_manager.start_async()

    async def _on_ws_trade(self, trade: WebSocketTrade) -> None:
        """Handle trade from WebSocket"""
        # Record for gap detection
        self.gap_detector.record_ws_trade(trade.transaction_hash, trade.timestamp)

        # Process trade
        await self._process_trade(trade, source="websocket")
        self.trades_from_ws += 1

    async def _rest_backup_loop(self) -> None:
        """REST polling when WebSocket unhealthy"""
        logger.info("Starting REST backup loop...")

        while self.running:
            await asyncio.sleep(self.rest_poller.poll_interval)

            if self.ws_manager.is_healthy():
                continue

            logger.info("WebSocket unhealthy, running REST backup...")

            try:
                trades = await self.rest_poller.poll_trades_async()

                for trade in trades:
                    await self._process_trade(trade, source="rest_backup")
                    self.trades_from_rest += 1

            except Exception as e:
                logger.error(f"REST backup error: {e}")

    async def _gap_detection_loop(self) -> None:
        """Gap detection every 60 seconds"""
        logger.info("Starting gap detection loop...")

        while self.running:
            await asyncio.sleep(self.gap_detector.check_interval)

            try:
                gaps = await self.gap_detector.detect_gaps(self.rest_poller)

                for gap in gaps:
                    await self._fill_gap(gap)

            except Exception as e:
                logger.error(f"Gap detection error: {e}")

    async def _fill_gap(self, gap: GapInfo) -> None:
        """Fill a detected gap"""
        logger.info(
            f"Filling gap: {gap.missing_count} trades from "
            f"{gap.start_time.isoformat()} to {gap.end_time.isoformat()}"
        )

        for trade in gap.missing_trades:
            await self._process_trade(trade, source="gap_fill")
            self.trades_from_gap_fill += 1

        self.gap_detector.record_recovery(gap.missing_count)

    async def _checkpoint_loop(self) -> None:
        """Save checkpoint every 10 seconds"""
        logger.info("Starting checkpoint loop...")

        while self.running:
            await asyncio.sleep(self.checkpoint_manager.save_interval)
            self.checkpoint_manager.save()

    async def _stats_loop(self) -> None:
        """Log stats periodically"""
        while self.running:
            await asyncio.sleep(60)  # Log every minute
            stats = self.get_stats()
            logger.info(
                f"Capture stats: {stats['total_trades_captured']} trades, "
                f"{stats['total_wallets_discovered']} wallets, "
                f"WS: {stats['trades_from_ws']}, REST: {stats['trades_from_rest']}, "
                f"Gap fill: {stats['trades_from_gap_fill']}"
            )

    async def _process_trade(
        self,
        trade: WebSocketTrade | RESTTrade,
        source: str,
    ) -> bool:
        """
        Process trade through deduplication and storage.

        Args:
            trade: Trade to process
            source: Capture source (websocket, rest_backup, gap_fill, restart_fill)

        Returns:
            True if trade was new and saved, False if duplicate
        """
        # Check deduplication
        if not self.deduplicator.check_and_record(trade.transaction_hash):
            return False  # Duplicate

        # Save to database
        await self._save_trade(trade, source)

        # Update checkpoint
        self.checkpoint_manager.update(trade.timestamp, trade.transaction_hash)

        # Track stats
        self.total_trades_captured += 1

        # Discover wallet
        if trade.wallet_address:
            await self._discover_wallet(trade.wallet_address)

        return True

    async def _save_trade(
        self,
        trade: WebSocketTrade | RESTTrade,
        source: str,
    ) -> None:
        """Save trade to database"""
        session = get_session()
        try:
            captured_trade = CapturedTrade(
                transaction_hash=trade.transaction_hash,
                timestamp=trade.timestamp,
                wallet_address=trade.wallet_address,
                market_id=trade.market_id,
                condition_id=trade.condition_id,
                side=trade.side,
                outcome=trade.outcome,
                price=trade.price,
                size=trade.size,
                value=trade.value,
                fee=trade.fee,
                capture_source=source,
            )
            session.add(captured_trade)
            session.commit()
        except Exception as e:
            # Likely duplicate key - that's fine
            session.rollback()
            if "UNIQUE constraint failed" not in str(e):
                logger.error(f"Failed to save trade: {e}")
        finally:
            session.close()

    async def _discover_wallet(self, address: str) -> None:
        """Add wallet to registry if new"""
        session = get_session()
        try:
            with WalletRepository(session) as repo:
                wallet = repo.get_by_address(address)
                if not wallet:
                    repo.create(address)
                    self.total_wallets_discovered += 1

                    if self.on_new_wallet:
                        result = self.on_new_wallet(address)
                        if asyncio.iscoroutine(result):
                            await result
        except Exception as e:
            logger.error(f"Failed to discover wallet: {e}")
        finally:
            session.close()

    def stop(self) -> None:
        """Stop all capture components"""
        logger.info("Stopping Trade Capture Orchestrator...")
        self.running = False
        self.ws_manager.stop()
        self.rest_poller.stop()
        self.gap_detector.stop()
        self.checkpoint_manager.force_save()

    def get_stats(self) -> dict:
        """Get comprehensive statistics"""
        uptime = None
        if self.start_time:
            uptime = (datetime.utcnow() - self.start_time).total_seconds()

        return {
            "running": self.running,
            "uptime_seconds": uptime,
            "total_trades_captured": self.total_trades_captured,
            "total_wallets_discovered": self.total_wallets_discovered,
            "trades_from_ws": self.trades_from_ws,
            "trades_from_rest": self.trades_from_rest,
            "trades_from_gap_fill": self.trades_from_gap_fill,
            "websocket": self.ws_manager.get_stats(),
            "rest_backup": self.rest_poller.get_stats(),
            "gap_detector": self.gap_detector.get_stats(),
            "checkpoint": self.checkpoint_manager.get_status(),
            "deduplicator": self.deduplicator.get_stats(),
        }

    def is_healthy(self) -> bool:
        """Check if the capture system is healthy"""
        # WebSocket should be connected
        if not self.ws_manager.is_healthy():
            return False

        # Should be capturing trades
        if self.total_trades_captured == 0 and self.start_time:
            uptime = (datetime.utcnow() - self.start_time).total_seconds()
            if uptime > 60:
                logger.warning("No trades captured in over 60 seconds")
                return False

        return True


def run_realtime_capture(on_new_wallet: Callable[[str], Any] = None) -> None:
    """
    Run the realtime trade capture system.

    Args:
        on_new_wallet: Optional callback when new wallet discovered
    """
    orchestrator = TradeCaptureOrchestrator(on_new_wallet=on_new_wallet)

    try:
        asyncio.run(orchestrator.start())
    except KeyboardInterrupt:
        logger.info("Received interrupt, shutting down...")
        orchestrator.stop()
