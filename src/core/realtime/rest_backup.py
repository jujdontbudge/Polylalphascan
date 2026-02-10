"""
REST Backup Poller - REST polling with offset-based pagination.

Used as backup when WebSocket is unhealthy and for gap detection.
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Optional
from dataclasses import dataclass

from src.api.data_client import DataClient, Trade
from config.settings import realtime_config

logger = logging.getLogger(__name__)


@dataclass
class RESTTrade:
    """Trade from REST API (compatible with WebSocketTrade)"""
    transaction_hash: str
    timestamp: datetime
    wallet_address: str
    market_id: str
    condition_id: str
    side: str
    outcome: str
    price: float
    size: float
    value: float
    fee: float = 0.0

    @classmethod
    def from_trade(cls, trade: Trade) -> "RESTTrade":
        """Convert from data_client Trade"""
        return cls(
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
        )


class RESTBackupPoller:
    """REST polling backup with offset-based pagination"""

    def __init__(
        self,
        poll_interval: float = None,
        overlap_seconds: int = None,
    ):
        """
        Initialize REST backup poller.

        Args:
            poll_interval: Seconds between polls (default from config)
            overlap_seconds: Overlap window for boundary safety (default from config)
        """
        self.poll_interval = poll_interval or realtime_config.REST_POLL_INTERVAL
        self.overlap_seconds = overlap_seconds or realtime_config.REST_OVERLAP_SECONDS
        self.client = DataClient()

        # Stats
        self.polls_completed = 0
        self.trades_fetched = 0
        self.last_poll_time: Optional[datetime] = None
        self.running = False

    def poll_trades(
        self,
        since_timestamp: datetime = None,
        limit: int = 500,
    ) -> List[RESTTrade]:
        """
        Fetch trades using offset-based pagination.

        Args:
            since_timestamp: Fetch trades after this time
            limit: Max trades per request

        Returns:
            List of trades
        """
        # Apply overlap to ensure no boundary gaps
        if since_timestamp:
            start_time = since_timestamp - timedelta(seconds=self.overlap_seconds)
        else:
            start_time = datetime.utcnow() - timedelta(minutes=5)

        all_trades = []
        offset = 0
        max_pages = 20  # Safety limit

        while offset < max_pages * limit:
            try:
                # Fetch page of trades
                trades = self.client.get_recent_trades(limit=limit)

                if not trades:
                    break

                # Filter by timestamp (client-side since API may not support it)
                for trade in trades:
                    if trade.timestamp >= start_time:
                        all_trades.append(RESTTrade.from_trade(trade))

                # Check if we've reached trades older than our window
                oldest_in_batch = min(t.timestamp for t in trades)
                if oldest_in_batch < start_time:
                    break

                # Check if we got fewer items than requested (last page)
                if len(trades) < limit:
                    break

                offset += limit

            except Exception as e:
                logger.error(f"REST poll error: {e}")
                break

        self.polls_completed += 1
        self.trades_fetched += len(all_trades)
        self.last_poll_time = datetime.utcnow()

        logger.debug(f"REST poll: fetched {len(all_trades)} trades")
        return all_trades

    async def poll_trades_async(
        self,
        since_timestamp: datetime = None,
        limit: int = 500,
    ) -> List[RESTTrade]:
        """Async version of poll_trades"""
        # Run sync method in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.poll_trades(since_timestamp, limit)
        )

    async def run_backup_loop(
        self,
        on_trade: callable,
        is_primary_healthy: callable,
    ) -> None:
        """
        Run backup polling loop.

        Args:
            on_trade: Callback for each trade
            is_primary_healthy: Callable that returns True if primary (WS) is healthy
        """
        self.running = True
        last_timestamp = datetime.utcnow() - timedelta(minutes=5)

        while self.running:
            await asyncio.sleep(self.poll_interval)

            # Only poll if primary is unhealthy
            if is_primary_healthy():
                continue

            logger.info("Primary unhealthy, running REST backup poll...")

            try:
                trades = await self.poll_trades_async(since_timestamp=last_timestamp)

                for trade in trades:
                    await self._process_trade(trade, on_trade)

                if trades:
                    last_timestamp = max(t.timestamp for t in trades)

            except Exception as e:
                logger.error(f"REST backup loop error: {e}")

    async def _process_trade(self, trade: RESTTrade, on_trade: callable) -> None:
        """Process a single trade through callback"""
        try:
            result = on_trade(trade)
            if asyncio.iscoroutine(result):
                await result
        except Exception as e:
            logger.error(f"Error in trade callback: {e}")

    def stop(self) -> None:
        """Stop the backup poller"""
        self.running = False

    def get_stats(self) -> dict:
        """Get poller statistics"""
        return {
            "polls_completed": self.polls_completed,
            "trades_fetched": self.trades_fetched,
            "last_poll_time": (
                self.last_poll_time.isoformat()
                if self.last_poll_time else None
            ),
            "running": self.running,
            "poll_interval": self.poll_interval,
            "overlap_seconds": self.overlap_seconds,
        }

    def close(self) -> None:
        """Close the REST client"""
        self.client.close()
