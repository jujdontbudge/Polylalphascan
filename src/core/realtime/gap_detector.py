"""
Gap Detector - Detect missing trades by comparing WebSocket vs REST.

Periodically compares trades received via WebSocket against REST snapshots
to identify and fill gaps.
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Set, Optional
from dataclasses import dataclass, field

from .rest_backup import RESTBackupPoller, RESTTrade
from config.settings import realtime_config

logger = logging.getLogger(__name__)


@dataclass
class GapInfo:
    """Information about a detected gap"""
    start_time: datetime
    end_time: datetime
    missing_count: int
    missing_trades: List[RESTTrade] = field(default_factory=list)


class GapDetector:
    """Detect missing trades by comparing WebSocket vs REST"""

    def __init__(
        self,
        check_interval: float = None,
        lookback_minutes: int = 5,
    ):
        """
        Initialize gap detector.

        Args:
            check_interval: Seconds between gap checks (default from config)
            lookback_minutes: Minutes to look back for gap detection
        """
        self.check_interval = check_interval or realtime_config.GAP_CHECK_INTERVAL
        self.lookback_minutes = lookback_minutes

        # Track trades received via WebSocket
        self.ws_trade_ids: Set[str] = set()
        self.ws_trade_times: dict = {}  # hash -> timestamp

        # Stats
        self.gaps_detected = 0
        self.trades_recovered = 0
        self.last_check_time: Optional[datetime] = None
        self.running = False

    def record_ws_trade(self, trade_hash: str, timestamp: datetime = None) -> None:
        """
        Record trade received via WebSocket.

        Args:
            trade_hash: Transaction hash
            timestamp: Trade timestamp
        """
        self.ws_trade_ids.add(trade_hash)
        self.ws_trade_times[trade_hash] = timestamp or datetime.utcnow()

        # Trim old entries to prevent memory growth
        self._trim_old_entries()

    def _trim_old_entries(self) -> None:
        """Remove entries older than lookback window"""
        cutoff = datetime.utcnow() - timedelta(minutes=self.lookback_minutes * 2)
        to_remove = [
            h for h, t in self.ws_trade_times.items()
            if t and t < cutoff
        ]
        for h in to_remove:
            self.ws_trade_ids.discard(h)
            del self.ws_trade_times[h]

    async def detect_gaps(
        self,
        rest_poller: RESTBackupPoller,
    ) -> List[GapInfo]:
        """
        Find trades in REST that weren't seen via WebSocket.

        Args:
            rest_poller: REST backup poller for fetching comparison data

        Returns:
            List of gap information
        """
        since = datetime.utcnow() - timedelta(minutes=self.lookback_minutes)
        rest_trades = await rest_poller.poll_trades_async(since_timestamp=since)

        missing = []
        for trade in rest_trades:
            if trade.transaction_hash not in self.ws_trade_ids:
                missing.append(trade)

        self.last_check_time = datetime.utcnow()

        if missing:
            self.gaps_detected += 1
            logger.warning(
                f"GAP DETECTED: {len(missing)} trades missed via WebSocket "
                f"in last {self.lookback_minutes} minutes"
            )

            # Group into gap info
            gaps = self._group_into_gaps(missing)
            return gaps

        logger.debug(f"Gap check: no gaps detected ({len(rest_trades)} trades verified)")
        return []

    def _group_into_gaps(self, missing: List[RESTTrade]) -> List[GapInfo]:
        """Group missing trades into gap periods"""
        if not missing:
            return []

        # Sort by timestamp
        sorted_trades = sorted(missing, key=lambda t: t.timestamp)

        gaps = []
        current_gap_trades = [sorted_trades[0]]

        for trade in sorted_trades[1:]:
            # If more than 30 seconds gap, start new gap period
            last_trade = current_gap_trades[-1]
            if (trade.timestamp - last_trade.timestamp).total_seconds() > 30:
                # Save current gap
                gaps.append(GapInfo(
                    start_time=current_gap_trades[0].timestamp,
                    end_time=current_gap_trades[-1].timestamp,
                    missing_count=len(current_gap_trades),
                    missing_trades=current_gap_trades,
                ))
                current_gap_trades = [trade]
            else:
                current_gap_trades.append(trade)

        # Add final gap
        if current_gap_trades:
            gaps.append(GapInfo(
                start_time=current_gap_trades[0].timestamp,
                end_time=current_gap_trades[-1].timestamp,
                missing_count=len(current_gap_trades),
                missing_trades=current_gap_trades,
            ))

        return gaps

    async def run_detection_loop(
        self,
        rest_poller: RESTBackupPoller,
        on_gap_found: callable = None,
    ) -> None:
        """
        Run gap detection loop.

        Args:
            rest_poller: REST backup poller
            on_gap_found: Callback when gaps are found
        """
        self.running = True

        while self.running:
            await asyncio.sleep(self.check_interval)

            try:
                gaps = await self.detect_gaps(rest_poller)

                if gaps and on_gap_found:
                    for gap in gaps:
                        result = on_gap_found(gap)
                        if asyncio.iscoroutine(result):
                            await result

            except Exception as e:
                logger.error(f"Gap detection error: {e}")

    def record_recovery(self, count: int) -> None:
        """Record recovered trades"""
        self.trades_recovered += count

    def stop(self) -> None:
        """Stop the gap detector"""
        self.running = False

    def get_stats(self) -> dict:
        """Get gap detection statistics"""
        return {
            "gaps_detected": self.gaps_detected,
            "trades_recovered": self.trades_recovered,
            "ws_trades_tracked": len(self.ws_trade_ids),
            "last_check_time": (
                self.last_check_time.isoformat()
                if self.last_check_time else None
            ),
            "running": self.running,
            "check_interval": self.check_interval,
            "lookback_minutes": self.lookback_minutes,
        }

    def clear(self) -> None:
        """Clear tracked trades"""
        self.ws_trade_ids.clear()
        self.ws_trade_times.clear()
