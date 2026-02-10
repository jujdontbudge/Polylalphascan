"""
WebSocket Manager - Real-time trade capture with auto-reconnection.

Connects to Polymarket's RTDS (Real-Time Data Service) endpoint
and subscribes to all trades.
"""
import asyncio
import json
import logging
from datetime import datetime
from typing import Optional, Callable, Any
from dataclasses import dataclass

from config.settings import realtime_config

logger = logging.getLogger(__name__)

try:
    import websockets
    from websockets.exceptions import (
        ConnectionClosed,
        ConnectionClosedError,
        ConnectionClosedOK,
    )
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    logger.warning("websockets not installed, WebSocket capture unavailable")


@dataclass
class WebSocketTrade:
    """Trade received from WebSocket"""
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
    def from_ws_message(cls, data: dict) -> Optional["WebSocketTrade"]:
        """Parse trade from WebSocket message"""
        try:
            # Parse timestamp
            timestamp_str = data.get("timestamp", data.get("createdAt", ""))
            if isinstance(timestamp_str, str) and timestamp_str:
                timestamp = datetime.fromisoformat(
                    timestamp_str.replace("Z", "+00:00")
                )
            else:
                timestamp = datetime.utcnow()

            # Extract wallet address
            wallet = (
                data.get("proxyWallet") or
                data.get("maker") or
                data.get("taker") or
                data.get("user") or
                ""
            )

            # Calculate value
            size = float(data.get("size", data.get("amount", 0)) or 0)
            price = float(data.get("price", 0) or 0)
            value = float(data.get("value", data.get("usdcSize", 0)) or 0)
            if value == 0 and size > 0 and price > 0:
                value = size * price

            return cls(
                transaction_hash=data.get("transactionHash", data.get("id", "")),
                timestamp=timestamp,
                wallet_address=wallet,
                market_id=data.get("marketId", data.get("market", "")),
                condition_id=data.get("conditionId", ""),
                side=data.get("side", data.get("type", "BUY")).upper(),
                outcome=data.get("outcome", data.get("outcomeName", "")),
                price=price,
                size=size,
                value=value,
                fee=float(data.get("fee", data.get("feeAmount", 0)) or 0),
            )
        except Exception as e:
            logger.error(f"Failed to parse WebSocket trade: {e}")
            return None


class WebSocketManager:
    """Real-time WebSocket trade capture with auto-reconnection"""

    def __init__(
        self,
        on_trade: Callable[[WebSocketTrade], Any] = None,
        ws_url: str = None,
        ping_interval: float = None,
        reconnect_max: float = None,
    ):
        """
        Initialize WebSocket manager.

        Args:
            on_trade: Callback for each trade received
            ws_url: WebSocket URL (default from config)
            ping_interval: Ping interval seconds (default from config)
            reconnect_max: Max reconnection delay (default from config)
        """
        if not WEBSOCKETS_AVAILABLE:
            raise ImportError("websockets package required for WebSocket capture")

        self.ws_url = ws_url or realtime_config.RTDS_WS_URL
        self.ping_interval = ping_interval or realtime_config.WS_PING_INTERVAL
        self.reconnect_max = reconnect_max or realtime_config.WS_RECONNECT_MAX

        self.on_trade = on_trade or (lambda t: None)
        self.running = False
        self.connected = False
        self.ws = None

        # Stats
        self.trades_received = 0
        self.reconnection_count = 0
        self.last_trade_time: Optional[datetime] = None
        self.connection_start: Optional[datetime] = None

    async def connect_and_subscribe(self) -> None:
        """Connect to RTDS and subscribe to all trades"""
        logger.info(f"Connecting to WebSocket: {self.ws_url}")

        async with websockets.connect(
            self.ws_url,
            ping_interval=self.ping_interval,
            ping_timeout=10,
        ) as ws:
            self.ws = ws
            self.connected = True
            self.connection_start = datetime.utcnow()

            logger.info("WebSocket connected, subscribing to all trades...")

            # Subscribe to ALL trades (no filter = receive everything)
            subscribe_msg = json.dumps({
                "subscriptions": [{
                    "topic": "activity",
                    "type": "trades"
                    # No market filter = ALL markets
                }]
            })
            await ws.send(subscribe_msg)
            logger.info("Subscribed to trade stream")

            # Process incoming messages
            async for message in ws:
                if not self.running:
                    break

                await self._process_message(message)

    async def _process_message(self, message: str) -> None:
        """Process incoming WebSocket message"""
        try:
            data = json.loads(message)

            # Handle different message types
            msg_type = data.get("type", data.get("event", ""))

            if msg_type in ("trade", "trades", "activity"):
                # Single trade or trade update
                trade = WebSocketTrade.from_ws_message(data)
                if trade and trade.wallet_address:
                    self.trades_received += 1
                    self.last_trade_time = datetime.utcnow()
                    await self._handle_trade(trade)

            elif "trades" in data:
                # Batch of trades
                for trade_data in data["trades"]:
                    trade = WebSocketTrade.from_ws_message(trade_data)
                    if trade and trade.wallet_address:
                        self.trades_received += 1
                        self.last_trade_time = datetime.utcnow()
                        await self._handle_trade(trade)

            elif msg_type == "subscribed":
                logger.info(f"Subscription confirmed: {data}")

            elif msg_type == "error":
                logger.error(f"WebSocket error: {data}")

        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON message: {message[:100]}...")
        except Exception as e:
            logger.error(f"Error processing message: {e}")

    async def _handle_trade(self, trade: WebSocketTrade) -> None:
        """Handle a trade - call the callback"""
        try:
            result = self.on_trade(trade)
            if asyncio.iscoroutine(result):
                await result
        except Exception as e:
            logger.error(f"Error in trade callback: {e}")

    async def run_with_reconnection(self) -> None:
        """Run with exponential backoff reconnection"""
        self.running = True
        delay = 1.0

        while self.running:
            try:
                await self.connect_and_subscribe()
                delay = 1.0  # Reset delay on successful connection
            except ConnectionClosed as e:
                logger.warning(f"WebSocket connection closed: {e}")
                self.connected = False
                self.reconnection_count += 1
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                self.connected = False
                self.reconnection_count += 1

            if self.running:
                logger.info(f"Reconnecting in {delay:.1f}s...")
                await asyncio.sleep(delay)
                delay = min(delay * 2, self.reconnect_max)

    def start(self) -> None:
        """Start WebSocket capture (blocking)"""
        asyncio.run(self.run_with_reconnection())

    async def start_async(self) -> None:
        """Start WebSocket capture (async)"""
        await self.run_with_reconnection()

    def stop(self) -> None:
        """Stop WebSocket capture"""
        self.running = False
        self.connected = False
        if self.ws:
            asyncio.create_task(self.ws.close())

    def is_healthy(self) -> bool:
        """Check if WebSocket is healthy"""
        if not self.connected:
            return False

        # Check if we've received trades recently (within 60 seconds)
        if self.last_trade_time:
            since_last = (datetime.utcnow() - self.last_trade_time).total_seconds()
            if since_last > 60:
                logger.warning(f"No trades received in {since_last:.0f}s")
                return False

        return True

    def get_stats(self) -> dict:
        """Get WebSocket statistics"""
        uptime = None
        if self.connection_start:
            uptime = (datetime.utcnow() - self.connection_start).total_seconds()

        return {
            "connected": self.connected,
            "running": self.running,
            "trades_received": self.trades_received,
            "reconnection_count": self.reconnection_count,
            "last_trade_time": (
                self.last_trade_time.isoformat()
                if self.last_trade_time else None
            ),
            "uptime_seconds": uptime,
            "healthy": self.is_healthy(),
        }
