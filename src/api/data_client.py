"""Data API client for wallet trades and positions"""
import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

from .base_client import BaseClient
from config.settings import api_config

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Individual trade data"""
    id: str
    transaction_hash: str
    timestamp: datetime
    market_id: str
    condition_id: str
    wallet_address: str
    side: str  # 'BUY' or 'SELL'
    outcome: str
    price: float
    size: float
    value: float
    fee: float
    asset_id: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Trade":
        """Create Trade from API response dict"""
        timestamp_str = data.get("timestamp", data.get("createdAt", ""))
        if isinstance(timestamp_str, str) and timestamp_str:
            try:
                timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            except ValueError:
                timestamp = datetime.now()
        else:
            timestamp = datetime.now()

        # Extract wallet address - try multiple field names
        wallet = (
            data.get("proxyWallet") or  # Primary field from trades API
            data.get("maker") or
            data.get("user") or
            data.get("wallet_address") or
            data.get("taker") or
            ""
        )

        # Calculate value if not present
        size = float(data.get("size", data.get("amount", 0)) or 0)
        price = float(data.get("price", 0) or 0)
        value = float(data.get("value", data.get("usdcSize", 0)) or 0)
        if value == 0 and size > 0 and price > 0:
            value = size * price

        return cls(
            id=data.get("id", data.get("tradeId", "")),
            transaction_hash=data.get("transactionHash", data.get("transaction_hash", "")),
            timestamp=timestamp,
            market_id=data.get("marketId", data.get("market_id", data.get("market", ""))),
            condition_id=data.get("conditionId", data.get("condition_id", "")),
            wallet_address=wallet,
            side=data.get("side", data.get("type", "BUY")).upper(),
            outcome=data.get("outcome", data.get("outcomeName", data.get("title", ""))),
            price=price,
            size=size,
            value=value,
            fee=float(data.get("fee", data.get("feeAmount", 0)) or 0),
            asset_id=data.get("assetId", data.get("asset_id", data.get("asset", ""))),
        )


@dataclass
class Position:
    """Wallet position data"""
    market_id: str
    condition_id: str
    wallet_address: str
    outcome: str
    outcome_index: int
    size: float
    avg_price: float
    current_price: float
    value: float
    unrealized_pnl: float
    realized_pnl: float
    asset_id: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Position":
        """Create Position from API response dict"""
        size = float(data.get("size", data.get("amount", 0)) or 0)
        avg_price = float(data.get("avgPrice", data.get("averagePrice", 0)) or 0)
        current_price = float(data.get("currentPrice", data.get("price", 0)) or 0)
        value = size * current_price
        unrealized_pnl = (current_price - avg_price) * size

        return cls(
            market_id=data.get("marketId", data.get("market_id", "")),
            condition_id=data.get("conditionId", data.get("condition_id", "")),
            wallet_address=data.get("user", data.get("wallet_address", "")),
            outcome=data.get("outcome", data.get("outcomeName", "")),
            outcome_index=int(data.get("outcomeIndex", data.get("outcome_index", 0)) or 0),
            size=size,
            avg_price=avg_price,
            current_price=current_price,
            value=value,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=float(data.get("realizedPnl", data.get("realized_pnl", 0)) or 0),
            asset_id=data.get("assetId", data.get("asset_id")),
        )


@dataclass
class WalletActivity:
    """Summary of wallet activity"""
    address: str
    trades: List[Trade] = field(default_factory=list)
    positions: List[Position] = field(default_factory=list)
    total_volume: float = 0.0
    total_trades: int = 0
    total_pnl: float = 0.0
    first_trade: Optional[datetime] = None
    last_trade: Optional[datetime] = None

    def calculate_summary(self):
        """Calculate summary statistics from trades"""
        if self.trades:
            self.total_trades = len(self.trades)
            self.total_volume = sum(t.value for t in self.trades)
            self.first_trade = min(t.timestamp for t in self.trades)
            self.last_trade = max(t.timestamp for t in self.trades)

        if self.positions:
            self.total_pnl = sum(p.realized_pnl + p.unrealized_pnl for p in self.positions)


class DataClient(BaseClient):
    """Client for Polymarket Data API (trades/positions)"""

    def __init__(self):
        super().__init__(
            base_url=api_config.DATA_API_BASE,
            delay_seconds=api_config.DATA_API_DELAY,
        )

    def get_trades_by_condition(
        self,
        condition_id: str,
        limit: int = 500,
        max_trades: Optional[int] = None,
    ) -> List[Trade]:
        """
        Get trades for a specific market condition.

        Args:
            condition_id: Market condition ID
            limit: Trades per page
            max_trades: Maximum total trades to fetch

        Returns:
            List of Trade objects
        """
        max_pages = None
        if max_trades:
            max_pages = (max_trades // limit) + 1

        # The API uses 'market' parameter for condition ID
        response = self.get_paginated(
            "trades",
            params={"market": condition_id},
            page_size=limit,
            max_pages=max_pages,
        )
        return [Trade.from_dict(t) for t in response]

    def get_wallet_trades(
        self,
        wallet_address: str,
        limit: int = 500,
        max_trades: Optional[int] = None,
    ) -> List[Trade]:
        """
        Get all trades for a wallet address.

        Args:
            wallet_address: Wallet address to query
            limit: Trades per page
            max_trades: Maximum total trades to fetch

        Returns:
            List of Trade objects
        """
        max_pages = None
        if max_trades:
            max_pages = (max_trades // limit) + 1

        response = self.get_paginated(
            "activity",
            params={"user": wallet_address, "type": "TRADE"},
            page_size=limit,
            max_pages=max_pages,
        )
        return [Trade.from_dict(t) for t in response]

    def get_wallet_positions(
        self,
        wallet_address: str,
    ) -> List[Position]:
        """
        Get current positions for a wallet.

        Args:
            wallet_address: Wallet address to query

        Returns:
            List of Position objects
        """
        response = self.get(
            "positions",
            params={"user": wallet_address},
        )

        if not response:
            return []

        positions_data = response if isinstance(response, list) else response.get("positions", [])
        return [Position.from_dict(p) for p in positions_data]

    def get_wallet_activity(
        self,
        wallet_address: str,
        include_positions: bool = True,
        max_trades: Optional[int] = None,
    ) -> WalletActivity:
        """
        Get full activity summary for a wallet.

        Args:
            wallet_address: Wallet address to query
            include_positions: Whether to fetch current positions
            max_trades: Maximum trades to fetch

        Returns:
            WalletActivity object with trades and positions
        """
        activity = WalletActivity(address=wallet_address)

        # Fetch trades
        activity.trades = self.get_wallet_trades(
            wallet_address,
            max_trades=max_trades,
        )

        # Fetch positions
        if include_positions:
            activity.positions = self.get_wallet_positions(wallet_address)

        # Calculate summary
        activity.calculate_summary()

        return activity

    def extract_wallets_from_trades(
        self,
        condition_id: str,
        max_trades: int = 1000,
    ) -> List[str]:
        """
        Extract unique wallet addresses from market trades.

        Args:
            condition_id: Market condition ID
            max_trades: Maximum trades to scan

        Returns:
            List of unique wallet addresses
        """
        trades = self.get_trades_by_condition(
            condition_id,
            max_trades=max_trades,
        )

        wallets = set()
        for trade in trades:
            if trade.wallet_address:
                wallets.add(trade.wallet_address.lower())

        return list(wallets)

    def get_wallet_stats(self, wallet_address: str) -> Dict[str, Any]:
        """
        Get quick stats for lite filtering.

        Args:
            wallet_address: Wallet address

        Returns:
            Dict with volume and trade count
        """
        # Fetch limited trades to get quick stats
        trades = self.get_wallet_trades(
            wallet_address,
            limit=100,
            max_trades=100,
        )

        volume = sum(t.value for t in trades)
        trade_count = len(trades)

        return {
            "address": wallet_address,
            "volume": volume,
            "trade_count": trade_count,
            "has_more": trade_count >= 100,
        }

    def get_recent_trades(
        self,
        limit: int = 100,
        market_ids: Optional[List[str]] = None,
    ) -> List[Trade]:
        """
        Get recent trades across all markets.

        Args:
            limit: Maximum trades to return
            market_ids: Optional list of market IDs to filter

        Returns:
            List of Trade objects
        """
        params = {"limit": limit}
        if market_ids:
            params["market_ids"] = ",".join(market_ids)

        response = self.get("trades", params)

        if not response:
            return []

        trades_data = response if isinstance(response, list) else response.get("data", [])
        return [Trade.from_dict(t) for t in trades_data]

    def get_market_trades(
        self,
        market_id: str,
        limit: int = 500,
    ) -> List[Trade]:
        """Get trades for a specific market"""
        response = self.get_paginated(
            "trades",
            params={"market_id": market_id},
            page_size=limit,
        )
        return [Trade.from_dict(t) for t in response]
