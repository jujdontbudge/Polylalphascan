"""CLOB API client for order book data"""
import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from .base_client import BaseClient
from config.settings import api_config

logger = logging.getLogger(__name__)


@dataclass
class OrderBookEntry:
    """Single order book entry"""
    price: float
    size: float
    side: str  # 'BID' or 'ASK'

    @classmethod
    def from_dict(cls, data: Dict[str, Any], side: str) -> "OrderBookEntry":
        """Create OrderBookEntry from API response"""
        return cls(
            price=float(data.get("price", data.get("p", 0)) or 0),
            size=float(data.get("size", data.get("s", 0)) or 0),
            side=side,
        )


@dataclass
class OrderBook:
    """Full order book for a market"""
    market_id: str
    asset_id: str
    bids: List[OrderBookEntry]
    asks: List[OrderBookEntry]
    timestamp: Optional[str] = None

    @property
    def best_bid(self) -> Optional[float]:
        """Get best bid price"""
        if self.bids:
            return max(b.price for b in self.bids)
        return None

    @property
    def best_ask(self) -> Optional[float]:
        """Get best ask price"""
        if self.asks:
            return min(a.price for a in self.asks)
        return None

    @property
    def spread(self) -> Optional[float]:
        """Get bid-ask spread"""
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None

    @property
    def mid_price(self) -> Optional[float]:
        """Get mid price"""
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return None

    @property
    def total_bid_liquidity(self) -> float:
        """Total liquidity on bid side"""
        return sum(b.size * b.price for b in self.bids)

    @property
    def total_ask_liquidity(self) -> float:
        """Total liquidity on ask side"""
        return sum(a.size * a.price for a in self.asks)

    @classmethod
    def from_dict(cls, data: Dict[str, Any], market_id: str, asset_id: str) -> "OrderBook":
        """Create OrderBook from API response"""
        bids_data = data.get("bids", [])
        asks_data = data.get("asks", [])

        bids = [OrderBookEntry.from_dict(b, "BID") for b in bids_data]
        asks = [OrderBookEntry.from_dict(a, "ASK") for a in asks_data]

        return cls(
            market_id=market_id,
            asset_id=asset_id,
            bids=bids,
            asks=asks,
            timestamp=data.get("timestamp"),
        )


@dataclass
class OpenOrder:
    """Open order on the CLOB"""
    id: str
    market_id: str
    asset_id: str
    owner: str
    side: str  # 'BUY' or 'SELL'
    price: float
    original_size: float
    remaining_size: float
    created_at: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OpenOrder":
        """Create OpenOrder from API response"""
        return cls(
            id=data.get("id", data.get("orderId", "")),
            market_id=data.get("marketId", data.get("market_id", "")),
            asset_id=data.get("assetId", data.get("asset_id", "")),
            owner=data.get("owner", data.get("maker", "")),
            side=data.get("side", "").upper(),
            price=float(data.get("price", 0) or 0),
            original_size=float(data.get("originalSize", data.get("size", 0)) or 0),
            remaining_size=float(data.get("remainingSize", data.get("sizeRemaining", 0)) or 0),
            created_at=data.get("createdAt", data.get("timestamp")),
        )


class CLOBClient(BaseClient):
    """Client for Polymarket CLOB API (order book)"""

    def __init__(self):
        super().__init__(
            base_url=api_config.CLOB_API_BASE,
            delay_seconds=api_config.DEFAULT_DELAY,
        )

    def get_order_book(
        self,
        token_id: str,
        market_id: Optional[str] = None,
    ) -> Optional[OrderBook]:
        """
        Get order book for a token/asset.

        Args:
            token_id: Token/asset ID
            market_id: Optional market ID for context

        Returns:
            OrderBook object
        """
        response = self.get(f"book", params={"token_id": token_id})

        if not response:
            return None

        return OrderBook.from_dict(
            response,
            market_id=market_id or "",
            asset_id=token_id,
        )

    def get_order_books(
        self,
        token_ids: List[str],
    ) -> Dict[str, OrderBook]:
        """
        Get order books for multiple tokens.

        Args:
            token_ids: List of token IDs

        Returns:
            Dict mapping token_id to OrderBook
        """
        order_books = {}
        for token_id in token_ids:
            book = self.get_order_book(token_id)
            if book:
                order_books[token_id] = book
        return order_books

    def get_open_orders(
        self,
        wallet_address: str,
        market_id: Optional[str] = None,
    ) -> List[OpenOrder]:
        """
        Get open orders for a wallet.

        Args:
            wallet_address: Wallet address
            market_id: Optional market filter

        Returns:
            List of OpenOrder objects
        """
        params = {"owner": wallet_address}
        if market_id:
            params["market_id"] = market_id

        response = self.get("orders", params)

        if not response:
            return []

        orders_data = response if isinstance(response, list) else response.get("orders", [])
        return [OpenOrder.from_dict(o) for o in orders_data]

    def get_market_makers(
        self,
        token_id: str,
        min_liquidity: float = 100.0,
    ) -> List[str]:
        """
        Extract market makers (providers of liquidity) from order book.

        Args:
            token_id: Token/asset ID
            min_liquidity: Minimum liquidity to qualify

        Returns:
            List of wallet addresses providing liquidity
        """
        # This would require order-level data with maker info
        # For now, return empty - actual implementation depends on API
        book = self.get_order_book(token_id)
        if not book:
            return []

        # In practice, we'd need access to individual order data
        # to extract maker addresses
        return []

    def get_spread(self, token_id: str) -> Optional[Dict[str, float]]:
        """
        Get bid-ask spread info for a token.

        Args:
            token_id: Token/asset ID

        Returns:
            Dict with spread metrics
        """
        book = self.get_order_book(token_id)
        if not book:
            return None

        return {
            "best_bid": book.best_bid,
            "best_ask": book.best_ask,
            "spread": book.spread,
            "mid_price": book.mid_price,
            "bid_liquidity": book.total_bid_liquidity,
            "ask_liquidity": book.total_ask_liquidity,
        }

    def get_liquidity_depth(
        self,
        token_id: str,
        price_depth: float = 0.05,
    ) -> Optional[Dict[str, float]]:
        """
        Calculate liquidity within price depth of mid.

        Args:
            token_id: Token/asset ID
            price_depth: Percentage from mid (0.05 = 5%)

        Returns:
            Dict with liquidity metrics
        """
        book = self.get_order_book(token_id)
        if not book or not book.mid_price:
            return None

        mid = book.mid_price
        upper = mid * (1 + price_depth)
        lower = mid * (1 - price_depth)

        bid_depth = sum(b.size * b.price for b in book.bids if b.price >= lower)
        ask_depth = sum(a.size * a.price for a in book.asks if a.price <= upper)

        return {
            "mid_price": mid,
            "bid_depth": bid_depth,
            "ask_depth": ask_depth,
            "total_depth": bid_depth + ask_depth,
            "depth_ratio": bid_depth / ask_depth if ask_depth > 0 else 0,
        }
