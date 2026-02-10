"""Gamma API client for markets and events"""
import json
import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from .base_client import BaseClient
from config.settings import api_config

logger = logging.getLogger(__name__)


def parse_prices(prices_data) -> List[float]:
    """Parse outcome prices which may be string, list, or JSON string"""
    if not prices_data:
        return []

    # If it's a string, try to parse as JSON
    if isinstance(prices_data, str):
        try:
            prices_data = json.loads(prices_data)
        except (json.JSONDecodeError, ValueError):
            return []

    # Now it should be a list
    if isinstance(prices_data, list):
        result = []
        for p in prices_data:
            try:
                result.append(float(p))
            except (ValueError, TypeError):
                result.append(0.0)
        return result

    return []


def parse_outcomes(outcomes_data) -> List[str]:
    """Parse outcomes which may be string or list"""
    if not outcomes_data:
        return []

    if isinstance(outcomes_data, str):
        try:
            outcomes_data = json.loads(outcomes_data)
        except (json.JSONDecodeError, ValueError):
            return [outcomes_data]

    if isinstance(outcomes_data, list):
        return [str(o) for o in outcomes_data]

    return []


@dataclass
class Market:
    """Polymarket market data"""
    id: str
    condition_id: str
    question: str
    slug: str
    outcomes: List[str]
    outcome_prices: List[float]
    volume: float
    volume_24h: float
    liquidity: float
    end_date: Optional[str]
    active: bool
    closed: bool
    market_type: str
    tags: List[str]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Market":
        """Create Market from API response dict"""
        return cls(
            id=data.get("id", ""),
            condition_id=data.get("conditionId", data.get("condition_id", "")),
            question=data.get("question", ""),
            slug=data.get("slug", ""),
            outcomes=parse_outcomes(data.get("outcomes", [])),
            outcome_prices=parse_prices(data.get("outcomePrices", data.get("outcome_prices", []))),
            volume=float(data.get("volume", 0) or 0),
            volume_24h=float(data.get("volume24hr", data.get("volume_24h", 0)) or 0),
            liquidity=float(data.get("liquidity", 0) or 0),
            end_date=data.get("endDate", data.get("end_date")),
            active=data.get("active", False),
            closed=data.get("closed", False),
            market_type=data.get("marketType", data.get("market_type", "binary")),
            tags=data.get("tags", []),
        )


def parse_tags(tags_data) -> List[str]:
    """Parse tags which may be list of strings or list of objects"""
    if not tags_data:
        return []

    result = []
    for tag in tags_data:
        if isinstance(tag, str):
            result.append(tag.lower())
        elif isinstance(tag, dict):
            # Tags come as objects: {"id": "...", "label": "Politics", "slug": "politics"}
            slug = tag.get("slug", tag.get("label", ""))
            if slug:
                result.append(slug.lower())
    return result


@dataclass
class Event:
    """Polymarket event data (contains multiple markets)"""
    id: str
    slug: str
    title: str
    description: str
    markets: List[Market]
    volume: float
    liquidity: float
    end_date: Optional[str]
    active: bool
    closed: bool
    tags: List[str]
    category: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Event":
        """Create Event from API response dict"""
        markets_data = data.get("markets", [])
        markets = [Market.from_dict(m) for m in markets_data]

        # Extract category from first market if available
        category = ""
        if markets_data:
            category = markets_data[0].get("category", "")

        return cls(
            id=data.get("id", ""),
            slug=data.get("slug", ""),
            title=data.get("title", ""),
            description=data.get("description", ""),
            markets=markets,
            volume=float(data.get("volume", 0) or 0),
            liquidity=float(data.get("liquidity", 0) or 0),
            end_date=data.get("endDate", data.get("end_date")),
            active=data.get("active", False),
            closed=data.get("closed", False),
            tags=parse_tags(data.get("tags", [])),
            category=category,
        )

    def matches_category(self, search_terms: List[str]) -> bool:
        """Check if event matches any of the search terms"""
        search_lower = [s.lower() for s in search_terms]

        # Check tags
        for tag in self.tags:
            if any(term in tag for term in search_lower):
                return True

        # Check category
        if self.category and any(term in self.category.lower() for term in search_lower):
            return True

        # Check title and slug
        title_lower = self.title.lower()
        slug_lower = self.slug.lower()
        for term in search_lower:
            if term in title_lower or term in slug_lower:
                return True

        return False


class GammaClient(BaseClient):
    """Client for Polymarket Gamma API (markets/events)"""

    def __init__(self):
        super().__init__(
            base_url=api_config.GAMMA_API_BASE,
            delay_seconds=api_config.GAMMA_API_DELAY,
        )

    def get_events_by_tag(
        self,
        tag: str,
        active: bool = True,
        limit: int = 100,
    ) -> List[Event]:
        """
        Get events by tag/category.

        Args:
            tag: Event category (politics, sports, crypto, etc.)
            active: Only return active (not closed) events
            limit: Maximum events to return

        Returns:
            List of Event objects matching the tag
        """
        # Use closed=false to get actually open events
        params = {
            "tag_slug": tag.lower(),
            "closed": "false",
            "limit": limit,
            "order": "volume",
        }
        response = self.get("events", params)

        if not response:
            # Fallback: fetch all and filter client-side
            return self._get_filtered_events([tag], active, limit)

        events_data = response if isinstance(response, list) else response.get("data", [])
        events = [Event.from_dict(e) for e in events_data]

        # Filter by tag match and only include events with volume
        filtered = [
            e for e in events
            if e.matches_category([tag]) and e.volume > 0
        ]

        return filtered[:limit]

    def _get_filtered_events(
        self,
        tags: List[str],
        active: bool = True,
        limit: int = 100,
    ) -> List[Event]:
        """Fetch all events and filter by tags client-side"""
        params = {
            "closed": "false",
            "limit": 500,  # Fetch more to filter
            "order": "volume",
        }
        response = self.get("events", params)

        if not response:
            return []

        events_data = response if isinstance(response, list) else response.get("data", [])
        events = [Event.from_dict(e) for e in events_data]

        # Filter by tag match
        filtered = [e for e in events if e.matches_category(tags)]

        return filtered[:limit]

    def get_active_events(
        self,
        tags: Optional[List[str]] = None,
        min_volume: float = 0,
        min_liquidity: float = 0,
    ) -> List[Event]:
        """
        Get all active events, optionally filtered by tags.

        Args:
            tags: List of category tags to filter by
            min_volume: Minimum total volume
            min_liquidity: Minimum liquidity

        Returns:
            List of Event objects
        """
        all_events = []

        if tags:
            for tag in tags:
                events = self.get_events_by_tag(tag, active=True)
                all_events.extend(events)
        else:
            # Get all open events (not closed)
            response = self.get_paginated(
                "events",
                params={"closed": "false"},
                page_size=100,
            )
            all_events = [Event.from_dict(e) for e in response]

        # Filter by volume and liquidity
        filtered = [
            e for e in all_events
            if e.volume >= min_volume and e.liquidity >= min_liquidity
        ]

        # Remove duplicates by ID
        seen = set()
        unique_events = []
        for event in filtered:
            if event.id not in seen:
                seen.add(event.id)
                unique_events.append(event)

        return unique_events

    def get_market(self, market_id: str) -> Optional[Market]:
        """Get a single market by ID"""
        response = self.get(f"markets/{market_id}")
        if response:
            return Market.from_dict(response)
        return None

    def get_markets(
        self,
        active: bool = True,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Market]:
        """Get markets with pagination"""
        params = {
            "closed": "false" if active else "true",
            "limit": limit,
            "offset": offset,
        }
        response = self.get("markets", params)

        if not response:
            return []

        markets_data = response if isinstance(response, list) else response.get("data", [])
        return [Market.from_dict(m) for m in markets_data]

    def get_high_volume_markets(
        self,
        min_liquidity: float = 1000,
        limit: int = 100,
    ) -> List[Market]:
        """Get markets with significant liquidity (best for wallet discovery)"""
        params = {
            "closed": "false",
            "liquidity_num_min": str(min_liquidity),
            "limit": limit,
            "order": "volume",
            "ascending": "false",  # Highest volume first
        }
        response = self.get("markets", params)

        if not response:
            return []

        markets_data = response if isinstance(response, list) else response.get("data", [])
        return [Market.from_dict(m) for m in markets_data]

    def get_top_events(
        self,
        limit: int = 50,
        min_volume: float = 0,
    ) -> List[Event]:
        """
        Get top events by volume (best for finding active traders).

        Args:
            limit: Maximum events to return
            min_volume: Minimum volume threshold

        Returns:
            List of Event objects sorted by volume
        """
        params = {
            "closed": "false",
            "limit": limit,
            "order": "volume",
            "ascending": "false",
        }
        response = self.get("events", params)

        if not response:
            return []

        events_data = response if isinstance(response, list) else response.get("data", [])
        events = [Event.from_dict(e) for e in events_data]

        # Filter by volume
        if min_volume > 0:
            events = [e for e in events if e.volume >= min_volume]

        return events

    def get_all_active_markets(
        self,
        min_volume: float = 0,
        min_liquidity: float = 0,
    ) -> List[Market]:
        """Get all active markets with filtering"""
        response = self.get_paginated(
            "markets",
            params={"closed": "false"},
            page_size=100,
        )
        markets = [Market.from_dict(m) for m in response]

        # Filter
        return [
            m for m in markets
            if m.volume >= min_volume and m.liquidity >= min_liquidity
        ]

    def search_markets(self, query: str, limit: int = 50) -> List[Market]:
        """Search markets by text query"""
        params = {"q": query, "limit": limit}
        response = self.get("markets", params)

        if not response:
            return []

        markets_data = response if isinstance(response, list) else response.get("data", [])
        return [Market.from_dict(m) for m in markets_data]

    def get_condition_ids(self, tags: Optional[List[str]] = None) -> List[str]:
        """
        Get all condition IDs for active markets.

        Args:
            tags: Optional list of tags to filter by

        Returns:
            List of condition IDs
        """
        events = self.get_active_events(tags=tags)
        condition_ids = []
        for event in events:
            for market in event.markets:
                if market.condition_id and market.condition_id not in condition_ids:
                    condition_ids.append(market.condition_id)
        return condition_ids
