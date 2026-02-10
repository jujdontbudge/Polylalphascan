"""Step 1: Market Discovery - Find active markets from watering hole categories"""
import logging
from typing import List, Set, Dict, Any
from dataclasses import dataclass, field

from src.api import GammaClient
from src.api.gamma_client import Market, Event
from config.settings import scanner_config

logger = logging.getLogger(__name__)


@dataclass
class DiscoveryResult:
    """Results from market discovery"""
    events: List[Event] = field(default_factory=list)
    markets: List[Market] = field(default_factory=list)
    condition_ids: Set[str] = field(default_factory=set)
    stats: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_events(self) -> int:
        return len(self.events)

    @property
    def total_markets(self) -> int:
        return len(self.markets)

    @property
    def total_conditions(self) -> int:
        return len(self.condition_ids)


class MarketDiscovery:
    """
    Step 1: Discover active markets from watering hole categories.

    Fetches markets from predefined categories (politics, sports, crypto, etc.)
    that are known to have algorithmic traders.
    """

    def __init__(self, gamma_client: GammaClient = None):
        self.client = gamma_client or GammaClient()
        self.tags = scanner_config.WATERING_HOLE_TAGS

    def discover_top_events(
        self,
        limit: int = 50,
        min_volume: float = 100000,
    ) -> DiscoveryResult:
        """
        Discover top events by volume (best method for finding active traders).

        Args:
            limit: Maximum events to return
            min_volume: Minimum volume threshold (default $100k)

        Returns:
            DiscoveryResult with high-volume events
        """
        result = DiscoveryResult()

        logger.info(f"Discovering top {limit} events by volume...")

        events = self.client.get_top_events(limit=limit, min_volume=min_volume)

        for event in events:
            result.events.append(event)

            for market in event.markets:
                if not market.closed:
                    result.markets.append(market)
                    if market.condition_id:
                        result.condition_ids.add(market.condition_id)

        # Categorize for stats
        categories = {}
        for event in events:
            cat = self._categorize_event(event)
            if cat not in categories:
                categories[cat] = {"count": 0, "volume": 0}
            categories[cat]["count"] += 1
            categories[cat]["volume"] += event.volume

        result.stats["categories"] = categories
        result.stats["total_volume"] = sum(e.volume for e in events)
        result.stats["events_found"] = result.total_events
        result.stats["markets_found"] = result.total_markets
        result.stats["conditions_found"] = result.total_conditions

        logger.info(
            f"Discovery complete: {result.total_events} events, "
            f"{result.total_markets} markets, ${result.stats['total_volume']:,.0f} volume"
        )

        return result

    def _categorize_event(self, event: Event) -> str:
        """Categorize an event by its title/content"""
        title_lower = event.title.lower()

        if any(x in title_lower for x in ['trump', 'biden', 'election', 'congress', 'senate', 'president', 'vote', 'governor']):
            return 'Politics'
        elif any(x in title_lower for x in ['nfl', 'nba', 'soccer', 'nhl', 'ufc', 'mma', 'super bowl', 'championship', 'premier league', 'champions league']):
            return 'Sports'
        elif any(x in title_lower for x in ['bitcoin', 'btc', 'eth', 'ethereum', 'crypto', 'solana', 'xrp', 'token']):
            return 'Crypto'
        elif any(x in title_lower for x in ['earnings', 'stock', 'fed', 'rate', 'gdp', 'inflation', 'market', 'treasury']):
            return 'Finance'
        elif any(x in title_lower for x in ['weather', 'temperature', 'hurricane', 'storm']):
            return 'Weather'
        else:
            return 'Other'

    def discover(
        self,
        tags: List[str] = None,
        min_volume: float = 0,
        min_liquidity: float = 0,
    ) -> DiscoveryResult:
        """
        Discover active markets from specified tags.

        Args:
            tags: Categories to search (default: watering hole tags)
            min_volume: Minimum volume filter
            min_liquidity: Minimum liquidity filter

        Returns:
            DiscoveryResult with events, markets, and condition IDs
        """
        tags = tags or self.tags
        result = DiscoveryResult()
        result.stats["tags_searched"] = tags

        logger.info(f"Starting market discovery for tags: {tags}")

        seen_event_ids = set()
        seen_market_ids = set()

        for tag in tags:
            logger.info(f"Fetching events for tag: {tag}")
            events = self.client.get_events_by_tag(tag, active=True, limit=100)

            for event in events:
                # Skip duplicates
                if event.id in seen_event_ids:
                    continue
                seen_event_ids.add(event.id)

                # Apply volume/liquidity filters
                if event.volume < min_volume or event.liquidity < min_liquidity:
                    continue

                result.events.append(event)

                # Extract markets
                for market in event.markets:
                    if market.id in seen_market_ids:
                        continue
                    seen_market_ids.add(market.id)

                    if not market.active or market.closed:
                        continue

                    result.markets.append(market)

                    # Collect condition IDs
                    if market.condition_id:
                        result.condition_ids.add(market.condition_id)

        # Update stats
        result.stats["events_found"] = result.total_events
        result.stats["markets_found"] = result.total_markets
        result.stats["conditions_found"] = result.total_conditions

        logger.info(
            f"Discovery complete: {result.total_events} events, "
            f"{result.total_markets} markets, {result.total_conditions} conditions"
        )

        return result

    def discover_all_active(
        self,
        min_volume: float = 1000,
        min_liquidity: float = 100,
    ) -> DiscoveryResult:
        """
        Discover all active markets (not just watering holes).

        Args:
            min_volume: Minimum volume filter
            min_liquidity: Minimum liquidity filter

        Returns:
            DiscoveryResult with all active markets
        """
        result = DiscoveryResult()

        logger.info("Fetching all active markets...")
        markets = self.client.get_all_active_markets(
            min_volume=min_volume,
            min_liquidity=min_liquidity,
        )

        for market in markets:
            result.markets.append(market)
            if market.condition_id:
                result.condition_ids.add(market.condition_id)

        result.stats["markets_found"] = result.total_markets
        result.stats["conditions_found"] = result.total_conditions

        logger.info(
            f"All-market discovery complete: {result.total_markets} markets"
        )

        return result

    def get_high_volume_markets(
        self,
        min_volume: float = 10000,
        limit: int = 50,
    ) -> List[Market]:
        """Get markets with highest volume"""
        result = self.discover(min_volume=min_volume)
        markets = sorted(result.markets, key=lambda m: m.volume, reverse=True)
        return markets[:limit]

    def get_condition_ids(self, tags: List[str] = None) -> List[str]:
        """
        Get condition IDs for wallet harvesting.

        Args:
            tags: Categories to search

        Returns:
            List of condition IDs
        """
        result = self.discover(tags=tags)
        return list(result.condition_ids)

    def close(self):
        """Close the API client"""
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
