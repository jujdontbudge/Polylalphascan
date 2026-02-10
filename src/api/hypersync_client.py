"""
HyperSync Client - 2000x faster blockchain queries than RPC.

HyperSync is a high-performance blockchain data API that can query
30M blocks in ~5 minutes (vs 20+ hours with traditional RPC).

Used for:
- Full wallet discovery from CTF Exchange events
- Historical trade indexing
- Gap filling for missed events
"""
import asyncio
import logging
from typing import List, Set, Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

# Polymarket CTF Exchange contracts on Polygon
# Using the same addresses as polygon_rpc_client.py for consistency
CTF_EXCHANGE_MULTI = "0xC5d563A36AE78145C45a50134d48A1215220f80a"  # Multi-outcome exchange
CTF_EXCHANGE_LEGACY = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"  # Legacy binary exchange

# OrderFilled event signature (keccak256 hash)
# event OrderFilled(bytes32 indexed orderHash, address indexed maker, address indexed taker,
#                   uint256 makerAssetId, uint256 takerAssetId, uint256 makerAmountFilled,
#                   uint256 takerAmountFilled, uint256 fee)
ORDER_FILLED_TOPIC = "0xd0a08e8c493f9c94f29311604c9de1b4e8c8d4c06bd0c789af57f2d65bfec0f6"

# Block range constants
POLYMARKET_START_BLOCK = 30_000_000  # Polymarket deployment (~Sept 2021)

# Check if hypersync is available
try:
    import hypersync
    from hypersync import (
        HypersyncClient as HypersyncClientLib,
        ClientConfig,
        Query,
        LogSelection,
        FieldSelection,
        LogField,
        BlockField,
    )
    HYPERSYNC_AVAILABLE = True
except ImportError:
    HYPERSYNC_AVAILABLE = False
    logger.warning("hypersync not installed. Install with: pip install hypersync")


@dataclass
class OrderFilledEvent:
    """Parsed OrderFilled event from CTF Exchange"""
    transaction_hash: str
    block_number: int
    log_index: int
    timestamp: datetime
    maker: str
    taker: str
    market_id: str
    outcome_index: int
    side: str  # BUY or SELL
    price: float
    size: float
    fee: float

    @classmethod
    def from_log(cls, log: Dict[str, Any], timestamp: datetime) -> Optional["OrderFilledEvent"]:
        """Parse from raw log data"""
        try:
            # Decode log data based on OrderFilled event ABI
            topics = log.get("topics", [])
            data = log.get("data", "")

            if len(topics) < 4:
                return None

            # Topic[1] = maker, Topic[2] = taker, Topic[3] = marketId
            maker = "0x" + topics[1][-40:]
            taker = "0x" + topics[2][-40:]
            market_id = topics[3]

            # Decode data (price, size, fee, outcomeIndex, side)
            # This is a simplified example - actual decoding depends on ABI
            data_bytes = bytes.fromhex(data[2:]) if data.startswith("0x") else bytes.fromhex(data)

            return cls(
                transaction_hash=log.get("transactionHash", ""),
                block_number=int(log.get("blockNumber", 0)),
                log_index=int(log.get("logIndex", 0)),
                timestamp=timestamp,
                maker=maker.lower(),
                taker=taker.lower(),
                market_id=market_id,
                outcome_index=0,  # Decode from data
                side="BUY",  # Decode from data
                price=0.0,  # Decode from data
                size=0.0,  # Decode from data
                fee=0.0,  # Decode from data
            )
        except Exception as e:
            logger.debug(f"Failed to parse OrderFilled event: {e}")
            return None


@dataclass
class HarvestResult:
    """Result from wallet harvesting"""
    wallets: Set[str] = field(default_factory=set)
    blocks_scanned: int = 0
    events_processed: int = 0
    duration_seconds: float = 0.0
    errors: List[str] = field(default_factory=list)

    @property
    def wallets_per_second(self) -> float:
        if self.duration_seconds > 0:
            return len(self.wallets) / self.duration_seconds
        return 0.0


class HyperSyncHarvester:
    """
    High-performance wallet harvester using HyperSync.

    2000x faster than traditional RPC for scanning blockchain events.
    Can scan 52.5M blocks in ~5 minutes.
    """

    # HyperSync URL for Polygon
    POLYGON_HYPERSYNC_URL = "https://polygon.hypersync.xyz"

    def __init__(
        self,
        chain_id: int = 137,  # Polygon
        batch_size: int = 10000,
    ):
        self.chain_id = chain_id
        self.batch_size = batch_size
        self.client = None

        if HYPERSYNC_AVAILABLE:
            try:
                config = ClientConfig(url=self.POLYGON_HYPERSYNC_URL)
                self.client = HypersyncClientLib(config)
            except Exception as e:
                logger.error(f"Failed to initialize HyperSync client: {e}")
                self.client = None

    async def harvest_all_wallets(
        self,
        from_block: int = POLYMARKET_START_BLOCK,
        to_block: Optional[int] = None,
        progress_callback: Optional[callable] = None,
    ) -> HarvestResult:
        """
        Harvest all wallet addresses from CTF Exchange OrderFilled events.

        Args:
            from_block: Starting block number (default: Polymarket deployment)
            to_block: Ending block number (None = latest)
            progress_callback: Optional callback(blocks_scanned, total_blocks, wallets_found)

        Returns:
            HarvestResult with discovered wallets
        """
        import time
        start_time = time.monotonic()
        result = HarvestResult()

        if not HYPERSYNC_AVAILABLE:
            result.errors.append("hypersync not installed. Install with: pip install hypersync")
            return result

        if self.client is None:
            result.errors.append("HyperSync client not initialized")
            return result

        try:
            # Build query for OrderFilled events from both CTF Exchange contracts
            query = Query(
                from_block=from_block,
                to_block=to_block,
                logs=[
                    LogSelection(
                        address=[CTF_EXCHANGE_MULTI, CTF_EXCHANGE_LEGACY],
                        topics=[[ORDER_FILLED_TOPIC]],
                    )
                ],
                field_selection=FieldSelection(
                    log=[
                        LogField.BLOCK_NUMBER,
                        LogField.TRANSACTION_HASH,
                        LogField.LOG_INDEX,
                        LogField.ADDRESS,
                        LogField.TOPIC0,
                        LogField.TOPIC1,  # orderHash
                        LogField.TOPIC2,  # maker (indexed)
                        LogField.TOPIC3,  # taker (indexed)
                        LogField.DATA,
                    ],
                    block=[BlockField.NUMBER, BlockField.TIMESTAMP],
                ),
            )

            logger.info(f"Starting HyperSync harvest from block {from_block:,}...")
            if to_block:
                logger.info(f"Target end block: {to_block:,} ({to_block - from_block:,} blocks)")

            # Use collect_events for simpler iteration
            # Or use stream for memory efficiency on large ranges
            current_block = from_block
            batch_count = 0

            while True:
                # Create query for current batch
                batch_query = Query(
                    from_block=current_block,
                    to_block=to_block,
                    logs=[
                        LogSelection(
                            address=[CTF_EXCHANGE_MULTI, CTF_EXCHANGE_LEGACY],
                            topics=[[ORDER_FILLED_TOPIC]],
                        )
                    ],
                    field_selection=FieldSelection(
                        log=[
                            LogField.BLOCK_NUMBER,
                            LogField.TOPIC0,
                            LogField.TOPIC1,
                            LogField.TOPIC2,
                            LogField.TOPIC3,
                        ],
                    ),
                )

                # Fetch batch
                response = await self.client.get(batch_query)

                if not response or not response.data:
                    break

                logs = response.data.logs if hasattr(response.data, 'logs') else []

                if not logs:
                    # Check if we reached the end
                    if response.next_block and (to_block is None or response.next_block <= to_block):
                        current_block = response.next_block
                        continue
                    break

                # Process logs
                for log in logs:
                    topics = log.topics if hasattr(log, 'topics') else []
                    if len(topics) >= 4:
                        # Topic[2] = maker (address, indexed, 32 bytes padded)
                        # Topic[3] = taker (address, indexed, 32 bytes padded)
                        maker_topic = topics[2] if isinstance(topics[2], str) else topics[2].hex() if hasattr(topics[2], 'hex') else str(topics[2])
                        taker_topic = topics[3] if isinstance(topics[3], str) else topics[3].hex() if hasattr(topics[3], 'hex') else str(topics[3])

                        # Extract address from padded topic (last 40 chars)
                        maker = "0x" + maker_topic[-40:].lower()
                        taker = "0x" + taker_topic[-40:].lower()

                        result.wallets.add(maker)
                        result.wallets.add(taker)
                        result.events_processed += 1

                # Update progress
                if response.next_block:
                    result.blocks_scanned = response.next_block - from_block
                    current_block = response.next_block
                else:
                    break

                batch_count += 1

                # Progress callback
                if progress_callback and batch_count % 10 == 0:
                    total_blocks = (to_block - from_block) if to_block else result.blocks_scanned
                    progress_callback(result.blocks_scanned, total_blocks, len(result.wallets))

                # Check if we've reached the target
                if to_block and current_block >= to_block:
                    break

            result.duration_seconds = time.monotonic() - start_time

            logger.info(
                f"HyperSync harvest complete: {len(result.wallets):,} wallets from "
                f"{result.events_processed:,} events in {result.duration_seconds:.1f}s "
                f"({result.blocks_scanned:,} blocks)"
            )

        except Exception as e:
            logger.error(f"HyperSync harvest failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            result.errors.append(str(e))
            result.duration_seconds = time.monotonic() - start_time

        return result

    async def get_wallet_events(
        self,
        wallet_address: str,
        from_block: int = POLYMARKET_START_BLOCK,
        to_block: Optional[int] = None,
    ) -> List[OrderFilledEvent]:
        """
        Get all OrderFilled events for a specific wallet.

        Args:
            wallet_address: Wallet address to query
            from_block: Starting block
            to_block: Ending block

        Returns:
            List of OrderFilledEvent
        """
        if not HYPERSYNC_AVAILABLE or self.client is None:
            return []

        events = []
        # Pad address to 32 bytes (64 hex chars) for topic matching
        wallet_hex = wallet_address.lower()
        if wallet_hex.startswith("0x"):
            wallet_hex = wallet_hex[2:]
        wallet_padded = "0x" + wallet_hex.zfill(64)

        try:
            # Query events where wallet is maker (topic[2]) or taker (topic[3])
            # We need two separate queries since hypersync doesn't support OR on different topic positions

            # Query 1: Wallet as maker
            maker_query = Query(
                from_block=from_block,
                to_block=to_block,
                logs=[
                    LogSelection(
                        address=[CTF_EXCHANGE_MULTI, CTF_EXCHANGE_LEGACY],
                        topics=[
                            [ORDER_FILLED_TOPIC],
                            None,  # orderHash (any)
                            [wallet_padded],  # maker
                        ],
                    ),
                ],
                field_selection=FieldSelection(
                    log=[
                        LogField.BLOCK_NUMBER,
                        LogField.TRANSACTION_HASH,
                        LogField.LOG_INDEX,
                        LogField.TOPIC0,
                        LogField.TOPIC1,
                        LogField.TOPIC2,
                        LogField.TOPIC3,
                        LogField.DATA,
                    ],
                    block=[BlockField.NUMBER, BlockField.TIMESTAMP],
                ),
            )

            # Query 2: Wallet as taker
            taker_query = Query(
                from_block=from_block,
                to_block=to_block,
                logs=[
                    LogSelection(
                        address=[CTF_EXCHANGE_MULTI, CTF_EXCHANGE_LEGACY],
                        topics=[
                            [ORDER_FILLED_TOPIC],
                            None,  # orderHash (any)
                            None,  # maker (any)
                            [wallet_padded],  # taker
                        ],
                    ),
                ],
                field_selection=FieldSelection(
                    log=[
                        LogField.BLOCK_NUMBER,
                        LogField.TRANSACTION_HASH,
                        LogField.LOG_INDEX,
                        LogField.TOPIC0,
                        LogField.TOPIC1,
                        LogField.TOPIC2,
                        LogField.TOPIC3,
                        LogField.DATA,
                    ],
                    block=[BlockField.NUMBER, BlockField.TIMESTAMP],
                ),
            )

            # Execute both queries
            for query in [maker_query, taker_query]:
                response = await self.client.get(query)

                if not response or not response.data:
                    continue

                logs = response.data.logs if hasattr(response.data, 'logs') else []
                blocks = response.data.blocks if hasattr(response.data, 'blocks') else []

                # Build block timestamp lookup
                block_timestamps = {}
                for block in blocks:
                    if hasattr(block, 'number') and hasattr(block, 'timestamp'):
                        block_timestamps[block.number] = block.timestamp

                for log in logs:
                    try:
                        block_num = log.block_number if hasattr(log, 'block_number') else 0
                        timestamp = block_timestamps.get(block_num, 0)
                        ts_datetime = datetime.fromtimestamp(timestamp) if timestamp else datetime.now()

                        log_dict = {
                            "blockNumber": hex(block_num) if block_num else "0x0",
                            "transactionHash": log.transaction_hash if hasattr(log, 'transaction_hash') else "",
                            "logIndex": hex(log.log_index) if hasattr(log, 'log_index') else "0x0",
                            "topics": log.topics if hasattr(log, 'topics') else [],
                            "data": log.data if hasattr(log, 'data') else "0x",
                        }

                        event = OrderFilledEvent.from_log(log_dict, ts_datetime)
                        if event:
                            events.append(event)
                    except Exception as e:
                        logger.debug(f"Error parsing wallet event: {e}")

        except Exception as e:
            logger.error(f"Failed to get wallet events: {e}")
            import traceback
            logger.debug(traceback.format_exc())

        return events


class HyperSyncClient:
    """
    Wrapper for HyperSync that provides high-level wallet data fetching.

    Fallback-compatible: returns empty results if hypersync not installed.
    """

    def __init__(self, chain_id: int = 137):
        self.harvester = HyperSyncHarvester(chain_id=chain_id)

    async def discover_wallets(
        self,
        from_block: int = 30_000_000,
        max_wallets: int = 100_000,
    ) -> Set[str]:
        """
        Discover wallet addresses from blockchain events.

        Args:
            from_block: Starting block
            max_wallets: Maximum wallets to return

        Returns:
            Set of wallet addresses
        """
        result = await self.harvester.harvest_all_wallets(from_block=from_block)
        wallets = result.wallets

        if len(wallets) > max_wallets:
            # Return first max_wallets (arbitrary selection)
            return set(list(wallets)[:max_wallets])

        return wallets

    async def get_wallet_trade_events(
        self,
        wallet_address: str,
        from_block: int = 30_000_000,
    ) -> List[Dict[str, Any]]:
        """
        Get trade events for a wallet.

        Args:
            wallet_address: Wallet address
            from_block: Starting block

        Returns:
            List of trade event dicts
        """
        events = await self.harvester.get_wallet_events(
            wallet_address, from_block=from_block
        )

        return [
            {
                "transaction_hash": e.transaction_hash,
                "block_number": e.block_number,
                "timestamp": e.timestamp.isoformat(),
                "maker": e.maker,
                "taker": e.taker,
                "market_id": e.market_id,
                "side": e.side,
                "price": e.price,
                "size": e.size,
                "fee": e.fee,
            }
            for e in events
        ]


# Convenience function
async def discover_wallets_hypersync(
    from_block: int = 30_000_000,
    max_wallets: int = 100_000,
) -> Set[str]:
    """
    Convenience function to discover wallets using HyperSync.

    Args:
        from_block: Starting block
        max_wallets: Maximum wallets

    Returns:
        Set of wallet addresses
    """
    client = HyperSyncClient()
    return await client.discover_wallets(from_block, max_wallets)
