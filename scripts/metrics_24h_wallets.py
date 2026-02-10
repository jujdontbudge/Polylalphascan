#!/usr/bin/env python3
"""
Compute full 117 metrics for wallets active in the last N hours.

- Discovers wallets active in the last N hours (default: 24)
- Fetches FULL trade history + current positions for each wallet
- Computes all 117 metrics for 3 timeframes (all_time, 7d, 31d)
- Writes Parquet parts incrementally (RAM efficient, resumable)

Usage:
  python scripts/metrics_24h_wallets.py --source hypersync --hours 24
  python scripts/metrics_24h_wallets.py --source captured --hours 24
  python scripts/metrics_24h_wallets.py --source file --input exports/metrics_24h/wallets.parquet
"""
import argparse
import asyncio
import json
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Iterable, Tuple

import aiohttp
import pandas as pd

# Add project root
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import api_config
from src.api.async_data_client import AsyncRateLimiter
from src.api.data_client import Trade, Position
from src.core.analyzer.metric_engine import MetricEngine
from src.core.analyzer.groups.base_group import TIME_PERIODS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CTF_MULTI = "0xC5d563A36AE78145C45a50134d48A1215220f80a"
CTF_LEGACY = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"
ORDER_FILLED = "0xd0a08e8c493f9c94f29311604c9de1b4e8c8d4c06bd0c789af57f2d65bfec0f6"


def _chunked(items: List[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), size):
        yield items[i:i + size]


def _topic_to_address(topic: object) -> Optional[str]:
    if topic is None:
        return None
    if hasattr(topic, "hex"):
        hex_str = topic.hex()
    else:
        hex_str = str(topic)
    if hex_str.startswith("0x"):
        hex_str = hex_str[2:]
    if len(hex_str) < 40:
        return None
    return "0x" + hex_str[-40:].lower()


def load_wallets_from_captured_trades(
    db_path: Path,
    hours: int = 24,
) -> Set[str]:
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    cutoff_iso = cutoff.isoformat()

    con = sqlite3.connect(db_path)
    try:
        cur = con.cursor()
        cur.execute(
            "SELECT DISTINCT wallet_address FROM captured_trades WHERE timestamp >= ?",
            (cutoff_iso,),
        )
        rows = cur.fetchall()
        wallets = {r[0].lower() for r in rows if r and r[0]}
        logger.info(f"Found {len(wallets):,} wallets in captured_trades (last {hours}h)")
        return wallets
    finally:
        con.close()


def load_wallets_from_file(path: Path) -> Set[str]:
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
        col = "address" if "address" in df.columns else "wallet_address" if "wallet_address" in df.columns else None
        if col is None:
            raise ValueError("Parquet file must contain 'address' or 'wallet_address' column")
        wallets = set(df[col].astype(str).str.lower())
    else:
        with open(path, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            wallets = set(data)
        else:
            wallets = set(data.get("wallets", []))
        wallets = {w.lower() for w in wallets if isinstance(w, str)}

    logger.info(f"Loaded {len(wallets):,} wallets from {path}")
    return wallets


async def load_wallets_hypersync(hours: int = 24, safety_hours: int = 2) -> Set[str]:
    try:
        from hypersync import HypersyncClient, ClientConfig, Query, LogSelection, FieldSelection, LogField, BlockField
    except ImportError:
        raise RuntimeError("hypersync not installed. Install with: pip install hypersync")

    config = ClientConfig(url="https://polygon.hypersync.xyz")
    client = HypersyncClient(config)

    height = await client.get_height()
    blocks_per_hour = 1800  # Polygon ~2s block time
    from_block = max(0, height - int((hours + safety_hours) * blocks_per_hour))

    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    cutoff_ts = int(cutoff.timestamp())

    logger.info(
        f"HyperSync: scanning blocks {from_block:,} to {height:,} (target last {hours}h)"
    )

    wallets: Set[str] = set()
    current = from_block

    while current < height:
        query = Query(
            from_block=current,
            to_block=height,
            logs=[LogSelection(
                address=[CTF_MULTI, CTF_LEGACY],
                topics=[[ORDER_FILLED]],
            )],
            field_selection=FieldSelection(
                log=[LogField.BLOCK_NUMBER, LogField.TOPIC2, LogField.TOPIC3],
                block=[BlockField.NUMBER, BlockField.TIMESTAMP],
            ),
        )

        response = await client.get(query)
        if not response or not response.data:
            break

        logs = response.data.logs if hasattr(response.data, "logs") else []
        blocks = response.data.blocks if hasattr(response.data, "blocks") else []

        block_ts: Dict[int, int] = {}
        for b in blocks:
            if hasattr(b, "number") and hasattr(b, "timestamp"):
                block_ts[int(b.number)] = int(b.timestamp)

        for log in logs:
            block_num = int(getattr(log, "block_number", 0) or 0)
            ts = block_ts.get(block_num)
            if ts is None or ts < cutoff_ts:
                continue

            topics = getattr(log, "topics", []) or []
            if len(topics) < 4:
                continue

            maker = _topic_to_address(topics[2])
            taker = _topic_to_address(topics[3])
            if maker:
                wallets.add(maker)
            if taker:
                wallets.add(taker)

        if getattr(response, "next_block", None):
            current = response.next_block
        else:
            break

    logger.info(f"HyperSync: found {len(wallets):,} wallets (last {hours}h)")
    return wallets


async def _fetch_json(
    session: aiohttp.ClientSession,
    url: str,
    params: Dict[str, str],
    limiter: AsyncRateLimiter,
    max_retries: int = 3,
) -> Optional[Dict]:
    for attempt in range(max_retries):
        await limiter.acquire()
        try:
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    return await resp.json()
                if resp.status == 429:
                    await asyncio.sleep(2 ** attempt)
                else:
                    await asyncio.sleep(0.5)
        except asyncio.TimeoutError:
            await asyncio.sleep(0.5)
        except Exception:
            await asyncio.sleep(0.5)
    return None


async def _fetch_wallet_trades(
    address: str,
    session: aiohttp.ClientSession,
    limiter: AsyncRateLimiter,
    max_trades: Optional[int],
) -> List[Trade]:
    endpoint = f"{api_config.DATA_API_BASE.rstrip('/')}/activity"
    limit = 500
    offset = 0
    trades: List[Trade] = []

    while True:
        params = {
            "user": address,
            "type": "TRADE",
            "limit": str(limit),
            "offset": str(offset),
        }
        data = await _fetch_json(session, endpoint, params, limiter)
        if data is None:
            break

        items = data if isinstance(data, list) else data.get("data", [])
        if not items:
            break

        for item in items:
            trades.append(Trade.from_dict(item))
            if max_trades and len(trades) >= max_trades:
                break

        if max_trades and len(trades) >= max_trades:
            break

        if len(items) < limit:
            break

        offset += limit

    return trades


async def _fetch_wallet_positions(
    address: str,
    session: aiohttp.ClientSession,
    limiter: AsyncRateLimiter,
) -> List[Position]:
    endpoint = f"{api_config.DATA_API_BASE.rstrip('/')}/positions"
    params = {"user": address}
    data = await _fetch_json(session, endpoint, params, limiter)
    if not data:
        return []

    items = data if isinstance(data, list) else data.get("positions", [])
    return [Position.from_dict(item) for item in items]


def _compute_metric_rows(
    metric_engine: MetricEngine,
    address: str,
    trades: List[Trade],
    positions: List[Position],
) -> List[dict]:
    result = metric_engine.calculate_all_periods(address, trades, positions)
    rows: List[dict] = []
    computed_at = datetime.utcnow().isoformat()

    metric_ids = list(range(1, 118))

    for period in TIME_PERIODS:
        period_result = result.results.get(period)
        if not period_result:
            continue

        row = {
            "wallet_address": address,
            "time_period": period,
            "quant_score": period_result.quant_score,
            "classification": period_result.classification,
            "metrics_calculated": period_result.total_calculated,
            "errors": "|".join(period_result.errors) if period_result.errors else "",
            "computed_at": computed_at,
        }
        for mid in metric_ids:
            row[f"metric_{mid:03d}"] = period_result.metrics.get(mid)

        rows.append(row)

    return rows


def _next_part_index(directory: Path) -> int:
    if not directory.exists():
        return 1
    parts = sorted(directory.glob("part-*.parquet"))
    if not parts:
        return 1
    last = parts[-1].stem
    try:
        return int(last.split("-")[-1]) + 1
    except Exception:
        return len(parts) + 1


def _load_processed(processed_dir: Path) -> Set[str]:
    if not processed_dir.exists():
        return set()
    processed: Set[str] = set()
    for part in sorted(processed_dir.glob("part-*.parquet")):
        try:
            df = pd.read_parquet(part, columns=["address"])
            processed.update(df["address"].astype(str).str.lower())
        except Exception:
            continue
    return processed


def _flush_parquet_part(
    rows: List[dict],
    output_dir: Path,
    part_idx: int,
    columns: Optional[List[str]] = None,
) -> None:
    if not rows:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if columns:
        df = df.reindex(columns=columns)
    part_path = output_dir / f"part-{part_idx:05d}.parquet"
    df.to_parquet(part_path, index=False)


async def compute_metrics_streaming(
    wallets: List[str],
    max_concurrent: int,
    requests_per_second: float,
    metrics_dir: Path,
    processed_dir: Path,
    resume: bool,
    max_trades_per_wallet: Optional[int],
    batch_wallets: int,
) -> None:
    processed = _load_processed(processed_dir) if resume else set()
    if processed:
        logger.info(f"Resume enabled: {len(processed):,} wallets already processed")

    limiter = AsyncRateLimiter(
        requests_per_second=requests_per_second,
        burst_size=max(10, int(requests_per_second)),
    )

    connector = aiohttp.TCPConnector(limit=max_concurrent, limit_per_host=max_concurrent)
    timeout = aiohttp.ClientTimeout(total=60)

    metrics_part_idx = _next_part_index(metrics_dir)
    processed_part_idx = _next_part_index(processed_dir)

    buffer_rows: List[dict] = []
    buffer_wallets = 0
    buffer_processed: List[dict] = []

    total = len(wallets)
    done = 0

    metric_engine = MetricEngine()
    metric_columns = [
        "wallet_address",
        "time_period",
        "quant_score",
        "classification",
        "metrics_calculated",
        "errors",
        "computed_at",
    ] + [f"metric_{mid:03d}" for mid in range(1, 118)]

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        async def _compute_one(addr: str) -> Tuple[str, Optional[List[dict]], str]:
            try:
                trades = await _fetch_wallet_trades(addr, session, limiter, max_trades_per_wallet)
                if not trades:
                    return addr, None, "no_trades"
                positions = await _fetch_wallet_positions(addr, session, limiter)
                rows = await asyncio.to_thread(_compute_metric_rows, metric_engine, addr, trades, positions)
                return addr, rows, "ok"
            except Exception:
                return addr, None, "error"

        for chunk in _chunked(wallets, max_concurrent):
            tasks = []
            for addr in chunk:
                if addr in processed:
                    done += 1
                    continue
                tasks.append(asyncio.create_task(_compute_one(addr)))

            for coro in asyncio.as_completed(tasks):
                addr, rows, status = await coro
                done += 1

                if rows:
                    buffer_rows.extend(rows)
                    buffer_wallets += 1

                buffer_processed.append({
                    "address": addr,
                    "status": status,
                    "processed_at": datetime.utcnow().isoformat(),
                })

                if buffer_wallets >= batch_wallets:
                    _flush_parquet_part(buffer_rows, metrics_dir, metrics_part_idx, columns=metric_columns)
                    metrics_part_idx += 1
                    buffer_rows = []
                    buffer_wallets = 0

                if len(buffer_processed) >= batch_wallets:
                    _flush_parquet_part(
                        buffer_processed,
                        processed_dir,
                        processed_part_idx,
                        columns=["address", "status", "processed_at"],
                    )
                    processed_part_idx += 1
                    buffer_processed = []

                if done % 100 == 0 or done == total:
                    logger.info(f"Processed {done:,}/{total:,} wallets...")

    if buffer_rows:
        _flush_parquet_part(buffer_rows, metrics_dir, metrics_part_idx, columns=metric_columns)
    if buffer_processed:
        _flush_parquet_part(buffer_processed, processed_dir, processed_part_idx, columns=["address", "status", "processed_at"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute 117 metrics (all_time, 7d, 31d) for wallets active in last N hours"
    )
    parser.add_argument(
        "--source",
        choices=["captured", "hypersync", "file"],
        required=True,
        help="Wallet source: captured_trades DB, HyperSync, or file",
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=24,
        help="Lookback window in hours (default: 24)",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Wallet file (parquet or json; required for --source file)",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/polyscan.db",
        help="Path to polyscan.db (for --source captured)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of wallets processed",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=10,
        help="Max concurrent HTTP requests (default: 10)",
    )
    parser.add_argument(
        "--rps",
        type=float,
        default=10.0,
        help="Requests per second cap for Data API (default: 10)",
    )
    parser.add_argument(
        "--max-trades",
        type=int,
        default=None,
        help="Max trades per wallet (default: no limit = full history)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="exports/metrics_24h",
        help="Output directory (parquet parts)",
    )
    parser.add_argument(
        "--wallets-parquet",
        type=str,
        default=None,
        help="Write discovered wallets to this parquet file (optional)",
    )
    parser.add_argument(
        "--reuse-wallets",
        action="store_true",
        help="If --wallets-parquet exists, reuse it instead of discovering again",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume (recompute all wallets)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Wallets per parquet part (default: 50)",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    metrics_dir = output_dir / "metrics"
    processed_dir = output_dir / "processed"

    wallets_path = Path(args.wallets_parquet) if args.wallets_parquet else output_dir / "wallets.parquet"

    # Resolve wallet list
    wallets: Set[str]
    if args.reuse_wallets and wallets_path.exists():
        wallets = load_wallets_from_file(wallets_path)
    else:
        if args.source == "captured":
            wallets = load_wallets_from_captured_trades(Path(args.db_path), args.hours)
        elif args.source == "hypersync":
            wallets = asyncio.run(load_wallets_hypersync(args.hours))
        else:
            if not args.input:
                raise SystemExit("--input is required when --source file")
            wallets = load_wallets_from_file(Path(args.input))

        wallets_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"address": sorted(wallets)}).to_parquet(wallets_path, index=False)
        logger.info(f"Wrote {len(wallets):,} wallets to {wallets_path}")

    if args.limit:
        wallets = set(sorted(wallets)[:args.limit])

    if not wallets:
        raise SystemExit("No wallets found for given source/time window")

    wallet_list = sorted(wallets)
    logger.info(f"Computing 117 metrics for {len(wallet_list):,} wallets...")

    asyncio.run(
        compute_metrics_streaming(
            wallet_list,
            max_concurrent=args.max_concurrent,
            requests_per_second=args.rps,
            metrics_dir=metrics_dir,
            processed_dir=processed_dir,
            resume=not args.no_resume,
            max_trades_per_wallet=args.max_trades,
            batch_wallets=args.batch_size,
        )
    )


if __name__ == "__main__":
    main()
