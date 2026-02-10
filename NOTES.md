# NOTES.md -- PolyScan Alpha Finder v2.0
# Everything needed to resume work without re-explaining.
# Last updated: 2026-02-03

=========================================================================
## 1. PROJECT PURPOSE
=========================================================================

PolyScan Alpha Finder is a Polymarket wallet intelligence platform.
It discovers, analyzes, ranks, and classifies trading wallets to find
the best algorithmic traders (bots/quants) for copy-trading.

Primary goals:
- Discover every wallet that trades on Polymarket (zero missed wallets)
- Calculate 117 trading metrics across 15 categories per wallet
- Maintain Top 100 leaderboards per metric (117 x 3 time periods)
- Classify wallets as bot / quant / human / unknown
- Generate a curated "copy roster" of best wallets to follow
- Export results as Excel or JSON

The system has TWO data ingestion paths:
  1. Polymarket REST/WebSocket APIs (rate-limited)
  2. Direct Polygon RPC blockchain queries (unlimited, newer)


=========================================================================
## 2. KEY ARCHITECTURE / WORKFLOW
=========================================================================

### 2a. 5-Stage Pipeline (scripts/run_scanner.py --mode full)

```
  Stage 1: Market Discovery
    GammaClient -> top events by volume -> list of condition_ids
                                                    |
  Stage 2: Wallet Harvesting                        v
    DataClient -> trades per condition -> unique wallet addresses
                                                    |
  Stage 3: Lite Filter                              v
    DataClient -> quick stats per wallet -> discard low-activity
                                                    |
  Stage 4: Deep Scan                                v
    DataClient -> full trade history -> MetricEngine (117 metrics)
    -> save to wallet_metrics table -> classify (bot/quant/human)
                                                    |
  Stage 5: Ranking                                  v
    RankingEngine -> sort all wallets per metric -> Top 100 per
    metric per time period -> metric_rankings table
```

### 2b. Realtime Trade Capture (scripts/run_realtime_capture.py)

```
  WebSocket (wss://ws-live-data.polymarket.com)
      |
      v
  Deduplicator (Bloom filter -> LRU cache -> DB unique constraint)
      |
      v
  captured_trades table
      |
  REST Backup Poller (10s interval, when WS unhealthy)
      |
  Gap Detector (compares WS vs REST every 60s)
      |
  Checkpoint Manager (saves progress every 10s)
```

### 2c. Direct Polygon RPC Harvester (scripts/harvest_rpc.py)

```
  Polygon blockchain -> eth_getLogs (OrderFilled events)
      |
      v
  5 free RPC providers (round-robin, health checks, failover)
      |
      v
  Parse maker/taker addresses from CTF Exchange contracts
      |
      v
  wallets table (via WalletRepository)
```


=========================================================================
## 3. CURRENT STATUS
=========================================================================

### What is done and working:
- [x] Full database schema (7 tables, all indexed)
- [x] 4 API clients (Gamma, Data, CLOB, Polygon RPC)
- [x] Market discovery (tag-based and top-events)
- [x] Wallet harvesting (API-based and RPC-based)
- [x] Lite filter with configurable thresholds
- [x] 117 metrics defined in YAML, loaded into DB
- [x] All 15 metric group calculators (A through O)
- [x] MetricEngine orchestrating all group calculations
- [x] Deep scanner fetching full trade history + running metrics
- [x] Bot/quant classifier using Group N metrics
- [x] Ranking engine with Top 100 leaderboards (3 time periods)
- [x] Copy roster generator (weighted scoring)
- [x] Excel exporter (leaderboards, roster, wallet profiles)
- [x] JSON exporter
- [x] Realtime WebSocket + REST backup trade capture
- [x] Bloom filter + LRU deduplication
- [x] Gap detection and checkpoint recovery
- [x] Continuous trade monitor (REST-only fallback)
- [x] Parallel API harvester (1000 markets concurrent)
- [x] Direct Polygon RPC harvester (OrderFilled events)
- [x] RPC provider pool (rotation, health, failover)
- [x] CLI scripts for all modes
- [x] Unit tests (database, config, metrics, dedup, checkpoint)

### Database state as of last run:
- wallets: 15,956 rows (most from RPC harvesting)
- metrics: 117 rows (all loaded from YAML)
- wallet_metrics: 3,159 rows (53 wallets deep-scanned)
- metric_rankings: 3,159 rows
- captured_trades: 0 rows (realtime capture not run yet)
- scan_history: 0 rows
- Classifications: 28 quant, 1 human, 15,927 unknown

### What is pending / incomplete:
- [ ] Most wallets (15,903 of 15,956) have not been deep-scanned
- [ ] PnL showing $0 for top wallets -- needs investigation
      (likely the market-based PnL calc needs resolved positions)
- [ ] captured_trades table is empty (realtime capture not run)
- [ ] No .env file exists; all config is hardcoded in settings.py
- [ ] No CI/CD or deployment automation
- [ ] No cron/scheduler for periodic re-scanning
- [ ] RPC provider list needs maintenance (free tiers change often)
- [ ] Some free RPC providers (Ankr, Blast) now require API keys
- [ ] No PostgreSQL migration path (SQLite only for now)


=========================================================================
## 4. IMPORTANT FILES + WHERE TO LOOK
=========================================================================

### Configuration
  config/settings.py             All config dataclasses + global instances
  config/metrics_config.yaml     117 metric definitions (985 lines)
  config/__init__.py             Exports: api_config, scanner_config,
                                 classifier_config, metrics_config,
                                 rpc_config

### API Clients
  src/api/base_client.py         HTTP client base (rate limit, retry, pagination)
  src/api/gamma_client.py        Market/event discovery (gamma-api.polymarket.com)
  src/api/data_client.py         Trade history & positions (data-api.polymarket.com)
  src/api/clob_client.py         Order book data (clob.polymarket.com)
  src/api/polygon_rpc_client.py  Direct blockchain RPC (Polygon JSON-RPC)

### Database
  src/database/models.py         SQLAlchemy ORM: Wallet, Metric, WalletMetric,
                                 MetricRanking, CapturedTrade, ScanHistory,
                                 ScannerState + init_db/get_session
  src/database/repositories/
    wallet_repository.py         CRUD for wallets
    metric_repository.py         CRUD for metrics + YAML loading
    ranking_repository.py        CRUD for rankings

### Scanner / Discovery
  src/core/scanner/discovery.py         Market discovery (Stage 1)
  src/core/scanner/harvester.py         API-based wallet harvesting (Stage 2)
  src/core/scanner/filter.py            Lite filter (Stage 3)
  src/core/scanner/parallel_harvester.py  Async parallel harvester (1000 markets)
  src/core/scanner/rpc_harvester.py     Polygon RPC-based harvester
  src/core/scanner/continuous_monitor.py  REST-only continuous capture

### Analyzer (117 Metrics)
  src/core/analyzer/metric_engine.py    Orchestrates all 15 groups
  src/core/analyzer/deep_scanner.py     Stage 4: full wallet analysis
  src/core/analyzer/groups/
    base_group.py                       Abstract base with utility methods
    group_a_profitability.py            IDs 1-10:  PnL, ROI, profit factor
    group_b_risk.py                     IDs 11-21: Sharpe, Sortino, drawdown, VaR
    group_c_winloss.py                  IDs 22-32: Win rate, expectancy, Kelly
    group_d_activity.py                 IDs 33-40: Trade count, volume, active days
    group_e_timing.py                   IDs 41-48: Hold times, entry timing
    group_f_money.py                    IDs 49-57: Position sizing, diversification
    group_g_strategy.py                 IDs 58-66: Buy/sell ratio, momentum
    group_h_psychological.py            IDs 67-74: Disposition, revenge, FOMO
    group_i_alpha.py                    IDs 75-83: Alpha, info ratio, beta
    group_j_hygiene.py                  IDs 84-94: Wallet age, data quality
    group_k_copyability.py              IDs 95-100: Frequency, scalability
    group_l_market_intel.py             IDs 101-104: Timing accuracy, prediction
    group_m_execution.py                IDs 105-108: Fill rate, price improvement
    group_n_bot_quant.py                IDs 109-113: Bot detection signals
    group_o_portfolio.py                IDs 114-117: Portfolio heat, correlation

### Classifier / Ranker
  src/core/classifier/bot_detector.py   Bot vs quant vs human classification
  src/core/classifier/copy_roster.py    Copy-trading candidate scoring
  src/core/ranker/ranking_engine.py     Top 100 leaderboard maintenance

### Realtime Capture
  src/core/realtime/orchestrator.py     Coordinates all realtime components
  src/core/realtime/websocket_manager.py  WebSocket trade streaming
  src/core/realtime/rest_backup.py      REST polling backup
  src/core/realtime/gap_detector.py     Gap detection between WS and REST
  src/core/realtime/checkpoint_manager.py  Progress saving / crash recovery
  src/core/realtime/deduplicator.py     Bloom filter + LRU cache dedup

### Export
  src/export/excel_exporter.py   Excel workbooks (openpyxl)
  src/export/json_exporter.py    JSON files

### Scripts (entry points)
  scripts/run_scanner.py              Main pipeline (all modes)
  scripts/run_realtime_capture.py     24/7 WebSocket + REST capture
  scripts/harvest_rpc.py              Polygon RPC blockchain harvester
  scripts/harvest_1000_markets.py     API-based mass market harvester
  scripts/fast_live_test.py           Quick connectivity test
  scripts/live_test.py                Full integration test
  scripts/migrate_to_time_periods.py  DB migration script

### Tests
  tests/test_database_models.py   ORM and schema
  tests/test_config.py            Configuration loading
  tests/test_metrics.py           Metric calculations
  tests/test_deduplicator.py      Bloom filter + LRU cache
  tests/test_checkpoint.py        Checkpoint save/restore

### Data / Output
  data/polyscan.db          SQLite database (persistent)
  exports/                  Generated Excel/JSON files


=========================================================================
## 5. COMMANDS TO RUN
=========================================================================

### Install dependencies
  pip install -r requirements.txt

### Initialize database (creates tables + loads 117 metrics)
  python scripts/run_scanner.py --mode init

### Full 5-stage pipeline
  python scripts/run_scanner.py --mode full --wallets 100

### Individual stages
  python scripts/run_scanner.py --mode discover
  python scripts/run_scanner.py --mode harvest
  python scripts/run_scanner.py --mode filter
  python scripts/run_scanner.py --mode scan --wallets 500
  python scripts/run_scanner.py --mode rank --recalc
  python scripts/run_scanner.py --mode classify
  python scripts/run_scanner.py --mode export --format excel

### Realtime capture (24/7, use Ctrl+C to stop)
  python scripts/run_realtime_capture.py
  python scripts/run_realtime_capture.py --verbose

### REST-only continuous capture
  python scripts/run_scanner.py --mode continuous

### RPC harvester (direct blockchain)
  python scripts/harvest_rpc.py --hours 1          # last hour
  python scripts/harvest_rpc.py --hours 24         # last day
  python scripts/harvest_rpc.py --days 7           # last week
  python scripts/harvest_rpc.py --blocks 82000000 82100000
  python scripts/harvest_rpc.py --all-history      # ~20 min
  python scripts/harvest_rpc.py --hours 1 --no-save   # dry run

### API-based mass harvester
  python scripts/harvest_1000_markets.py --markets 1000 --concurrent 150

### Run tests
  pytest tests/
  pytest tests/test_metrics.py -v

### Quick connectivity check
  python scripts/fast_live_test.py


=========================================================================
## 6. ENVIRONMENT VARIABLES / SECRETS
=========================================================================

Currently NONE are required. All configuration is hardcoded in
config/settings.py. There is no .env file.

If adding paid RPC providers in the future, you would add:
  ANKR_API_KEY          -- For https://rpc.ankr.com/polygon/{key}
  ALCHEMY_API_KEY       -- For Alchemy Polygon RPC
  INFURA_PROJECT_ID     -- For Infura Polygon RPC
  POLYGONSCAN_API_KEY   -- For verified contract ABIs

The project lists python-dotenv in requirements.txt but does not
currently call load_dotenv() anywhere. No secrets are embedded.


=========================================================================
## 7. EXTERNAL SERVICES / SERVERS
=========================================================================

### Polymarket APIs (no auth required)
  https://gamma-api.polymarket.com       Market/event discovery
  https://data-api.polymarket.com        Trade history, positions
  https://clob.polymarket.com            Order book data
  wss://ws-live-data.polymarket.com      WebSocket trade stream

### Polygon RPC Providers (no auth, free tier)
  https://1rpc.io/matic                  ~10 req/s, 500 block range
  https://polygon-rpc.com               ~5 req/s, limited block range
  https://polygon-bor-rpc.publicnode.com ~10 req/s, 5000 block range
  https://polygon-mainnet.g.alchemy.com/v2/demo  ~5 req/s (demo key)
  https://polygon.drpc.org              ~10 req/s, 5000 block range

### Polygon Contracts (Polymarket CTF Exchange)
  0xC5d563A36AE78145C45a50134d48A1215220f80a   CTF Exchange (multi-outcome)
  0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E   CTF Exchange (legacy binary)

### Local
  data/polyscan.db    SQLite database (file-based, no server)
  Port: none          No web server or API exposed


=========================================================================
## 8. KNOWN BUGS / CAVEATS
=========================================================================

### Bugs
1. PnL is $0 for most wallets including top-volume ones.
   The market-based PnL calculation in group_a_profitability.py
   requires both BUY and SELL trades for the same market. If a
   wallet only has open positions (buys with no sells yet), PnL
   shows as 0. May also need to account for resolved markets.

2. blocks_scanned in RPCHarvestResult over-counts because
   chunk.size is added once per contract queried (2 contracts).
   A 500-block chunk shows as 1000 blocks scanned.

### Caveats
3. Free RPC providers are unreliable. Ankr now requires API keys.
   Blast returns 403. LlamaNodes DNS fails. Only 1RPC, PublicNode,
   and DrpcPublic work consistently. The provider list in
   polygon_rpc_client.py needs periodic updates.

4. 1RPC actual block range limit is 500 (not 5000 as documented).
   The chunk size in rpc_harvester.py was lowered to 500 to match.

5. polygon-rpc.com returns "Block range is too large" even for
   500-block chunks when querying high-activity contracts. It
   may limit by log count rather than block count.

6. SQLite is single-writer. Running realtime capture + scanner
   simultaneously may cause "database is locked" errors under
   heavy write load. Consider PostgreSQL for production.

7. The Polymarket Data API rate limit headers are not parsed.
   The system uses fixed delays rather than dynamic backoff
   based on X-RateLimit-Remaining.

8. aiohttp brotli decoding fails with some RPC providers.
   The PolygonRPCClient sets Accept-Encoding: gzip, deflate
   to work around this.

9. Only 53 of 15,956 wallets have been deep-scanned. The
   remaining ~15,900 are in "unknown" classification.

10. The Alchemy demo key (polygon-mainnet.g.alchemy.com/v2/demo)
    is heavily rate-limited and often returns 429.


=========================================================================
## 9. NEXT STEPS / TODOS
=========================================================================

### High priority
- [ ] Deep scan remaining ~15,900 wallets (run in batches of 500)
      Command: python scripts/run_scanner.py --mode scan --wallets 500
      Repeat until all wallets are scanned.

- [ ] Debug PnL calculation: check if resolved markets are handled
      correctly. Look at group_a_profitability.py calculate_pnl_by_market.
      May need to fetch wallet positions to account for unrealized PnL.

- [ ] Fix blocks_scanned double-counting in rpc_harvester.py
      (counts once per contract per chunk instead of once per chunk)

### Medium priority
- [ ] Add paid RPC provider support (Alchemy/Infura/QuickNode)
      Would increase throughput from ~40 req/s to 200+ req/s.

- [ ] Run realtime capture for 24+ hours to populate captured_trades.

- [ ] Set up cron job for periodic re-scanning:
        Every hour: harvest new wallets via RPC
        Every 6 hours: deep scan unscanned wallets (batch 500)
        Daily at 2am: full ranking recalculation
        Daily: export leaderboards + roster

- [ ] Consider PostgreSQL migration for concurrent read/write.

### Low priority
- [ ] Add more RPC providers or paid tier keys as free ones degrade.
- [ ] Implement incremental ranking updates (currently full recalc).
- [ ] Add a simple web dashboard for viewing leaderboards.
- [ ] Add Telegram/Discord alerts for new elite wallet discoveries.
- [ ] Integrate with Polymarket betting API for automated copy-trading.
- [ ] Add backtesting framework to validate copy roster performance.


=========================================================================
## 10. ASSUMPTIONS AND CRITICAL CONTEXT
=========================================================================

### Architecture assumptions
- The system runs on a single machine. There is no distributed
  processing, message queue, or microservice architecture.
- SQLite is the only database backend. The ORM uses SQLAlchemy
  so migration to PostgreSQL is structurally possible but untested.
- All API clients are unauthenticated. Polymarket public APIs do
  not require API keys as of this writing.

### Data model assumptions
- A "wallet" is a Polygon address (0x..., 42 chars). The system
  uses proxy wallet addresses from the Polymarket CLOB system,
  not underlying EOA addresses.
- Condition IDs are the primary market identifier. Each Polymarket
  event can have multiple conditions (e.g., Yes/No outcomes).
- Trade data from the Data API uses 'proxyWallet' as the wallet field.
- RPC-harvested wallets come from OrderFilled event maker/taker fields.

### Metric assumptions
- 117 metrics across 15 groups (A-O). Metric IDs 1-117 are fixed
  and loaded from config/metrics_config.yaml.
- Three time periods: all_time, 7d, 31d. Total ranking slots:
  117 metrics x 3 periods x 100 ranks = 35,100.
- higher_is_better flag determines sort direction for ranking.
- Quant score (0-100) is derived from Group N (IDs 109-113):
    >=80 = bot, 50-79 = quant, <=20 = human

### Blockchain assumptions
- Polygon block time is approximately 2 seconds (~1,800 blocks/hour).
- Polymarket CTF Exchange contracts are queried starting from
  block 30,000,000 (approximate deployment). Current block: ~82.5M.
- OrderFilled event topic:
  0xd0a08e8c493f9c94f29311604c9de1b4e8c8d4c06bd0c789af57f2d65bfec0f6
  = keccak256("OrderFilled(bytes32,address,address,uint256,uint256,uint256,uint256,uint256)")
- Both CTF Exchange contracts emit the same OrderFilled event.

### RPC provider assumptions
- Free providers can change limits or require keys at any time.
- The system degrades gracefully when providers go unhealthy
  (round-robin skips them, retries on other providers).
- Chunk size of 500 blocks is the safe minimum across all providers.

### Rate limit assumptions
- Gamma API: 300 req/10s (0.1s delay between requests)
- Data API: 200 req/10s (0.2s delay)
- CLOB API: default 0.2s delay
- RPC combined: ~40 req/s across working free providers
- These are conservative estimates; actual limits may be higher.

### Python version
- Python 3.12 (anaconda3 distribution)
- All code uses f-strings, dataclasses, type hints, asyncio
- No Python 3.13+ features used

### File paths
- Project root: /Users/johansardar/polyscanalphaX/
- Database: data/polyscan.db (relative to project root)
- Exports: exports/ (relative to project root)
- All scripts use sys.path.insert(0, project_root) for imports
