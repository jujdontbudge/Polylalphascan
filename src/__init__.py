"""
PolyScan Alpha Finder v2.0 - Source Package

High-throughput wallet analysis for Polymarket with 1000+ wallets/second target.

Modules:
- api: Data clients (REST, Goldsky, HyperSync, unified router)
- core/analyzer: Metric calculation engines (standard, parallel, optimized)
- core/indexer: Local trade index for <10ms lookups
- cache: Redis caching layer
- workers: Celery distributed workers
- monitoring: Prometheus metrics
- database: SQLAlchemy models and repositories
- export: Data export utilities
"""
