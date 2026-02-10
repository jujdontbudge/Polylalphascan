"""SQLAlchemy models for PolyScan Alpha Finder v2.0"""
from datetime import datetime
from typing import Optional
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    Boolean,
    DateTime,
    Text,
    ForeignKey,
    Index,
    CheckConstraint,
    UniqueConstraint,
)
from sqlalchemy.orm import (
    declarative_base,
    relationship,
    sessionmaker,
    Session,
)

from config.settings import DATABASE_URL

Base = declarative_base()


class Wallet(Base):
    """Wallet registry table"""
    __tablename__ = "wallets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    address = Column(String(42), unique=True, nullable=False, index=True)
    first_seen_at = Column(DateTime, default=datetime.utcnow)
    last_scanned_at = Column(DateTime)
    is_elite_flag = Column(Boolean, default=False)
    classification = Column(String(20), default="unknown")  # bot, quant, human, unknown
    quant_score = Column(Float)
    total_volume = Column(Float, default=0.0)
    total_trades = Column(Integer, default=0)
    total_pnl = Column(Float, default=0.0)
    active_positions = Column(Integer, default=0)
    unique_markets = Column(Integer, default=0)

    # Relationships
    metrics = relationship("WalletMetric", back_populates="wallet", cascade="all, delete-orphan")
    rankings = relationship("MetricRanking", back_populates="wallet", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Wallet(address={self.address[:10]}..., classification={self.classification})>"


class Metric(Base):
    """Metric definitions table (117 rows)"""
    __tablename__ = "metrics"

    id = Column(Integer, primary_key=True)  # 1-117
    group_id = Column(String(1), nullable=False, index=True)  # 'A'-'O'
    code = Column(String(50), unique=True, nullable=False, index=True)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    higher_is_better = Column(Boolean, default=True)
    formula = Column(Text)

    # Relationships
    wallet_metrics = relationship("WalletMetric", back_populates="metric")
    rankings = relationship("MetricRanking", back_populates="metric")

    def __repr__(self):
        return f"<Metric(id={self.id}, code={self.code}, group={self.group_id})>"


# Time period constants
TIME_PERIOD_ALL = "all_time"
TIME_PERIOD_7D = "7d"
TIME_PERIOD_31D = "31d"
TIME_PERIODS = [TIME_PERIOD_ALL, TIME_PERIOD_7D, TIME_PERIOD_31D]


class WalletMetric(Base):
    """Current metric values per wallet per time period"""
    __tablename__ = "wallet_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    wallet_id = Column(Integer, ForeignKey("wallets.id", ondelete="CASCADE"), nullable=False)
    metric_id = Column(Integer, ForeignKey("metrics.id", ondelete="CASCADE"), nullable=False)
    time_period = Column(String(10), nullable=False, default=TIME_PERIOD_ALL)  # all_time, 7d, 31d
    value = Column(Float)
    percentile = Column(Float)  # 0-100
    calculated_at = Column(DateTime, default=datetime.utcnow)

    # Unique constraint per wallet, metric, AND time period
    __table_args__ = (
        UniqueConstraint("wallet_id", "metric_id", "time_period", name="uq_wallet_metric_period"),
        Index("idx_wallet_metrics_wallet", "wallet_id"),
        Index("idx_wallet_metrics_metric", "metric_id"),
        Index("idx_wallet_metrics_period", "time_period"),
    )

    # Relationships
    wallet = relationship("Wallet", back_populates="metrics")
    metric = relationship("Metric", back_populates="wallet_metrics")

    def __repr__(self):
        return f"<WalletMetric(wallet_id={self.wallet_id}, metric_id={self.metric_id}, period={self.time_period}, value={self.value})>"


class MetricRanking(Base):
    """Top 100 rankings per metric per time period (35,100 slots total: 117 × 3 × 100)"""
    __tablename__ = "metric_rankings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    metric_id = Column(Integer, ForeignKey("metrics.id", ondelete="CASCADE"), nullable=False)
    wallet_id = Column(Integer, ForeignKey("wallets.id", ondelete="CASCADE"), nullable=False)
    time_period = Column(String(10), nullable=False, default=TIME_PERIOD_ALL)  # all_time, 7d, 31d
    rank = Column(Integer, nullable=False)
    value = Column(Float, nullable=False)
    percentile = Column(Float)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Constraints - now include time_period in uniqueness
    __table_args__ = (
        CheckConstraint("rank >= 1 AND rank <= 100", name="chk_rank_range"),
        UniqueConstraint("metric_id", "time_period", "rank", name="uq_metric_period_rank"),
        UniqueConstraint("metric_id", "time_period", "wallet_id", name="uq_metric_period_wallet"),
        Index("idx_rankings_metric_period", "metric_id", "time_period", "rank"),
        Index("idx_rankings_wallet", "wallet_id"),
        Index("idx_rankings_period", "time_period"),
    )

    # Relationships
    metric = relationship("Metric", back_populates="rankings")
    wallet = relationship("Wallet", back_populates="rankings")

    def __repr__(self):
        return f"<MetricRanking(metric_id={self.metric_id}, period={self.time_period}, rank={self.rank}, value={self.value})>"


class ScanHistory(Base):
    """Scan history for tracking progress"""
    __tablename__ = "scan_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    scan_type = Column(String(20))  # 'full', 'incremental', 'market'
    wallets_discovered = Column(Integer, default=0)
    wallets_scanned = Column(Integer, default=0)
    wallets_qualified = Column(Integer, default=0)
    markets_scanned = Column(Integer, default=0)
    status = Column(String(20), default="running")  # running, completed, failed
    error_message = Column(Text)

    def __repr__(self):
        return f"<ScanHistory(id={self.id}, status={self.status}, wallets_scanned={self.wallets_scanned})>"


class ScannerState(Base):
    """Scanner state for checkpoint persistence (continuous monitoring)"""
    __tablename__ = "scanner_state"

    key = Column(String(50), primary_key=True)
    value = Column(Text)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<ScannerState(key={self.key}, value={self.value[:20] if self.value else None}...)>"


class CapturedTrade(Base):
    """Captured trades from realtime WebSocket + REST backup system"""
    __tablename__ = "captured_trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    transaction_hash = Column(String(66), unique=True, nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    wallet_address = Column(String(42), nullable=False, index=True)
    market_id = Column(String(100))
    condition_id = Column(String(100))
    side = Column(String(10))  # BUY or SELL
    outcome = Column(String(100))
    price = Column(Float)
    size = Column(Float)
    value = Column(Float)
    fee = Column(Float, default=0.0)
    captured_at = Column(DateTime, default=datetime.utcnow)
    capture_source = Column(String(20))  # 'websocket', 'rest_backup', 'gap_fill'

    __table_args__ = (
        Index("idx_captured_trades_wallet", "wallet_address"),
        Index("idx_captured_trades_timestamp", "timestamp"),
        Index("idx_captured_trades_market", "market_id"),
    )

    def __repr__(self):
        return f"<CapturedTrade(hash={self.transaction_hash[:10]}..., wallet={self.wallet_address[:10]}...)>"


# Engine and session management
_engine = None
_SessionLocal = None


def get_engine():
    """Get or create database engine"""
    global _engine
    if _engine is None:
        _engine = create_engine(DATABASE_URL, echo=False)
    return _engine


def get_session() -> Session:
    """Get a new database session"""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine())
    return _SessionLocal()


def init_db():
    """Initialize database and create all tables"""
    engine = get_engine()
    Base.metadata.create_all(engine)


def drop_db():
    """Drop all tables (use with caution!)"""
    engine = get_engine()
    Base.metadata.drop_all(engine)


def reset_db():
    """Drop and recreate all tables"""
    drop_db()
    init_db()
