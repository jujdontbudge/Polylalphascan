"""Repository for wallet data access"""
from datetime import datetime
from typing import List, Optional, Dict, Any
from sqlalchemy import select, update, func
from sqlalchemy.orm import Session

from src.database.models import Wallet, WalletMetric, get_session


class WalletRepository:
    """Data access layer for wallets"""

    def __init__(self, session: Optional[Session] = None):
        self.session = session or get_session()
        self._owns_session = session is None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._owns_session:
            self.session.close()

    def get_by_id(self, wallet_id: int) -> Optional[Wallet]:
        """Get wallet by ID"""
        return self.session.get(Wallet, wallet_id)

    def get_by_address(self, address: str) -> Optional[Wallet]:
        """Get wallet by address"""
        stmt = select(Wallet).where(Wallet.address == address.lower())
        return self.session.execute(stmt).scalar_one_or_none()

    def get_or_create(self, address: str) -> Wallet:
        """Get existing wallet or create new one"""
        address = address.lower()
        wallet = self.get_by_address(address)
        if not wallet:
            wallet = Wallet(
                address=address,
                first_seen_at=datetime.utcnow(),
            )
            self.session.add(wallet)
            self.session.commit()
        return wallet

    def create(self, address: str, **kwargs) -> Wallet:
        """Create a new wallet"""
        wallet = Wallet(
            address=address.lower(),
            first_seen_at=datetime.utcnow(),
            **kwargs,
        )
        self.session.add(wallet)
        self.session.commit()
        return wallet

    def bulk_create(self, addresses: List[str]) -> List[Wallet]:
        """Bulk create wallets, skipping existing ones"""
        addresses = [a.lower() for a in addresses]

        # Get existing addresses
        stmt = select(Wallet.address).where(Wallet.address.in_(addresses))
        existing = {row[0] for row in self.session.execute(stmt).all()}

        # Create new wallets
        new_wallets = []
        for address in addresses:
            if address not in existing:
                wallet = Wallet(
                    address=address,
                    first_seen_at=datetime.utcnow(),
                )
                self.session.add(wallet)
                new_wallets.append(wallet)

        if new_wallets:
            self.session.commit()

        return new_wallets

    def update(self, wallet_id: int, **kwargs) -> Optional[Wallet]:
        """Update wallet attributes"""
        wallet = self.get_by_id(wallet_id)
        if wallet:
            for key, value in kwargs.items():
                if hasattr(wallet, key):
                    setattr(wallet, key, value)
            self.session.commit()
        return wallet

    def mark_scanned(
        self,
        wallet_id: int,
        total_volume: float = None,
        total_trades: int = None,
        total_pnl: float = None,
    ) -> Optional[Wallet]:
        """Mark wallet as scanned with stats"""
        updates = {"last_scanned_at": datetime.utcnow()}
        if total_volume is not None:
            updates["total_volume"] = total_volume
        if total_trades is not None:
            updates["total_trades"] = total_trades
        if total_pnl is not None:
            updates["total_pnl"] = total_pnl
        return self.update(wallet_id, **updates)

    def set_classification(
        self,
        wallet_id: int,
        classification: str,
        quant_score: float = None,
    ) -> Optional[Wallet]:
        """Set wallet classification"""
        updates = {"classification": classification}
        if quant_score is not None:
            updates["quant_score"] = quant_score
        return self.update(wallet_id, **updates)

    def get_all(self, limit: int = None, offset: int = 0) -> List[Wallet]:
        """Get all wallets with pagination"""
        stmt = select(Wallet).offset(offset)
        if limit:
            stmt = stmt.limit(limit)
        return list(self.session.execute(stmt).scalars().all())

    def get_unscanned(self, limit: int = 100) -> List[Wallet]:
        """Get wallets that haven't been deep scanned"""
        stmt = (
            select(Wallet)
            .where(Wallet.last_scanned_at.is_(None))
            .limit(limit)
        )
        return list(self.session.execute(stmt).scalars().all())

    def get_by_classification(
        self,
        classification: str,
        limit: int = None,
    ) -> List[Wallet]:
        """Get wallets by classification"""
        stmt = select(Wallet).where(Wallet.classification == classification)
        if limit:
            stmt = stmt.limit(limit)
        return list(self.session.execute(stmt).scalars().all())

    def get_elite_wallets(self, limit: int = 100) -> List[Wallet]:
        """Get elite flagged wallets"""
        stmt = (
            select(Wallet)
            .where(Wallet.is_elite_flag == True)
            .order_by(Wallet.quant_score.desc())
            .limit(limit)
        )
        return list(self.session.execute(stmt).scalars().all())

    def get_top_by_volume(self, limit: int = 100) -> List[Wallet]:
        """Get top wallets by volume"""
        stmt = (
            select(Wallet)
            .order_by(Wallet.total_volume.desc())
            .limit(limit)
        )
        return list(self.session.execute(stmt).scalars().all())

    def get_top_by_pnl(self, limit: int = 100) -> List[Wallet]:
        """Get top wallets by PnL"""
        stmt = (
            select(Wallet)
            .order_by(Wallet.total_pnl.desc())
            .limit(limit)
        )
        return list(self.session.execute(stmt).scalars().all())

    def count(self) -> int:
        """Count total wallets"""
        stmt = select(func.count(Wallet.id))
        return self.session.execute(stmt).scalar() or 0

    def count_by_classification(self) -> Dict[str, int]:
        """Count wallets by classification"""
        stmt = (
            select(Wallet.classification, func.count(Wallet.id))
            .group_by(Wallet.classification)
        )
        results = self.session.execute(stmt).all()
        return {cls: count for cls, count in results}

    def delete(self, wallet_id: int) -> bool:
        """Delete a wallet"""
        wallet = self.get_by_id(wallet_id)
        if wallet:
            self.session.delete(wallet)
            self.session.commit()
            return True
        return False

    def exists(self, address: str) -> bool:
        """Check if wallet exists"""
        stmt = select(func.count(Wallet.id)).where(
            Wallet.address == address.lower()
        )
        return (self.session.execute(stmt).scalar() or 0) > 0

    def get_addresses(self) -> List[str]:
        """Get all wallet addresses"""
        stmt = select(Wallet.address)
        return [row[0] for row in self.session.execute(stmt).all()]
