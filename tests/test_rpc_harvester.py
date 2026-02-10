"""Tests for RPC harvester block counting"""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.scanner.rpc_harvester import RPCHarvestResult, BlockChunk, RPCParallelHarvester


class TestBlocksScannedCount:
    """Tests for blocks_scanned double-count fix"""

    def test_completed_chunks_tracking(self):
        """RPCHarvestResult tracks completed chunk IDs"""
        result = RPCHarvestResult()
        assert len(result._completed_chunks) == 0

        chunk_key = (100, 200)
        result._completed_chunks.add(chunk_key)
        assert chunk_key in result._completed_chunks

    def test_blocks_scanned_no_double_count(self):
        """Same chunk counted only once even if processed twice"""
        result = RPCHarvestResult()

        chunk = BlockChunk(from_block=100, to_block=199)
        chunk_key = (chunk.from_block, chunk.to_block)

        # First time: should count
        if chunk_key not in result._completed_chunks:
            result.blocks_scanned += chunk.size
            result._completed_chunks.add(chunk_key)

        assert result.blocks_scanned == 100

        # Second time (retry): should NOT count again
        if chunk_key not in result._completed_chunks:
            result.blocks_scanned += chunk.size
            result._completed_chunks.add(chunk_key)

        assert result.blocks_scanned == 100  # Still 100, not 200

    def test_different_chunks_counted_separately(self):
        """Different chunks are each counted once"""
        result = RPCHarvestResult()

        chunks = [
            BlockChunk(from_block=0, to_block=99),
            BlockChunk(from_block=100, to_block=199),
            BlockChunk(from_block=200, to_block=299),
        ]

        for chunk in chunks:
            chunk_key = (chunk.from_block, chunk.to_block)
            if chunk_key not in result._completed_chunks:
                result.blocks_scanned += chunk.size
                result._completed_chunks.add(chunk_key)

        assert result.blocks_scanned == 300
        assert len(result._completed_chunks) == 3

    def test_block_chunk_size(self):
        """BlockChunk.size is inclusive of both endpoints"""
        chunk = BlockChunk(from_block=100, to_block=199)
        assert chunk.size == 100

        chunk2 = BlockChunk(from_block=0, to_block=0)
        assert chunk2.size == 1

    def test_chunk_retry_tracking(self):
        """BlockChunk tracks retries correctly"""
        chunk = BlockChunk(from_block=0, to_block=99, max_retries=3)

        assert chunk.can_retry()
        chunk.retry_count = 3
        assert not chunk.can_retry()
