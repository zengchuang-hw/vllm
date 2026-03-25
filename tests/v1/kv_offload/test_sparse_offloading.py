# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to vLLM project
"""
Tests for sparse KV offloading components
"""

import pytest
import numpy as np
import torch

from vllm.v1.kv_offload.sparse_manager import SparseOffloadingManager
from vllm.v1.kv_offload.block_repr_manager import BlockReprManager
from vllm.v1.kv_offload.worker.optimized_swap import (
    OptimizedSwapHandler,
    merge_contiguous_blocks,
)


class TestSparseOffloadingManager:
    """Test SparseOffloadingManager functionality."""
    
    def test_init(self):
        """Test initialization."""
        from vllm.v1.kv_offload.lru_manager import LRUOffloadingManager
        from vllm.v1.kv_offload.backends.cpu import CPUBackend
        
        backend = CPUBackend(block_size=16, num_blocks=100)
        backing_manager = LRUOffloadingManager(backend=backend)
        
        manager = SparseOffloadingManager(
            backing_manager=backing_manager,
            sparse_topk=1024,
            block_size=16,
            num_layers=32,
            num_kv_heads=8,
            head_dim=128,
        )
        
        assert manager.sparse_topk == 1024
        assert manager.num_layers == 32
        assert manager.num_kv_heads == 8
        assert manager.head_dim == 128
    
    def test_lookup_with_sparse(self):
        """Test sparse-aware lookup."""
        from vllm.v1.kv_offload.lru_manager import LRUOffloadingManager
        from vllm.v1.kv_offload.backends.cpu import CPUBackend
        
        backend = CPUBackend(block_size=16, num_blocks=100)
        backing_manager = LRUOffloadingManager(backend=backend)
        
        manager = SparseOffloadingManager(
            backing_manager=backing_manager,
            sparse_topk=32,
            block_size=16,
            num_layers=4,
            num_kv_heads=8,
            head_dim=128,
        )
        
        # Test without query - no blocks stored, so no hits
        block_hashes = [1, 2, 3, 4, 5]
        selected_blocks, scores = manager.lookup_with_sparse(
            block_hashes, query=None, layer_idx=0
        )
        
        # Since no blocks are stored, lookup returns 0 hits
        assert len(selected_blocks) == 0
        assert len(scores) == 0
    
    def test_update_block_repr(self):
        """Test block representation update."""
        manager = SparseOffloadingManager(
            backing_manager=None,  # Dummy for testing
            sparse_topk=1024,
            block_size=16,
            num_layers=4,
            num_kv_heads=8,
            head_dim=128,
        )
        
        # Create a dummy block representation
        block_repr = np.random.randn(8, 128).astype(np.float32)
        
        # Update representation
        manager.update_block_repr(layer_idx=0, block_hash=123, block_repr=block_repr)
        
        # Verify representation is stored
        assert manager.block_reprs[0] is not None
        assert len(manager.block_reprs[0]) > 0


class TestBlockReprManager:
    """Test BlockReprManager functionality."""
    
    def test_init(self):
        """Test initialization."""
        manager = BlockReprManager(
            num_layers=4,
            num_blocks=100,
            num_kv_heads=8,
            head_dim=128,
            block_size=16,
            device="cpu",
        )
        
        assert manager.num_layers == 4
        assert manager.num_blocks == 100
        assert manager.num_kv_heads == 8
        assert manager.head_dim == 128
    
    def test_generate_repr(self):
        """Test block representation generation."""
        manager = BlockReprManager(
            num_layers=4,
            num_blocks=100,
            num_kv_heads=8,
            head_dim=128,
            block_size=16,
            device="cpu",
        )
        
        # Create dummy KV cache
        kv_cache = torch.randn(
            2, 100, 16, 8, 128, dtype=torch.float16, device="cpu"
        )
        
        # Generate representations for blocks
        block_ids = [0, 1, 2]
        reprs = manager.generate_repr(kv_cache, layer_idx=0, block_ids=block_ids)
        
        assert reprs.shape == (len(block_ids), 8, 128)
        assert reprs.dtype == torch.float32
    
    def test_update_and_get_repr(self):
        """Test update and get block representations."""
        manager = BlockReprManager(
            num_layers=4,
            num_blocks=100,
            num_kv_heads=8,
            head_dim=128,
            block_size=16,
            device="cpu",
        )
        
        # Create dummy representation
        block_repr = torch.randn(8, 128, dtype=torch.float32, device="cpu")
        
        # Update representation
        manager.update(layer_idx=0, block_ids=[5], reprs=block_repr.unsqueeze(0))
        
        # Get representation
        retrieved_repr = manager.get_repr(layer_idx=0, block_id=5)
        
        assert retrieved_repr is not None
        assert torch.allclose(retrieved_repr, block_repr)
    



class TestMergeContiguousBlocks:
    """Test contiguous block merging."""
    
    def test_merge_simple(self):
        """Test simple contiguous merging."""
        block_mapping = np.array([
            [0, 10],
            [1, 11],
            [2, 12],
            [5, 15],
            [6, 16],
        ], dtype=np.int64)
        
        merged_mappings, total_merged = merge_contiguous_blocks(block_mapping)
        
        # Should merge into 2 chunks: [0,1,2] and [5,6]
        assert total_merged == 2
        assert len(merged_mappings) == 2
        
        # Check first chunk
        assert merged_mappings[0] == (0, 10, 3)
        # Check second chunk
        assert merged_mappings[1] == (5, 15, 2)
    
    def test_merge_all_contiguous(self):
        """Test merging when all blocks are contiguous."""
        block_mapping = np.array([
            [0, 10],
            [1, 11],
            [2, 12],
            [3, 13],
        ], dtype=np.int64)
        
        merged_mappings, total_merged = merge_contiguous_blocks(block_mapping)
        
        # Should merge into 1 chunk
        assert total_merged == 1
        assert len(merged_mappings) == 1
        assert merged_mappings[0] == (0, 10, 4)
    
    def test_merge_none_contiguous(self):
        """Test merging when no blocks are contiguous."""
        block_mapping = np.array([
            [0, 10],
            [5, 15],
            [10, 20],
        ], dtype=np.int64)
        
        merged_mappings, total_merged = merge_contiguous_blocks(block_mapping)
        
        # Should not merge
        assert total_merged == 3
        assert len(merged_mappings) == 3
    
    def test_merge_empty(self):
        """Test merging empty array."""
        block_mapping = np.array([], dtype=np.int64).reshape(0, 2)
        
        merged_mappings, total_merged = merge_contiguous_blocks(block_mapping)
        
        assert total_merged == 0
        assert len(merged_mappings) == 0


class TestOptimizedSwapHandler:
    """Test OptimizedSwapHandler functionality."""
    
    def test_init(self):
        """Test initialization."""
        src_tensors = [torch.randn(10, 16, 8, 128, dtype=torch.float16, device="cpu")]
        dst_tensors = [torch.randn(10, 16, 8, 128, dtype=torch.float16, device="cpu")]
        
        handler = OptimizedSwapHandler(
            src_tensors=src_tensors,
            dst_tensors=dst_tensors,
            src_block_size_factor=1,
            dst_block_size_factor=1,
            enable_merge=True,
        )
        
        assert handler.enable_merge is True
        assert handler.gpu_to_cpu is False
    
    def test_stats(self):
        """Test statistics tracking."""
        src_tensors = [torch.randn(10, 16, 8, 128, dtype=torch.float16, device="cpu")]
        dst_tensors = [torch.randn(10, 16, 8, 128, dtype=torch.float16, device="cpu")]
        
        handler = OptimizedSwapHandler(
            src_tensors=src_tensors,
            dst_tensors=dst_tensors,
            src_block_size_factor=1,
            dst_block_size_factor=1,
            enable_merge=True,
        )
        
        # Get initial stats
        stats = handler.get_stats()
        assert stats["total_transfers"] == 0
        assert stats["total_merged_transfers"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])