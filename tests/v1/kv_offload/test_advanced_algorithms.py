# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to vLLM project
"""
Tests for Phase 3 advanced algorithms in sparse KV offloading

Tests for:
1. Advanced cache policies (LRU+Hot, LayerWise LRU, Hot Score)
2. Adaptive top-K selection
3. Multi-strategy fusion
4. Triton kernels for sparse selection
"""

import pytest
import numpy as np
import torch

from vllm.v1.kv_offload.cache_policies import (
    LRUWithHotCache,
    LayerWiseLRUCache,
    HotScoreCache,
)
from vllm.v1.kv_offload.adaptive_selection import (
    AdaptiveTopKSelector,
    MultiStrategySelector,
)


class TestLRUWithHotCache:
    """Test LRU with hotness score cache policy."""
    
    def test_init(self):
        """Test initialization."""
        cache = LRUWithHotCache(
            capacity=100,
            decay_factor=0.44,
            window_size=110,
        )
        
        assert cache.capacity == 100
        assert cache.decay_factor == 0.44
        assert cache.window_size == 110
    
    def test_cache_hit(self):
        """Test cache hit with hotness update."""
        cache = LRUWithHotCache(capacity=100)
        
        # Insert blocks
        for i in range(50):
            slot_id, hit = cache.get(i, hot_score=1.0)
            assert not hit
        
        # Cache hit should update hotness
        slot_id, hit = cache.get(25, hot_score=2.0)
        assert hit
        assert slot_id >= 0
    
    def test_cache_miss(self):
        """Test cache miss and eviction."""
        cache = LRUWithHotCache(capacity=10)
        
        # Fill cache
        for i in range(10):
            slot_id, hit = cache.get(i, hot_score=1.0)
            assert not hit
        
        # Cache miss should trigger eviction
        slot_id, hit = cache.get(15, hot_score=1.0)
        assert not hit
        assert slot_id >= 0
    
    def test_pin_unpin(self):
        """Test pin and unpin functionality."""
        cache = LRUWithHotCache(capacity=10)
        
        # Insert blocks
        for i in range(5):
            cache.get(i, hot_score=1.0)
        
        # Pin a block
        assert cache.pin_block(3) is True
        
        # Try to evict (should skip pinned block)
        for i in range(10, 20):
            cache.get(i, hot_score=1.0)
        
        # Unpin the block
        assert cache.unpin_block(3) is True


class TestLayerWiseLRUCache:
    """Test layer-wise LRU cache policy."""
    
    def test_init(self):
        """Test initialization."""
        cache = LayerWiseLRUCache(
            capacity=100,
            num_layers=4,
            num_cpu_blocks=50,
        )
        
        assert cache.capacity == 100
        assert cache.num_layers == 4
        assert cache.num_cpu_blocks == 50
    
    def test_layer_isolation(self):
        """Test layer isolation in eviction."""
        cache = LayerWiseLRUCache(
            capacity=20,
            num_layers=4,
            num_cpu_blocks=50,
        )
        
        # Fill layer 0
        for i in range(10):
            cache.get((0, i), hot_score=1.0)
        
        # Fill layer 1
        for i in range(10):
            cache.get((1, i), hot_score=1.0)
        
        # Evicting from layer 0 should not affect layer 1
        for i in range(10, 20):
            cache.get((0, i), hot_score=1.0)
        
        # Layer 1 blocks should still be in cache
        slot_id, hit = cache.get((1, 5), hot_score=1.0)
        assert hit


class TestHotScoreCache:
    """Test hotness score cache policy."""
    
    def test_init(self):
        """Test initialization."""
        cache = HotScoreCache(capacity=100, decay_factor=0.2)
        
        assert cache.capacity == 100
        assert cache.decay_factor == 0.2
    
    def test_hotness_update(self):
        """Test hotness score update and eviction."""
        cache = HotScoreCache(capacity=10)
        
        # Insert blocks with different hotness scores
        hotness_scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        for i, score in enumerate(hotness_scores):
            cache.get(i, hot_score=score)
        
        # Low hotness block should be evicted first
        slot_id, hit = cache.get(10, hot_score=10.0)
        assert not hit
        
        # High hotness block should still be in cache
        slot_id, hit = cache.get(4, hot_score=6.0)
        assert hit


class TestAdaptiveTopKSelector:
    """Test adaptive top-K selector."""
    
    def test_init(self):
        """Test initialization."""
        selector = AdaptiveTopKSelector(
            base_topk=10240000,
            min_topk=4096,
            max_topk=2048000,
            strategy="hit_rate",
        )
        
        assert selector.base_topk == 10240000
        assert selector.strategy == "hit_rate"
    
    def test_hit_rate_based(self):
        """Test hit rate based top-K adjustment."""
        selector = AdaptiveTopKSelector(
            base_topk=10240000,
            max_topk=20480000,
            strategy="hit_rate",
        )
        
        # Low hit rate -> increase top-K
        topk = selector.get_topk(
            seq_len=16384,
            cache_hit_rate=0.3,
        )
        assert topk > selector.base_topk
        
        # High hit rate -> decrease top-K
        topk = selector.get_topk(
            seq_len=16384,
            cache_hit_rate=0.8,
        )
        assert topk < selector.base_topk
    
    def test_sequence_length_based(self):
        """Test sequence length based top-K adjustment."""
        selector = AdaptiveTopKSelector(
            base_topk=10240000,
            strategy="sequence_length",
        )
        
        # Short sequence -> lower top-K
        topk = selector.get_topk(seq_len=4096)
        assert topk < selector.base_topk
        
        # Long sequence -> higher top-K
        topk = selector.get_topk(seq_len=65536)
        assert topk > selector.base_topk
    
    def test_hybrid_strategy(self):
        """Test hybrid strategy combining multiple approaches."""
        selector = AdaptiveTopKSelector(
            base_topk=10240000,
            strategy="hybrid",
        )
        
        topk = selector.get_topk(
            seq_len=16384,
            cache_hit_rate=0.6,
        )
        # Should be between hit_rate and sequence_length based values
        assert selector.min_topk <= topk <= selector.max_topk


class TestMultiStrategySelector:
    """Test multi-strategy fusion selector."""
    
    def test_init(self):
        """Test initialization."""
        selector = MultiStrategySelector(
            strategies=["lru", "hot_score", "similarity"],
            weights=[0.4, 0.3, 0.3],
            consensus="weighted",
        )
        
        assert len(selector.strategies) == 3
        assert selector.consensus == "weighted"
    
    def test_weighted_consensus(self):
        """Test weighted consensus selection."""
        selector = MultiStrategySelector(
            strategies=["lru", "hot_score", "similarity"],
            weights=[0.5, 0.3, 0.2],
            consensus="weighted",
        )
        
        candidate_blocks = list(range(100))
        scores_list = [
            np.random.rand(100).tolist() for _ in range(3)
        ]
        
        selected_blocks, scores = selector.select_blocks(
            candidate_blocks=candidate_blocks,
            scores_list=scores_list,
            top_k=10,
        )
        
        assert len(selected_blocks) == 10
        assert len(scores) == 10
    
    def test_voting_consensus(self):
        """Test voting consensus selection."""
        selector = MultiStrategySelector(
            strategies=["lru", "hot_score", "similarity"],
            consensus="voting",
        )
        
        candidate_blocks = list(range(50))
        scores_list = [
            np.random.rand(50).tolist() for _ in range(3)
        ]
        
        selected_blocks, scores = selector.select_blocks(
            candidate_blocks=candidate_blocks,
            scores_list=scores_list,
            top_k=5,
        )
        
        assert len(selected_blocks) == 5
        assert len(scores) == 5
    
    def test_union_selection(self):
        """Test union selection."""
        selector = MultiStrategySelector(
            strategies=["lru", "hot_score"],
            consensus="union",
        )
        
        candidate_blocks = list(range(30))
        scores_list = [
            np.random.rand(30).tolist() for _ in range(2)
        ]
        
        selected_blocks, scores = selector.select_blocks(
            candidate_blocks=candidate_blocks,
            scores_list=scores_list,
            top_k=8,
        )
        
        # Union may select more than top_k
        assert len(selected_blocks) >= 8


class TestTritonKernels:
    """Test Triton kernels for sparse KV selection."""
    
    def test_block_repr_gen(self):
        """Test block representation generation kernel."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
            return

        try:
            from vllm.v1.kv_offload.sparse_kernels import kv_repr_gen
        except ImportError:
            pytest.skip("Triton not available")
            return

        # Create test data
        num_blocks = 10
        block_size = 16
        num_heads = 8
        head_dim = 128

        kv_cache = torch.randn(
            2, num_blocks, block_size, num_heads, head_dim,
            dtype=torch.bfloat16, device='cuda'
        )

        block_repr = torch.zeros(
            (num_blocks, num_heads, head_dim),
            dtype=torch.bfloat16, device='cuda'
        )

        mapping = torch.zeros((num_blocks, 2), device='cuda', dtype=torch.int32)
        for i in range(num_blocks):
            mapping[i] = [i, i]

        # Generate block representations
        result = kv_repr_gen(
            kv_cache=kv_cache,
            block_repr=block_repr,
            mapping=mapping,
            num_mappings=num_blocks,
            block_size=block_size,
            num_kv_heads=num_heads,
            head_dim=head_dim,
        )
        
        # Verify output
        assert result.shape == block_repr.shape
        assert result.device == block_repr.device
    
    def test_sparse_kv_selection(self):
        """Test sparse KV selection kernel."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
            return

        try:
            from vllm.v1.kv_offload.sparse_kernels import sparse_kv_selection
        except ImportError:
            pytest.skip("Triton not available")
            return

        # Create test data
        batch_size = 4
        max_blocks_per_seq = 10
        block_size = 16
        num_heads = 8
        num_q_heads = 32
        head_dim = 128
        top_k = 5
        
        block_table = torch.zeros(
            (batch_size, max_blocks_per_seq),
            dtype=torch.int32, device='cuda'
        )
        
        seq_lens = torch.zeros(batch_size, dtype=torch.int32, device='cuda')
        seq_lens[0] = 100
        seq_lens[1] = 120
        seq_lens[2] = 80
        seq_lens[3] = 110
        
        k_repr = torch.randn(
            100, num_heads, head_dim,
            dtype=torch.bfloat16, device='cuda'
        )
        
        query = torch.randn(
            8, num_q_heads, head_dim,
            dtype=torch.bfloat16, device='cuda'
        )
        
        query_start_loc = torch.zeros(
            (batch_size + 1,),
            dtype=torch.int32, device='cuda'
        )
        query_start_loc[0] = 0
        query_start_loc[1] = 2
        query_start_loc[2] = 4
        query_start_loc[3] = 6
        query_start_loc[4] = 8
        
        scores = torch.zeros(
            (batch_size, max_blocks_per_seq),
            dtype=torch.bfloat16, device='cuda'
        )
        
        # Perform sparse selection
        topk_choices, topk_scores = sparse_kv_selection(
            block_table=block_table,
            batch_size=batch_size,
            block_size=block_size,
            max_num_blocks_this_batch=max_blocks_per_seq,
            seq_lens=seq_lens,
            k_repr=k_repr,
            query=query,
            query_start_loc=query_start_loc,
            top_k=top_k,
            scores=scores,
        )
        
        # Verify output
        assert topk_choices.shape == (batch_size, top_k)
        assert topk_scores.shape == (batch_size, top_k)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])