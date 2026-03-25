# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to vLLM project
"""
Sparse Offloading Manager for hotness-aware KV cache management

This module implements a sparse-optimized offloading manager that:
1. Maintains block representations for intelligent block scoring
2. Implements query-aware sparse selection to identify hot blocks
3. Supports advanced cache policies (LRU+Hot, LayerWise LRU)
4. Provides adaptive top-K selection
"""

from collections.abc import Iterable
from typing import Optional, Tuple, List, Dict
import numpy as np

from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.abstract import (
    LoadStoreSpec,
    OffloadingEvent,
    OffloadingManager,
    PrepareStoreOutput,
)
from vllm.v1.kv_offload.lru_manager import LRUOffloadingManager
from vllm.logger import init_logger

logger = init_logger(__name__)


class SparseOffloadingManager(OffloadingManager):
    """
    Sparse-optimized OffloadingManager that extends standard OffloadingManager
    with intelligent block selection based on query similarity.
    
    Key features:
    - Maintains block representations for similarity computation
    - Query-aware sparse selection (top-k hot blocks)
    - Compatible with standard KV offload interfaces
    - Supports advanced cache policies
    - Adaptive top-K selection
    """

    def __init__(
        self,
        backing_manager: OffloadingManager,
        sparse_topk: int = 1024,
        block_size: int = 16,
        num_layers: int = 32,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        cache_policy: str = "lru-layerwise",
        enable_adaptive: bool = False,
    ):
        """
        Initialize SparseOffloadingManager.
        
        Args:
            backing_manager: Standard OffloadingManager for block lifecycle management
            sparse_topk: Maximum number of KV tokens to select per layer
            block_size: Block size in tokens
            num_layers: Number of transformer layers
            num_kv_heads: Number of KV attention heads
            head_dim: Dimension of each attention head
            cache_policy: Cache replacement policy
            enable_adaptive: Enable adaptive top-K selection
        """
        self.backing_manager = backing_manager
        self.sparse_topk = sparse_topk
        self.block_size = block_size
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.cache_policy = cache_policy
        self.enable_adaptive = enable_adaptive
        
        # Block representations storage
        # Using dictionary for sparse storage
        self.block_reprs: Dict[int, Dict[BlockHash, np.ndarray]] = {
            layer_idx: {} for layer_idx in range(num_layers)
        }
        
        # Statistics
        self.total_lookups = 0
        self.total_hits = 0
        self.total_selections = 0
        
        # Adaptive top-K selector
        self.adaptive_selector = None
        if enable_adaptive:
            try:
                from vllm.v1.kv_offload.adaptive_selection import AdaptiveTopKSelector
                self.adaptive_selector = AdaptiveTopKSelector(
                    base_topk=sparse_topk,
                    strategy="hit_rate",
                )
                logger.info("Adaptive top-K selection enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize adaptive selector: {e}")
        
        logger.info(
            f"Initialized SparseOffloadingManager with "
            f"sparse_topk={sparse_topk}, block_size={block_size}, "
            f"num_layers={num_layers}, cache_policy={cache_policy}"
        )

    def lookup_with_sparse(
        self,
        block_hashes: Iterable[BlockHash],
        query: Optional[np.ndarray] = None,
        layer_idx: int = 0,
    ) -> Tuple[List[BlockHash], List[float]]:
        """
        Sparse-aware lookup that selects hot blocks based on query similarity.
        
        Args:
            block_hashes: Hashes identifying blocks to lookup
            query: Query vector for similarity scoring (optional)
            layer_idx: Current layer index
            
        Returns:
            Tuple of (selected_blocks, scores):
            - selected_blocks: List of selected block hashes
            - scores: List of corresponding similarity scores
        """
        self.total_lookups += 1
        block_hashes_list = list(block_hashes)
        
        if not block_hashes_list:
            return [], []
        
        # Standard lookup to find offloaded blocks
        num_offloaded = self.backing_manager.lookup(block_hashes_list)
        
        if num_offloaded is None:
            # Cannot perform lookup now
            return [], []
        
        # Get offloaded blocks
        offloaded_blocks = block_hashes_list[:num_offloaded]
        
        if not offloaded_blocks:
            return [], []
        
        # If query is provided and block representations exist, perform sparse selection
        if query is not None and len(self.block_reprs[layer_idx]) > 0:
            selected_blocks, scores = self._select_hot_blocks(
                offloaded_blocks, query, layer_idx
            )
            self.total_selections += 1
            return selected_blocks, scores
        
        # Fallback: return all offloaded blocks
        self.total_hits += len(offloaded_blocks)
        return offloaded_blocks, [1.0] * len(offloaded_blocks)

    def _select_hot_blocks(
        self,
        block_hashes: List[BlockHash],
        query: np.ndarray,
        layer_idx: int,
    ) -> Tuple[List[BlockHash], List[float]]:
        """
        Select top-k hot blocks based on query similarity.
        
        Args:
            block_hashes: List of candidate block hashes
            query: Query vector [num_heads, head_dim]
            layer_idx: Current layer index
            
        Returns:
            Tuple of (selected_blocks, scores)
        """
        if not block_hashes:
            return [], []
        
        # Simple similarity computation
        layer_reprs = self.block_reprs[layer_idx]
        if not layer_reprs:
            # No representations available, return all blocks
            return block_hashes, [1.0] * len(block_hashes)
        
        # Compute similarity scores
        scores = []
        valid_blocks = []
        
        for block_hash in block_hashes:
            if block_hash in layer_reprs:
                block_repr_vec = layer_reprs[block_hash]
                # Compute dot product similarity
                score = float(np.sum(query * block_repr_vec))
                scores.append(score)
                valid_blocks.append(block_hash)
            else:
                # Block representation not available
                scores.append(0.0)
                valid_blocks.append(block_hash)
        
        # Select top-k blocks
        num_select = min(len(valid_blocks), self.sparse_topk // self.block_size)
        
        if num_select >= len(valid_blocks):
            # Select all blocks
            sorted_indices = list(range(len(valid_blocks)))
        else:
            # Get indices of top-k scores
            sorted_indices = np.argsort(scores)[-num_select:].tolist()
            sorted_indices.reverse()  # Descending order
        
        selected_blocks = [valid_blocks[i] for i in sorted_indices]
        selected_scores = [scores[i] for i in sorted_indices]
        
        return selected_blocks, selected_scores

    def update_block_repr(
        self,
        layer_idx: int,
        block_hash: BlockHash,
        block_repr: np.ndarray,
    ):
        """
        Update block representation for a specific block.
        
        Args:
            layer_idx: Layer index
            block_hash: Hash of block
            block_repr: Block representation [num_kv_heads, head_dim]
        """
        if layer_idx not in self.block_reprs:
            self.block_reprs[layer_idx] = {}
        
        # Store representation (copy to avoid external modification)
        self.block_reprs[layer_idx][block_hash] = block_repr.copy()

    def get_adaptive_topk(
        self,
        seq_len: int,
        cache_hit_rate: Optional[float] = None,
    ) -> int:
        """
        Get adaptive top-K value.
        
        Args:
            seq_len: Current sequence length
            cache_hit_rate: Current cache hit rate (0-1)
            
        Returns:
            Adaptive top-K value
        """
        if self.enable_adaptive and self.adaptive_selector is not None:
            return self.adaptive_selector.get_topk(
                seq_len=seq_len,
                cache_hit_rate=cache_hit_rate,
            )
        else:
            return self.sparse_topk

    def prepare_sparse_load(
        self,
        selected_blocks: List[BlockHash],
    ) -> LoadStoreSpec:
        """
        Prepare load for only selected hot blocks.
        
        Args:
            selected_blocks: List of selected block hashes
            
        Returns:
            LoadStoreSpec for loading selected blocks
        """
        if not selected_blocks:
            # Return empty spec
            return self.backing_manager.prepare_load([])
        
        return self.backing_manager.prepare_load(selected_blocks)

    def get_stats(self) -> Dict:
        """
        Get selection statistics.
        
        Returns:
            Dictionary with statistics
        """
        overall_hit_rate = (
            self.total_hits / self.total_lookups
            if self.total_lookups > 0
            else 0.0
        )
        
        stats = {
            "total_lookups": self.total_lookups,
            "total_hits": self.total_hits,
            "total_selections": self.total_selections,
            "overall_hit_rate": overall_hit_rate,
            "cache_policy": self.cache_policy,
            "sparse_topk": self.sparse_topk,
        }
        
        if self.adaptive_selector:
            stats["adaptive_stats"] = self.adaptive_selector.get_stats()
        
        return stats

    # Delegate standard OffloadingManager methods to backing manager
    def lookup(self, block_hashes: Iterable[BlockHash]) -> int | None:
        return self.backing_manager.lookup(block_hashes)

    def prepare_load(self, block_hashes: Iterable[BlockHash]) -> LoadStoreSpec:
        return self.backing_manager.prepare_load(block_hashes)

    def touch(self, block_hashes: Iterable[BlockHash]):
        self.backing_manager.touch(block_hashes)

    def complete_load(self, block_hashes: Iterable[BlockHash]):
        self.backing_manager.complete_load(block_hashes)

    def prepare_store(
        self, block_hashes: Iterable[BlockHash]
    ) -> PrepareStoreOutput | None:
        return self.backing_manager.prepare_store(block_hashes)

    def complete_store(self, block_hashes: Iterable[BlockHash], success: bool = True):
        self.backing_manager.complete_store(block_hashes, success)

    def take_events(self) -> Iterable[OffloadingEvent]:
        return self.backing_manager.take_events()