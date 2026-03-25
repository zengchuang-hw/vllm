# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to vLLM project
"""
Block Representation Manager for sparse KV offloading

This module manages block representations for efficient similarity computation:
1. Generates compact representations from KV blocks
2. Updates representations when blocks are modified
3. Provides efficient query-block similarity computation
"""

from typing import List, Optional, Tuple
import numpy as np
import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


class BlockReprManager:
    """
    Manages block representations for sparse KV selection.
    
    Features:
    - Generates mean representations from K vectors
    - Supports incremental updates
    - Efficient similarity computation
    """

    def __init__(
        self,
        num_layers: int,
        num_blocks: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int = 16,
        device: str = "cuda",
    ):
        """
        Initialize BlockReprManager.
        
        Args:
            num_layers: Number of transformer layers
            num_blocks: Maximum number of blocks per layer
            num_kv_heads: Number of KV attention heads
            head_dim: Dimension of each attention head
            block_size: Block size in tokens
            device: Device for storing representations
        """
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.device = device
        
        # Block representations: [num_layers, num_blocks, num_kv_heads, head_dim]
        # Stored as float32 for accurate similarity computation
        self.block_reprs: List[Optional[torch.Tensor]] = [None] * num_layers
        
        # Track which blocks have valid representations
        self.valid_blocks: List[set] = [set() for _ in range(num_layers)]
        
        # Statistics
        self.total_updates = 0
        self.total_computations = 0
        
        logger.info(
            f"Initialized BlockReprManager with "
            f"num_layers={num_layers}, num_blocks={num_blocks}, "
            f"num_kv_heads={num_kv_heads}, head_dim={head_dim}"
        )

    def generate_repr(
        self,
        kv_cache: torch.Tensor,
        layer_idx: int,
        block_ids: List[int],
    ) -> torch.Tensor:
        """
        Generate block representations from KV cache.
        
        Args:
            kv_cache: KV cache tensor [2, num_blocks, block_size, num_kv_heads, head_dim]
            layer_idx: Layer index
            block_ids: List of block IDs to generate representations for
            
        Returns:
            Block representations tensor [len(block_ids), num_kv_heads, head_dim]
        """
        if not block_ids:
            return torch.empty(0, self.num_kv_heads, self.head_dim, device=self.device)
        
        key_cache = kv_cache[0]  # [num_blocks, block_size, num_kv_heads, head_dim]
        
        # Extract blocks
        blocks = key_cache[block_ids]  # [len(block_ids), block_size, num_kv_heads, head_dim]
        
        # Compute mean representation across tokens
        # blocks: [len(block_ids), block_size, num_kv_heads, head_dim]
        # Mean across block_size dimension (dim=1)
        reprs = blocks.float().mean(dim=1)  # [len(block_ids), num_kv_heads, head_dim]
        
        # Store to device
        reprs = reprs.to(device=self.device)
        
        self.total_computations += len(block_ids)
        
        return reprs

    def update(
        self,
        layer_idx: int,
        block_ids: List[int],
        reprs: torch.Tensor,
    ):
        """
        Update block representations.
        
        Args:
            layer_idx: Layer index
            block_ids: List of block IDs
            reprs: Block representations [len(block_ids), num_kv_heads, head_dim]
        """
        if not block_ids:
            return
        
        # Initialize layer representations if needed
        if self.block_reprs[layer_idx] is None:
            self.block_reprs[layer_idx] = torch.zeros(
                self.num_blocks,
                self.num_kv_heads,
                self.head_dim,
                dtype=torch.float32,
                device=self.device,
            )
        
        # Update representations
        layer_reprs = self.block_reprs[layer_idx]
        
        for block_id, repr in zip(block_ids, reprs):
            if 0 <= block_id < self.num_blocks:
                layer_reprs[block_id] = repr
                self.valid_blocks[layer_idx].add(block_id)
        
        self.total_updates += len(block_ids)

    def get_repr(
        self,
        layer_idx: int,
        block_id: int,
    ) -> Optional[torch.Tensor]:
        """
        Get representation for a specific block.
        
        Args:
            layer_idx: Layer index
            block_id: Block ID
            
        Returns:
            Block representation tensor [num_kv_heads, head_dim] or None
        """
        if (self.block_reprs[layer_idx] is None or
            block_id not in self.valid_blocks[layer_idx]):
            return None
        
        return self.block_reprs[layer_idx][block_id]

    def compute_similarity(
        self,
        query: torch.Tensor,
        layer_idx: int,
        block_ids: List[int],
    ) -> List[float]:
        """
        Compute similarity scores between query and blocks.
        
        Args:
            query: Query vector [num_kv_heads, head_dim]
            layer_idx: Layer index
            block_ids: List of block IDs
            
        Returns:
            List of similarity scores
        """
        if self.block_reprs[layer_idx] is None:
            # No representations available
            return [0.0] * len(block_ids)
        
        layer_reprs = self.block_reprs[layer_idx]
        scores = []
        
        for block_id in block_ids:
            if block_id in self.valid_blocks[layer_idx]:
                # Compute dot product similarity
                block_repr = layer_reprs[block_id]
                score = float(torch.sum(query * block_repr))
                scores.append(score)
            else:
                # Block representation not available
                scores.append(0.0)
        
        return scores

    def get_stats(self) -> dict:
        """
        Get statistics.
        
        Returns:
            Dictionary with statistics
        """
        valid_counts = [len(s) for s in self.valid_blocks]
        
        return {
            "total_updates": self.total_updates,
            "total_computations": self.total_computations,
            "valid_blocks_per_layer": valid_counts,
            "total_valid_blocks": sum(valid_counts),
            "avg_valid_blocks": sum(valid_counts) / len(valid_counts) if valid_counts else 0,
        }

    def reset_stats(self):
        """Reset statistics counters."""
        self.total_updates = 0
        self.total_computations = 0