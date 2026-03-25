# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to vLLM project
"""
Sparse KV Offloading Module

This module provides sparse-optimized KV offloading capabilities:
- SparseOffloadingManager: Intelligent block selection based on query similarity
- OptimizedSwapHandler: Efficient transfer with contiguous block merging
- BlockReprManager: Block representation management for similarity computation
- SparseCPUOffloadingSpec: Complete sparse offloading specification

Phase 3 Advanced Algorithms:
- Advanced cache policies (LRU+Hot, LayerWise LRU, Hot Score)
- Adaptive top-K selection
- Multi-strategy fusion
- Triton kernels for sparse selection
"""

from vllm.v1.kv_offload.sparse_manager import SparseOffloadingManager
from vllm.v1.kv_offload.worker.optimized_swap import (
    OptimizedSwapHandler,
    OptimizedCpuGpuHandlers,
)
from vllm.v1.kv_offload.block_repr_manager import BlockReprManager
from vllm.v1.kv_offload.sparse_cpu import SparseCPUOffloadingSpec
from vllm.v1.kv_offload.cache_policies import (
    LRUWithHotCache,
    LayerWiseLRUCache,
    HotScoreCache,
)
from vllm.v1.kv_offload.adaptive_selection import (
    AdaptiveTopKSelector,
    MultiStrategySelector,
)

__all__ = [
    # Core components
    "SparseOffloadingManager",
    "OptimizedSwapHandler",
    "OptimizedCpuGpuHandlers",
    "BlockReprManager",
    "SparseCPUOffloadingSpec",
    # Phase 3 advanced algorithms
    "LRUWithHotCache",
    "LayerWiseLRUCache",
    "HotScoreCache",
    "AdaptiveTopKSelector",
    "MultiStrategySelector",
]