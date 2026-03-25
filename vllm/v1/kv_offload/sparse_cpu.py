# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to vLLM project
"""
Sparse CPU Offloading Spec for hotness-aware KV cache management

This module extends CPUOffloadingSpec with sparse optimization capabilities:
1. SparseOffloadingManager for intelligent block selection
2. OptimizedSwapHandler for efficient transfers
3. BlockReprManager for block representation management
"""

from collections.abc import Iterator
from typing import TYPE_CHECKING

import torch

from vllm.config import VllmConfig
from vllm.platforms import current_platform
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.abstract import LoadStoreSpec, OffloadingManager
from vllm.v1.kv_offload.backends.cpu import CPUBackend
from vllm.v1.kv_offload.lru_manager import LRUOffloadingManager
from vllm.v1.kv_offload.mediums import CPULoadStoreSpec, GPULoadStoreSpec
from vllm.v1.kv_offload.reuse_manager import FilterReusedOffloadingManager
from vllm.v1.kv_offload.spec import OffloadingSpec
from vllm.v1.kv_offload.arc_manager import ARCOffloadingManager
from vllm.v1.kv_offload.sparse_manager import SparseOffloadingManager
from vllm.v1.kv_offload.worker.optimized_swap import OptimizedCpuGpuHandlers
from vllm.v1.kv_offload.worker.worker import OffloadingHandler
from vllm.logger import init_logger

if TYPE_CHECKING:
    pass

logger = init_logger(__name__)


class SparseCPUOffloadingSpec(OffloadingSpec):
    """
    Sparse-optimized CPU offloading specification.
    
    Extends CPUOffloadingSpec with:
    - SparseOffloadingManager for intelligent block selection
    - OptimizedSwapHandler for efficient transfers
    - BlockReprManager for block representation management
    """

    def __init__(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig):
        super().__init__(vllm_config, kv_cache_config)
        
        cpu_bytes_to_use = self.extra_config.get("cpu_bytes_to_use")
        if not cpu_bytes_to_use:
            raise Exception(
                "cpu_bytes_to_use must be specified in kv_connector_extra_config"
            )
        
        # Calculate block sizes
        assert kv_cache_config is not None
        page_sizes = {
            kv_cache_group.kv_cache_spec.page_size_bytes
            for kv_cache_group in kv_cache_config.kv_cache_groups
        }
        assert len(page_sizes) == 1
        page_size_bytes = page_sizes.pop()
        
        kv_bytes_per_block = (
            page_size_bytes
            * len(kv_cache_config.kv_cache_tensors)
            * vllm_config.parallel_config.world_size
        )
        
        kv_bytes_per_offloaded_block = kv_bytes_per_block * self.block_size_factor
        self.num_blocks = (
            int(cpu_bytes_to_use) // kv_bytes_per_offloaded_block
            if kv_bytes_per_offloaded_block > 0
            else 0
        )
        
        # Sparse configuration
        self.sparse_topk = int(
            self.extra_config.get("sparse_topk", 10240000)
        )
        self.copy_method = self.extra_config.get("copy_method", "merged")
        self.cache_policy = self.extra_config.get("cache_policy", "lru-layerwise")
        self.enable_merge = self.copy_method in ["merged", "gather-scatter"]
        
        # Get model configuration for block representation
        self.num_layers = self._get_num_layers(vllm_config)
        self.num_kv_heads = self._get_num_kv_heads(vllm_config)
        self.head_dim = self._get_head_dim(vllm_config)
        
        # scheduler-side
        self._manager: OffloadingManager | None = None
        
        # worker-side
        self._handlers: OptimizedCpuGpuHandlers | None = None
        
        # Block representation manager (worker-side)
        self._block_repr_manager = None
        
        # Eviction policy
        self.eviction_policy: str = self.extra_config.get("eviction_policy", "lru")
        
        logger.info(
            f"Initialized SparseCPUOffloadingSpec with "
            f"sparse_topk={self.sparse_topk}, copy_method={self.copy_method}, "
            f"cache_policy={self.cache_policy}, enable_merge={self.enable_merge}"
        )

    def _get_num_layers(self, vllm_config: VllmConfig) -> int:
        """Get number of transformer layers from config."""
        # Try to get from model config
        model_config = vllm_config.model_config
        if hasattr(model_config, "hf_config"):
            hf_config = model_config.hf_config
            if hasattr(hf_config, "num_hidden_layers"):
                return hf_config.num_hidden_layers
            elif hasattr(hf_config, "num_layers"):
                return hf_config.num_layers
        # Default fallback
        return 32

    def _get_num_kv_heads(self, vllm_config: VllmConfig) -> int:
        """Get number of KV attention heads from config."""
        model_config = vllm_config.model_config
        if hasattr(model_config, "hf_config"):
            hf_config = model_config.hf_config
            if hasattr(hf_config, "num_key_value_heads"):
                return hf_config.num_key_value_heads
            elif hasattr(hf_config, "num_attention_heads"):
                return hf_config.num_attention_heads
        return 8

    def _get_head_dim(self, vllm_config: VllmConfig) -> int:
        """Get attention head dimension from config."""
        model_config = vllm_config.model_config
        if hasattr(model_config, "hf_config"):
            hf_config = model_config.hf_config
            if hasattr(hf_config, "hidden_size"):
                num_heads = self._get_num_kv_heads(vllm_config)
                return hf_config.hidden_size // num_heads
        return 128

    def get_manager(self) -> OffloadingManager:
        """
        Get SparseOffloadingManager for scheduler-side block management.
        
        Returns:
            SparseOffloadingManager instance
        """
        if not self._manager:
            kv_events_config = self.vllm_config.kv_events_config
            enable_events = (
                kv_events_config is not None and kv_events_config.enable_kv_cache_events
            )
            
            assert len(self.gpu_block_size) == 1
            gpu_block_size = self.gpu_block_size[0]
            offloaded_block_size = gpu_block_size * self.block_size_factor
            
            # Create CPU backend
            backend = CPUBackend(
                block_size=offloaded_block_size, num_blocks=self.num_blocks
            )
            
            # Create base offloading manager
            if self.eviction_policy == "lru":
                base_manager = LRUOffloadingManager(
                    backend=backend, enable_events=enable_events
                )
            elif self.eviction_policy == "arc":
                base_manager = ARCOffloadingManager(
                    backend=backend, enable_events=enable_events
                )
            else:
                raise ValueError(
                    f"Unknown eviction policy: {self.eviction_policy}. "
                    f"Supported policies: lru, arc"
                )
            
            # Apply reuse filter if configured
            store_threshold = int(self.extra_config.get("store_threshold", 0))
            if store_threshold >= 2:
                max_tracker_size = int(
                    self.extra_config.get("max_tracker_size", 64_000)
                )
                base_manager = FilterReusedOffloadingManager(
                    backing=base_manager,
                    store_threshold=store_threshold,
                    max_tracker_size=max_tracker_size,
                )
            
            # Wrap with sparse manager
            self._manager = SparseOffloadingManager(
                backing_manager=base_manager,
                sparse_topk=self.sparse_topk,
                block_size=gpu_block_size,
                num_layers=self.num_layers,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                cache_policy=self.cache_policy,
            )
        
        return self._manager

    def get_handlers(
        self,
        kv_caches: dict[str, torch.Tensor],
        attn_backends: dict[str, type[AttentionBackend]],
    ) -> Iterator[tuple[type[LoadStoreSpec], type[LoadStoreSpec], OffloadingHandler]]:
        """
        Get optimized offloading handlers.
        
        Args:
            kv_caches: Dictionary of layer_name -> GPU KV cache tensor
            attn_backends: Dictionary of layer_name -> AttentionBackend
            
        Yields:
            Tuples of (src_type, dst_type, offloading_handler)
        """
        if not self._handlers:
            if not current_platform.is_cuda_alike():
                raise Exception(
                    "Sparse CPU Offloading is currently only supported on CUDA-alike GPUs"
                )
            
            assert len(self.gpu_block_size) == 1
            gpu_block_size = self.gpu_block_size[0]
            
            # Create optimized handlers
            self._handlers = OptimizedCpuGpuHandlers(
                attn_backends=attn_backends,
                gpu_block_size=gpu_block_size,
                cpu_block_size=gpu_block_size * self.block_size_factor,
                num_cpu_blocks=self.num_blocks,
                gpu_caches=kv_caches,
                enable_merge=self.enable_merge,
            )
            
            logger.info(
                f"Created OptimizedCpuGpuHandlers with "
                f"enable_merge={self.enable_merge}"
            )
        
        assert self._handlers is not None
        
        # Yield handlers for both directions
        yield GPULoadStoreSpec, CPULoadStoreSpec, self._handlers.gpu_to_cpu_handler
        yield CPULoadStoreSpec, GPULoadStoreSpec, self._handlers.cpu_to_gpu_handler

    def get_block_repr_manager(self):
        """
        Get block representation manager for worker-side.
        
        Returns:
            BlockReprManager instance or None
        """
        if self._block_repr_manager is None:
            from vllm.v1.kv_offload.block_repr_manager import BlockReprManager
            
            self._block_repr_manager = BlockReprManager(
                num_layers=self.num_layers,
                num_blocks=self.num_blocks,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                block_size=self.gpu_block_size[0],
                device="cuda",
            )
        
        return self._block_repr_manager