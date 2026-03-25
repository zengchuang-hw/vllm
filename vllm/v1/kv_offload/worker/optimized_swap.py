# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to vLLM project
"""
Optimized Swap Handler for sparse KV offloading

This module implements optimized transfer mechanisms with:
1. Contiguous block merging to reduce transfer overhead
2. Batch transfer operations for improved efficiency
3. Block representation management and updates
"""

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.v1.kv_offload.mediums import BlockIDsLoadStoreSpec
from vllm.v1.kv_offload.worker.worker import (
    OffloadingHandler,
    TransferResult,
    TransferSpec,
)

logger = init_logger(__name__)


def expand_block_ids(
    block_ids: np.ndarray,
    block_size_factor: int,
    output: np.ndarray,
    skip_count: int = 0,
):
    """
    Convert a list of block IDs to a list of matching block ids,
    assuming each block is composed of actual block_size_factor blocks.
    Outputs to output tensor.
    The first skip_count blocks will be skipped.
    Note that skip_count must be less than block_size_factor.

    For example, if block_ids = [0, 1, 3] and block_size_factor =  4,
    then it yields [0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15]
    since 0 maps to [0, 1, 2, 3]
    1 maps to [4, 5, 6, 7]
    and 3 maps to [12, 13, 14, 15]
    """
    assert skip_count < block_size_factor

    first_range = np.arange(skip_count, block_size_factor)
    full_range = np.arange(0, block_size_factor)

    output_idx = 0
    for i, block_id in enumerate(block_ids):
        base_block_id = block_id * block_size_factor
        indices = first_range if i == 0 else full_range
        output_end_idx = output_idx + len(indices)
        output[output_idx:output_end_idx] = base_block_id + indices
        output_idx = output_end_idx


@dataclass
class Transfer:
    job_id: int
    stream: torch.cuda.Stream
    start_event: torch.Event
    end_event: torch.Event
    num_bytes: int
    merged_blocks: int  # Number of blocks merged into this transfer


def merge_contiguous_blocks(
    block_mapping: np.ndarray,
) -> tuple[list[tuple[int, int, int]], int]:
    """
    Merge contiguous blocks to reduce transfer overhead.
    
    Args:
        block_mapping: Array of shape [N, 2] where each row is [src, dst]
        
    Returns:
        Tuple of (merged_mappings, total_merged):
        - merged_mappings: List of (src_start, dst_start, count) tuples
        - total_merged: Total number of blocks after merging
    """
    if len(block_mapping) == 0:
        return [], 0
    
    # Sort by source block ID
    sorted_indices = np.argsort(block_mapping[:, 0])
    sorted_mapping = block_mapping[sorted_indices]
    
    # Merge contiguous blocks
    merged_mappings = []
    current_src_start = int(sorted_mapping[0, 0])
    current_dst_start = int(sorted_mapping[0, 1])
    current_count = 1
    
    for i in range(1, len(sorted_mapping)):
        src_id = int(sorted_mapping[i, 0])
        dst_id = int(sorted_mapping[i, 1])
        
        # Check if contiguous
        if (src_id == current_src_start + current_count and
            dst_id == current_dst_start + current_count):
            current_count += 1
        else:
            # Save current chunk and start new one
            merged_mappings.append(
                (current_src_start, current_dst_start, current_count)
            )
            current_src_start = src_id
            current_dst_start = dst_id
            current_count = 1
    
    # Add last chunk
    merged_mappings.append((current_src_start, current_dst_start, current_count))
    
    total_merged = len(merged_mappings)
    return merged_mappings, total_merged


class OptimizedSwapHandler(OffloadingHandler):
    """
    Optimized OffloadingHandler with contiguous block merging
    and batch transfer support.
    
    Key optimizations:
    - Merges contiguous blocks to reduce transfer calls
    - Supports batch operations for multiple blocks
    - Efficient stream and event pooling
    """

    def __init__(
        self,
        src_tensors: list[torch.Tensor],
        dst_tensors: list[torch.Tensor],
        src_block_size_factor: int = 1,
        dst_block_size_factor: int = 1,
        enable_merge: bool = True,
    ):
        """
        Initialize OptimizedSwapHandler.
        
        Args:
            src_tensors: List of source KV cache tensors
            dst_tensors: List of destination KV cache tensors
            src_block_size_factor: Blocks per KV block in source
            dst_block_size_factor: Blocks per KV block in destination
            enable_merge: Enable contiguous block merging
        """
        assert len(src_tensors) == len(dst_tensors)
        
        self.src_tensors = src_tensors
        self.dst_tensors = dst_tensors
        min_block_size_factor = min(src_block_size_factor, dst_block_size_factor)
        self.src_block_size_factor = src_block_size_factor // min_block_size_factor
        self.dst_block_size_factor = dst_block_size_factor // min_block_size_factor
        self.enable_merge = enable_merge
        
        # Calculate block size in bytes
        if len(src_tensors) > 0:
            self.block_size_in_bytes = [
                tensor.element_size() * tensor.stride(0) * min_block_size_factor
                for tensor in src_tensors
            ]
            self.total_block_size_in_bytes = sum(self.block_size_in_bytes)
        else:
            self.block_size_in_bytes = []
            self.total_block_size_in_bytes = 0
        
        # Determine transfer direction
        self.gpu_to_cpu = src_tensors[0].is_cuda if src_tensors else False
        self.transfer_type = ("GPU", "CPU") if self.gpu_to_cpu else ("CPU", "GPU")
        
        # Transfer management
        self._transfer_events: dict[int, torch.Event] = {}
        self._transfers: deque[Transfer] = deque()
        
        # Stream and event pooling for efficiency
        self._stream_pool: list[torch.cuda.Stream] = []
        self._event_pool: list[torch.Event] = []
        
        # Statistics
        self.total_transfers = 0
        self.total_merged_transfers = 0

    def transfer_async(self, job_id: int, spec: TransferSpec) -> bool:
        """
        Initiates an optimized asynchronous transfer.
        
        Args:
            job_id: Unique transfer job ID
            spec: (src, dst) transfer specification
            
        Returns:
            True if transfer submitted successfully
        """
        src_spec, dst_spec = spec
        assert isinstance(src_spec, BlockIDsLoadStoreSpec)
        assert isinstance(dst_spec, BlockIDsLoadStoreSpec)
        
        src_blocks = src_spec.block_ids
        dst_blocks = dst_spec.block_ids
        
        if len(src_blocks) == 0:
            return True  # Nothing to transfer
        
        # Expand block IDs to handle block size factors
        src_sub_block_count = src_blocks.size * self.src_block_size_factor
        dst_sub_block_count = dst_blocks.size * self.dst_block_size_factor
        src_sub_blocks_to_skip = -dst_blocks.size % self.src_block_size_factor
        
        assert dst_sub_block_count == src_sub_block_count - src_sub_blocks_to_skip
        
        src_to_dst = np.empty((dst_sub_block_count, 2), dtype=np.int64)
        expand_block_ids(
            src_blocks,
            self.src_block_size_factor,
            src_to_dst[:, 0],
            skip_count=src_sub_blocks_to_skip,
        )
        expand_block_ids(dst_blocks, self.dst_block_size_factor, src_to_dst[:, 1])
        
        # Build block mapping
        block_mapping = src_to_dst
        
        # Merge contiguous blocks
        if self.enable_merge:
            merged_mappings, num_merged = merge_contiguous_blocks(block_mapping)
            self.total_merged_transfers += num_merged
        else:
            # No merging: treat each block as separate
            merged_mappings = [
                (int(block_mapping[i, 0]), int(block_mapping[i, 1]), 1)
                for i in range(len(block_mapping))
            ]
            num_merged = len(merged_mappings)
        
        # Get or create stream and events
        stream = self._stream_pool.pop() if self._stream_pool else torch.cuda.Stream()
        start_event = (
            self._event_pool.pop()
            if self._event_pool
            else torch.Event(enable_timing=True)
        )
        end_event = (
            self._event_pool.pop()
            if self._event_pool
            else torch.Event(enable_timing=True)
        )
        
        # Wait for previous transfers if needed
        if self.gpu_to_cpu:
            # Wait for model computation before GPU->CPU transfer
            stream.wait_stream(torch.cuda.current_stream())
        
        if self._transfers:
            last_transfer = self._transfers[-1]
            stream.wait_event(last_transfer.end_event)
        
        # Execute transfer
        with torch.cuda.stream(stream):
            start_event.record(stream)
            
            for src_tensor, dst_tensor, block_size_bytes in zip(
                self.src_tensors,
                self.dst_tensors,
                self.block_size_in_bytes,
            ):
                self._execute_transfer(
                    src_tensor, dst_tensor, merged_mappings, block_size_bytes
                )
            
            end_event.record(stream)
        
        # Track transfer
        total_bytes = dst_sub_block_count * self.total_block_size_in_bytes
        
        self._transfer_events[job_id] = end_event
        self._transfers.append(
            Transfer(
                job_id=job_id,
                stream=stream,
                start_event=start_event,
                end_event=end_event,
                num_bytes=total_bytes,
                merged_blocks=num_merged,
            )
        )
        
        self.total_transfers += 1
        return True

    def _execute_transfer(
        self,
        src_tensor: torch.Tensor,
        dst_tensor: torch.Tensor,
        merged_mappings: list[tuple[int, int, int]],
        block_size_bytes: int,
    ):
        """
        Execute transfer with merged contiguous blocks.
        
        Args:
            src_tensor: Source tensor
            dst_tensor: Destination tensor
            merged_mappings: List of (src_start, dst_start, count) tuples
            block_size_bytes: Size of each block in bytes
        """
        # Create full mapping from merged chunks
        full_mapping = []
        for src_start, dst_start, count in merged_mappings:
            full_mapping.extend(
                [[src_start + i, dst_start + i] for i in range(count)]
            )
        
        # Execute swap with full mapping
        mapping_tensor = torch.tensor(
            full_mapping,
            dtype=torch.int64,
            device="cpu",
        )
        
        ops.swap_blocks(
            src_tensor,
            dst_tensor,
            block_size_bytes,
            mapping_tensor,
        )

    def get_finished(self) -> list[TransferResult]:
        """
        Get finished transfers since last call.
        
        Returns:
            List of TransferResult with completion status
        """
        results = []
        
        while self._transfers and self._transfers[0].end_event.query():
            transfer = self._transfers.popleft()
            
            # Calculate transfer time
            transfer_time = (
                transfer.start_event.elapsed_time(transfer.end_event) * 1e-3
            )  # Convert to milliseconds
            
            result = TransferResult(
                job_id=transfer.job_id,
                success=True,
                transfer_size=transfer.num_bytes,
                transfer_time=transfer_time,
                transfer_type=self.transfer_type,
            )
            
            results.append(result)
            
            # Return stream and events to pool
            self._stream_pool.append(transfer.stream)
            self._event_pool.append(transfer.end_event)
            self._event_pool.append(transfer.start_event)
            
            del self._transfer_events[transfer.job_id]
        
        return results

    def wait(self, job_ids: set[int]) -> None:
        """
        Wait for specific jobs to finish (blocking).
        
        Args:
            job_ids: Set of job IDs to wait for
        """
        for job_id in job_ids:
            event = self._transfer_events.get(job_id)
            if event is not None:
                event.synchronize()

    def get_stats(self) -> dict:
        """
        Get transfer statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "total_transfers": self.total_transfers,
            "total_merged_transfers": self.total_merged_transfers,
            "avg_merge_ratio": (
                self.total_merged_transfers / self.total_transfers
                if self.total_transfers > 0
                else 0
            ),
            "transfer_type": " -> ".join(self.transfer_type),
        }


class OptimizedCpuGpuHandlers:
    """
    Factory for creating optimized CPU<->GPU offloading handlers.
    """

    def __init__(
        self,
        gpu_block_size: int,
        cpu_block_size: int,
        num_cpu_blocks: int,
        gpu_caches: dict[str, torch.Tensor],
        attn_backends: dict[str, type],
        enable_merge: bool = True,
    ):
        """
        Initialize OptimizedCpuGpuHandlers.
        
        Args:
            gpu_block_size: GPU block size in tokens
            cpu_block_size: CPU block size in tokens
            num_cpu_blocks: Number of CPU blocks
            gpu_caches: Dictionary of layer_name -> GPU KV cache tensor
            attn_backends: Dictionary of layer_name -> AttentionBackend
            enable_merge: Enable contiguous block merging
        """
        assert gpu_caches
        assert cpu_block_size % gpu_block_size == 0
        
        self.enable_merge = enable_merge
        
        # Parse GPU tensors to find kernel block size
        kernel_block_size = None
        parsed_gpu_tensors = []
        
        for layer_name, gpu_tensor in gpu_caches.items():
            gpu_shape = gpu_tensor.shape
            attn_backend = attn_backends[layer_name]
            
            # Determine tensor layout
            test_shape = attn_backend.get_kv_cache_shape(
                num_blocks=1234, block_size=16, num_kv_heads=8, head_size=256
            )
            
            has_layers_dim = False
            split_k_and_v = False
            
            if len(gpu_shape) != len(test_shape):
                # Cross-layers tensor
                has_layers_dim = True
                test_shape = (80,) + test_shape
            elif test_shape[0] != 1234:
                # Split K and V
                split_k_and_v = True

            # Find kernel block size in gpu_shape
            # test_shape contains placeholder values like 1234 and 16
            # We need to find which dimension in gpu_shape corresponds to the block_size dimension (16)
            if has_layers_dim:
                try:
                    kv_cache_stride_order = attn_backend.get_kv_cache_stride_order(
                        include_num_layers_dimension=has_layers_dim
                    )
                    test_shape = tuple(
                        test_shape[i] for i in kv_cache_stride_order
                    )
                except (AttributeError, NotImplementedError):
                    # Fallback: use test_shape as is (contains logical layout)
                    pass

            # Find kernel block size
            block_size_idx = test_shape.index(16)
            if kernel_block_size is not None:
                assert kernel_block_size == gpu_shape[block_size_idx]
            else:
                kernel_block_size = gpu_shape[block_size_idx]
                assert gpu_block_size % kernel_block_size == 0
            
            parsed_gpu_tensors.append((gpu_tensor, split_k_and_v))
        
        assert kernel_block_size is not None
        
        # Calculate block size factors
        cpu_block_size_factor = cpu_block_size // kernel_block_size
        gpu_block_size_factor = gpu_block_size // kernel_block_size
        num_cpu_kernel_blocks = num_cpu_blocks * cpu_block_size_factor
        
        # Allocate CPU tensors
        gpu_tensors = []
        cpu_tensors = []
        
        for gpu_tensor, split_k_and_v in parsed_gpu_tensors:
            cpu_shape = list(gpu_tensor.shape)
            cpu_shape[1 if split_k_and_v else 0] = num_cpu_kernel_blocks
            
            cpu_tensor = torch.zeros(
                cpu_shape,
                dtype=gpu_tensor.dtype,
                device="cpu",
                pin_memory=True,
            )
            
            gpu_tensors.extend(
                gpu_tensor.unbind(0) if split_k_and_v else [gpu_tensor]
            )
            cpu_tensors.extend(
                cpu_tensor.unbind(0) if split_k_and_v else [cpu_tensor]
            )
        
        # Create optimized handlers
        self.gpu_to_cpu_handler = OptimizedSwapHandler(
            src_tensors=gpu_tensors,
            dst_tensors=cpu_tensors,
            src_block_size_factor=gpu_block_size_factor,
            dst_block_size_factor=cpu_block_size_factor,
            enable_merge=enable_merge,
        )
        
        self.cpu_to_gpu_handler = OptimizedSwapHandler(
            src_tensors=cpu_tensors,
            dst_tensors=gpu_tensors,
            src_block_size_factor=cpu_block_size_factor,
            dst_block_size_factor=gpu_block_size_factor,
            enable_merge=enable_merge,
        )
        
        logger.info(
            "Created OptimizedCpuGpuHandlers with "
            "gpu_block_size=%d, cpu_block_size=%d, "
            "num_cpu_blocks=%d, enable_merge=%s",
            gpu_block_size,
            cpu_block_size,
            num_cpu_blocks,
            enable_merge,
        )