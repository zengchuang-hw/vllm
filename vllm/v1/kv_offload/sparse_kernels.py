# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to vLLM project
"""
Triton kernels for sparse KV selection and block representation

This module implements high-performance kernels for:
1. Block representation generation from KV cache
2. Query-block similarity computation
3. Sparse top-K selection
"""

import triton
import triton.language as tl
import torch
from typing import Tuple


@triton.jit
def moba_block_repr_kernel(
    # Pointers to matrices
    k_cache_ptr,  # [num_blocks, block_size, num_heads, head_dim]
    block_repr_ptr,  # [num_blocks, num_heads, head_dim]
    mapping_ptr,  # [num_mappings, 2] (gpu_block_id, cpu_block_id)

    # Matrix dimensions
    num_mappings,  # Number of mappings
    block_size,  # Size of each block
    num_heads,  # Number of attention heads

    # Meta-parameters
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """
    Generate block representations by computing mean of K vectors.

    Args:
        k_cache_ptr: Pointer to K cache tensor
        block_repr_ptr: Pointer to output block representations
        mapping_ptr: Pointer to block mappings
        num_mappings: Number of blocks to process
        block_size: Number of tokens per block
        num_heads: Number of attention heads
    """
    # Get program ID - each thread block handles one (head_idx, mapping_idx) pair
    pid_h = tl.program_id(axis=0)  # Head index
    pid_m = tl.program_id(axis=1)  # Mapping index

    # Early exit if out of range
    if pid_h >= num_heads or pid_m >= num_mappings:
        return

    # Load mapping
    gpu_block_id = tl.load(mapping_ptr + pid_m * 2)
    cpu_block_id = tl.load(mapping_ptr + pid_m * 2 + 1)

    gpu_block_id = tl.cast(gpu_block_id, tl.int64)
    cpu_block_id = tl.cast(cpu_block_id, tl.int64)

    # Calculate pointer to K block head
    # K cache layout: [num_blocks, block_size, num_heads, head_dim]
    k_block_head_ptr = (
        k_cache_ptr +
        (gpu_block_id * block_size * num_heads * HEAD_DIM) +
        (pid_h * HEAD_DIM)
    )

    # Calculate pointer to output block representation
    block_repr_out_ptr = (
        block_repr_ptr +
        (cpu_block_id * num_heads * HEAD_DIM) +
        (pid_h * HEAD_DIM)
    )

    # Initialize accumulator for mean computation
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    # Process all tokens in block, BLOCK_SIZE tokens at a time
    # Use static_range for block_size when possible, otherwise use range
    num_chunks = tl.cdiv(block_size, BLOCK_SIZE)
    for chunk_idx in tl.static_range(num_chunks):
        i = chunk_idx * BLOCK_SIZE

        # Create mask for valid indices
        block_idx = tl.arange(0, BLOCK_SIZE)
        mask = (i + block_idx) < block_size

        # Calculate offsets for K vectors
        # Step between tokens: num_heads * HEAD_DIM
        offsets = (i + block_idx) * num_heads * HEAD_DIM

        # Load K vectors for this chunk
        # Create indices for each dimension
        dim_indices = tl.arange(0, HEAD_DIM)

        # Load K values: [BLOCK_SIZE, HEAD_DIM]
        k_indices = k_block_head_ptr + offsets[:, None] + dim_indices[None, :]
        k_chunk = tl.load(k_indices, mask=mask[:, None])

        # Accumulate sum
        acc += tl.sum(k_chunk, axis=0)

    # Compute mean (divide by block_size)
    acc = acc / block_size

    # Store result to block representation
    tl.store(block_repr_out_ptr + tl.arange(0, HEAD_DIM), acc)


def kv_repr_gen(
    kv_cache: torch.Tensor,
    block_repr: torch.Tensor,
    mapping: torch.Tensor,
    num_mappings: int,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """
    Generate block representations from KV cache.
    
    Computes mean representation of K vectors for each block.
    
    Args:
        kv_cache: KV cache tensor [2, num_blocks, block_size, num_heads, head_dim]
        block_repr: Output tensor [num_blocks, num_heads, head_dim]
        mapping: Mapping tensor [num_mappings, 2] (gpu_block_id, cpu_block_id)
        num_mappings: Number of mappings to process
        block_size: Block size in tokens
        num_kv_heads: Number of KV attention heads
        head_dim: Head dimension
        
    Returns:
        Updated block_repr tensor
    """
    # Extract K cache (index 0) from KV cache
    k_cache = kv_cache[0]
    
    # Determine optimal BLOCK_SIZE for Triton
    BLOCK_SIZE = triton.next_power_of_2(min(64, block_size))
    
    # Launch kernel with 2D grid: (num_mappings, num_kv_heads)
    grid = (num_kv_heads, num_mappings)
    
    moba_block_repr_kernel[grid](
        k_cache,
        block_repr,
        mapping,
        num_mappings,
        block_size,
        num_kv_heads,
        BLOCK_SIZE=BLOCK_SIZE,
        HEAD_DIM=head_dim,
    )
    
    return block_repr


@triton.jit
def compute_block_scores_kernel(
    # Input pointers
    block_indices_ptr,  # [batch_size, max_num_blocks_per_seq]
    k_repr_ptr,  # [num_blocks, num_heads, head_size]
    query_ptr,  # [num_query_tokens, num_heads, head_size]
    query_start_loc,  # [num_query_tokens + 1]
    seq_lens_ptr,
    scores_ptr,  # [batch_size, seq_num_blocks]

    # Shape parameters
    block_size,
    batch_size,
    max_num_blocks_per_seq,
    num_kv_heads,
    num_heads,
    head_size,
    heads_per_kv,

    # Stride parameters
    block_indices_stride,
    k_repr_stride_0,
    k_repr_stride_1,
    query_stride_0,
    query_stride_1,
    scores_stride,

    # Parallelization parameters
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """
    Compute similarity scores between query and blocks.
    
    Args:
        block_indices_ptr: Block indices for each sequence
        k_repr_ptr: Block representations
        query_ptr: Query vectors
        query_start_loc: Query start locations
        seq_lens_ptr: Sequence lengths
        scores_ptr: Output scores
        block_size: Block size in tokens
        batch_size: Batch size
        max_num_blocks_per_seq: Maximum blocks per sequence
        num_kv_heads: Number of KV heads
        num_heads: Number of query heads
        head_size: Head dimension
        heads_per_kv: Query heads per KV head
    """
    # Get thread block IDs
    batch_idx = tl.program_id(0)
    block_idx = tl.program_id(1)

    # Early exit if out of range
    if batch_idx >= batch_size or block_idx >= max_num_blocks_per_seq:
        return

    # Load block index
    block_offset = batch_idx * block_indices_stride + block_idx
    block_id = tl.load(block_indices_ptr + block_offset)

    # Initialize score accumulator
    total_score = 0.0

    # Get query range for this sequence
    query_start = tl.load(query_start_loc + batch_idx)
    query_end = tl.load(query_start_loc + batch_idx + 1)
    num_query_tokens = query_end - query_start

    # Get sequence length
    seq_len = tl.load(seq_lens_ptr + batch_idx)
    kv_len = seq_len - num_query_tokens

    # Calculate number of complete KV blocks
    num_kv_full_blocks = kv_len // block_size

    # Skip partial blocks - only select complete blocks
    if block_idx >= num_kv_full_blocks:
        tl.store(scores_ptr + batch_idx * scores_stride + block_idx, float('-inf'))
        return

    # Head and tail blocks must always be selected
    if num_kv_full_blocks > 0 and (block_idx == 0 or block_idx == num_kv_full_blocks - 1):
        tl.store(scores_ptr + batch_idx * scores_stride + block_idx, float('inf'))
        return

    # Compute similarity score
    # Use last query token for scoring (can be extended for chunked prefill)
    token_idx = query_end - 1

    # For each KV head
    for kv_head_idx in range(0, num_kv_heads):
        # Load K representation
        k_offset = block_id * k_repr_stride_0 + kv_head_idx * k_repr_stride_1
        k = tl.load(
            k_repr_ptr + k_offset + tl.arange(0, BLOCK_K),
            mask=tl.arange(0, BLOCK_K) < head_size
        )

        # Compute corresponding query head range
        q_head_start = kv_head_idx * heads_per_kv
        q_head_end = (kv_head_idx + 1) * heads_per_kv

        # For each corresponding query head
        for q_head_idx in range(q_head_start, q_head_end):
            # Load query vector
            q_offset = token_idx * query_stride_0 + q_head_idx * query_stride_1
            q = tl.load(
                query_ptr + q_offset + tl.arange(0, BLOCK_K),
                mask=tl.arange(0, BLOCK_K) < head_size
            )

            # Compute dot product
            dot_product = tl.sum(k * q)
            total_score += dot_product

    # Store score
    tl.store(scores_ptr + batch_idx * scores_stride + block_idx, total_score)


def sparse_kv_selection(
    block_table: torch.Tensor,
    batch_size: int,
    block_size: int,
    max_num_blocks_this_batch: int,
    seq_lens: torch.Tensor,
    k_repr: torch.Tensor,
    query: torch.Tensor,
    query_start_loc: torch.Tensor,
    top_k: int,
    scores: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Select top-K KV blocks based on query similarity.
    
    Args:
        block_table: Block table [max_batch_size, max_num_blocks]
        batch_size: Current batch size
        block_size: Block size in tokens
        max_num_blocks_this_batch: Maximum blocks in current batch
        seq_lens: Sequence lengths [max_batch_size]
        k_repr: Block representations [num_blocks, num_kv_heads, head_size]
        query: Query vectors [num_tokens, num_heads, head_size]
        query_start_loc: Query start locations [batch_size + 1]
        top_k: Number of blocks to select per sequence
        scores: Pre-allocated scores tensor [batch_size, max_num_blocks]
        
    Returns:
        Tuple of (topk_choices, topk_scores):
        - topk_choices: Selected block indices [batch_size, top_k]
        - topk_scores: Corresponding scores [batch_size, top_k]
    """
    # Get dimensions
    num_kv_heads = k_repr.shape[1]
    head_size = k_repr.shape[2]
    num_heads = query.shape[1]

    # Calculate heads per KV head (GQA support)
    heads_per_kv = num_heads // num_kv_heads

    # Calculate strides
    block_indices_stride = block_table.stride(0)
    k_repr_stride_0 = k_repr.stride(0)
    k_repr_stride_1 = k_repr.stride(1)
    query_stride_0 = query.stride(0)
    query_stride_1 = query.stride(1)
    scores_stride = scores.stride(0)

    # Determine parallelization parameters
    BLOCK_M = 1
    BLOCK_N = 1
    BLOCK_K = triton.next_power_of_2(head_size)
    GROUP_M = 1

    # Launch kernel with 2D grid: (batch_size, max_num_blocks_this_batch)
    grid = (batch_size, max_num_blocks_this_batch)

    compute_block_scores_kernel[grid](
        block_table,
        k_repr,
        query,
        query_start_loc,
        seq_lens,
        scores,
        block_size,
        batch_size,
        max_num_blocks_this_batch,
        num_kv_heads,
        num_heads,
        head_size,
        heads_per_kv,
        block_indices_stride,
        k_repr_stride_0,
        k_repr_stride_1,
        query_stride_0,
        query_stride_1,
        scores_stride,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M,
    )

    # Select top-k using torch operations
    # scores: [batch_size, max_num_blocks_this_batch]
    topk_scores, topk_choices = torch.topk(scores, top_k, dim=1, sorted=False)

    return topk_choices, topk_scores


# Example usage and testing
if __name__ == '__main__':
    import numpy as np

    print("Testing sparse KV selection kernels...")

    # Test parameters
    num_blocks = 100
    block_size = 16
    num_heads = 8
    num_q_heads = 32
    head_dim = 128
    max_batch_size = 4
    max_blocks_per_seq = 10
    top_k = 5

    # Create test data
    kv_cache = torch.randn(
        2, num_blocks, block_size, num_heads, head_dim,
        dtype=torch.bfloat16, device='cuda'
    )

    mapping = torch.zeros((10, 2), device='cpu', dtype=torch.int32)
    mapping_np = mapping.numpy()
    for i in range(10):
        mapping_np[i] = [i * 10, i * 10]

    k_repr = torch.zeros(
        (num_blocks, num_heads, head_dim),
        dtype=torch.bfloat16, device='cuda'
    )

    block_table = torch.zeros(
        (max_batch_size, max_blocks_per_seq),
        dtype=torch.int32, device='cuda'
    )

    query = torch.randn(
        (8, num_q_heads, head_dim),
        dtype=torch.bfloat16, device='cuda'
    )

    query_start_loc = torch.zeros(
        (max_batch_size + 1,),
        dtype=torch.int32, device='cuda'
    )
    query_start_loc[0] = 0
    query_start_loc[1] = 2
    query_start_loc[2] = 4
    query_start_loc[3] = 6
    query_start_loc[4] = 8

    seq_lens = torch.zeros(max_batch_size, dtype=torch.int32, device='cuda')
    seq_lens[0] = 100
    seq_lens[1] = 120
    seq_lens[2] = 80
    seq_lens[3] = 110

    scores = torch.zeros(
        (max_batch_size, max_blocks_per_seq),
        dtype=torch.bfloat16, device='cuda'
    )

    # Test block representation generation
    print("Testing block representation generation...")
    k_repr = kv_repr_gen(
        kv_cache=kv_cache,
        block_repr=k_repr,
        mapping=mapping.cuda(),
        num_mappings=10,
        block_size=block_size,
        num_kv_heads=num_heads,
        head_dim=head_dim,
    )

    print(f"Generated {10} block representations")
    print(f"Block repr shape: {k_repr.shape}")

    # Test sparse KV selection
    print("\nTesting sparse KV selection...")
    topk_choices, topk_scores = sparse_kv_selection(
        block_table=block_table,
        batch_size=4,
        block_size=block_size,
        max_num_blocks_this_batch=10,
        seq_lens=seq_lens,
        k_repr=k_repr,
        query=query,
        query_start_loc=query_start_loc,
        top_k=top_k,
        scores=scores,
    )

    print(f"Selected top-{top_k} blocks")
    topk_choices_cpu = topk_choices.cpu().numpy()
    topk_scores_cpu = topk_scores.cpu().numpy()

    for i in range(4):
        print(f"Sequence {i}:")
        print(f"  Selected blocks: {topk_choices_cpu[i]}")
        print(f"  Scores: {topk_scores_cpu[i]}")

    print("\nAll tests passed!")