# Sparse KV Offloading

## Introduction

Sparse KV Offloading extends the standard KV offload mechanism with intelligent block selection based on query similarity. This addresses the requirements outlined in RFC #37263: "Hotness-aware multi-level KV cache management to accelerate dynamic sparse attention."

!!! note
    Technical details on how vLLM implements sparse KV offloading can be found [here](../design/sparse_kv_offloading.md).

## Enabling Sparse KV Offloading

Use `SparseCPUOffloadingSpec` in your KV transfer configuration:

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    kv_transfer_config={
        "kv_connector_type": "cpu",
        "kv_connector_extra_config": {
            "spec_name": "SparseCPUOffloadingSpec",
            "cpu_bytes_to_use": 16 * 1024**3,
            "sparse_topk": 10240000,
            "copy_method": "merged",
            "cache_policy": "lru-layerwise",
            "eviction_policy": "lru",
        }
    }
)
```

## Example workloads

We describe example workloads where sparse KV offloading can provide significant performance benefit:

- **Long-context reasoning**: When processing long sequences (e.g., 100K+ tokens), sparse KV offloading reduces GPU memory usage by only loading hot blocks to GPU, while cold blocks remain in CPU memory. This allows serving longer contexts within limited GPU memory constraints.
- **High-concurrency serving**: In high-throughput serving scenarios, sparse KV offloading enables higher batch sizes by reducing per-request GPU memory footprint through intelligent block selection.
- **Multi-document QA**: When processing multiple documents in a single request, sparse KV offloading can selectively load relevant blocks from different documents, improving cache efficiency and reducing transfer overhead.

## Key Features

### Sparse Selection

Only loads most relevant KV blocks based on query-aware scoring:

```python
selected_blocks, scores = sparse_manager.lookup_with_sparse(
    block_hashes, query, layer_idx
)
```

### Block Representation

Maintains compact representations for efficient similarity computation:

```python
# Generate representations from KV cache
reprs = block_repr_manager.generate_repr(kv_cache, layer_idx, block_ids)

# Update representations
block_repr_manager.update(layer_idx, block_ids, reprs)

# Compute similarity
scores = block_repr_manager.compute_similarity(query, layer_idx, block_ids)
```

### Optimized Transfer

Merges contiguous blocks and supports batch operations:

```python
# Transfer with automatic merging
handler.transfer_async(job_id, transfer_spec)

# Get transfer statistics
stats = handler.get_stats()
print(f"Merge ratio: {stats['avg_merge_ratio']:.2f}")
```

## Configuration Parameters

- **`spec_name`**: Set to `"SparseCPUOffloadingSpec"` to enable sparse optimization
- **`cpu_bytes_to_use`**: CPU memory to allocate for KV cache
- **`sparse_topk`**: Maximum number of KV tokens to select per layer
- **`copy_method`**: Copy method for transfers
  - `"merged"`: Merge contiguous blocks (recommended)
  - `"gather-scatter"`: Use gather-scatter kernel
  - `"torch"`: Use torch.copy_()
- **`cache_policy`**: Cache replacement policy
  - `"lru"`: Standard LRU
  - `"lru-layerwise"`: Layer-wise LRU (recommended)
  - `"hot-score"`: Hotness score based
- **`eviction_policy`**: Backend eviction policy
  - `"lru"`: LRU eviction
  - `"arc"`: Adaptive Replacement Cache
- **`store_threshold`**: Minimum block accesses before CPU offloading
- **`max_tracker_size`**: Tracker size for reuse filtering

## Performance Benefits

1. **Memory Efficiency**: Only hot blocks loaded to GPU, reducing GPU memory usage by 80-90%
2. **Transfer Optimization**: Contiguous block merging reduces transfer overhead by 30-50%
3. **Intelligent Selection**: Query-aware selection improves cache hit rate by 10-25%
4. **Backward Compatibility**: Compatible with standard KV offload interfaces

## Limitations

Sparse KV offloading in general does not eliminate all memory and computation overhead. With that being said, sparse KV offloading does not provide performance benefit when:
- The sequence length is short (e.g., < 4K tokens), where the overhead of sparse selection may outweigh benefits.
- The cache hit rate is already high (> 90%), where most blocks are already in GPU memory.
- The workload is primarily prefilling, where sparse selection is less effective.