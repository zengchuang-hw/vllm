from typing import Tuple, List, Dict, Optional
from vllm.utils import cdiv
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata
import torch
import numpy as np

from vllm.debug_config import global_debug_config, DebugConfig
from vllm.v1.attention.backends.sparse_select import sparse_kv_selection, kv_repr_gen
from vllm.v1.cache import AbstractCache, LRUCache, LFUCache, HotScoreCache, LayerWiseLRUCache, LRUWithHotCache
from vllm.config import CopyMethod, CachePolicy

class GPUCacheManager:
    """
    管理 GPU Cache Block 与 CPU KV Cache Block 的映射关系
    """
    def __init__(
        self,
        num_gpu_cache_blocks: int,
        num_cpu_blocks: int,
        num_layers: int,
        block_repr_tensor: Optional[torch.Tensor],
        block_size: int,
        max_batch_size: int,
        max_num_batch_tokens: int,
        max_seq_len: int,
        sparse_topk: Optional[int] = None,
        copy_method: CopyMethod = CopyMethod.MERGED,
        cache_policy: CachePolicy = CachePolicy.LRU_LAYERWISE,
    ):
        self.num_gpu_cache_blocks = num_gpu_cache_blocks
        self.block_size = block_size
        self.num_cpu_blocks = num_cpu_blocks
        self.num_layers = num_layers
        self.device = block_repr_tensor.device
        self.copy_method = copy_method
        assert self.copy_method in [CopyMethod.NON_MERGED, CopyMethod.MERGED, CopyMethod.TORCH], "Invalid copy method"

        self.cache_policy = cache_policy
        if self.cache_policy == CachePolicy.LRU:
            self.cache = LRUCache(num_gpu_cache_blocks)
        elif self.cache_policy == CachePolicy.LFU:
            self.cache = LFUCache(num_gpu_cache_blocks)
        elif self.cache_policy == CachePolicy.HOT_SCORE:
            self.cache = HotScoreCache(num_gpu_cache_blocks)
        elif self.cache_policy == CachePolicy.LRU_LAYERWISE:
            self.cache = LayerWiseLRUCache(num_gpu_cache_blocks, num_layers, num_cpu_blocks=num_cpu_blocks)
        elif self.cache_policy == CachePolicy.LRU_WITH_HOT_SCORE:
            self.cache = LRUWithHotCache(num_gpu_cache_blocks)
        self.origin_attn_metadata: Optional[FlashAttentionMetadata] = None

        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.max_blocks_per_seq = cdiv(self.max_seq_len, self.block_size)
        self.max_num_batch_tokens = max_num_batch_tokens

        # !!! 每个 layer 的 CPU tensor 独立, 防止 CPU->GPU 拷贝操作在 device 执行的时候, CPU tensor 的内容已经被下一 layer 修改
        self.selected_logical_block_ids_cpu_tensor_list = [
            torch.zeros((self.max_batch_size, self.max_blocks_per_seq), dtype=torch.int64, device="cpu", pin_memory=True)
            for _ in range(self.num_layers)
        ]
        self.selected_logical_block_ids_np_list = [t.numpy() for t in self.selected_logical_block_ids_cpu_tensor_list]
        self.selected_logical_block_ids_cpu_tensor: Optional[torch.Tensor] = None
        self.selected_logical_block_ids_np: Optional[np.ndarray] = None

        self.num_selected_existing_full_blocks_cpu_tensor = torch.zeros(self.max_batch_size, dtype=torch.int32, device="cpu", pin_memory=True)
        self.num_selected_existing_full_blocks_np: Optional[np.ndarray] = self.num_selected_existing_full_blocks_cpu_tensor.numpy()

        self.new_slot_mapping = torch.zeros(self.max_num_batch_tokens, dtype=torch.int64, device=self.device)
        self.new_slot_mapping_cpu_list = [
            torch.zeros(self.max_num_batch_tokens, dtype=torch.int64, device="cpu", pin_memory=True)
            for _ in range(self.num_layers)
        ]
        self.new_slot_mapping_np_list = [t.numpy() for t in self.new_slot_mapping_cpu_list]
        self.new_slot_mapping_cpu: Optional[torch.Tensor] = None
        self.new_slot_mapping_np: Optional[np.ndarray] = None

        # 将 new_block_table 和 new_seq_lens 底层拼在一起, 降低一次 tensor.copy_() kernel launch
        self.merged_new_block_table_and_seq_lens = torch.zeros(self.max_batch_size + self.max_batch_size * self.max_blocks_per_seq, dtype=torch.int32, device=self.device)
        self.merged_new_block_table_and_seq_lens_cpu_list = [
            torch.zeros(self.max_batch_size + self.max_batch_size * self.max_blocks_per_seq, dtype=torch.int32, device="cpu", pin_memory=True)
            for _ in range(self.num_layers)
        ]
        self.merged_new_block_table_and_seq_lens_cpu: Optional[torch.Tensor] = None

        self.new_block_table = self.merged_new_block_table_and_seq_lens[self.max_batch_size:].view(self.max_batch_size, self.max_blocks_per_seq)
        self.new_block_table_cpu_list = [t[self.max_batch_size:].view(self.max_batch_size, self.max_blocks_per_seq) for t in self.merged_new_block_table_and_seq_lens_cpu_list]
        self.new_block_table_np_list = [t.numpy() for t in self.new_block_table_cpu_list]
        self.new_block_table_cpu: Optional[torch.Tensor] = None
        self.new_block_table_np: Optional[np.ndarray] = None

        self.new_seq_lens = self.merged_new_block_table_and_seq_lens[:self.max_batch_size]
        self.new_seq_lens_cpu_list = [t[:self.max_batch_size] for t in self.merged_new_block_table_and_seq_lens_cpu_list]
        self.new_seq_lens_np_list = [t.numpy() for t in self.new_seq_lens_cpu_list]
        self.new_seq_lens_cpu: Optional[torch.Tensor] = None
        self.new_seq_lens_np: Optional[np.ndarray] = None

        self.curr_num_actual_tokens = 0
        self.curr_batch_size = 0
        self.cur_max_num_selected_blocks = 0

        self.swap_in_cpu_tensor_list = [
            torch.zeros((self.num_gpu_cache_blocks, 2), dtype=torch.int64, device="cpu", pin_memory=True)
            for _ in range(self.num_layers)
        ]
        self.swap_in_np_list = [t.numpy() for t in self.swap_in_cpu_tensor_list]
        self.swap_in_cpu_tensor: Optional[torch.Tensor] = None
        self.swap_in_np: Optional[np.ndarray] = None
        
        self.num_swap_in_blocks = 0

        self.swap_out_cpu_tensor_list = [
            torch.zeros((self.num_gpu_cache_blocks, 2), dtype=torch.int64, device="cpu", pin_memory=True)
            for _ in range(self.num_layers)
        ]
        self.swap_out_np_list = [t.numpy() for t in self.swap_out_cpu_tensor_list]
        self.swap_out_cpu_tensor: Optional[torch.Tensor] = None
        self.swap_out_np: Optional[np.ndarray] = None

        self.num_swap_out_blocks = 0

        # KV Cache 写入 GPU
        self.kv_swap_in_stream = torch.cuda.Stream()
        self.kv_swap_in_event = torch.cuda.Event()

        # KV Cache 写回 CPU
        self.kv_swap_out_stream = torch.cuda.Stream()
        self.kv_swap_out_event = torch.cuda.Event()

        self.main_stream = torch.cuda.current_stream()

        # reshape 将新产生的 KV 写入 KV Cache
        self.kv_cache_update_event = torch.cuda.Event()

        self.sparse_topk = sparse_topk
        self.block_repr: Optional[torch.Tensor] = None
        if block_repr_tensor is not None:
            self.num_kv_heads = int(block_repr_tensor.shape[-2])
            self.head_dim = int(block_repr_tensor.shape[-1])
            self.block_repr = torch.zeros((self.num_layers, self.num_cpu_blocks) + block_repr_tensor.shape, dtype=block_repr_tensor.dtype, device=self.device)
            self.scores = torch.zeros((self.max_batch_size, self.max_blocks_per_seq), dtype=block_repr_tensor.dtype, device=self.device)
            self.scores_narrowed = self.scores
            self.num_top_k = 0

        self.max_num_existing_full_blocks: int = 0

        # slice tensor to match the current batch
        self.new_block_table_narrowed: Optional[torch.Tensor] = None
        self.new_block_table_cpu_narrowed: Optional[torch.Tensor] = None
        self.merged_new_block_table_and_seq_lens_narrowed: Optional[torch.Tensor] = None
        self.merged_new_block_table_and_seq_lens_cpu_narrowed: Optional[torch.Tensor] = None
        self.new_slot_mapping_narrowed: Optional[torch.Tensor] = None
        self.new_slot_mapping_cpu_narrowed: Optional[torch.Tensor] = None
        self.new_seq_lens_narrowed: Optional[torch.Tensor] = None
        self.new_seq_lens_cpu_narrowed: Optional[torch.Tensor] = None

    def _batch_prepare(self, attn_metadata: FlashAttentionMetadata):
        """
        batch 中的第一层, 计算 batch 中的 seq_len, query_len, kv_len 等信息
        Args:
            attn_metadata: FlashAttentionMetadata
        Returns:
        """
        assert isinstance(attn_metadata, FlashAttentionMetadata), "Only Support FlashAttetion"

        self.origin_attn_metadata = attn_metadata

        self.old_seq_lens_np = self.origin_attn_metadata.seq_lens_np

        self.curr_num_actual_tokens = self.origin_attn_metadata.num_actual_tokens

        self.curr_batch_size = self.origin_attn_metadata.num_seqs

        self.query_start_loc_np = self.origin_attn_metadata.query_start_loc_np

        self.old_block_table_np = self.origin_attn_metadata.block_table_np

        self.in_block_offset_np = self.origin_attn_metadata.in_block_offset_np.astype(np.int64)


        self.kv_lens_list = [0] * self.curr_batch_size
        self.num_existing_full_blocks = [0] * self.curr_batch_size
        self.num_full_blocks = [0] * self.curr_batch_size
        self.num_new_full_blocks = [0] * self.curr_batch_size
        self.num_partial_blocks = [0] * self.curr_batch_size

        for seq_idx in range(self.curr_batch_size):
            # 这里的隐含条件是 seq = [ kv | q ]
            # q 的长度
            seq_len = int(self.old_seq_lens_np[seq_idx])
            query_len = int(self.query_start_loc_np[seq_idx + 1]) - int(self.query_start_loc_np[seq_idx])
            # kv 长度
            self.kv_lens_list[seq_idx] = seq_len - query_len

            # 已经 cache 的完整 block 数量
            self.num_existing_full_blocks[seq_idx] = self.kv_lens_list[seq_idx] // self.block_size
            self.num_full_blocks[seq_idx] = seq_len // self.block_size
            # num_new_full_blocks 新填满的 block 数量(可能是之前未填满的 block, 也可能是新分配的完整 block)
            self.num_new_full_blocks[seq_idx] = self.num_full_blocks[seq_idx] - self.num_existing_full_blocks[seq_idx]
            # 未填满的 block(有可能是新产生的, 也有可能是上一轮为填满, 这一轮仍未填满)
            self.num_partial_blocks[seq_idx] = cdiv(seq_len, self.block_size) - self.num_full_blocks[seq_idx]

    def layer_prepare(self, layer_idx: int, attn_metadata: FlashAttentionMetadata):
        """
        根据 layer_idx 选择 cpu tensor. 如果是第一层, 还需要计算 batch 中的 seq_len, query_len, kv_len 等信息
        Args:
            layer_idx: int
            attn_metadata: FlashAttentionMetadata
        Returns:
        """
        if layer_idx == 0:
            self._batch_prepare(attn_metadata=attn_metadata)

        self.selected_logical_block_ids_cpu_tensor = self.selected_logical_block_ids_cpu_tensor_list[layer_idx]
        self.selected_logical_block_ids_np = self.selected_logical_block_ids_np_list[layer_idx]

        self.new_slot_mapping_cpu = self.new_slot_mapping_cpu_list[layer_idx]
        self.new_slot_mapping_np = self.new_slot_mapping_np_list[layer_idx]

        self.merged_new_block_table_and_seq_lens_cpu = self.merged_new_block_table_and_seq_lens_cpu_list[layer_idx]

        self.new_block_table_cpu = self.new_block_table_cpu_list[layer_idx]
        self.new_block_table_np = self.new_block_table_np_list[layer_idx]

        self.new_seq_lens_cpu = self.new_block_table_cpu_list[layer_idx]
        self.new_seq_lens_np = self.new_seq_lens_np_list[layer_idx]

        self.swap_in_np = self.swap_in_np_list[layer_idx]
        self.swap_in_cpu_tensor = self.swap_in_cpu_tensor_list[layer_idx]

        self.swap_out_np = self.swap_out_np_list[layer_idx]
        self.swap_out_cpu_tensor = self.swap_out_cpu_tensor_list[layer_idx]

        self.new_block_table_narrowed = self.new_block_table.narrow(0, 0, self.curr_batch_size)
        self.new_block_table_cpu_narrowed = self.new_block_table_cpu.narrow(0, 0, self.curr_batch_size)

        self.merged_new_block_table_and_seq_lens_narrowed = self.merged_new_block_table_and_seq_lens.narrow(0, 0, self.max_batch_size + self.curr_batch_size * self.max_blocks_per_seq)
        self.merged_new_block_table_and_seq_lens_cpu_narrowed = self.merged_new_block_table_and_seq_lens_cpu.narrow(0, 0, self.max_batch_size + self.curr_batch_size * self.max_blocks_per_seq)

        self.new_slot_mapping_narrowed = self.new_slot_mapping.narrow(0, 0, self.curr_num_actual_tokens)
        self.new_slot_mapping_cpu_narrowed = self.new_slot_mapping_cpu.narrow(0, 0, self.curr_num_actual_tokens)

        self.new_seq_lens_narrowed = self.new_seq_lens.narrow(0, 0, self.curr_batch_size)
        self.new_seq_lens_cpu_narrowed = self.new_seq_lens_cpu.narrow(0, 0, self.curr_batch_size)


    def select_topk_naive(self, attn_metadata: FlashAttentionMetadata, layer_idx: int, query: torch.Tensor, top_k_tokens_override: Optional[int] = None,
                          num_init_tokens: Optional[int] = 4096, num_local_tokens: Optional[int] = 4096) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        根据 block_repr 选择 top-k 个 KV 块
        Args:
            attn_metadata: FlashAttentionMetadata 中包含的 block_table 和 slot_mapping 映射的是在 CPU block 中的位置
            layer_idx: int
            query: [num_tokens, num_q_heads, head_size]
            top_k_tokens_override: 由参数指定的 KV token 数量
            num_init_tokens: init tokens 数量
            num_local_tokens: local tokens 数量
            # TODO: add support for init & local tokens
        Returns:
            selected_logical_blocks_np: batch 中每个 seq 选择的 KV 块在 seq 中的位置
            top_k_block_scores_np: batch 中每个 seq 选择的 KV 块的 attn score
            num_selected_blocks_cpu_np: batch 中每个 seq 实际选择的 KV 块数量
        """
        assert self.block_repr is not None, "block_repr 不能为空"
        if layer_idx == 0:
            top_k_tokens = self.sparse_topk
            if top_k_tokens_override is not None:
                top_k_tokens = top_k_tokens_override
            self.max_num_existing_full_blocks = 0
            self.num_top_k = top_k_tokens // self.block_size
            self.curr_batch_size=attn_metadata.num_seqs
            for i in range(self.curr_batch_size):
                query_len = int(attn_metadata.query_start_loc_np[i + 1] - attn_metadata.query_start_loc_np[i])
                seq_len = int(attn_metadata.seq_lens_np[i])
                kv_len = seq_len - query_len
                num_existing_full_blocks = kv_len // self.block_size
                self.max_num_existing_full_blocks = max(self.max_num_existing_full_blocks, num_existing_full_blocks)
                self.num_selected_existing_full_blocks_np[i] = min(num_existing_full_blocks, self.num_top_k)
            self.scores_narrowed = self.scores[:self.curr_batch_size, :self.max_num_existing_full_blocks]
        
        if self.max_num_existing_full_blocks <= self.num_top_k:
            # select all
            top_k_logical_block_ids_np, top_k_block_scores_np, _ = self.select_local(attn_metadata=attn_metadata, layer_idx=layer_idx, query=query, num_init_tokens=1024000000)
            return top_k_logical_block_ids_np, top_k_block_scores_np, self.num_selected_existing_full_blocks_np

        top_k_logical_block_ids_tensor, top_k_block_scores = sparse_kv_selection(block_table=attn_metadata.block_table,
                                                             batch_size=self.curr_batch_size,
                                                             block_size=self.block_size,
                                                             max_num_blocks_this_batch=self.max_num_existing_full_blocks,
                                                             seq_lens=attn_metadata.seq_lens,
                                                             k_repr=self.block_repr[layer_idx],  # 取出当前 layer 的 block repr
                                                             query=query,
                                                             query_start_loc=attn_metadata.query_start_loc,
                                                             top_k=self.num_top_k,
                                                             scores=self.scores_narrowed)
        # !!! 这里的 selected_logical_blocks 是 GPU tensor (调用 select_sparse_kernel), 会触发 CUDA stream 同步
        # TODO: 性能优化, 如果将 allocate 逻辑实现在 GPU 侧可以避免 GPU->CPU 的同步
        selected_logical_blocks_np = top_k_logical_block_ids_tensor.cpu().numpy()
        top_k_block_scores_np = top_k_block_scores.to(torch.float16).cpu().numpy()

        return selected_logical_blocks_np, top_k_block_scores_np, self.num_selected_existing_full_blocks_np



    def select_local(self, attn_metadata: FlashAttentionMetadata, layer_idx: int, query: torch.Tensor, num_init_tokens: int = 4096, num_local_tokens: int = 4096) -> Tuple[np.ndarray, np.ndarray]:
        """
        根据 block_repr 选择 top-k 个 KV 块。同时, 生产的新 KV 块的 block_repr 会被缓存。
        Args:
            attn_metadata: FlashAttentionMetadata 中包含的 block_table 和 slot_mapping 映射的是在 CPU block 中的位置
            layer_idx: int
            query: [num_tokens, num_q_heads, head_size]
            num_init_tokens: int
            num_local_tokens: int
        Returns:
            selected_logical_blocks_np: batch 中每个 seq 选择的 KV 块在 seq 中的位置
            top_k_block_scores_np: batch 中每个 seq 选择的 KV 块的 attn score
            num_selected_blocks_cpu_np: batch 中每个 seq 实际选择的 KV 块数量
        """
        num_init_blocks = num_init_tokens // self.block_size
        num_local_blocks = num_local_tokens // self.block_size

        self.curr_batch_size = attn_metadata.num_seqs
        self.cur_max_num_selected_blocks = num_init_blocks + num_local_blocks

        query_start_loc_np = attn_metadata.query_start_loc_np
        seq_lens = attn_metadata.seq_lens_np

        for i in range(self.curr_batch_size):
            query_len = query_start_loc_np[i + 1] - query_start_loc_np[i]
            seq_len = seq_lens[i]
            kv_len = seq_len - query_len
            num_existing_full_blocks = kv_len // self.block_size  # 当前 seq 的完整 kv 块数量

            if self.cur_max_num_selected_blocks > num_existing_full_blocks:  # 选择所有的完整 kv 块
                self.selected_logical_block_ids_np[i, :num_existing_full_blocks] = np.arange(0, num_existing_full_blocks, dtype=np.int32)
                self.num_selected_existing_full_blocks_np[i] = num_existing_full_blocks
            else:
                self.selected_logical_block_ids_np[i, :num_init_blocks] = np.arange(0, num_init_blocks, dtype=np.int32)
                self.selected_logical_block_ids_np[i, num_init_blocks:self.cur_max_num_selected_blocks] = np.arange(num_existing_full_blocks - num_local_blocks, num_existing_full_blocks, dtype=np.int32)
                self.num_selected_existing_full_blocks_np[i] = self.cur_max_num_selected_blocks

        selected_logical_blocks_np = self.selected_logical_block_ids_cpu_tensor.numpy()

        top_k_block_scores_np = np.ones_like(selected_logical_blocks_np, dtype=np.float16)

        return selected_logical_blocks_np, top_k_block_scores_np, self.num_selected_existing_full_blocks_np



    def allocate(self, layer_idx: int, selected_logical_block_ids_np: np.ndarray, selected_logical_block_scores_np: np.ndarray, num_selected_blocks: np.ndarray) -> Tuple[FlashAttentionMetadata, torch.Tensor, int, torch.Tensor, int]:
        """
        Parameters:
        layer_idx(int):  当前 Attention layer index
        selected_logical_block_ids_np: np.ndarray, [batch_size, max_num_seleced_blocks]: 稀疏注意力中当前 layer 选择的 KV 块位置。
        selected_logical_block_scores_np: np.ndarray 稀疏注意力中当前 layer 选择的 KV 块 attn score, 用于 cache 热度更新
        num_selected_blocks: np.ndarray, [batch_size]: 每个序列实际选择的 KV 块数量。
        Returns:
        Tuple[FlashAttentionMetadata, Dict[int, int], Dict[int, int]]:
            新 FlashAttentionMetadata 中包含的 block_table 和 slot_mapping 映射到已分配的 GPU block
            swap_in_mapping: 需要 KV 换入的 CPU block ID: GPU block ID
            swap_out_mapping: 需要 KV 换出的 GPU block ID: CPU block ID
        """

        self.cache.add_timer()
        query_pos_in_slot_mapping = 0

        max_sparse_seq_len: int = 0

        self.new_slot_mapping_np[:self.curr_num_actual_tokens] = self.in_block_offset_np

        self.num_swap_in_blocks = 0
        self.num_swap_out_blocks = 0

        for seq_idx in range(self.curr_batch_size):
            if global_debug_config.test:
                assert self.num_existing_full_blocks[seq_idx] >= 0 and self.num_new_full_blocks[seq_idx] >= 0 and (self.num_partial_blocks[seq_idx] == 0 or self.num_partial_blocks[seq_idx] == 1)

            new_block_table_next_pos = 0

            # if isinstance(self.cache, LayerWiseLRUCache):
            #     cache_size = [self.cache.layer_lists[layer_id].size for layer_id in range(self.num_layers)]
            #     print (f"before layer {layer_idx} Cache Size: {cache_size}")
            # 已经缓存的完整 KV 块稀疏化选择
            for j in reversed(range(num_selected_blocks[seq_idx])):
                logical_block_id = int(selected_logical_block_ids_np[seq_idx][j])
                if logical_block_id >= self.num_existing_full_blocks[seq_idx]:
                    continue  # 如果 seq 的 block 数量 < topk, 选择集中可能有溢出的 logical_block_id

                block_id = int(self.old_block_table_np[seq_idx][logical_block_id])

                slot_id, hit = self.cache.get((layer_idx, block_id), float(selected_logical_block_scores_np[seq_idx][j]))
                self.new_block_table_np[seq_idx][new_block_table_next_pos] = slot_id

                new_block_table_next_pos += 1

                if not hit:
                    # 只从 CPU 中 swap in cache miss的 GPU Cache Block
                    self.swap_in_np[self.num_swap_in_blocks] = [block_id, slot_id]
                    self.num_swap_in_blocks += 1

            # 新产生的完整 KV 块全部选择
            for logical_block_id in range(self.num_existing_full_blocks[seq_idx], self.num_new_full_blocks[seq_idx] + self.num_existing_full_blocks[seq_idx]):
                block_id = int(self.old_block_table_np[seq_idx][logical_block_id])
                slot_id, hit = self.cache.get((layer_idx, block_id), 1.0)

                # 填满的 block 不需要 pin 在 GPU 中
                # timer 机制保证了这一层新分配的 GPU block 不会被 evict
                self.cache.unpin_block((layer_idx, block_id))

                self.new_block_table_np[seq_idx][new_block_table_next_pos] = slot_id
                new_block_table_next_pos += 1

                # 等计算完成后 swap out 所有 GPU Cache Block 到 CPU 中
                self.swap_out_np[self.num_swap_out_blocks] = [slot_id, block_id]
                self.num_swap_out_blocks += 1

                # 将当前 block 中的 query_tokens 的 slot_mapping 设置为 slot 位置
                num_query_tokens_in_this_block = self.block_size
                # 减掉剩余 KV Cache
                num_query_tokens_in_this_block -= max(0, self.kv_lens_list[seq_idx] - logical_block_id * self.block_size)
                # 截断末尾

                num_query_tokens_in_this_block -= max(0, (logical_block_id + 1) * self.block_size - int(self.old_seq_lens_np[seq_idx]))

                self.new_slot_mapping_np[query_pos_in_slot_mapping: num_query_tokens_in_this_block + query_pos_in_slot_mapping] += slot_id * self.block_size
                query_pos_in_slot_mapping += num_query_tokens_in_this_block

            if self.num_partial_blocks[seq_idx] == 1:
                logical_block_id = self.num_full_blocks[seq_idx]  # = num_existing_full_blocks[i] + num_new_full_blocks[i]
                block_id = self.old_block_table_np[seq_idx][logical_block_id]

                slot_id, hit = self.cache.get((layer_idx, block_id), 0.0)
                self.cache.pin_block((layer_idx, block_id))
                self.new_block_table_np[seq_idx][new_block_table_next_pos] = slot_id
                new_block_table_next_pos += 1

                num_query_tokens_in_this_block = self.block_size
                # 减掉剩余 KV Cache
                num_query_tokens_in_this_block -= max(0, self.kv_lens_list[seq_idx] - logical_block_id * self.block_size)
                # 截断末尾
                num_query_tokens_in_this_block -= max(0, (logical_block_id + 1) * self.block_size - int(self.old_seq_lens_np[seq_idx]))

                self.new_slot_mapping_np[query_pos_in_slot_mapping: num_query_tokens_in_this_block + query_pos_in_slot_mapping] += slot_id * self.block_size
                query_pos_in_slot_mapping += num_query_tokens_in_this_block

            # 每个请求经过稀疏化后实际选择了多少 token
            num_not_selected_existing_full_blocks = self.num_existing_full_blocks[seq_idx] - int(num_selected_blocks[seq_idx])
            self.new_seq_lens_np[seq_idx] = int(self.old_seq_lens_np[seq_idx]) - num_not_selected_existing_full_blocks * self.block_size
            max_sparse_seq_len = max(max_sparse_seq_len, int(self.new_seq_lens_np[seq_idx]))

        # commit slot_mapping, block_table, seq_lens 到 GPU
        self.new_slot_mapping_narrowed.copy_(self.new_slot_mapping_cpu_narrowed, non_blocking=True)

        self.merged_new_block_table_and_seq_lens_narrowed.copy_(self.merged_new_block_table_and_seq_lens_cpu_narrowed, non_blocking=True)

        new_attn_metadata = FlashAttentionMetadata(
            num_actual_tokens=self.curr_num_actual_tokens,
            max_query_len=self.origin_attn_metadata.max_query_len,
            num_seqs=self.curr_batch_size,
            query_start_loc=self.origin_attn_metadata.query_start_loc,  # 无需更改
            query_start_loc_cpu=self.origin_attn_metadata.query_start_loc_cpu,
            query_start_loc_np=self.query_start_loc_np,
            max_seq_len=max_sparse_seq_len,

            seq_lens=self.new_seq_lens_narrowed,  # 需要更改
            seq_lens_cpu=self.new_seq_lens_cpu,
            seq_lens_np=self.new_seq_lens_np,

            block_table=self.new_block_table_narrowed,  # 需要更改
            block_table_cpu=self.new_block_table_cpu,
            block_table_np=self.new_block_table_np,

            slot_mapping=self.new_slot_mapping_narrowed,  # 需要更改
            slot_mapping_cpu=self.new_slot_mapping_cpu,
            slot_mapping_np=self.new_slot_mapping_np,

            # For cascade attention. Not Implemented
            use_cascade=self.origin_attn_metadata.use_cascade,
            common_prefix_len=self.origin_attn_metadata.common_prefix_len,
            cu_prefix_query_lens=self.origin_attn_metadata.cu_prefix_query_lens.clone() if self.origin_attn_metadata.cu_prefix_query_lens is not None else None,
            prefix_kv_lens=self.origin_attn_metadata.prefix_kv_lens.clone() if self.origin_attn_metadata.prefix_kv_lens is not None else None,
            suffix_kv_lens=self.origin_attn_metadata.suffix_kv_lens.clone() if self.origin_attn_metadata.suffix_kv_lens is not None else None,
            # For logging.
            num_input_tokens=self.origin_attn_metadata.num_input_tokens,
        )

        return new_attn_metadata, self.swap_in_cpu_tensor, self.num_swap_in_blocks, self.swap_out_cpu_tensor, self.num_swap_out_blocks


    def gen_repr(self, layer_idx: int, gpu_cache: torch.Tensor, swap_out_mapping_cpu: torch.Tensor, num_swap_out_mapping: int):
        if self.block_repr is None:
            return
        swap_out_mapping = swap_out_mapping_cpu.cuda(self.device, non_blocking=True)
        kv_repr_gen(
            kv_cache=gpu_cache,
            block_repr=self.block_repr[layer_idx],
            mapping=swap_out_mapping,
            num_mappings=num_swap_out_mapping,
            block_size=self.block_size,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim)