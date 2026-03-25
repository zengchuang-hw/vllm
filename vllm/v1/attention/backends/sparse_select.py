import triton
import triton.language as tl
import torch
from typing import Tuple

@triton.jit
def moba_block_repr_kernel(
        # Pointers to matrices
        k_cache_ptr,  # [num_gpu_blocks, block_size, num_heads, head_dim]
        block_repr_ptr,  # [num_cpu_blocks, num_heads, head_dim]
        mapping_ptr,  # [num_mappings, 2] (gpu_block_id, cpu_block_id)

        # Matrix dimensions
        num_mappings,  # Number of mappings
        block_size,  # Size of each block
        num_heads,  # Number of attention heads

        # Meta-parameters
        BLOCK_SIZE: tl.constexpr,
        HEAD_DIM: tl.constexpr,
):
    # 获取程序ID - 每个线程块处理一个(mapping_idx, head_idx)对
    # pid_m 是 mapping 的索引,pid_h 是 head 的索引
    pid_m = tl.program_id(axis=0)  # Mapping index
    pid_h = tl.program_id(axis=1)  # Head index

    # 如果超出范围则提前退出
    if pid_m >= num_mappings or pid_h >= num_heads:
        return

    # 加载映射信息
    gpu_block_id = tl.load(mapping_ptr + pid_m * 2)
    cpu_block_id = tl.load(mapping_ptr + pid_m * 2 + 1)

    gpu_block_id = tl.cast(gpu_block_id, tl.int64)
    cpu_block_id = tl.cast(cpu_block_id, tl.int64)

    # 计算指向 K block 中特定 head 的指针
    k_block_head_ptr = k_cache_ptr + (gpu_block_id * block_size * num_heads * HEAD_DIM) + (pid_h * HEAD_DIM)

    # 计算指向输出 block 表示的指针
    block_repr_out_ptr = block_repr_ptr + (cpu_block_id * num_heads * HEAD_DIM) + (pid_h * HEAD_DIM)

    # 初始化累加器,用于计算均值
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    # 处理 block 中的所有 token,每次处理 BLOCK_SIZE 个
    for i in range(0, block_size, BLOCK_SIZE):
        # 创建有效索引的掩码
        block_idx = tl.arange(0, BLOCK_SIZE)

        # 计算当前 chunk 中每个 token 的 key 向量的偏移
        # 注意：k_cache 的布局是 [num_gpu_blocks, block_size, num_heads, HEAD_DIM]
        # 所以每个 token 之间的步长是 num_heads * HEAD_DIM
        offsets = (i + block_idx) * num_heads * HEAD_DIM

        # 加载当前 chunk 的 key 向量
        # 为每个维度创建索引向量
        dim_indices = tl.arange(0, HEAD_DIM)

        # 为每个 token 的每个维度加载数据
        # 使用二维加载,第一维是 token 索引,第二维是 HEAD_DIM 索引
        k_indices = k_block_head_ptr + offsets[:, None] + dim_indices[None, :]
        k_chunk = tl.load(k_indices)

        # 累加和
        acc += tl.sum(k_chunk, axis=0)

    # 计算均值（除以 block_size）
    acc = acc / block_size

    # 将结果存储到 block_repr, 会自动将 bfloat16 acc 转换为 float32
    tl.store(block_repr_out_ptr + tl.arange(0, HEAD_DIM), acc)


def kv_repr_gen(
        kv_cache: torch.Tensor,
        block_repr: torch.Tensor,
        mapping: torch.Tensor,
        num_mappings: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
):
    """
    计算 MoBA block 表示, 将 K cache 中新生产的 k block 计算均值表示，存在同位置上的 block_repr 中。
    
    Args:
        kv_cache: 形状为 [2, num_gpu_blocks, block_size, num_heads, head_dim]
        block_repr: [num_cpu_blocks, num_kv_heads, head_dim]
        mapping: [num_gpu_blocks, 2] 映射 tensor, 前 num_mappings 个是要处理的 (gpu_block_id, cpu_block_id) 对
        num_mappings: int
        block_size: int,
        num_kv_heads: int,
        head_dim: int
    """
    # 确定处理 chunk 的大小

    # 启动内核
    # 使用二维网格,第一维是 mapping 数量,第二维是 head 数量
    grid = (num_mappings, num_kv_heads)
    moba_block_repr_kernel[grid](
        kv_cache,
        block_repr,
        mapping,
        num_mappings,
        block_size,
        num_kv_heads,
        HEAD_DIM=head_dim,
        BLOCK_SIZE=block_size,
    )

    return block_repr


@triton.jit
def compute_block_scores_kernel(
        block_indices_ptr,  # [batch_size, max_num_blocks_per_seq] 当前批次的 block 索引
        k_repr_ptr,  # [num_blocks, num_kv_heads, head_size] 每个 block 的代表性特征
        query_ptr,  # [num_query_tokens, num_heads, head_size] 查询
        query_start_loc,  # [num_query_tokens + 1]
        seq_lens_ptr,
        scores_ptr,  # [batch_size, seq_num_blocks] 输出分数

        # 形状参数
        block_size,
        batch_size,  # 批次大小
        max_num_blocks_per_seq,  # 每个序列的 block 数量
        num_kv_heads,  # KV 的头数量
        num_heads,  # Q 的头数量
        head_size,  # 每个头的维度
        heads_per_kv,  # 每个 KV 头对应的查询头数量

        # 数据类型和步长
        block_indices_stride,  # block_indices 的步长
        k_repr_stride_0,  # k_repr 的步长
        k_repr_stride_1,  # k_repr 的步长
        query_stride_0,  # query 的步长
        query_stride_1,  # query 的步长
        scores_stride,  # scores 的步长

        # 并行化参数
        BLOCK_M: tl.constexpr,  # 线程块的行数
        BLOCK_N: tl.constexpr,  # 线程块的列数
        BLOCK_K: tl.constexpr,  # 线程块的内部维度
        GROUP_M: tl.constexpr,  # 线程组的行数
):
    """计算每个 block 的分数"""
    # 获取当前线程块处理的索引
    batch_idx = tl.program_id(0)
    block_idx = tl.program_id(1)

    batch_size = tl.cast(batch_size, tl.int32)
    max_num_blocks_per_seq = tl.cast(max_num_blocks_per_seq, tl.int32)

    # 如果超出范围,则返回
    if batch_idx >= batch_size or block_idx >= max_num_blocks_per_seq:
        return

    # 加载 seq 的当前 block 索引
    block_offset = batch_idx * block_indices_stride + block_idx
    block_id = tl.load(block_indices_ptr + block_offset)

    # 初始化分数
    total_score = 0.0

    query_start = tl.load(query_start_loc + batch_idx)
    query_end = tl.load(query_start_loc + batch_idx + 1)
    num_query_tokens = query_end - query_start

    seq_len = tl.load(seq_lens_ptr + batch_idx)
    kv_len = seq_len - num_query_tokens

    # 计算当前请求有多少完整的 KV 块
    num_kv_full_blocks = kv_len // block_size

    # 不是完整的 KV Cache 块,不选择,直接赋最小值
    if block_idx >= num_kv_full_blocks:
        tl.store(scores_ptr + batch_idx * scores_stride + block_idx, float('-inf'))
        return
    
    # 头尾块一定被选择
    if num_kv_full_blocks > 0 and (block_idx == 0 or block_idx == num_kv_full_blocks - 1):
        tl.store(scores_ptr + batch_idx * scores_stride + block_idx, float('inf'))
        return

    # 对每个 token 计算分数
    # FIXME: 当前只用 q 的最后一个 token. chunked prefill 时需要用上 chunk 内所有的 q
    for token_idx in range(query_end - 1, query_end):
        # 对每个 KV 头计算分数
        for kv_head_idx in range(0, num_kv_heads):
            # 加载 k_repr
            k_offset = block_id * k_repr_stride_0 + kv_head_idx * k_repr_stride_1
            k = tl.load(k_repr_ptr + k_offset + tl.arange(0, BLOCK_K), mask=tl.arange(0, BLOCK_K) < head_size)

            # 对应的查询头范围
            q_head_start = kv_head_idx * heads_per_kv
            q_head_end = (kv_head_idx + 1) * heads_per_kv

            # 对每个对应的查询头计算分数
            for q_head_idx in range(q_head_start, q_head_end):
                # 加载查询
                q_offset = token_idx * query_stride_0 + q_head_idx * query_stride_1
                q = tl.load(query_ptr + q_offset + tl.arange(0, BLOCK_K), mask=tl.arange(0, BLOCK_K) < head_size)

                # 计算点积
                dot_product = tl.sum(k * q)
                total_score += dot_product

    # 只要比较大小,无需平均分数
    # final_score = total_score / (num_query_tokens * num_heads)

    # 写入输出
    tl.store(scores_ptr + batch_idx * scores_stride + block_idx, total_score)


def sparse_kv_selection(
        block_table: torch.Tensor,  # [max_batch_size, max_num_blocks]
        batch_size: int,
        block_size: int,
        max_num_blocks_this_batch: int,
        seq_lens: torch.Tensor,  # [max_batch_size]
        k_repr: torch.Tensor,  # [num_cpu_blocks, num_kv_heads, head_size]
        query: torch.Tensor,  # [max_num_batched_query_tokens, num_heads, head_size]
        query_start_loc,  # [batch_size + 1]
        top_k: int,  # 每个 seq 选择的 top-k 个 block,
        scores: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    根据查询和 KV 块的代表性特征,为每个 seq 选择 top-k 个 KV 块。

    Args:
        block_table: [max_batch_size, max_num_blocks] 每个 seq 的 block 位置
        batch_size: int,
        block_size: int,
        max_num_blocks_this_batch: 当前 batch 中 seq 的最大 block 数量
        seq_lens: [max_batch_size] batch 中每个 seq 的长度, 包含 len(kv_cache) + len(q)
        k_repr: [num_cpu_blocks, num_kv_heads, head_size] 每个 block 的代表性特征
        query: [max_num_batched_query_tokens, num_q_heads, head_size] 查询向量
        query_start_loc: [batch_size + 1] 标志 batch 中每个 query 的始末位置    query: [ q0 | q1 q1 | q2 q2]  query_start_loc = [0,1,3,5]
        top_k: 每个 seq 选择的 top-k 个 block 数量
        scores: [batch_size, max_num_blocks_this_batch] 预分配的 block scores

    Returns:
        topk_choices: [batch_size, top_k] 选中的 block 索引, 若某个 seq 的 block 数量 < topk, 该行的 [topk: ] 之后的 block 索引没有意义
        topk_scores: [batch_size, topk] 选中的 block 对应的 attn scores
    """
    # 获取维度
    num_kv_heads = k_repr.shape[1]
    head_size = k_repr.shape[2]
    num_heads = query.shape[1]

    # 计算每个 KV 头对应的查询头数量
    heads_per_kv = num_heads // num_kv_heads  # GQA 结构

    # 计算步长
    block_indices_stride = block_table.stride(0)
    k_repr_stride_0 = k_repr.stride(0)
    k_repr_stride_1 = k_repr.stride(1)
    query_stride_0 = query.stride(0)
    query_stride_1 = query.stride(1)
    scores_stride = scores.stride(0)

    # 计算并行化参数
    BLOCK_M = 1
    BLOCK_N = 1
    BLOCK_K = triton.next_power_of_2(head_size)
    GROUP_M = 1

    # 启动计算分数的 kernel
    grid = (batch_size, max_num_blocks_this_batch)

    # TODO: query 形状动态变化,会触发多次编译
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

    # scores: [batch_size, max_num_blocks_this_batch], 若某个 seq 不足 max_num_blocks_per_req, 后面是 pad -inf,
    topk_scores, topk_choices = torch.topk(scores, top_k, dim=1, sorted=False)
    # topk_choices [batch_size, top_k] 每一行是一个 seq 的 top_k logical_block_ids, 若某个 seq 不足 topk blocks 时, 会有超过其实际 block 数量的 logical_block_id 出现
    return topk_choices, topk_scores


if __name__ == '__main__':
    # 示例用法
    num_gpu_blocks = 800
    block_size = 4096
    num_heads = 8
    num_q_heads = 40
    head_dim = 128
    num_cpu_blocks = 600
    num_mappings = 4
    max_block_per_seq = 5
    max_batch_size = 4
    top_k = 2

    # 创建示例数据
    kv_cache = torch.zeros(2, num_gpu_blocks, block_size, num_heads, head_dim, device='cuda', dtype=torch.bfloat16)

    kv_cache.fill_(-1)

    mapping = torch.zeros((num_mappings, 2), device='cpu', dtype=torch.int32)

    mapping_np = mapping.numpy()

    mapping_np[0] = [600, 500]  # gpu_block_id -> cpu_block_id
    mapping_np[1] = [601, 501]
    mapping_np[2] = [602, 502]
    mapping_np[3] = [605, 505]

    k_repr = torch.zeros((num_cpu_blocks, num_heads, head_dim), dtype=torch.bfloat16, device='cuda')
    k_repr.fill_(10.0)

    block_table = torch.zeros((max_batch_size, max_block_per_seq), dtype=torch.int32, device='cuda')
    block_table.fill_(-1)

    # seq1: cpu block id: [500, 501, 502, 503]
    #       gpu block id: [600, 601, 602, 603]
    block_table[0, 0] = 500  # logical_block_id -> cpu_block_id
    block_table[0, 1] = 501
    block_table[0, 2] = 502
    block_table[0, 3] = 503

    # seq2: cpu block id: [505, 506]
    #       gpu block id: [605, 606]
    block_table[1, 0] = 505
    block_table[1, 1] = 506

    query = torch.ones((4, num_q_heads, head_dim), dtype=torch.bfloat16, device='cuda')
    query_start_loc = torch.ones((max_batch_size + 1,), dtype=torch.int32, device='cuda')
    # seq1 query[0:3)
    # seq2 query[3:4)
    query_start_loc[0] = 0
    query_start_loc[1] = 3
    query_start_loc[2] = 4

    kv_cache[0, 600, :, :, :].fill_(1)
    kv_cache[0, 601, :, :, :].fill_(2)
    kv_cache[0, 602, :, :, :].fill_(3)
    kv_cache[0, 605, :, :, :].fill_(4)

    # 计算 block 表示
    k_repr = kv_repr_gen(kv_cache=kv_cache,
                         block_repr=k_repr,
                         mapping=mapping.cuda(),
                         num_mappings=num_mappings,
                         block_size=block_size,
                         num_kv_heads=num_heads,
                         head_dim=head_dim)

    print(k_repr[500])

    scores = torch.zeros((max_batch_size, max_block_per_seq), dtype=torch.bfloat16, device="cuda")

    seq_lens = torch.zeros(max_batch_size, dtype=torch.int32, device="cuda")
    seq_lens[0] = 3 * block_size + 0 + 3 # seq1: kkkk | kkkk | kkkk | qqq
    seq_lens[1] = 1 * block_size + 1 + 1   # seq2: kkkk | kq

    output = sparse_kv_selection(
        block_table=block_table,
        block_size=block_size,
        batch_size=2,
        max_num_blocks_this_batch=max_block_per_seq,
        seq_lens=seq_lens,
        k_repr=k_repr,
        query=query,
        query_start_loc=query_start_loc,
        top_k=top_k,
        scores=scores.narrow(0, 0, 2),
    )

    print(output[:2, :])
    print(scores[:2, :])
