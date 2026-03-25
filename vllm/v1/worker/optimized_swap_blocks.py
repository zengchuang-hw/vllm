import os
import time
from typing import Optional, List
from torch.utils.cpp_extension import load
import torch
import torch.profiler
import argparse
import contextlib
from vllm.config import CopyMethod
from vllm.debug_config import global_debug_config


def get_arch_flag():
    """获取当前默认 GPU 的 -gencode 编译标志"""
    if not torch.cuda.is_available():
        return None

    # 获取默认设备的计算能力 (major, minor)
    # 例如对于 RTX 4090, major=8, minor=9
    major, minor = torch.cuda.get_device_capability()

    # 格式化为 '-gencode=arch=compute_XX,code=sm_XX'
    # 注意 sm_XX 中 XX 是 major 和 minor 的拼接
    return f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}"


# 定义你想要的编译优化标志
# 对于 C++ 编译器 (g++)
cxx_flags = [
    "-O3",  # 开启最高级别的优化
    "-std=c++17",  # 使用 C++17 标准
    "-Wall",  # 开启所有警告，有助于代码质量
    # '-g'               # (可选) 生成调试信息，方便使用 gdb 调试
]

# 对于 CUDA 编译器 (nvcc)
# 注意：在方案一中我们没有 .cu 文件，所以这个参数不起作用
# 但如果是方案二，这个参数就至关重要了
cuda_flags = [
    "-O3",
    # 例如，为特定的 GPU 架构生成代码 (Ampere)
    get_arch_flag(),
]


swap_blocks_ops = load(
    name="optimized_swap_blocks",  # 建议换个名字以区分不同的编译配置
    sources=[
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "optimized_swap_blocks.cu"
        )
    ],
    extra_cflags=cxx_flags,
    extra_cuda_cflags=cuda_flags,  # 如果有 .cu 文件则添加此行
    verbose=True,
)

# 此时，swap_blocks_ops 变量就是一个已加载的 Python 模块！


def swap_blocks(
    src_kv_cache: torch.Tensor,
    dst_kv_cache: torch.Tensor,
    block_mapping: torch.Tensor,
    copy_method: CopyMethod,
    swap_func: Optional[callable] = None,
    print_verbose: bool = global_debug_config.swap_copy_ops,
) -> None:
    swap_in = False 
    if src_kv_cache.is_cpu and dst_kv_cache.is_cuda:
        swap_in = True
    elif src_kv_cache.is_cuda and dst_kv_cache.is_cpu:
        swap_in = False
    else:
        raise ValueError(f"neither GPU->CPU nor CPU->GPU, src_kv_cache {src_kv_cache.device} and dst_kv_cache {dst_kv_cache.device}")
    
    if copy_method == CopyMethod.CUSTOM:
        if swap_func is None:
            raise ValueError("swap_func must be provided when copy_method is 'defined'")
        else:
            swap_func(src_kv_cache, dst_kv_cache, block_mapping)
        return

    if copy_method == CopyMethod.TORCH:
        src_key_cache = src_kv_cache[0]
        src_value_cache = src_kv_cache[1]
        dst_key_cache = dst_kv_cache[0]
        dst_value_cache = dst_kv_cache[1]

        block_mapping_np = block_mapping.numpy()
        for i in range(block_mapping_np.shape[0]):
            src_block_id = block_mapping[i][0]
            dst_block_id = block_mapping[i][1]

            dst_key_cache[dst_block_id].copy_(src_key_cache[src_block_id])
            dst_value_cache[dst_block_id].copy_(src_value_cache[src_block_id])

        return

    if copy_method == CopyMethod.MERGED:
        src_key_cache = src_kv_cache[0]
        dst_key_cache = dst_kv_cache[0]

        _ = swap_blocks_ops.swap_contiguous_blocks(
            src_key_cache, dst_key_cache, block_mapping
        )

        src_value_cache = src_kv_cache[1]
        dst_value_cache = dst_kv_cache[1]

        num_copy_ops = swap_blocks_ops.swap_contiguous_blocks(
            src_value_cache, dst_value_cache, block_mapping
        )

        if print_verbose:
            if swap_in:
                swap_direction = "in"
            else:
                swap_direction = "out"
            print(
                f"swap {swap_direction} {block_mapping.size(0)} blocks -> merge into {num_copy_ops} copy ops. Method: {copy_method}"
            )
        return

    if copy_method == CopyMethod.NON_MERGED:
        src_key_cache = src_kv_cache[0]
        dst_key_cache = dst_kv_cache[0]

        num_copy_ops = swap_blocks_ops.swap_blocks_raw(
            src_key_cache, dst_key_cache, block_mapping
        )

        src_value_cache = src_kv_cache[1]
        dst_value_cache = dst_kv_cache[1]

        num_copy_ops = swap_blocks_ops.swap_blocks_raw(
            src_value_cache, dst_value_cache, block_mapping
        )

        if print_verbose:
            if swap_in:
                swap_direction = "in"
            else:
                swap_direction = "out"

            print(f"swap {swap_direction} {num_copy_ops} blocks. Method: {copy_method}")
        return

    if copy_method == CopyMethod.GATHER_SCATTER:
        src_key_cache = src_kv_cache[0]
        src_value_cache = src_kv_cache[1]
        dst_key_cache = dst_kv_cache[0]
        dst_value_cache = dst_kv_cache[1]
        
        # Check if CPU -> GPU or GPU -> CPU
        if swap_in:
             swap_blocks_ops.swap_blocks_optimized(
                src_key_cache, src_value_cache,
                dst_key_cache, dst_value_cache,
                block_mapping
            )
        else:
             swap_blocks_ops.swap_blocks_optimized(
                src_key_cache, src_value_cache,
                dst_key_cache, dst_value_cache,
                block_mapping
            )

        if print_verbose:
            if swap_in:
                swap_direction = "in"
            else:
                swap_direction = "out"
            print(f"swap {swap_direction} {block_mapping.size(0)} blocks. Method: {copy_method}")
        return

    raise ValueError(f"Unknown copy_method: {copy_method}")


# ===================================================================
# =================== 调用和测试部分 =================================
# ===================================================================


# 测试拷贝准确性
def __test_accuracy(
    method: CopyMethod, swap_in: bool, profiler: Optional[torch.profiler.profile] = None
):
    print(get_arch_flag())
    print(f"使用的拷贝方法: {method}")

    # 准备测试数据
    num_blocks = 1024
    num_swapping_blocks = 100
    block_size = 16
    hidden_size = 128
    dtype = torch.bfloat16
    device = "cuda:0"

    gpu_cache = torch.randn(
        2, num_blocks, block_size, hidden_size, dtype=dtype, device=device
    )
    cpu_cache = torch.randn(
        2,
        num_blocks,
        block_size,
        hidden_size,
        dtype=dtype,
        device="cpu",
        pin_memory=True,
    )

    import random
    
    def generate_random_mapping(num_blocks, num_pairs):
        # Generate random unique source blocks
        src_blocks = random.sample(range(num_blocks), num_pairs)
        # Generate random unique dest blocks
        dst_blocks = random.sample(range(num_blocks), num_pairs)
        
        mapping_list = []
        for s, d in zip(src_blocks, dst_blocks):
            mapping_list.append([s, d])
            
        # Add some contiguous blocks to test merging logic specifically
        # Try to find a free range
        used_src = set(src_blocks)
        used_dst = set(dst_blocks)
        
        # Add a contiguous chunk of size 10 if possible
        start_src = 0
        while start_src < num_blocks - 10:
             if not any(b in used_src for b in range(start_src, start_src + 10)):
                 break
             start_src += 1
             
        start_dst = 0
        while start_dst < num_blocks - 10:
             if not any(b in used_dst for b in range(start_dst, start_dst + 10)):
                 break
             start_dst += 1
             
        if start_src < num_blocks - 10 and start_dst < num_blocks - 10:
            for i in range(10):
                mapping_list.append([start_src + i, start_dst + i])

        return mapping_list

    # print(block_mapping.shape)
    if not swap_in:
        # Generate random mapping for GPU -> CPU
        block_mapping_list = generate_random_mapping(num_blocks, num_swapping_blocks)
        block_mapping = torch.tensor(block_mapping_list, dtype=torch.int64, device="cpu")

        print("测试 GPU -> CPU")
        # 调用我们自己写的 C++ 函数，用法和之前完全一样
        torch.cuda.synchronize()
        if profiler:
            profiler.start()

        swap_blocks(gpu_cache, cpu_cache, block_mapping, copy_method=method)
        if profiler:
            profiler.stop()
            print(f"Trace log saved to {args.trace_log_dir}")
        else:
            torch.cuda.synchronize()

        # 验证结果
        for block_mapping_task in block_mapping_list:
            is_correct_task = torch.allclose(
                gpu_cache[0][block_mapping_task[0]],
                cpu_cache[0][block_mapping_task[1]].cuda(device=device),
            )
            is_correct_task = torch.allclose(
                gpu_cache[1][block_mapping_task[0]],
                cpu_cache[1][block_mapping_task[1]].cuda(device=device),
            )
            # print(f"任务验证结果: {is_correct_task}")
            if not is_correct_task:
                exit(-1)

    if swap_in:
        # Generate random mapping for CPU -> GPU
        block_mapping_list = generate_random_mapping(num_blocks, num_swapping_blocks)
        block_mapping = torch.tensor(block_mapping_list, dtype=torch.int64, device="cpu")

        print("测试 CPU -> GPU")
        # 调用我们自己写的 C++ 函数，用法和之前完全一样
        swap_blocks(cpu_cache, gpu_cache, block_mapping, copy_method=method)
        torch.cuda.synchronize()

        # 验证结果
        for block_mapping_task in block_mapping_list:
            is_correct_task = torch.allclose(
                cpu_cache[0][block_mapping_task[0]].cuda(device=device),
                gpu_cache[0][block_mapping_task[1]],
            )
            is_correct_task = torch.allclose(
                cpu_cache[1][block_mapping_task[0]].cuda(device=device),
                gpu_cache[1][block_mapping_task[1]],
            )
            # print(f"任务验证结果: {is_correct_task}")
            if not is_correct_task:
                exit(-1)

    print("拷贝功能验证通过。")


# 测试拷贝性能
def __test_performance(
    num_total_blocks: int,
    block_size: int,
    hidden_size: int,
    num_tasks: int,
    num_contiguous_blocks_per_task: int,
    test_times: int = 1,
    warmup_iterations: int = 2,
    methods: List[CopyMethod] = [],
    swap_in: bool = True,  # CPU -> GPU
    custom_swap_func: Optional[callable] = None,
    profiler: Optional[torch.profiler.profile] = None,
):

    def __get_random_block_mapping_tensor(
        num_total_blocks: int, num_tasks: int, num_blocks_per_task: int
    ):
        # 随机生成不重叠的 block 映射，返回 block_mapping_list，每一项为 [block_id, block_id]，共 num_tasks * num_blocks_per_task 个，用于性能测试
        # 参数:
        #   num_total_blocks: 所有可用的块数（int）
        #   num_tasks: 区间（任务）数，每个任务是一段连续的块，不与其他任务重叠（int）
        #   num_blocks_per_task: 每个任务区间中包含多少个连续块（int）
        # 返回:
        #   block_mapping_list: List[List[int, int]]，每个元素为 [block_id, block_id]，总共 num_tasks * num_blocks_per_task 个

        import random

        block_mapping_list = []

        # 得到 num_tasks 个区间, 每个区间是 num_blocks_per_task 个连续的块, 区间不能重叠

        # 确保有足够的块
        assert (
            num_tasks * num_blocks_per_task <= num_total_blocks
        ), f"需要 {num_tasks * num_blocks_per_task} 个块，但只有 {num_total_blocks} 个块"

        # 随机选择 num_tasks 个不重叠的起始位置
        selected_starts = []
        used_blocks = set()

        # 所有可能的起始位置（确保区间不会超出范围）
        max_start = num_total_blocks - num_blocks_per_task
        possible_starts = list(range(max_start + 1))
        random.shuffle(possible_starts)

        # 选择不重叠的区间
        for start in possible_starts:
            if len(selected_starts) >= num_tasks:
                break

            # 检查这个区间是否与已选择的区间重叠
            overlap = False
            for block_id in range(start, start + num_blocks_per_task):
                if block_id in used_blocks:
                    overlap = True
                    break

            if not overlap:
                selected_starts.append(start)
                # 标记这些块为已使用
                for block_id in range(start, start + num_blocks_per_task):
                    used_blocks.add(block_id)

        # 收集所有 block_id
        all_block_ids = []
        for start in selected_starts:
            for i in range(num_blocks_per_task):
                all_block_ids.append(start + i)

        # 打乱顺序
        random.shuffle(all_block_ids)

        # 转换成 [[block_id, block_id], ...] 格式
        block_mapping_list = [[block_id, block_id] for block_id in all_block_ids]

        return torch.tensor(block_mapping_list, dtype=torch.int64, device="cpu")

    dtype = torch.bfloat16
    device = "cuda:0"

    gpu_cache = torch.randn(
        2, num_total_blocks, block_size, hidden_size, dtype=dtype, device=device
    )
    cpu_cache = torch.randn(
        2,
        num_total_blocks,
        block_size,
        hidden_size,
        dtype=dtype,
        device="cpu",
        pin_memory=True,
    )

    # ===== warm-up 阶段 =====
    warmup_iterations = 2  # 预热迭代次数
    print("开始 warm-up...")
    for method in methods:
        for _ in range(warmup_iterations):
            if swap_in:
                swap_blocks(
                    cpu_cache,
                    gpu_cache,
                    __get_random_block_mapping_tensor(
                        num_total_blocks,
                        num_tasks=num_tasks,
                        num_blocks_per_task=num_contiguous_blocks_per_task,
                    ),
                    copy_method=method,
                    swap_func=custom_swap_func,
                )
            else:
                swap_blocks(
                    gpu_cache,
                    cpu_cache,
                    __get_random_block_mapping_tensor(
                        num_total_blocks,
                        num_tasks=num_tasks,
                        num_blocks_per_task=num_contiguous_blocks_per_task,
                    ),
                    copy_method=method,
                    swap_func=custom_swap_func,
                )
            torch.cuda.synchronize()
    print("Warm-up 完成，开始正式测试...\n")
    # ==============================

    for method in methods:
        for test_time in range(test_times):
            block_mapping = __get_random_block_mapping_tensor(
                num_total_blocks,
                num_tasks=num_tasks,
                num_blocks_per_task=num_contiguous_blocks_per_task,
            )
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            if profiler:
                profiler.start()
            
            with torch.profiler.record_function(f"SwapBlocks_{method}") if profiler else contextlib.nullcontext():
                if swap_in:
                    swap_blocks(
                        cpu_cache,
                        gpu_cache,
                        block_mapping,
                        copy_method=method,
                        swap_func=custom_swap_func,
                    )
                else:
                    swap_blocks(
                        gpu_cache,
                        cpu_cache,
                        block_mapping,
                        copy_method=method,
                        swap_func=custom_swap_func,
                    )
            
            if profiler:
                profiler.stop()
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            block_id_list = sorted(
                [
                    block_mapping.cpu().numpy().tolist()[i][0]
                    for i in range(block_mapping.size(0))
                ]
            )
            elapsed_time_ms = (end_time - start_time) * 1000
            transferred_data_size = (
                block_mapping.size(0) * block_size * hidden_size * 2 * 2
            )
            transferred_data_size_gb = transferred_data_size / 1024 / 1024 / 1024
            real_bandwith = transferred_data_size_gb / elapsed_time_ms * 1000
            print(
                f"{method} 传输数据大小: {transferred_data_size_gb:.3f} GB  执行时间: {elapsed_time_ms:.3f} ms  实际带宽: {real_bandwith:.3f} GB/s"
            )
            # print(f"block_ids: {block_id_list}")


if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument(
        "--copy-method",
        type=str,
        help="copy method",
        required=False,
        choices=[method.value for method in CopyMethod],
        default=CopyMethod.GATHER_SCATTER,
    )
    argparse.add_argument(
        "--block-size", type=int, help="block size", required=False, default=16
    )
    argparse.add_argument(
        "--hidden-size", type=int, help="hidden size", required=False, default=128
    )
    argparse.add_argument(
        "--num-tasks",
        type=int,
        help="number of copy tasks from CPU->GPU",
        required=False,
        default=1000,
    )

    argparse.add_argument(
        "--num-blocks-scale",
        type=int,
        help="测试时 block 数量相较于传输的 block 数量的倍数",
        required=False,
        default=10,
    )

    argparse.add_argument(
        "--num-contiguous-blocks-per-task",
        type=int,
        help="number of contiguous blocks per task",
        required=False,
        default=1,
    )
    argparse.add_argument(
        "--test-times",
        type=int,
        help="number of test times",
        required=False,
        default=3,
    )
    argparse.add_argument(
        "--warmup-iterations",
        type=int,
        help="warmup iterations",
        required=False,
        default=2,
    )
    argparse.add_argument(
        "--swap-in",
        type=lambda x: (str(x).lower() == 'true'),
        help="swap in or swap out",
        required=False,
        default=True,
    )
    argparse.add_argument(
        "--trace-log-dir", type=str, help="trace log dir", required=False
    )
    args = argparse.parse_args()
    copy_method = CopyMethod(args.copy_method)

    profiler = None

    if args.trace_log_dir:
        print(f"trace log dir: {args.trace_log_dir}")
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            with_stack=True,
            profile_memory=True,
            record_shapes=True,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                args.trace_log_dir, use_gzip=True
            ),
        )

    __test_accuracy(method=copy_method, swap_in=args.swap_in)

    __test_performance(
        num_total_blocks=args.num_tasks * args.num_contiguous_blocks_per_task * 10,
        block_size=args.block_size,
        hidden_size=args.hidden_size,
        num_tasks=args.num_tasks,
        num_contiguous_blocks_per_task=args.num_contiguous_blocks_per_task,
        test_times=args.test_times,
        warmup_iterations=args.warmup_iterations,
        swap_in=args.swap_in,
        methods=[copy_method],
        profiler=profiler,
    )
