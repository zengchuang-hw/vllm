#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector_types.h>

// Helper for PTX optimized load/store
// Uses ld.global.nc (Non-Coherent) to bypass L1 cache for loads
// Uses st.global.cg (Cache-Global) to bypass L1 cache for stores
__device__ __forceinline__ void copy_int4_opt(const int4* __restrict__ src, int4* __restrict__ dst) {
    int4 val;
    // Load 128-bit data (4 x 32-bit) using non-coherent cache modifier
    asm volatile("ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%4];"
        : "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w)
        : "l"(src)
        : "memory");
        
    // Store 128-bit data using cache-global modifier
    asm volatile("st.global.cg.v4.u32 [%0], {%1, %2, %3, %4};"
        : 
        : "l"(dst), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w)
        : "memory");
}

// CUDA Kernel: Gather-Scatter Copy
// Optimized for small block transfers by fusing multiple copies into one kernel launch.
// Only Supports CPU (Pinned) <-> GPU.
__global__ void gather_scatter_kernel(
    const char* __restrict__ src_key_base,
    const char* __restrict__ src_value_base,
    char* __restrict__ dst_key_base,
    char* __restrict__ dst_value_base,
    const int64_t* __restrict__ block_mapping,
    const int block_size_in_bytes,
    const int num_blocks) {

    // Each CUDA block handles one data block copy
    const int block_idx = blockIdx.x;
    if (block_idx >= num_blocks) return;

    // Load mapping: [src_block_index, dst_block_index]
    const int64_t src_idx = block_mapping[block_idx * 2];
    const int64_t dst_idx = block_mapping[block_idx * 2 + 1];

    // Calculate pointers
    const char* src_key_ptr = src_key_base + src_idx * block_size_in_bytes;
    char* dst_key_ptr = dst_key_base + dst_idx * block_size_in_bytes;

    const char* src_value_ptr = src_value_base + src_idx * block_size_in_bytes;
    char* dst_value_ptr = dst_value_base + dst_idx * block_size_in_bytes;

    // Vectorized copy using int4 (16 bytes per thread per iter)
    // Reinterpret pointers as int4
    const int4* src_key_vec = reinterpret_cast<const int4*>(src_key_ptr);
    int4* dst_key_vec = reinterpret_cast<int4*>(dst_key_ptr);

    const int4* src_value_vec = reinterpret_cast<const int4*>(src_value_ptr);
    int4* dst_value_vec = reinterpret_cast<int4*>(dst_value_ptr);

    // Number of int4 vectors per block (for one cache type, e.g. Key or Value)
    // Assumption: block_size_in_bytes is a multiple of 16.
    const int num_vecs = block_size_in_bytes / sizeof(int4);
    
    // Total vectors to copy = Key vectors + Value vectors
    const int total_vecs = num_vecs * 2;

    // Grid-stride loop (within the block)
    // We flatten the Key and Value copy tasks into a single loop space [0, total_vecs)
    // Range [0, num_vecs) -> Key copy
    // Range [num_vecs, total_vecs) -> Value copy
    // This allows us to use more threads (e.g. 512) to parallelize Key and Value copies
    // without warp divergence (since num_vecs is typically a multiple of 32).
    for (int i = threadIdx.x; i < total_vecs; i += blockDim.x) {
        if (i < num_vecs) {
            copy_int4_opt(&src_key_vec[i], &dst_key_vec[i]);
        } else {
            int v_idx = i - num_vecs;
            copy_int4_opt(&src_value_vec[v_idx], &dst_value_vec[v_idx]);
        }
    }
    
    // Handle remaining bytes is skipped here as we assume block_size is aligned to 16 bytes
    // which is true for all standard vLLM configurations (16*128*2 = 4096 bytes).
}

// Optimized swap function using custom kernel
void swap_blocks_optimized(
    torch::Tensor& src_key,
    torch::Tensor& src_value,
    torch::Tensor& dst_key,
    torch::Tensor& dst_value,
    const torch::Tensor& block_mapping) {

    // Device checks
    torch::Device src_device = src_key.device();
    torch::Device dst_device = dst_key.device();
    
    // Determine stream (use dst device stream)
    torch::Device device_to_guard = dst_device.is_cuda() ? dst_device : src_device;
    at::cuda::OptionalCUDAGuard device_guard(device_to_guard);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Handle block_mapping
    // If block_mapping is on CPU, we MUST copy it to GPU for the kernel to access.
    // We create a temporary tensor on the same device as the stream we are using.
    torch::Tensor mapping_gpu;
    if (block_mapping.device().is_cpu()) {
        mapping_gpu = block_mapping.to(device_to_guard, /*non_blocking=*/true);
    } else {
        mapping_gpu = block_mapping;
    }

    const int64_t num_blocks = block_mapping.size(0);
    if (num_blocks == 0) return;
    
    const int64_t block_size_in_bytes = src_key.stride(0) * src_key.element_size();

    // Kernel Configuration
    // 1 Block per Task
    dim3 grid(num_blocks);
    // 512 threads per block to fully cover Key (256 vecs) + Value (256 vecs)
    // for standard block size (16 * 128 * 2 bytes = 4096 bytes per Key/Value block)
    dim3 block(512); 

    gather_scatter_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const char*>(src_key.data_ptr()),
        reinterpret_cast<const char*>(src_value.data_ptr()),
        reinterpret_cast<char*>(dst_key.data_ptr()),
        reinterpret_cast<char*>(dst_value.data_ptr()),
        reinterpret_cast<const int64_t*>(mapping_gpu.data_ptr()),
        static_cast<int>(block_size_in_bytes),
        static_cast<int>(num_blocks)
    );
    
    // Check for launch errors
    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
    // }
}

// 合并连续块的拷贝
size_t swap_contiguous_blocks(
    torch::Tensor& src,
    torch::Tensor& dst,
    const torch::Tensor& block_mapping) {

    // --- 设备和拷贝类型检查 (与原版相同) ---
    torch::Device src_device = src.device();
    torch::Device dst_device = dst.device();
    cudaMemcpyKind memcpy_type;
    if (src_device.is_cuda() && dst_device.is_cuda()) {
        TORCH_CHECK(src_device.index() == dst_device.index(),
                    "src and dst must be on the same GPU");
        memcpy_type = cudaMemcpyDeviceToDevice;
    } else if (src_device.is_cuda() && dst_device.is_cpu()) {
        memcpy_type = cudaMemcpyDeviceToHost;
    } else if (src_device.is_cpu() && dst_device.is_cuda()) {
        memcpy_type = cudaMemcpyHostToDevice;
    } else {
        TORCH_CHECK(false, "Invalid device combination: src or dst must be a CUDA tensor.");
    }

    // --- 格式校验 ---
    TORCH_CHECK(block_mapping.device().is_cpu(), "block_mapping must be on CPU");
    TORCH_CHECK(block_mapping.dim() == 2 && block_mapping.size(1) == 2,
                "block_mapping must be a 2D tensor with shape [N, 2]");

    // ===============================================================
    // =================== 自动分组逻辑开始 =========================
    // ===============================================================
    auto block_mapping_accessor = block_mapping.accessor<int64_t, 2>();

    // 1. 将 Tensor 数据复制到 std::vector<std::pair> 以便排序
    std::vector<std::pair<int64_t, int64_t>> mapping_vec;
    mapping_vec.reserve(block_mapping_accessor.size(0));
    for(int64_t i = 0; i < block_mapping_accessor.size(0); ++i) {
        mapping_vec.push_back({block_mapping_accessor[i][0], block_mapping_accessor[i][1]});
    }

    // 2. 使用 std::sort 按 src_block (pair.first) 排序
    std::sort(mapping_vec.begin(), mapping_vec.end(),
              [](const auto& a, const auto& b) {
        return a.first < b.first;
    });

    // 3. 遍历与合并，结果存入新的 vector 中
    // 使用 tuple 存储 {src_start, dst_start, count}
    std::vector<std::tuple<int64_t, int64_t, int64_t>> grouped_mappings;
    if (!mapping_vec.empty()) {
        auto& first_block = mapping_vec[0];
        grouped_mappings.emplace_back(first_block.first, first_block.second, 1);

        for (size_t i = 1; i < mapping_vec.size(); ++i) {
            auto& prev_block = mapping_vec[i-1];
            auto& current_block = mapping_vec[i];

            if (current_block.first == prev_block.first + 1 &&
                current_block.second == prev_block.second + 1) {
                // 如果连续，增加最后一个元素的计数
                std::get<2>(grouped_mappings.back())++;
            } else {
                // 如果不连续，添加新块
                grouped_mappings.emplace_back(current_block.first, current_block.second, 1);
            }
        }
    }
    // ===============================================================
    // =================== 自动分组逻辑结束 =========================
    // ===============================================================




    // --- 获取指针和计算 block 大小 (与原版相同) ---
    char* src_ptr = static_cast<char*>(src.data_ptr());
    char* dst_ptr = static_cast<char*>(dst.data_ptr());
    const int64_t block_size_in_bytes = src.stride(0) * src.element_size();

    // --- 确保 device 是 GPU tensor 所在的 device
    const c10::cuda::OptionalCUDAGuard device_guard(
        src_device.is_cuda() ? src_device : dst_device);
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    // --- 核心改进：遍历新的 grouped_mappings ---

    for (const auto& task : grouped_mappings) {
        const int64_t src_block_start = std::get<0>(task);
        const int64_t dst_block_start = std::get<1>(task);
        const int64_t num_contiguous_blocks = std::get<2>(task);

        const int64_t src_offset = src_block_start * block_size_in_bytes;
        const int64_t dst_offset = dst_block_start * block_size_in_bytes;
        const int64_t total_bytes_to_copy = num_contiguous_blocks * block_size_in_bytes;

        // 执行一次大的、合并后的拷贝
        cudaMemcpyAsync(dst_ptr + dst_offset, src_ptr + src_offset,
                        total_bytes_to_copy, memcpy_type, stream);
    }

    return grouped_mappings.size(); // 返回合并后的连续块数量
}

// 与 vLLM 原有实现相似, 但使用 accessor 替代 .item() 避免每次访问都进行 gpu-cpu 同步
size_t swap_blocks_raw(torch::Tensor& src, torch::Tensor& dst,
                 const torch::Tensor& block_mapping) {
  torch::Device src_device = src.device();
  torch::Device dst_device = dst.device();
  cudaMemcpyKind memcpy_type;
  if (src_device.is_cuda() && dst_device.is_cuda()) {
    TORCH_CHECK(src_device.index() == dst_device.index(),
                "src and dst must be on the same GPU");
    memcpy_type = cudaMemcpyDeviceToDevice;
  } else if (src_device.is_cuda() && dst_device.is_cpu()) {
    memcpy_type = cudaMemcpyDeviceToHost;
  } else if (src_device.is_cpu() && dst_device.is_cuda()) {
    memcpy_type = cudaMemcpyHostToDevice;
  } else {
    TORCH_CHECK(false, "Invalid device combination");
  }

  // NOTE(youkaichao): keep in mind that `block_mapping` should be
  // a cpu tensor, otherwise every `item` call will require a gpu-cpu
  // synchronization.
  TORCH_CHECK(block_mapping.device().is_cpu(), "block_mapping must be on CPU");

  char* src_ptr = static_cast<char*>(src.data_ptr());
  char* dst_ptr = static_cast<char*>(dst.data_ptr());

  // We use the stride instead of numel in case the cache is padded for memory
  // alignment reasons, we assume the blocks data (inclusive of any padding)
  // is contiguous in memory
  const int64_t block_size_in_bytes = src.element_size() * src.stride(0);
  const at::cuda::OptionalCUDAGuard device_guard(
      src_device.is_cuda() ? src_device : dst_device);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  // NOTE(woosuk): This can be slow if the number of blocks is large.
  // 使用 accessor 替代 .item()
  auto block_mapping_accessor = block_mapping.accessor<int64_t, 2>();
  const int64_t num_blocks = block_mapping.size(0);
  for (size_t i = 0; i < num_blocks; i++) {
    int64_t src_block_number = block_mapping_accessor[i][0];
    int64_t dst_block_number = block_mapping_accessor[i][1];
    int64_t src_offset = src_block_number * block_size_in_bytes;
    int64_t dst_offset = dst_block_number * block_size_in_bytes;
    cudaMemcpyAsync(dst_ptr + dst_offset, src_ptr + src_offset,
                    block_size_in_bytes, memcpy_type, stream);
  }
  return num_blocks;
}

// Python 绑定
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("swap_contiguous_blocks", &swap_contiguous_blocks, "Swap contiguous blocks (JIT compiled)");
  m.def("swap_blocks_raw", &swap_blocks_raw, "Swap blocks (raw version)");
  m.def("swap_blocks_optimized", &swap_blocks_optimized, "Swap blocks optimized (Gather-Scatter)");
}