# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to vLLM project
"""
Advanced cache policies for sparse KV offloading

This module implements advanced cache replacement policies:
1. LRU with Hot Score: Combines recency and query similarity
2. LayerWise LRU: Per-layer LRU to avoid inter-layer interference
3. Hot Score Cache: Pure hotness-based replacement
"""

import heapq
import math
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import BlockHash

logger = init_logger(__name__)


class AbstractCache(ABC):
    """Abstract base class for cache policies."""

    def __init__(self, capacity: int):
        self.capacity = capacity

    @abstractmethod
    def get(
        self, block_key: BlockHash, hot_score: float = 0.0, layer_idx: int = -1
    ) -> tuple[int, bool]:
        """
        Get a block, allocating if necessary.

        Args:
            block_key: Block hash
            hot_score: Hotness score for the block
            layer_idx: Optional layer index for layer-wise isolation

        Returns:
            Tuple of (slot_id, hit)
        """
        pass

    @abstractmethod
    def pin_block(self, block_key: BlockHash) -> bool:
        """Pin a block to prevent eviction."""
        pass

    @abstractmethod
    def unpin_block(self, block_key: BlockHash) -> bool:
        """Unpin a block to allow eviction."""
        pass


class CacheEntry:
    """Cache entry with hotness tracking."""

    def __init__(
        self,
        slot_id: int,
        hot_score: float,
        version: int,
        is_pinned: bool,
        timer: int,
    ):
        self.slot_id: int = slot_id
        self.hot_score: float = hot_score
        self.version: int = version
        self.is_pinned: bool = is_pinned
        self.timer: int = timer


class LRUWithHotCache(AbstractCache):
    """
    LRU cache with hotness score integration.

    Combines:
    - LRU recency tracking
    - Query similarity hotness scores
    - Sliding window for temporal decay
    """

    def __init__(
        self,
        capacity: int,
        decay_factor: float = 0.44,
        window_size: int = 110,
    ):
        """
        Initialize LRUWithHotCache.

        Args:
            capacity: Cache capacity in slots
            decay_factor: Hotness score decay factor (0-1)
            window_size: Sliding window size for temporal decay
        """
        super().__init__(capacity)
        self.capacity = capacity
        self.decay_factor = decay_factor
        self.window_size = window_size

        # block_key -> CacheEntry
        self.cache: dict[BlockHash, CacheEntry] = {}
        # slot_id -> block_key
        self.reverse_mapping: dict[int, BlockHash] = {}
        # Free slots
        self.free_slots: set[int] = set(range(capacity))

        # Min-heap: (hot_score, version, block_key, timer)
        self.heap: list[tuple[float, int, BlockHash, int]] = []
        self.global_version = 0
        self.timer = 0

    def _log_scale_hot_score(self, hot_score: float) -> float:
        """Apply log scaling to hotness score."""
        if hot_score == float('inf'):
            hot_score = 1e6
        elif hot_score == float('-inf'):
            hot_score = -1e6
        sign = 1 if hot_score >= 0 else -1
        return sign * math.tanh(math.log1p(abs(hot_score)))

    def get(
        self, block_key: BlockHash, hot_score: float = 0.0, layer_idx: int = -1
    ) -> tuple[int, bool]:
        """Get a block, updating hotness score."""
        self.timer += 1
        scaled_hot = self._log_scale_hot_score(hot_score)

        # Cache hit
        if block_key in self.cache:
            entry = self.cache[block_key]
            # Time weight based on sliding window
            time_weight = max(0, 1 - (self.timer - entry.timer) / self.window_size)
            # Update hotness: decay old score + add new score
            entry.hot_score = (
                entry.hot_score * self.decay_factor + 
                scaled_hot * (1 - self.decay_factor) * time_weight
            )

            self.global_version += 1
            entry.version = self.global_version
            entry.timer = self.timer

            heapq.heappush(self.heap, (entry.hot_score, self.global_version, block_key, self.timer))
            return entry.slot_id, True

        # Cache miss
        if len(self.cache) >= self.capacity:
            self._evict()

        # Allocate new slot
        slot_id = self._allocate_slot()
        self.global_version += 1
        entry = CacheEntry(slot_id, scaled_hot, self.global_version, is_pinned=False, timer=self.timer)

        self.cache[block_key] = entry
        self.reverse_mapping[slot_id] = block_key
        heapq.heappush(self.heap, (scaled_hot, self.global_version, block_key, self.timer))

        return slot_id, False

    def _evict(self):
        """Evict lowest hotness non-pinned block."""
        # Clean up heap
        self._cleanup_heap()

        # Evict using heap
        while self.heap:
            hot_score, version, block_key, timer = heapq.heappop(self.heap)

            if block_key not in self.cache:
                continue  # Lazy delete

            entry = self.cache[block_key]
            if entry.version != version or entry.is_pinned or timer == self.timer:
                continue  # Skip outdated or pinned blocks

            # Evict this block
            self.free_slots.add(entry.slot_id)
            del self.reverse_mapping[entry.slot_id]
            del self.cache[block_key]
            return

        raise RuntimeError("Cannot evict: all blocks are pinned")

    def _allocate_slot(self) -> int:
        """Allocate a free slot."""
        if not self.free_slots:
            raise RuntimeError("No free slots available")
        return self.free_slots.pop()

    def _cleanup_heap(self):
        """Clean up heap by removing invalid entries."""
        valid_entries = [
            (hs, v, k, t) for hs, v, k, t in self.heap
            if k in self.cache and self.cache[k].version == v
        ]
        self.heap = valid_entries
        heapq.heapify(self.heap)

    def pin_block(self, block_key: BlockHash) -> bool:
        """Pin a block to prevent eviction."""
        if block_key not in self.cache:
            return False
        self.cache[block_key].is_pinned = True
        return True

    def unpin_block(self, block_key: BlockHash) -> bool:
        """Unpin a block to allow eviction."""
        if block_key not in self.cache:
            return False
        self.cache[block_key].is_pinned = False
        return True


class LayeredLRUNode:
    """Doubly linked list node for layer-wise LRU."""

    def __init__(self, slot_id: int, timer: int, is_pinned: bool):
        self.slot_id: int = slot_id
        self.timer: int = timer
        self.is_pinned: bool = is_pinned
        self.prev: LayeredLRUNode | None = None
        self.next: LayeredLRUNode | None = None


class LayeredLRUList:
    """Doubly linked list for layer-wise LRU."""

    def __init__(self):
        self.head = LayeredLRUNode(-1, -1, True)
        self.tail = LayeredLRUNode(-1, -1, True)
        self.head.next = self.tail
        self.tail.prev = self.head
        self.size = 0

    def add_to_head(self, node: LayeredLRUNode):
        """Add node to head of list (most recently used)."""
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node
        self.size += 1

    def remove_node(self, node: LayeredLRUNode):
        """Remove node from list."""
        node.prev.next = node.next
        node.next.prev = node.prev
        self.size -= 1

    def get_head(self) -> LayeredLRUNode | None:
        """Get head node (most recently used)."""
        if self.size == 0:
            return None
        return self.head.next

    def get_tail(self) -> LayeredLRUNode | None:
        """Get tail node (least recently used)."""
        if self.size == 0:
            return None
        return self.tail.prev

    def move_to_head(self, node: LayeredLRUNode):
        """Move node to head of list."""
        self.remove_node(node)
        self.add_to_head(node)


class LayerWiseLRUCache(AbstractCache):
    """
    Layer-wise LRU cache to avoid inter-layer interference.

    Each layer maintains its own LRU list, preventing
    one layer's blocks from evicting another layer's blocks.
    """

    def __init__(self, capacity: int, num_layers: int, num_cpu_blocks: int):
        """
        Initialize LayerWiseLRUCache.

        Args:
            capacity: Total cache capacity
            num_layers: Number of transformer layers
            num_cpu_blocks: Number of CPU blocks per layer
        """
        super().__init__(capacity)
        self.num_layers = num_layers
        self.num_cpu_blocks = num_cpu_blocks

        # Per-layer cache: [layer_idx][cpu_block_id] -> LayeredLRUNode
        self.cache_mapping_per_layer: List[List[LayeredLRUNode]] = [
            [LayeredLRUNode(-1, -1, False) for _ in range(num_cpu_blocks)]
            for _ in range(num_layers)
        ]

        # Per-layer LRU lists
        self.layer_lists: List[LayeredLRUList] = [
            LayeredLRUList() for _ in range(num_layers)
        ]

        # Free slots (shared across layers)
        self.free_slots: List[int] = list(range(capacity))

        # Round-robin eviction starting point
        self.evict_layer_idx = 0

        # Timer for tracking
        self.timer = 0

        # Mapping from block_hash to (layer_idx, cpu_block_id)
        # Note: BlockHash is a bytes hash value that doesn't contain original layer info.
        # We use round-robin assignment to distribute blocks across layers for isolation.
        # This provides deterministic mapping based on block_key hash value.
        # This is NOT true layer-wise isolation based on transformer layer index.
        # For true layer-wise isolation, the caller would need to pass layer_idx.
        self.block_key_mapping: Dict[BlockHash, Tuple[int, int]] = {}
        self.next_cpu_block_id: List[int] = [0] * num_layers

    def get(
        self, block_key: BlockHash, layer_idx: int = -1, hot_score: float = 0.0
    ) -> tuple[int, bool]:
        """
        Get a block from layer-wise cache.

        Args:
            block_key: Block hash
            layer_idx: Transformer layer index (optional, for true layer-wise isolation)
            hot_score: Hotness score (not used in pure LRU)

        Returns:
            Tuple of (slot_id, hit)
        """
        self.timer += 1

        # Check if block_key is already mapped
        if block_key in self.block_key_mapping:
            mapped_layer_idx, cpu_block_id = self.block_key_mapping[block_key]
            # If layer_idx is provided and differs from mapped layer, treat as miss
            if layer_idx >= 0 and mapped_layer_idx != layer_idx:
                return self._allocate_new_block(block_key, layer_idx)
            layer_idx = mapped_layer_idx
        else:
            # Assign new layer_idx and cpu_block_id for this block_key
            if layer_idx >= 0:
                # Use provided layer_idx for true layer-wise isolation
                pass
            else:
                # Fallback: round-robin assignment (not true layer-wise isolation)
                layer_idx = self.evict_layer_idx % self.num_layers
                self.evict_layer_idx += 1

            cpu_block_id = self.next_cpu_block_id[layer_idx] % self.num_cpu_blocks

            # Update mapping and next allocation
            self.block_key_mapping[block_key] = (layer_idx, cpu_block_id)
            self.next_cpu_block_id[layer_idx] = cpu_block_id + 1

        layer_cache_mapping = self.cache_mapping_per_layer[layer_idx]
        layer_list = self.layer_lists[layer_idx]
        node = layer_cache_mapping[cpu_block_id]

        # Cache hit
        if node.slot_id >= 0:
            node.timer = self.timer
            layer_list.move_to_head(node)
            return node.slot_id, True

        # Cache miss
        if len(self.free_slots) == 0:
            self._evict(layer_idx)

        # Allocate new slot
        slot_id = self._allocate_slot()
        node.slot_id = slot_id
        node.timer = self.timer
        node.is_pinned = False
        layer_list.add_to_head(node)

        return slot_id, False

    def _allocate_new_block(
        self, block_key: BlockHash, layer_idx: int
    ) -> tuple[int, bool]:
        """Allocate a new block entry for the given layer."""
        cpu_block_id = self.next_cpu_block_id[layer_idx] % self.num_cpu_blocks

        # Update mapping and next allocation
        self.block_key_mapping[block_key] = (layer_idx, cpu_block_id)
        self.next_cpu_block_id[layer_idx] = cpu_block_id + 1

        layer_cache_mapping = self.cache_mapping_per_layer[layer_idx]
        layer_list = self.layer_lists[layer_idx]
        node = layer_cache_mapping[cpu_block_id]

        # Cache miss
        if len(self.free_slots) == 0:
            self._evict(layer_idx)

        # Allocate new slot
        slot_id = self._allocate_slot()
        node.slot_id = slot_id
        node.timer = self.timer
        node.is_pinned = False
        layer_list.add_to_head(node)

        return slot_id, False

    def _evict(self, layer_idx: int):
        """Evict a block, preferring other layers."""
        # Calculate layer distance for round-robin
        max_size = max(l.size for l in self.layer_lists)
        _, layer_distance = max(
            (l.size, (idx - layer_idx) % self.num_layers)
            for idx, l in enumerate(self.layer_lists)
        )

        # Prefer evicting from different layer
        evict_layer_idx = (layer_idx + layer_distance) % self.num_layers
        if evict_layer_idx == layer_idx:
            evict_layer_idx = (evict_layer_idx - 1) % self.num_layers

        # Find evictable block
        num_layers_checked = 0
        while True:
            last_node = self.layer_lists[evict_layer_idx].get_tail()
            while last_node is not None and (
                last_node.timer == self.timer or last_node.is_pinned
            ):
                last_node = last_node.prev
                if last_node is None:
                    break

            if last_node is not None:
                # Evict this block
                self.free_slots.append(last_node.slot_id)
                last_node.slot_id = -1
                last_node.timer = -1
                self.layer_lists[evict_layer_idx].remove_node(last_node)
                return

            # Try next layer
            evict_layer_idx = (evict_layer_idx - 1) % self.num_layers
            num_layers_checked += 1
            if num_layers_checked >= self.num_layers:
                raise RuntimeError("Cannot evict: all blocks are pinned")

    def _allocate_slot(self) -> int:
        """Allocate a free slot."""
        if not self.free_slots:
            raise RuntimeError("No free slots available")
        return self.free_slots.pop()

    def pin_block(self, block_key: BlockHash) -> bool:
        """Pin a block to prevent eviction."""
        if block_key not in self.block_key_mapping:
            return False
        layer_idx, cpu_block_id = self.block_key_mapping[block_key]
        node = self.cache_mapping_per_layer[layer_idx][cpu_block_id]
        if node.slot_id == -1:
            return False
        node.is_pinned = True
        return True

    def unpin_block(self, block_key: BlockHash) -> bool:
        """Unpin a block to allow eviction."""
        if block_key not in self.block_key_mapping:
            return False
        layer_idx, cpu_block_id = self.block_key_mapping[block_key]
        node = self.cache_mapping_per_layer[layer_idx][cpu_block_id]
        if node.slot_id == -1:
            return False
        node.is_pinned = False
        return True


class HotScoreEntry:
    """Hot score cache entry."""

    def __init__(
        self,
        slot_id: int,
        hot_score: float,
        version: int,
        is_pinned: bool,
        timer: int,
    ):
        self.slot_id: int = slot_id
        self.hot_score: float = hot_score
        self.version: int = version
        self.is_pinned: bool = is_pinned
        self.timer: int = timer


class HotScoreCache(AbstractCache):
    """
    Pure hotness-based cache replacement.

    Uses minimum heap to track block hotness scores.
    """

    def __init__(self, capacity: int, decay_factor: float = 0.2):
        """
        Initialize HotScoreCache.

        Args:
            capacity: Cache capacity
            decay_factor: Hotness score decay factor
        """
        super().__init__(capacity)
        self.capacity = capacity
        self.decay_factor = decay_factor

        # block_key -> HotScoreEntry
        self.cache: dict[BlockHash, HotScoreEntry] = {}
        self.free_slots: set[int] = set(range(capacity))

        # Min-heap: (hot_score, version, block_key, timer)
        self.heap: list[tuple[float, int, BlockHash, int]] = []
        self.global_version = 0
        self.timer = 0

    def get(
        self, block_key: BlockHash, hot_score: float = 0.0, layer_idx: int = -1
    ) -> tuple[int, bool]:
        """Get a block, updating hotness score."""
        self.timer += 1
        # Cache hit
        if block_key in self.cache:
            entry = self.cache[block_key]
            # Decay old score and add new score
            new_hot_score = entry.hot_score * self.decay_factor + hot_score

            self.global_version += 1
            entry.hot_score = new_hot_score
            entry.version = self.global_version

            heapq.heappush(self.heap, (new_hot_score, self.global_version, block_key, self.timer))
            return entry.slot_id, True

        # Cache miss
        if len(self.cache) >= self.capacity:
            self._evict_by_heap()

        # Allocate new slot
        slot_id = self._allocate_slot()
        self.global_version += 1
        entry = HotScoreEntry(slot_id, hot_score, self.global_version, is_pinned=False, timer=self.timer)

        self.cache[block_key] = entry
        heapq.heappush(self.heap, (hot_score, self.global_version, block_key, self.timer))

        return slot_id, False

    def _evict_by_heap(self):
        """Evict lowest hotness non-pinned block."""
        while self.heap:
            hot_score, version, block_key, timer = heapq.heappop(self.heap)

            if block_key not in self.cache:
                continue  # Lazy delete

            entry = self.cache[block_key]
            if entry.version != version or entry.is_pinned or timer == self.timer:
                continue  # Skip outdated or pinned blocks

            # Evict this block
            self.free_slots.add(entry.slot_id)
            del self.cache[block_key]
            return

        raise RuntimeError("Cannot evict: all blocks are pinned")

    def _allocate_slot(self) -> int:
        """Allocate a free slot."""
        if not self.free_slots:
            raise RuntimeError("No free slots available")
        return self.free_slots.pop()

    def pin_block(self, block_key: BlockHash) -> bool:
        """Pin a block to prevent eviction."""
        if block_key not in self.cache:
            return False
        self.cache[block_key].is_pinned = True
        return True

    def unpin_block(self, block_key: BlockHash) -> bool:
        """Unpin a block to allow eviction."""
        if block_key not in self.cache:
            return False
        self.cache[block_key].is_pinned = False
        return True

    def cleanup_heap(self):
        """Clean up heap by removing invalid entries."""
        valid_entries = [
            (hs, v, k, t) for hs, v, k, t in self.heap
            if k in self.cache and self.cache[k].version == v
        ]
        self.heap = valid_entries
        heapq.heapify(self.heap)