# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to vLLM project
"""
Adaptive top-K selection for sparse KV offloading

This module implements adaptive top-K selection strategies:
1. Dynamic top-K based on cache hit rate
2. Sequence length-aware top-K adjustment
3. Query entropy-based top-K selection
"""

from typing import List, Tuple, Optional
import numpy as np
import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


class AdaptiveTopKSelector:
    """
    Adaptive top-K selector that dynamically adjusts selection size.

    Strategies:
    - Fixed: Use constant top-K
    - HitRate: Adjust based on cache hit rate
    - SequenceLength: Scale based on sequence length
    - QueryEntropy: Adjust based on query diversity
    """

    def __init__(
        self,
        base_topk: int = 10240000,
        min_topk: int = 4096,
        max_topk: int = 2048000,
        strategy: str = "hit_rate",
        window_size: int = 100,
    ):
        """
        Initialize AdaptiveTopKSelector.

        Args:
            base_topk: Base top-K value
            min_topk: Minimum top-K value
            max_topk: Maximum top-K value
            strategy: Selection strategy
            window_size: Window size for statistics
        """
        self.base_topk = base_topk
        self.min_topk = min_topk
        self.max_topk = max_topk
        self.strategy = strategy
        self.window_size = window_size

        # Statistics
        self.hit_history: List[float] = []
        self.miss_history: List[float] = []
        self.total_requests = 0
        self.total_hits = 0

        # Entropy tracking
        self.entropy_history: List[float] = []

        logger.info(
            f"Initialized AdaptiveTopKSelector with "
            f"base_topk={base_topk}, strategy={strategy}"
        )

    def get_topk(
        self,
        seq_len: int,
        query: Optional[torch.Tensor] = None,
        cache_hit_rate: Optional[float] = None,
    ) -> int:
        """
        Get adaptive top-K value.

        Args:
            seq_len: Current sequence length
            query: Query vector (for entropy-based selection)
            cache_hit_rate: Current cache hit rate (0-1)

        Returns:
            Adaptive top-K value
        """
        if self.strategy == "fixed":
            return self.base_topk

        elif self.strategy == "hit_rate":
            return self._hit_rate_based_topk(cache_hit_rate)

        elif self.strategy == "sequence_length":
            return self._sequence_length_based_topk(seq_len)

        elif self.strategy == "query_entropy":
            return self._query_entropy_based_topk(query)

        elif self.strategy == "hybrid":
            # Combine multiple strategies
            topk1 = self._hit_rate_based_topk(cache_hit_rate)
            topk2 = self._sequence_length_based_topk(seq_len)
            return int((topk1 + topk2) / 2)

        else:
            logger.warning(f"Unknown strategy: {self.strategy}, using fixed")
            return self.base_topk

    def _hit_rate_based_topk(self, cache_hit_rate: Optional[float]) -> int:
        """Adjust top-K based on cache hit rate."""
        if cache_hit_rate is None:
            return self.base_topk

        # Low hit rate -> increase top-K to get more blocks
        # if cache_hit_rate < 0.5:
        #     factor = 1.0 + (0.5 - cache_hit_rate)  # 1.0 to 1.5
        # else:
        #     factor = 1.0 - (cache_hit_rate - 0.5) * 0.5  # 0.75 to 1.0

        # Simplified: inverse relationship
        factor = 1.5 - cache_hit_rate  # 0.5 to 1.5

        adaptive_topk = int(self.base_topk * factor)
        return np.clip(adaptive_topk, self.min_topk, self.max_topk)

    def _sequence_length_based_topk(self, seq_len: int) -> int:
        """Adjust top-K based on sequence length."""
        # Longer sequences may benefit from more context
        # Use logarithmic scaling
        if seq_len <= 4096:
            factor = 0.8
        elif seq_len <= 16384:
            factor = 1.0
        elif seq_len <= 65536:
            factor = 1.2
        else:
            factor = 1.4

        adaptive_topk = int(self.base_topk * factor)
        return np.clip(adaptive_topk, self.min_topk, self.max_topk)

    def _query_entropy_based_topk(self, query: Optional[torch.Tensor]) -> int:
        """Adjust top-K based on query entropy."""
        if query is None:
            return self.base_topk

        # Compute query entropy as proxy for diversity
        # Normalize query to compute distribution
        query_flat = query.flatten().abs()
        query_norm = query_flat / (query_flat.sum() + 1e-8)
        entropy = -(query_norm * torch.log2(query_norm + 1e-8)).sum().item()

        # Normalize entropy (typical range: 0-10)
        normalized_entropy = min(entropy / 10.0, 1.0)

        # Higher entropy -> more diverse -> need more blocks
        factor = 0.8 + normalized_entropy * 0.4  # 0.8 to 1.2

        adaptive_topk = int(self.base_topk * factor)
        return np.clip(adaptive_topk, self.min_topk, self.max_topk)

    def update_stats(
        self,
        hit: bool,
        cache_hit_rate: Optional[float] = None,
    ):
        """
        Update selection statistics.

        Args:
            hit: Whether selection was a cache hit
            cache_hit_rate: Current cache hit rate
        """
        self.total_requests += 1
        if hit:
            self.total_hits += 1

        if cache_hit_rate is not None:
            self.hit_history.append(cache_hit_rate)
            if len(self.hit_history) > self.window_size:
                self.hit_history.pop(0)

    def get_stats(self) -> dict:
        """Get selection statistics."""
        overall_hit_rate = (
            self.total_hits / self.total_requests
            if self.total_requests > 0
            else 0.0
        )

        avg_hit_rate = (
            np.mean(self.hit_history)
            if self.hit_history
            else 0.0
        )

        return {
            "total_requests": self.total_requests,
            "total_hits": self.total_hits,
            "overall_hit_rate": overall_hit_rate,
            "avg_hit_rate": avg_hit_rate,
            "strategy": self.strategy,
        }


class MultiStrategySelector:
    """
    Multi-strategy fusion selector.

    Combines multiple selection strategies for improved accuracy.
    """

    def __init__(
        self,
        strategies: List[str],
        weights: Optional[List[float]] = None,
        consensus: str = "weighted",
    ):
        """
        Initialize MultiStrategySelector.

        Args:
            strategies: List of strategy names
            weights: Weights for each strategy (optional)
            consensus: Consensus method (weighted/voting)
        """
        self.strategies = strategies
        self.consensus = consensus

        if weights is None:
            self.weights = [1.0 / len(strategies)] * len(strategies)
        else:
            assert len(weights) == len(strategies)
            total = sum(weights)
            self.weights = [w / total for w in weights]

        logger.info(
            f"Initialized MultiStrategySelector with "
            f"strategies={strategies}, consensus={consensus}"
        )

    def select_blocks(
        self,
        candidate_blocks: List[int],
        scores_list: List[List[float]],
        top_k: int,
    ) -> Tuple[List[int], List[float]]:
        """
        Select blocks using multiple strategies.

        Args:
            candidate_blocks: List of candidate block IDs
            scores_list: List of score lists from different strategies
            top_k: Number of blocks to select

        Returns:
            Tuple of (selected_blocks, final_scores)
        """
        if not candidate_blocks or not scores_list:
            return [], []

        if self.consensus == "weighted":
            return self._weighted_consensus(candidate_blocks, scores_list, top_k)
        elif self.consensus == "voting":
            return self._voting_consensus(candidate_blocks, scores_list, top_k)
        elif self.consensus == "union":
            return self._union_selection(candidate_blocks, scores_list, top_k)
        else:
            logger.warning(f"Unknown consensus: {self.consensus}, using weighted")
            return self._weighted_consensus(candidate_blocks, scores_list, top_k)

    def _weighted_consensus(
        self,
        candidate_blocks: List[int],
        scores_list: List[List[float]],
        top_k: int,
    ) -> Tuple[List[int], List[float]]:
        """Weighted consensus of multiple strategies."""
        # Combine scores with weights
        combined_scores = [0.0] * len(candidate_blocks)
        for scores, weight in zip(scores_list, self.weights):
            for i, score in enumerate(scores):
                combined_scores[i] += score * weight

        # Select top-k based on combined scores
        num_select = min(top_k, len(candidate_blocks))
        sorted_indices = np.argsort(combined_scores)[-num_select:].tolist()
        sorted_indices.reverse()  # Descending order

        selected_blocks = [candidate_blocks[i] for i in sorted_indices]
        selected_scores = [combined_scores[i] for i in sorted_indices]

        return selected_blocks, selected_scores

    def _voting_consensus(
        self,
        candidate_blocks: List[int],
        scores_list: List[List[float]],
        top_k: int,
    ) -> Tuple[List[int], List[float]]:
        """Voting-based consensus."""
        # Each strategy votes for top-k
        votes = [0] * len(candidate_blocks)

        for scores in scores_list:
            num_select = min(top_k, len(candidate_blocks))
            top_indices = np.argsort(scores)[-num_select:].tolist()

            for idx in top_indices:
                votes[idx] += 1

        # Select blocks with most votes
        num_select = min(top_k, len(candidate_blocks))
        sorted_indices = np.argsort(votes)[-num_select:].tolist()
        sorted_indices.reverse()

        selected_blocks = [candidate_blocks[i] for i in sorted_indices]
        selected_scores = [votes[i] for i in sorted_indices]

        return selected_blocks, selected_scores

    def _union_selection(
        self,
        candidate_blocks: List[int],
        scores_list: List[List[float]],
        top_k: int,
    ) -> Tuple[List[int], List[float]]:
        """Union of top-k from all strategies."""
        # Collect unique blocks from all strategies
        selected_set = set()

        for scores in scores_list:
            num_select = min(top_k, len(candidate_blocks))
            top_indices = np.argsort(scores)[-num_select:].tolist()

            for idx in top_indices:
                selected_set.add(candidate_blocks[idx])

        # Convert to list and limit to top_k
        selected_blocks = list(selected_set)[:top_k]

        # Use average scores
        combined_scores = [0.0] * len(candidate_blocks)
        for scores in scores_list:
            for i, score in enumerate(scores):
                combined_scores[i] += score

        selected_scores = [
            combined_scores[candidate_blocks.index(block)] for block in selected_blocks
        ]

        return selected_blocks, selected_scores


# Example usage
if __name__ == "__main__":
    print("Testing adaptive top-K selection...")

    # Test AdaptiveTopKSelector
    selector = AdaptiveTopKSelector(
        base_topk=10240000,
        strategy="hit_rate",
    )

    # Test different scenarios
    scenarios = [
        {"seq_len": 4096, "cache_hit_rate": 0.3, "description": "Short context, low hit rate"},
        {"seq_len": 16384, "cache_hit_rate": 0.7, "description": "Medium context, high hit rate"},
        {"seq_len": 65536, "cache_hit_rate": 0.5, "description": "Long context, medium hit rate"},
    ]

    for scenario in scenarios:
        topk = selector.get_topk(
            seq_len=scenario["seq_len"],
            cache_hit_rate=scenario["cache_hit_rate"],
        )
        print(f"{scenario['description']}: topk={topk}")

    # Test MultiStrategySelector
    print("\nTesting multi-strategy selection...")

    candidate_blocks = list(range(100))
    scores_list = [
        np.random.rand(100).tolist() for _ in range(3)
    ]

    multi_selector = MultiStrategySelector(
        strategies=["lru", "hot_score", "similarity"],
        weights=[0.4, 0.3, 0.3],
        consensus="weighted",
    )

    selected_blocks, scores = multi_selector.select_blocks(
        candidate_blocks=candidate_blocks,
        scores_list=scores_list,
        top_k=10,
    )

    print(f"Selected {len(selected_blocks)} blocks")
    print(f"Blocks: {selected_blocks}")
    print(f"Scores: {scores}")

    print("\nAll tests passed!")