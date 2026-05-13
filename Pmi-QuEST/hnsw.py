"""
HNSW (Hierarchical Navigable Small World) approximate nearest neighbour index.
"""

import numpy as np
import heapq
import random
from typing import List, Tuple, Dict, Set, Optional
from scipy.sparse import csr_matrix, issparse
import math


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance (1 - cosine similarity) between two dense vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 1.0
    return 1.0 - float(np.dot(a, b) / (norm_a * norm_b))


class HNSWIndex:
    """
    HNSW index for approximate nearest neighbour search.

    Implements the algorithm from Malkov & Yashunin (2018).
    Configured to match H-QuEST defaults:
        M = 16       (neighbours per node per layer)
        ef_construction = 150  (beam width during construction)
        ef_search = 50         (beam width during search)

    Internally converts sparse TF-IDF vectors to dense for distance
    computation. For very large corpora, a production system would
    use hnswlib directly — this implementation is research-grade.
    """

    def __init__(self,
                 M: int = 16,
                 ef_construction: int = 150,
                 ef_search: int = 50,
                 seed: int = 42):
        """
        Args:
            M: Maximum number of bidirectional links per node per layer.
               Higher M = better recall, more memory.
            ef_construction: Size of dynamic candidate list during construction.
               Higher = better graph quality, slower build.
            ef_search: Size of dynamic candidate list during search.
               Higher = better recall, slower query.
            seed: Random seed for level generation.
        """
        self.M = M
        self.M_max0 = M * 2   # max links at layer 0 (denser bottom layer)
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.ml = 1.0 / np.log(M)  # level multiplier

        self._rng = random.Random(seed)
        self._vectors: List[np.ndarray] = []    # dense vectors, indexed by id
        self._graph: Dict[int, Dict[int, List[int]]] = {}  # node -> layer -> neighbours
        self._entry_point: Optional[int] = None
        self._max_layer: int = 0
        self.n_items: int = 0



    def add_items(self, matrix) -> None:
        """
        Add all vectors from a matrix to the index.

        Args:
            matrix: Dense numpy array or sparse csr_matrix,
                     shape (n_items, dim).
        """
        if issparse(matrix):
            matrix = matrix.toarray()

        n = matrix.shape[0]
        print(f"[HNSW] Building index for {n} vectors, M={self.M}, "
              f"ef_construction={self.ef_construction}")

        for i in range(n):
            self._insert(i, matrix[i])
            if i % 500 == 0 and i > 0:
                print(f"[HNSW] Inserted {i}/{n} vectors...")

        print(f"[HNSW] Index built. Max layer = {self._max_layer}, "
              f"entry point = {self._entry_point}")

    def _insert(self, idx: int, vec: np.ndarray) -> None:
        """Insert a single vector into the HNSW graph."""
        self._vectors.append(vec.astype(np.float32))
        level = self._random_level()
        self._graph[idx] = {lc: [] for lc in range(level + 1)}
        self.n_items += 1

        if self._entry_point is None:
            self._entry_point = idx
            self._max_layer = level
            return

        ep = self._entry_point
        # Greedy descent from top layer down to level+1
        for lc in range(self._max_layer, level, -1):
            ep = self._greedy_search(vec, ep, lc)

        # Insert with ef_construction beam from level down to 0
        for lc in range(min(level, self._max_layer), -1, -1):
            candidates = self._beam_search(vec, ep, lc, self.ef_construction)
            M_lc = self.M_max0 if lc == 0 else self.M
            neighbours = self._select_neighbours(vec, candidates, M_lc)

            self._graph[idx][lc] = neighbours
            # Add bidirectional links
            for nb in neighbours:
                if lc not in self._graph[nb]:
                    self._graph[nb][lc] = []
                self._graph[nb][lc].append(idx)
                # Prune if over limit
                if len(self._graph[nb][lc]) > M_lc:
                    nb_vec = self._vectors[nb]
                    self._graph[nb][lc] = self._select_neighbours(
                        nb_vec,
                        [(self._dist(nb_vec, self._vectors[x]), x)
                         for x in self._graph[nb][lc]],
                        M_lc
                    )
            if candidates:
                ep = candidates[0][1]  # closest found becomes new ep

        if level > self._max_layer:
            self._max_layer = level
            self._entry_point = idx


    def search(self,
               query: np.ndarray,
               k: int = 50) -> List[Tuple[float, int]]:
        """
        Find k approximate nearest neighbours for a query vector.

        Args:
            query: Dense query vector of shape (dim,).
            k: Number of results to return.

        Returns:
            List of (distance, index) tuples, sorted by distance ascending.
        """
        if self._entry_point is None:
            return []

        ep = self._entry_point
        # Greedy descent to layer 1
        for lc in range(self._max_layer, 0, -1):
            ep = self._greedy_search(query, ep, lc)

        # Beam search at layer 0 with ef_search
        candidates = self._beam_search(query, ep, 0, max(self.ef_search, k))
        candidates.sort()
        return candidates[:k]



    def _greedy_search(self, query: np.ndarray, ep: int, layer: int) -> int:
        """Single greedy step: move to closest neighbour at given layer."""
        best = ep
        best_dist = self._dist(query, self._vectors[ep])

        changed = True
        while changed:
            changed = False
            for nb in self._graph[best].get(layer, []):
                d = self._dist(query, self._vectors[nb])
                if d < best_dist:
                    best_dist = d
                    best = nb
                    changed = True
        return best

    def _beam_search(self,
                     query: np.ndarray,
                     ep: int,
                     layer: int,
                     ef: int) -> List[Tuple[float, int]]:
        """
        Beam search at a given layer, maintaining a candidate set of size ef.
        Returns list of (distance, index) pairs.
        """
        visited: Set[int] = {ep}
        d_ep = self._dist(query, self._vectors[ep])

        # candidates: min-heap by distance (closest first)
        candidates = [(d_ep, ep)]
        # results: max-heap by distance (furthest first, for pruning)
        results = [(-d_ep, ep)]

        while candidates:
            d_c, c = heapq.heappop(candidates)
            # Worst result distance
            d_worst = -results[0][0]
            if d_c > d_worst and len(results) >= ef:
                break

            for nb in self._graph[c].get(layer, []):
                if nb not in visited:
                    visited.add(nb)
                    d_nb = self._dist(query, self._vectors[nb])
                    d_worst = -results[0][0]

                    if d_nb < d_worst or len(results) < ef:
                        heapq.heappush(candidates, (d_nb, nb))
                        heapq.heappush(results, (-d_nb, nb))
                        if len(results) > ef:
                            heapq.heappop(results)

        return [(-d, idx) for d, idx in results]

    def _select_neighbours(self,
                           vec: np.ndarray,
                           candidates: List[Tuple[float, int]],
                           M: int) -> List[int]:
        """Select M closest neighbours from candidates."""
        candidates_sorted = sorted(candidates)[:M]
        return [idx for _, idx in candidates_sorted]

    def _dist(self, a: np.ndarray, b: np.ndarray) -> float:
        return _cosine_distance(a, b)

    def _random_level(self) -> int:
        """Sample layer level from geometric distribution."""
        return int(-math.log(self._rng.random()) * self.ml)


