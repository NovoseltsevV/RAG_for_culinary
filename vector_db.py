import numpy as np
from numpy import ndarray
from jaxtyping import Float
from collections import defaultdict
from typing import Any


class LSHdatabase:
    def __init__(self, k: int, L: int, dim: int) -> None:
        self.k = k
        self.L = L
        self.dim = dim
        self.hash_data = [defaultdict(list) for _ in range(self.L)]
        self.base_vectors = []
        for _ in range(self.L):
            base_vectors = np.random.randn(self.k, self.dim)
            norms = np.linalg.norm(base_vectors, axis=1)
            base_vectors = base_vectors / norms[:, None]
            self.base_vectors.append(base_vectors)
        self.vectors = {}
        self.info = defaultdict(dict)

    def get_hash(self, vector: Float[ndarray, "dim"]) -> list[tuple]:
        hashes = []
        for i in range(self.L):
            base_vectors = self.base_vectors[i]
            scores = (base_vectors @ vector >= 0)
            hash = tuple(scores.astype(int))
            hashes.append(hash)
        return hashes

    def add_vector(
        self, vector: Float[ndarray, "dim"], vector_id: Any
    ) -> None:
        self.vectors[vector_id] = vector
        hashes = self.get_hash(vector)
        for i in range(self.L):
            self.hash_data[i][hashes[i]].append(vector_id)
    
    def add_info(self, info: dict, vector_id: Any) -> None:
        self.info[vector_id] = info

    def cosine_similarity(
        self,
        a: Float[ndarray, "dim"],
        b: Float[ndarray, "dim"]
    ) -> float:
        similarity = np.dot(a, b)
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        return similarity / (a_norm * b_norm)

    def find_nearest(
        self,
        query_vector: Float[ndarray, "dim"],
        k_nearest: int = 10
    ) -> list:
        nearest_vectors_ids = []
        query_hashes = self.get_hash(query_vector)
        for i in range(self.L):
            nearest_vectors_ids.extend(self.hash_data[i][query_hashes[i]])
        nearest_vectors_ids = set(nearest_vectors_ids)
        similarities = []
        for id in nearest_vectors_ids:
            value = self.cosine_similarity(query_vector, self.vectors[id])
            similarities.append((id, value))
        similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
        output = similarities[:k_nearest]
        if self.info:
            output = [(item[0], item[1], self.info[item[0]]) for item in output]
        return output
