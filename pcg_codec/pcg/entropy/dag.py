from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Dag:
    layers: list[list[int]]  # L1..LD
    parents: dict[int, list[int]]  # block -> parent blocks

    @property
    def depth(self) -> int:
        return len(self.layers)


def build_dag_layers(blocks: int, depth: int, parents_k: int, seed: int) -> Dag:
    blocks = int(blocks)
    depth = int(depth)
    parents_k = int(parents_k)
    if blocks <= 0:
        raise ValueError("blocks must be positive")
    if depth <= 0:
        raise ValueError("depth must be positive")
    if depth > blocks:
        raise ValueError("depth cannot exceed blocks")
    rng = np.random.default_rng(int(seed))

    # Deterministic partition: round-robin assignment for near-equal layer sizes.
    layer_ids = [[] for _ in range(depth)]
    for b in range(blocks):
        layer_ids[b % depth].append(b)

    parents: dict[int, list[int]] = {}
    if depth == 1:
        for b in range(blocks):
            parents[b] = []
        return Dag(layers=layer_ids, parents=parents)

    # Layered bipartite: each block in layer ℓ depends on up to k parents from earlier layers.
    earlier: list[int] = []
    for layer_index, layer in enumerate(layer_ids):
        if layer_index == 0:
            for b in layer:
                parents[b] = []
            earlier.extend(layer)
            continue
        for b in layer:
            if len(earlier) == 0:
                parents[b] = []
                continue
            k = min(parents_k, len(earlier))
            pa = rng.choice(np.array(earlier, dtype=np.int64), size=k, replace=False).tolist()
            parents[b] = sorted(int(x) for x in pa)
        earlier.extend(layer)
    return Dag(layers=layer_ids, parents=parents)


def topological_depth(dag: Dag) -> int:
    # By construction, layers define a valid topological layering.
    return dag.depth

