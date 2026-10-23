"""Class-conditional remaining paths. Every sample carries intensity.

A path is drawn only from storms whose label at that age is the requested
class. The suffix is translated so it starts at the live storm's position.
Intensity and radius are the analog's, not a mean path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from model.config import CLASSES
from model.labels import label_at


@dataclass
class _Entry:
    obs: np.ndarray
    labels: np.ndarray  # class index at each row


class PathLibrary:
    def __init__(self, seasons: list, bin_seconds: int = 10):
        self.bin_seconds = bin_seconds
        self.entries: list[_Entry] = []
        self.index: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for season in seasons:
            for storm in season.storms:
                labels = np.array([
                    CLASSES.index(label_at(storm.pattern, storm.flip, float(age)))
                    for age in storm.obs[:, 0]
                ], dtype=int)
                entry = _Entry(obs=storm.obs, labels=labels)
                which = len(self.entries)
                self.entries.append(entry)
                for row, age in enumerate(storm.obs[:, 0]):
                    key = (int(labels[row]), int(age) // bin_seconds)
                    self.index.setdefault(key, []).append((which, row))

    def sample(self, class_name: str, age: float, n: int, rng: np.random.Generator, origin_xy: np.ndarray) -> list[np.ndarray]:
        class_index = CLASSES.index(class_name)
        bin_id = int(age) // self.bin_seconds
        pool = self.index.get((class_index, bin_id), [])
        if not pool:
            pool = self.index.get((class_index, bin_id - 1), []) + self.index.get((class_index, bin_id + 1), [])
        if not pool or n <= 0:
            return []
        picks = rng.integers(0, len(pool), size=n)
        out = []
        for pick in picks:
            which, row = pool[int(pick)]
            entry = self.entries[which]
            if int(entry.labels[row]) != class_index:
                raise RuntimeError("path library returned a storm with the wrong age-label")
            suffix = entry.obs[row:].copy()
            if suffix.shape[1] < 7:
                raise RuntimeError("path is missing intensity")
            suffix[:, 1] += float(origin_xy[0]) - suffix[0, 1]
            suffix[:, 2] += float(origin_xy[1]) - suffix[0, 2]
            out.append(suffix)
        return out

    def sample_mixture(
        self,
        probs: np.ndarray,
        age: float,
        n: int,
        rng: np.random.Generator,
        origin_xy: np.ndarray,
    ) -> list[np.ndarray]:
        probs = np.clip(probs, 0, None)
        total = float(probs.sum())
        if total <= 0 or n <= 0:
            return []
        probs = probs / total
        draws = rng.choice(len(CLASSES), size=n, p=probs)
        paths = []
        for class_index in draws:
            one = self.sample(CLASSES[int(class_index)], age, 1, rng, origin_xy)
            paths.extend(one)
        return paths
