"""AdaptiveDifficultyScheduler — curriculum learning for GRPO training.

Adjusts per-task training weights based on rolling score history.
Tasks where the agent performs well get lower weight (mastered).
Tasks where the agent struggles get higher weight (needs more practice).
Prevents gradient starvation on easy tasks and over-training on hard ones.

Usage:
    from training.adaptive_scheduler import AdaptiveDifficultyScheduler
    scheduler = AdaptiveDifficultyScheduler(task_ids)
    scheduler.update("billing_seat_adjustment", score=0.82)
    weights = scheduler.weights()
"""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any


class AdaptiveDifficultyScheduler:
    """Curriculum learning scheduler for multi-task GRPO training.

    Higher score → task weight decreases (agent has mastered it).
    Lower score → task weight increases (agent needs more practice).
    Weights are normalized to sum to 1.0 after each update.
    """

    WINDOW = 10
    MIN_WEIGHT = 0.05
    MAX_WEIGHT = 0.40
    BOOST_RATE = 0.05
    DECAY_RATE = 0.03

    def __init__(self, task_ids: list[str]) -> None:
        self._task_ids = list(task_ids)
        self._weights: dict[str, float] = {t: 1.0 / len(task_ids) for t in task_ids}
        self._history: dict[str, deque] = {t: deque(maxlen=self.WINDOW) for t in task_ids}
        self._update_count = 0

    def update(self, task_id: str, score: float) -> None:
        """Record a new score observation and adjust weights."""
        if task_id not in self._task_ids:
            return
        self._history[task_id].append(score)
        self._rebalance()
        self._update_count += 1

    def _rebalance(self) -> None:
        for task_id in self._task_ids:
            hist = self._history[task_id]
            if len(hist) < 3:
                continue
            avg = sum(hist) / len(hist)
            if avg >= 0.70:
                self._weights[task_id] = max(
                    self.MIN_WEIGHT, self._weights[task_id] - self.DECAY_RATE
                )
            elif avg < 0.35:
                self._weights[task_id] = min(
                    self.MAX_WEIGHT, self._weights[task_id] + self.BOOST_RATE
                )

        total = sum(self._weights.values())
        if total > 0:
            for task_id in self._task_ids:
                self._weights[task_id] = round(self._weights[task_id] / total, 4)

    def weights(self) -> dict[str, float]:
        """Return current normalized task weights."""
        return dict(self._weights)

    def sample_task(self, rng: Any = None) -> str:
        """Sample a task according to current weights."""
        import random

        r = rng or random
        tasks = list(self._weights.keys())
        weights = [self._weights[t] for t in tasks]
        return r.choices(tasks, weights=weights, k=1)[0]

    def mean_score(self, task_id: str) -> float | None:
        hist = self._history.get(task_id, deque())
        return sum(hist) / len(hist) if hist else None

    def summary(self) -> dict[str, Any]:
        return {
            "update_count": self._update_count,
            "weights": self.weights(),
            "mean_scores": {
                t: round(self.mean_score(t), 3) if self.mean_score(t) is not None else None
                for t in self._task_ids
            },
        }

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.summary(), indent=2))

    def load(self, path: Path) -> None:
        data = json.loads(path.read_text())
        for task_id, w in data.get("weights", {}).items():
            if task_id in self._weights:
                self._weights[task_id] = w
