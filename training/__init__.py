"""
Training package

Provides utilities for building trainers, computing metrics,
and managing reproducible training for sentiment analysis models.
"""

from .trainer import build_trainer, train_or_load_best
from .metrics import build_compute_metrics

__all__ = [
    "build_trainer",
    "train_or_load_best",
    "build_compute_metrics",
]
