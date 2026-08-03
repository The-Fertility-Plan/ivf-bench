"""Deterministic seeding utilities for reproducible synthetic data."""
from __future__ import annotations

import hashlib

import numpy as np


def case_seed(base_seed: int, case_id: str) -> int:
    """Generate a deterministic 32-bit seed for a specific case.

    Uses SHA-256 hash of base_seed + case_id so that adding or removing
    other cases never changes a given case's synthetic profile.
    """
    h = hashlib.sha256(f"{base_seed}:{case_id}".encode()).hexdigest()
    return int(h[:8], 16)


def make_rng(seed: int) -> np.random.Generator:
    """Create a numpy Generator with the given seed."""
    return np.random.default_rng(seed)
