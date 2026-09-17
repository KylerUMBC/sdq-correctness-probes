"""Shared fixtures for SDQ tests."""

import pytest
import torch


@pytest.fixture
def hidden_dim():
    """Small hidden dimension for fast tests."""
    return 32


@pytest.fixture
def latent_dim():
    return 16


@pytest.fixture
def gauge_dim():
    return 8


@pytest.fixture
def seq_len():
    """Typical short sequence length."""
    return 10


@pytest.fixture
def trajectory_pair(hidden_dim, seq_len):
    """Two trajectories of the same length (simulating aligned pair)."""
    torch.manual_seed(42)
    source = torch.randn(seq_len, hidden_dim)
    # Target is a perturbed version (simulating surface-form variation)
    target = source + 0.1 * torch.randn(seq_len, hidden_dim)
    return source, target


@pytest.fixture
def trajectory_triple(hidden_dim, seq_len):
    """Three trajectories for cycle consistency tests."""
    torch.manual_seed(42)
    base = torch.randn(seq_len, hidden_dim)
    t_i = base + 0.1 * torch.randn(seq_len, hidden_dim)
    t_j = base + 0.1 * torch.randn(seq_len, hidden_dim)
    t_k = base + 0.1 * torch.randn(seq_len, hidden_dim)
    return t_i, t_j, t_k
