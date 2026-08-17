"""Pruebas de dispositivo para ablación HQCNN (TASK-11)."""

from __future__ import annotations

from unittest.mock import patch

import torch

from src.experiments.ablacion_L import get_device_hqcnn


def test_get_device_hqcnn_prefiere_cuda() -> None:
    with patch("torch.cuda.is_available", return_value=True):
        assert get_device_hqcnn().type == "cuda"


def test_get_device_hqcnn_evita_mps() -> None:
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.backends.mps.is_available", return_value=True),
    ):
        assert get_device_hqcnn().type == "cpu"
