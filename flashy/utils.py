# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Various utilities.
"""
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
import os
import typing as tp

import torch
import torch.distributed

AnyPath = tp.Union[Path, str]


def averager(beta: float = 1):
    """
    Exponential Moving Average callback.
    Returns a single function that can be called to repeatidly update the EMA
    with a dict of metrics. The callback will return
    the new averaged dict of metrics.

    Note that for `beta=1`, this is just plain averaging.
    """
    fix: tp.Dict[str, float] = defaultdict(float)
    total: tp.Dict[str, float] = defaultdict(float)

    def _update(metrics: tp.Dict[str, tp.Any], weight: float = 1) -> tp.Dict[str, float]:
        counts = {}
        for key, value in list(metrics.items()):
            if key.startswith('_count_'):
                key = key.removeprefix('_count_')
                counts[key] = value
                del metrics[key]
        for key, value in metrics.items():
            count = counts.get(key, 1.)
            total[key] = total[key] * beta + count * weight * float(value)
            fix[key] = fix[key] * beta + count * weight
        return {key: tot / fix[key] for key, tot in total.items()}
    return _update


class TensorAverager:
    """Similar to `averager` but performs everything on the GPU to avoid sync points.

    Args:
        device: device on which to store the averages.
        beta: decay rate for the exponential moving average.
    """
    def __init__(self, device, beta: float = 1.):
        self.state = torch.zeros(2, 0, device=device)
        self.name_to_index: tp.Dict[str, int] = {}
        self.device = device
        self.beta = beta

    def update(self, metrics: tp.Dict[str, tp.Union[float, torch.Tensor]],
               weight: tp.Union[float, torch.Tensor] = 1.) -> None:
        """
        Update the moving average, given a dict of metrics, and an optional weight for the current
        update. Note that both the weight and metrics can be either float or Tensor.
        If Tensors, they MUST be on the proper device already.
        """
        for key in metrics:
            if key not in self.name_to_index:
                self.name_to_index[key] = len(self.name_to_index)
        if len(self.name_to_index) != self.state.shape[1]:
            new_state = torch.zeros(2, len(self.name_to_index), device=self.device)
            new_state[:, :self.state.shape[1]] = self.state
            self.state = new_state

        for key, value in metrics.items():
            if isinstance(value, float):
                value = torch.full([1], value, device=self.device)
            else:
                assert isinstance(value, torch.Tensor)
                assert value.numel() == 1
                value = value.detach().view(1)
            index = self.name_to_index[key]
            if self.beta != 1.:
                self.state[:, index] *= self.beta
            self.state[0, index: index + 1] += weight
            self.state[1, index: index + 1] += weight * value

    def get_tensor_averages(self) -> tp.Dict[str, torch.Tensor]:
        """
        Returns a dict with all the current values of the averages. The values of the dict
        are torch.Tensor on the original device, so that this doesn't trigger a sync point.
        """
        average = self.state[1] / self.state[0]
        return {key: average[index] for key, index in self.name_to_index.items()}

    def to_dict(self) -> tp.Dict[str, float]:
        """
        Returns a dict with all the current values of the averages as floats.
        This triggers a sync point.
        """
        average = (self.state[1] / self.state[0]).tolist()
        return {key: average[index] for key, index in self.name_to_index.items()}

    def all_reduce(self) -> 'TensorAverager':
        """Average metrics over all the GPUs."""
        if not torch.distributed.is_initialized():
            return self
        torch.distributed.all_reduce(self.state)
        return self

    def clear(self) -> None:
        """Reset the internal state."""
        self.state.zero_()


@contextmanager
def write_and_rename(path: AnyPath, mode: str = "wb", suffix: str = ".tmp", pid=False):
    """
    Write to a temporary file with the given suffix, then rename it
    to the right filename. As renaming a file is usually much faster
    than writing it, this removes (or highly limits as far as I understand NFS)
    the likelihood of leaving a half-written checkpoint behind, if killed
    at the wrong time.
    """
    tmp_path = str(path) + suffix
    if pid:
        tmp_path += f".{os.getpid()}"
    with open(tmp_path, mode) as f:
        yield f
    os.rename(tmp_path, path)


@contextmanager
def readonly(model: torch.nn.Module):
    """Temporarily switches off gradient computation for the given model.
    """
    state = []
    for p in model.parameters():
        state.append(p.requires_grad)
        p.requires_grad_(False)
    try:
        yield
    finally:
        for p, s in zip(model.parameters(), state):
            p.requires_grad_(s)
