# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fixed per-trajectory length weight on the policy-gradient term (``--algo.length_weight.*``).

    u_i = clip((T_i / T0) ** (1 - alpha), u_min, u_max),    w_i = u_i / c

T_i is the trajectory's policy-token count (prompt, tool feedback and padding excluded). T0 is the
median and c the mean of u over a reference length sample frozen before training, so the reference
set's mean weight is 1. Nothing is renormalized per batch. alpha = 1 gives w = 1 (plain Flash);
alpha < 1 up-weights long trajectories, alpha > 1 down-weights them.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence

import torch

# Length buckets (policy tokens) for the loss-mass diagnostics.
BUCKET_EDGES = (512, 1024, 2048, 4096)
BUCKET_NAMES = ("lt512", "512_1k", "1k_2k", "2k_4k", "ge4k")


def reference_stats_id(lengths: Sequence[int]) -> str:
    """Content hash of a reference length sample (order-independent)."""
    return hashlib.sha256(json.dumps(sorted(int(n) for n in lengths)).encode()).hexdigest()[:12]


def load_reference_lengths(path: str) -> List[int]:
    """A JSON file holding a list of lengths, or ``{"lengths": [...]}``."""
    with open(path) as f:
        data = json.load(f)
    lengths = data["lengths"] if isinstance(data, dict) else data
    if not lengths or any(int(n) < 1 for n in lengths):
        raise ValueError(f"{path}: reference lengths must be a nonempty list of positive token counts")
    return [int(n) for n in lengths]


@dataclass(frozen=True)
class LengthWeight:
    alpha: float
    reference_length: float  # T0
    raw_weight_min: float
    raw_weight_max: float
    calibration_scale: float  # c
    reference_stats_id: Optional[str] = None

    @classmethod
    def from_args(cls, lw) -> Optional["LengthWeight"]:
        """The frozen weight from ``args.algo.length_weight`` (calibrated by the CLI), or None if off."""
        if lw is None or not lw.enabled:
            return None
        return cls(
            lw.alpha,
            lw.reference_length,
            lw.raw_weight_min,
            lw.raw_weight_max,
            lw.calibration_scale,
            lw.reference_stats_id,
        )

    @classmethod
    def calibrate(cls, lengths: Sequence[int], alpha: float, raw_weight_min: float, raw_weight_max: float):
        """Freeze T0 = median and c = mean(u) on the reference sample."""
        t0 = float(statistics.median(lengths))
        partial = cls(alpha, t0, raw_weight_min, raw_weight_max, 1.0)
        c = partial.raw(torch.tensor(lengths)).mean().item()
        return cls(alpha, t0, raw_weight_min, raw_weight_max, c, reference_stats_id(lengths))

    def raw(self, lengths: torch.Tensor) -> torch.Tensor:
        u = (lengths.double() / self.reference_length).pow(1.0 - self.alpha)
        return u.clamp(self.raw_weight_min, self.raw_weight_max)

    def __call__(self, lengths: torch.Tensor) -> torch.Tensor:
        return (self.raw(lengths) / self.calibration_scale).float()

    def state_dict(self) -> Dict:
        return asdict(self)


def _quantile(x: torch.Tensor, q: float) -> float:
    return torch.quantile(x.double(), q).item()


def length_metrics(
    lengths: torch.Tensor,
    rewards: torch.Tensor,
    truncated: torch.Tensor,
    abs_advantages: torch.Tensor,
    weight: Optional[LengthWeight],
) -> Dict[str, float]:
    """Unweighted length statistics plus the policy-gradient loss-mass split by length.

    Per-trajectory loss mass is w_i * mean_t |A_i,t|: the scale of trajectory i's term in the
    seq-mean-token-mean surrogate (importance ratios ~1 and the sequence gate are not included).
    It is a proxy for the trajectory's gradient contribution, not a measured gradient norm.
    "Long" means T_i > T0 (the frozen reference median), or the batch median without a reference.
    """
    lengths, rewards = lengths.double(), rewards.double()
    truncated = truncated.bool()
    success = rewards > 0
    out: Dict[str, float] = {"length/success_frac": success.double().mean().item()}
    for name, selected in (("success", success), ("fail", ~success)):
        if selected.any():
            out[f"length/{name}_mean"] = lengths[selected].mean().item()
            out[f"length/{name}_p50"] = _quantile(lengths[selected], 0.5)
            out[f"length/{name}_p90"] = _quantile(lengths[selected], 0.9)
            out[f"length/{name}_truncated_rate"] = truncated[selected].double().mean().item()

    threshold = weight.reference_length if weight is not None else _quantile(lengths, 0.5)
    long_fail = (lengths > threshold) & ~success
    bucket = torch.bucketize(lengths, torch.tensor(BUCKET_EDGES, dtype=torch.float64), right=True)
    weights = {"unweighted": torch.ones_like(lengths)}
    if weight is not None:
        weights["weighted"] = weight(lengths).double()
        raw = weight.raw(lengths)
        w = weights["weighted"]
        out.update(
            {
                "lw/w_mean": w.mean().item(),
                "lw/w_p10": _quantile(w, 0.1),
                "lw/w_p50": _quantile(w, 0.5),
                "lw/w_p90": _quantile(w, 0.9),
                "lw/at_min_frac": (raw <= weight.raw_weight_min).double().mean().item(),
                "lw/at_max_frac": (raw >= weight.raw_weight_max).double().mean().item(),
            }
        )
        if truncated.any():
            out["lw/truncated_w_mean"] = w[truncated].mean().item()
    for tag, w in weights.items():
        mass = w * abs_advantages.double()
        total = mass.sum().clamp_min(1e-12)
        out[f"lw/long_fail_share_{tag}"] = (mass[long_fail].sum() / total).item()
        for k, name in enumerate(BUCKET_NAMES):
            out[f"lw/mass_share_{name}_{tag}"] = (mass[bucket == k].sum() / total).item()
    return out
