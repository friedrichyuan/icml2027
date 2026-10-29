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

"""Per-turn linear return baseline for FlashREINFORCE (``--algo.advantage.flash_baseline turn_linear``).

A turn is one contiguous run of policy tokens in a row's step-axis action mask: the step runner
masks env/tool feedback out between assistant actions, so each run is one assistant action, and a
pure-CoT rollout is a single turn. Every token of turn j shares A = R - b(s_j), where s_j holds only
what is known before turn j is generated (no reward, final length, action content or log-probs).

b is standardized features -> one linear layer, regressed on the terminal reward with plain MSE and
its own Adam. It never touches the actor graph. The caller must predict a batch BEFORE fitting on
it, so a trajectory's own reward never leaks into its advantage.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Tuple

import torch

# turn index; policy tokens generated in earlier turns; prompt tokens before the first action.
# Under the step runner every earlier turn ended in one env/tool step, so the turn index is also
# the tool-call count.
FEATURE_NAMES = ("turn", "prior_policy_tokens", "prompt_tokens")


def turn_spans(action_mask: torch.Tensor) -> List[List[Tuple[int, int]]]:
    """Contiguous True runs of each (B, L) mask row, as [start, end) step positions."""
    spans = []
    for row in action_mask.bool().cpu():
        padded = torch.cat([torch.zeros(1, dtype=torch.int8), row.to(torch.int8), torch.zeros(1, dtype=torch.int8)])
        edges = padded.diff()
        starts = (edges == 1).nonzero().flatten().tolist()
        ends = (edges == -1).nonzero().flatten().tolist()
        spans.append(list(zip(starts, ends)))
    return spans


def turn_features(spans: List[Tuple[int, int]]) -> torch.Tensor:
    """(num_turns, len(FEATURE_NAMES)) pre-action features for one row's turns."""
    rows, prior = [], 0
    # Step position k predicts token k + 1, so the first action starts after start + 1 prompt tokens.
    prompt_tokens = spans[0][0] + 1 if spans else 0
    for j, (start, end) in enumerate(spans):
        rows.append([j, prior, prompt_tokens])
        prior += end - start
    return torch.tensor(rows, dtype=torch.float64).reshape(-1, len(FEATURE_NAMES))


def turn_metrics(features: torch.Tensor, returns: torch.Tensor, advantages: torch.Tensor) -> Dict[str, float]:
    """Per-turn diagnostics, comparable across baselines: the prediction is R - A, i.e. the batch
    mean under batch_mean (in-sample) and the pre-update regressor under turn_linear (held-out)."""
    returns, advantages = returns.double(), advantages.double()
    preds = returns - advantages
    metrics = {
        "baseline/mse": (preds - returns).pow(2).mean().item(),
        "baseline/pred_mean": preds.mean().item(),
        "baseline/pred_std": preds.std(unbiased=False).item(),
        "baseline/adv_pos_frac": (advantages > 0).double().mean().item(),
        "baseline/adv_neg_frac": (advantages < 0).double().mean().item(),
        "baseline/num_turns": float(returns.numel()),
    }
    turn = features[:, FEATURE_NAMES.index("turn")]
    for name, selected in (("0", turn == 0), ("1", turn == 1), ("2", turn == 2), ("3plus", turn >= 3)):
        if selected.any():
            metrics[f"baseline/turn{name}_pred"] = preds[selected].mean().item()
            metrics[f"baseline/turn{name}_return"] = returns[selected].mean().item()
    return metrics


class TurnLinearBaseline:
    """Standardized features -> linear layer, fit to terminal reward by MSE with a separate Adam."""

    def __init__(self, lr: float):
        num_features = len(FEATURE_NAMES)
        self.linear = torch.nn.Linear(num_features, 1, dtype=torch.float64)
        torch.nn.init.zeros_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)
        self.optim = torch.optim.Adam(self.linear.parameters(), lr=lr)
        # Running feature moments (Chan/Welford merge). Features carry no outcome information, so
        # folding the current batch in before predicting is leak-free.
        self.count = 0
        self.mean = torch.zeros(num_features, dtype=torch.float64)
        self.m2 = torch.zeros(num_features, dtype=torch.float64)

    def update_feature_stats(self, features: torch.Tensor) -> None:
        n = features.size(0)
        if n == 0:
            return
        batch_mean = features.mean(0)
        delta = batch_mean - self.mean
        total = self.count + n
        self.m2 += ((features - batch_mean) ** 2).sum(0) + delta**2 * self.count * n / total
        self.mean += delta * n / total
        self.count = total

    def _standardize(self, features: torch.Tensor) -> torch.Tensor:
        std = (self.m2 / max(self.count, 1)).sqrt()
        # A constant feature (e.g. turn 0 on single-turn math) maps to 0 instead of 0/0.
        return torch.where(std > 0, (features - self.mean) / std.clamp_min(1e-12), torch.zeros_like(features))

    @torch.no_grad()
    def predict(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(self._standardize(features)).squeeze(-1)

    def fit(self, features: torch.Tensor, returns: torch.Tensor) -> float:
        """One full-batch Adam step on mean((b - R)^2) over every visited turn; returns the loss."""
        with torch.enable_grad():
            loss = (self.linear(self._standardize(features)).squeeze(-1) - returns.double()).pow(2).mean()
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
        return loss.item()

    def state_dict(self) -> Dict:
        # Deep copy: module/optimizer state_dicts alias the live tensors, and the trainer holds this
        # snapshot across steps until the next checkpoint save.
        return copy.deepcopy(
            {
                "linear": self.linear.state_dict(),
                "optim": self.optim.state_dict(),
                "count": self.count,
                "mean": self.mean,
                "m2": self.m2,
            }
        )

    def load_state_dict(self, state: Dict) -> None:
        self.linear.load_state_dict(state["linear"])
        self.optim.load_state_dict(state["optim"])
        self.count = state["count"]
        self.mean = state["mean"].clone()
        self.m2 = state["m2"].clone()
