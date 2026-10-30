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

"""Trajectory replay for Flash (``--train.replay.*``).

K = actor updates per new batch. After the fresh batch's update its trajectories enter the
buffer; each of the K - 1 extra updates trains on B trajectories drawn uniformly (no repeats
within a replay batch) from the eligible entries. A replayed trajectory keeps everything it was
generated and first scored with — behavior (rollout) log-probs, reward, first-use advantage —
and the actor recomputes the current log-probs, IS ratio and sequence gate on every update.

Eligibility: uses < max_uses (the first update counts) and age <= max_age, where age is the
number of learner optimizer updates since the trajectory's first update. Stale entries go first,
then the oldest beyond capacity (FIFO). Too few eligible entries skips the replay update.
"""

from __future__ import annotations

import copy
import random
import statistics
from dataclasses import dataclass
from typing import Dict, List, Optional

import ray

from molt.trainer.algorithm.experience import Experience


@dataclass
class ReplayEntry:
    experience: Experience  # heavy tensors local; advantages frozen at first use
    # Learner updates published when its batch entered training: the behavior policy is this
    # version or (async generator running ahead) an older one; its rollout log-probs are exact.
    insert_update: int
    uses: int = 1


def materialize(experiences: List[Experience]) -> List[Experience]:
    """Detached local copies: the runners' object-store blobs die with them, the buffer must not."""
    refs = [e.heavy_ref for e in experiences if e.heavy_ref is not None]
    blobs = iter(ray.get(refs)) if refs else iter(())
    out = []
    for e in experiences:
        c = copy.copy(e)
        c.info = dict(e.info)
        if e.heavy_ref is not None:
            for name, value in next(blobs).items():
                setattr(c, name, value)
            c.heavy_ref = None
        out.append(c)
    return out


class TrajectoryReplayBuffer:
    def __init__(self, capacity: int, max_uses: int, max_age: int, seed: int = 0):
        self.capacity = capacity
        self.max_uses = max_uses
        self.max_age = max_age
        self.entries: List[ReplayEntry] = []
        self.rng = random.Random(seed)

    def add(self, experiences: List[Experience], insert_update: int) -> None:
        self.entries.extend(ReplayEntry(e, insert_update) for e in experiences)

    def evict(self, learner_updates: int) -> Dict[str, int]:
        before = len(self.entries)
        self.entries = [
            e for e in self.entries if e.uses < self.max_uses and learner_updates - e.insert_update <= self.max_age
        ]
        stale = before - len(self.entries)
        overflow = max(0, len(self.entries) - self.capacity)
        self.entries = self.entries[overflow:]  # entries are in insertion order
        return {"evicted_stale": stale, "evicted_fifo": overflow}

    def sample(self, n: int) -> Optional[List[ReplayEntry]]:
        """n distinct entries uniformly, or None if fewer than n are eligible (no refill)."""
        if len(self.entries) < n:
            return None
        return self.rng.sample(self.entries, n)

    def stats(self, learner_updates: int, picked: Optional[List[ReplayEntry]] = None) -> Dict[str, float]:
        out = {"replay_buffer/size": float(len(self.entries))}
        if self.entries:
            out["replay_buffer/age_mean"] = statistics.fmean(learner_updates - e.insert_update for e in self.entries)
            out["replay_buffer/uses_mean"] = statistics.fmean(e.uses for e in self.entries)
        if picked:
            out["replay/age_mean"] = statistics.fmean(learner_updates - e.insert_update for e in picked)
            out["replay/age_max"] = float(max(learner_updates - e.insert_update for e in picked))
            out["replay/uses_mean"] = statistics.fmean(e.uses for e in picked)
        return out

    def state_dict(self) -> Dict:
        return {"entries": list(self.entries), "rng": self.rng.getstate()}

    def load_state_dict(self, state: Dict) -> None:
        self.entries = list(state["entries"])
        self.rng.setstate(state["rng"])
