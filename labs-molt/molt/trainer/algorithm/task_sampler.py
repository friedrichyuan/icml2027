"""Budget-aware task sampling (``--task_sampler.mode``).

Every new rollout draws its task bucket b, then a prompt of b in the original shuffled order, from

    p(b) = eta * p0(b) + (1 - eta) * q(b),      q(b) ∝ p0(b) * clip(raw_b, r_min, r_max)

with raw_b = 1 (original), S_b (signal_only), G_b (cost_only) or S_b * G_b (combined):

    S_b = (eps + L_b) / (eps + sum_k p0(k) L_k),   L_b = |p_new(b) - p_old(b)|
    G_b = (C_ref / C_b) ** gamma

p_new / p_old are Beta(1,1)-smoothed success rates of the bucket's two most recent non-overlapping
windows (L_b is a heuristic, not a measured learning gain), C_b the mean policy tokens per
trajectory in the newer window (truncated ones included) and C_ref the warmup cycle's mean, frozen
once known. A bucket without enough data stays neutral (S_b = G_b = 1).

The draw happens when a runner slot frees up, so a refreshed p applies to the very next dispatch
and never touches rollouts already in flight. Trajectories are archived under the scheduler cycle
(``version``) that dispatched them; windows are filled newest-first from whole cycles whose
completion rate reached ``cohort_completion_threshold``, so a cycle's slow tail cannot bias it.
Infrastructure drops resolve a trajectory without counting as a task failure. Nothing here feeds
the loss: the sampling probability is logged only.
"""

import json
import os
import time

import numpy as np
import torch

MODES = ("off", "original", "signal_only", "cost_only", "combined")
# Per-cycle, per-bucket counters (rows of a cohort array). Counts are trajectories; TOKENS sums
# policy tokens of the completed ones.
DISPATCHED, RESOLVED, COMPLETED, SUCCESS, TOKENS, INFRA = range(6)


class TaskSampler:
    def __init__(self, row_buckets, names, cfg, seed, log_dir=None):
        self.cfg = cfg
        self.names = list(names)
        rows = np.asarray(
            list(row_buckets)
        )  # list(): newer `datasets` return a lazy Column
        unknown = set(rows.tolist()) - set(self.names)
        if unknown:
            raise ValueError(
                f"task buckets {sorted(unknown)} are not in the bucket list {self.names}"
            )
        self.members = [np.flatnonzero(rows == b) for b in self.names]
        sizes = np.array([len(m) for m in self.members], dtype=float)
        self.p0 = sizes / sizes.sum()
        self.p = self.p0.copy()
        self.rng = np.random.default_rng(seed)
        self.order = [self.rng.permutation(m) for m in self.members]
        self.pos = [0] * len(self.names)
        self.remaining = 0  # draws left in this episode
        self.version = 0  # scheduler cycle = dispatch cohort
        self.new_batches = 0
        self.policy_version = 0
        self.c_ref = None
        self.cohorts = {}
        self.pending = []  # drawn, not yet bound to a rollout ref
        self.inflight = {}  # rollout ref -> dispatch record
        self.budget = {
            "policy_tokens": 0,
            "env_interactions": 0,
            "rollouts": 0,
            "dispatched": 0,
            "infra_failures": 0,
        }
        self.next_eval_tokens = cfg.eval_every_tokens
        self.final_eval_done = False
        self.wall_base, self.t0, self.eval_seconds = 0.0, time.time(), 0.0
        self.last = self._refresh_rows(*self._windows())
        self.log_dir = log_dir
        self._files = {}

    # ------------------------------------------------------------------ dispatch

    def _draw(self):
        b = int(self.rng.choice(len(self.names), p=self.p))
        if self.pos[b] == len(self.order[b]):
            self.order[b], self.pos[b] = self.rng.permutation(self.members[b]), 0
        idx = int(self.order[b][self.pos[b]])
        self.pos[b] += 1
        return b, idx

    def episode(self, dataset):
        """Prompt iterator for one episode: len(dataset) draws, each made when the slot is filled.
        A resume mid-episode keeps the remaining count, so the run's prompt budget is unchanged.
        """
        if self.remaining <= 0:
            self.remaining = len(dataset)
        while self.remaining > 0:
            b, idx = self._draw()
            self.remaining -= 1
            self.pending.append(
                {
                    "bucket": b,
                    "index": idx,
                    "prob": float(self.p[b]),
                    "version": self.version,
                    "policy_version": self.policy_version,
                    "dispatch_time": time.time(),
                }
            )
            datasource, prompt, label, image, tools = dataset[idx]
            yield [datasource], [prompt], [label], [image], [tools]

    def bind(self, refs, n_samples):
        """Pair the prompts drawn since the last dispatch with their rollout refs (same order)."""
        for ref, rec in zip(refs, self.pending, strict=True):
            rec["n"] = n_samples
            self.inflight[ref] = rec
            self._cohort(rec["version"])[DISPATCHED, rec["bucket"]] += n_samples
        self.budget["dispatched"] += n_samples * len(refs)
        self.pending = []

    def complete(self, ref, results):
        """Account one finished prompt group exactly once; ``results`` is run_group's
        ``[(experience | None, drop_reason)]``. Unknown refs (eval) are ignored."""
        rec = self.inflight.pop(ref, None)
        if rec is None:
            return
        b, cohort, now = rec["bucket"], self._cohort(rec["version"]), time.time()
        rollouts = {}
        for exp, drop_reason in results:
            if exp is None:
                self._resolve_infra(rec, cohort, 1, drop_reason)
                continue
            exp.info["task_bucket"] = torch.full(
                (len(exp.response_length),), b, dtype=torch.long
            )
            key = exp.rollout_ids[0] if exp.rollout_ids else id(exp)
            r = rollouts.setdefault(
                key, {"reward": None, "tokens": 0, "turns": 0, "truncated": False}
            )
            r["tokens"] += int(exp.response_length.sum())
            r["turns"] += (
                int(exp.info["num_turns"].sum()) if "num_turns" in exp.info else 1
            )
            r["truncated"] |= bool(exp.truncated is not None and exp.truncated.any())
            if "reward" in exp.info:
                r["reward"] = float(exp.info["reward"].flatten()[0])
        for key, r in rollouts.items():
            success = (
                r["reward"] is not None and r["reward"] >= self.cfg.success_threshold
            )
            cohort[[RESOLVED, COMPLETED], b] += 1
            cohort[SUCCESS, b] += success
            cohort[TOKENS, b] += r["tokens"]
            self.budget["policy_tokens"] += r["tokens"]
            self.budget["env_interactions"] += r["turns"]
            self.budget["rollouts"] += 1
            self._log(
                "trajectories",
                {
                    "trajectory_id": str(key),
                    **self._record_fields(rec),
                    "completion_time": now,
                    "reward": r["reward"],
                    "success": bool(success),
                    "policy_token_count": r["tokens"],
                    "environment_interaction_count": r["turns"],
                    "termination_reason": (
                        "truncated" if r["truncated"] else "terminated"
                    ),
                },
            )

    def _resolve_infra(self, rec, cohort, n, reason):
        cohort[[RESOLVED, INFRA], rec["bucket"]] += n
        self.budget["infra_failures"] += n
        self._log(
            "trajectories",
            {
                **self._record_fields(rec),
                "completion_time": time.time(),
                "termination_reason": f"infra:{reason}",
            },
        )

    def _record_fields(self, rec):
        return {
            "bucket_id": self.names[rec["bucket"]],
            "prompt_index": rec["index"],
            "dispatch_time": rec["dispatch_time"],
            "behavior_policy_version": rec["policy_version"],
            "scheduler_version": rec["version"],
            "sampling_probability": rec["prob"],
        }

    def _cohort(self, version):
        if version not in self.cohorts:
            self.cohorts[version] = np.zeros((6, len(self.names)))
        return self.cohorts[version]

    # ------------------------------------------------------------------ refresh

    def on_new_batch(self):
        self.new_batches += 1
        if self.new_batches % self.cfg.refresh_every_new_batches == 0:
            self.refresh()

    def _completion(self, cohort):
        dispatched = cohort[DISPATCHED].sum()
        return min(1.0, cohort[RESOLVED].sum() / dispatched) if dispatched else 1.0

    def _windows(self):
        """Per-bucket (completed, success, tokens) of the newest and the previous window, each filled
        with whole mature cycles until it holds min_samples_per_window trajectories."""
        m = self.cfg.min_samples_per_window
        mature = [
            v
            for v in sorted(self.cohorts, reverse=True)
            if self._completion(self.cohorts[v]) >= self.cfg.cohort_completion_threshold
        ]
        new, old = np.zeros((3, len(self.names))), np.zeros((3, len(self.names)))
        for b in range(len(self.names)):
            for v in mature:
                win = new if new[0, b] < m else old if old[0, b] < m else None
                if win is None:
                    break
                win[:, b] += self.cohorts[v][[COMPLETED, SUCCESS, TOKENS], b]
        return new, old, mature

    def refresh(self):
        """Close the current scheduler cycle and recompute p from the mature cycles."""
        new, old, mature = self._windows()
        if self.c_ref is None and 0 in mature and self.cohorts[0][COMPLETED].sum() > 0:
            warm = self.cohorts[0]
            self.c_ref = max(
                warm[TOKENS].sum() / warm[COMPLETED].sum(), self.cfg.cost_floor
            )
        rows = self._refresh_rows(new, old, mature)
        self.p = np.array([row["current_probability"] for row in rows])
        self.version += 1
        self.last = rows
        self._log(
            "refresh",
            {
                "scheduler_version": self.version,
                "new_batches": self.new_batches,
                "policy_version": self.policy_version,
                "time": time.time(),
                "mode": self.cfg.mode,
                "reference_cost": self.c_ref,
                "mature_cycles": len(mature),
                "missing_feedback_fraction": self._missing_feedback(),
                "buckets": rows,
            },
        )

    def _refresh_rows(self, new, old, mature):
        cfg, m = self.cfg, self.cfg.min_samples_per_window
        p_new, p_old = (new[1] + 1) / (new[0] + 2), (old[1] + 1) / (old[0] + 2)
        signal_ready = (new[0] >= m) & (old[0] >= m)
        cost_ready = (new[0] >= m) & (self.c_ref is not None)
        L = np.where(signal_ready, np.abs(p_new - p_old), 0.0)
        L_bar = (
            (self.p0 * L)[signal_ready].sum() / self.p0[signal_ready].sum()
            if signal_ready.any()
            else 0.0
        )
        S = np.where(
            signal_ready, (cfg.signal_epsilon + L) / (cfg.signal_epsilon + L_bar), 1.0
        )
        C = np.maximum(new[2] / np.maximum(new[0], 1), cfg.cost_floor)
        G = np.where(cost_ready, ((self.c_ref or 1.0) / C) ** cfg.cost_exponent, 1.0)
        raw = {
            "original": np.ones_like(S),
            "signal_only": S,
            "cost_only": G,
            "combined": S * G,
        }[cfg.mode]
        r = np.clip(raw, cfg.raw_weight_min, cfg.raw_weight_max)
        q = self.p0 * r / (self.p0 * r).sum()
        p = cfg.exploration_mix * self.p0 + (1 - cfg.exploration_mix) * q
        adaptive = {
            "original": np.zeros_like(signal_ready),
            "signal_only": signal_ready,
            "cost_only": cost_ready,
            "combined": signal_ready | cost_ready,
        }[cfg.mode]
        zeros = np.zeros((6, len(self.names)))
        total = sum(self.cohorts.values(), zeros)
        # Cycles closed before this refresh had at least one full cycle to report back.
        closed = sum((c for v, c in self.cohorts.items() if v < self.version), zeros)
        rows = []
        for b, name in enumerate(self.names):
            rows.append(
                {
                    "bucket_id": name,
                    "p0": float(self.p0[b]),
                    "current_probability": float(p[b]),
                    "probability_ratio": (
                        float(p[b] / self.p0[b]) if self.p0[b] > 0 else 0.0
                    ),
                    "old_success_rate": float(p_old[b]),
                    "new_success_rate": float(p_new[b]),
                    "learning_signal": float(L[b]),
                    "estimated_token_cost": float(C[b]) if new[0, b] else None,
                    "raw_weight": float(raw[b]),
                    "clipped_weight": float(r[b]),
                    "window_sample_counts": [int(new[0, b]), int(old[0, b])],
                    "statistics_ready": bool(signal_ready[b]),
                    "cost_ready": bool(cost_ready[b]),
                    "adaptive_enabled": bool(adaptive[b]),
                    "dispatched": int(total[DISPATCHED, b]),
                    "completed": int(total[RESOLVED, b]),
                    "in_flight": int(total[DISPATCHED, b] - total[RESOLVED, b]),
                    "infra_failures": int(total[INFRA, b]),
                    "feedback_completion_rate": (
                        float(closed[RESOLVED, b] / closed[DISPATCHED, b])
                        if closed[DISPATCHED, b]
                        else None
                    ),
                }
            )
        return rows

    def _missing_feedback(self):
        """Fraction of the closed cycles' trajectories with no feedback yet."""
        closed = [c for v, c in self.cohorts.items() if v <= self.version]
        dispatched = sum(c[DISPATCHED].sum() for c in closed)
        return (
            float(1 - sum(c[RESOLVED].sum() for c in closed) / dispatched)
            if dispatched
            else 0.0
        )

    def metrics(self):
        """Per-step scalars for the trainer log (state as of the last refresh)."""
        out = {"task_sampler/scheduler_version": float(self.version)}
        if self.c_ref is not None:
            out["task_sampler/reference_cost"] = self.c_ref
        for row in self.last:
            name = row["bucket_id"]
            out[f"task_sampler/prob/{name}"] = row["current_probability"]
            out[f"task_sampler/prob_ratio/{name}"] = row["probability_ratio"]
            out[f"task_sampler/success/{name}"] = row["new_success_rate"]
            out[f"task_sampler/signal/{name}"] = row["learning_signal"]
            out[f"task_sampler/weight/{name}"] = row["clipped_weight"]
            out[f"task_sampler/adaptive/{name}"] = float(row["adaptive_enabled"])
            if row["estimated_token_cost"] is not None:
                out[f"task_sampler/cost/{name}"] = row["estimated_token_cost"]
        return out

    # ------------------------------------------------------------------ budget / eval

    def wall_seconds(self):
        return self.wall_base + time.time() - self.t0

    def budget_metrics(self):
        return {
            **{f"budget/{k}": float(v) for k, v in self.budget.items()},
            "budget/wall_seconds": self.wall_seconds(),
            "budget/eval_seconds": self.eval_seconds,
        }

    def eval_due(self):
        tokens = self.budget["policy_tokens"]
        if self.cfg.max_policy_tokens and tokens >= self.cfg.max_policy_tokens:
            return not self.final_eval_done
        return bool(self.cfg.eval_every_tokens) and tokens >= self.next_eval_tokens

    def mark_eval(self, step, metrics, seconds):
        """Book an eval's cost and log it next to the budget spent so far (before this eval)."""
        self._log("eval", {"step": step, **self.budget_metrics(), **metrics})
        self.eval_seconds += seconds
        tokens, every = self.budget["policy_tokens"], self.cfg.eval_every_tokens
        if every:
            self.next_eval_tokens = (tokens // every + 1) * every
        if self.cfg.max_policy_tokens and tokens >= self.cfg.max_policy_tokens:
            self.final_eval_done = True

    # ------------------------------------------------------------------ logs / state

    def _log(self, name, row):
        if not self.log_dir:
            return
        if name not in self._files:
            os.makedirs(self.log_dir, exist_ok=True)
            self._files[name] = open(
                os.path.join(self.log_dir, f"{name}.jsonl"), "a", buffering=1
            )
        self._files[name].write(json.dumps(row) + "\n")

    def state_dict(self):
        return {
            "names": self.names,
            "p": self.p.tolist(),
            "rng": self.rng.bit_generator.state,
            "order": [o.tolist() for o in self.order],
            "pos": list(self.pos),
            "remaining": self.remaining,
            "version": self.version,
            "new_batches": self.new_batches,
            "c_ref": self.c_ref,
            "cohorts": {v: c.tolist() for v, c in self.cohorts.items()},
            "inflight": list(self.inflight.values()) + list(self.pending),
            "budget": dict(self.budget),
            "next_eval_tokens": self.next_eval_tokens,
            "final_eval_done": self.final_eval_done,
            "wall_seconds": self.wall_seconds(),
            "eval_seconds": self.eval_seconds,
        }

    def load_state_dict(self, state):
        if not state:
            return
        if state["names"] != self.names:
            raise ValueError(
                f"checkpoint task buckets {state['names']} != current {self.names}"
            )
        self.p = np.array(state["p"])
        self.rng.bit_generator.state = state["rng"]
        self.order = [np.array(o, dtype=np.int64) for o in state["order"]]
        self.pos = list(state["pos"])
        self.remaining = state["remaining"]
        self.version = state["version"]
        self.new_batches = state["new_batches"]
        self.c_ref = state["c_ref"]
        self.cohorts = {int(v): np.array(c) for v, c in state["cohorts"].items()}
        self.budget = dict(state["budget"])
        self.next_eval_tokens = state["next_eval_tokens"]
        self.final_eval_done = state["final_eval_done"]
        self.wall_base, self.t0, self.eval_seconds = (
            state["wall_seconds"],
            time.time(),
            state["eval_seconds"],
        )
        # Rollouts in flight at the checkpoint died with the old job: resolve them as infrastructure
        # losses so their cycle can still mature (they are not task failures).
        for rec in state["inflight"]:
            self._resolve_infra(
                rec, self._cohort(rec["version"]), rec.get("n", 1), "lost_on_resume"
            )
        self.last = self._refresh_rows(*self._windows())
