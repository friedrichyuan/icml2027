from types import SimpleNamespace

import numpy as np
import pytest
import torch

from molt.trainer.algorithm.task_sampler import (
    COMPLETED,
    DISPATCHED,
    INFRA,
    RESOLVED,
    TaskSampler,
)

NAMES = ["easy_short", "hard_long", "mid"]
ROWS = ["easy_short"] * 50 + ["hard_long"] * 30 + ["mid"] * 20


def cfg(**kw):
    base = dict(
        mode="combined",
        exploration_mix=0.3,
        cost_exponent=0.5,
        signal_epsilon=0.05,
        raw_weight_min=0.5,
        raw_weight_max=2.0,
        min_samples_per_window=8,
        refresh_every_new_batches=1,
        cohort_completion_threshold=0.95,
        cost_floor=32.0,
        success_threshold=0.5,
        eval_every_tokens=0,
        max_policy_tokens=0,
    )
    return SimpleNamespace(**{**base, **kw})


class Dataset:
    def __len__(self):
        return len(ROWS)

    def __getitem__(self, i):
        return "src", f"prompt-{i}", f"label-{i}", None, None


def stream(sampler):
    """Episodes back to back, like the generator's rounds."""
    while True:
        yield from sampler.episode(Dataset())


def exp(reward, tokens, rollout_id, truncated=False):
    return SimpleNamespace(
        response_length=torch.tensor([tokens]),
        truncated=torch.tensor([truncated]),
        rollout_ids=[rollout_id],
        info={"reward": torch.tensor([reward]), "num_turns": torch.tensor([1])},
    )


def dispatch(sampler, it, n):
    for _ in range(n):
        next(it)
    refs = [object() for _ in range(n)]
    sampler.bind(refs, 1)
    return refs


def run_cycle(sampler, it, n, outcome):
    """Dispatch n prompts and finish all of them with outcome(bucket) -> (reward, tokens)."""
    refs = dispatch(sampler, it, n)
    for ref in refs:
        b = sampler.inflight[ref]["bucket"]
        reward, tokens = outcome(sampler.names[b])
        sampler.complete(ref, [(exp(reward, tokens, id(ref)), None)])
    sampler.on_new_batch()


def test_p0_follows_bucket_shares_and_original_keeps_it():
    s = TaskSampler(ROWS, NAMES, cfg(mode="original"), seed=0)
    np.testing.assert_allclose(s.p0, [0.5, 0.3, 0.2])
    it = stream(s)
    for _ in range(4):
        run_cycle(
            s,
            it,
            30,
            lambda b: (float(b == "easy_short"), 100 if b == "easy_short" else 3000),
        )
    np.testing.assert_allclose(s.p, s.p0)


def test_full_exploration_mix_recovers_p0():
    s = TaskSampler(ROWS, NAMES, cfg(exploration_mix=1.0), seed=0)
    it = stream(s)
    for _ in range(4):
        run_cycle(s, it, 30, lambda b: (1.0, 100 if b == "easy_short" else 3000))
    np.testing.assert_allclose(s.p, s.p0)


@pytest.mark.parametrize("mode", ["signal_only", "cost_only", "combined"])
@pytest.mark.parametrize("outcome", ["all_success", "all_fail", "extreme_cost"])
def test_probabilities_stay_valid(mode, outcome):
    s = TaskSampler(ROWS, NAMES, cfg(mode=mode), seed=1)
    it = stream(s)
    fn = {
        "all_success": lambda b: (1.0, 500),
        "all_fail": lambda b: (0.0, 500),
        "extreme_cost": lambda b: (
            float(b == "mid"),
            1 if b == "easy_short" else 10**7,
        ),
    }[outcome]
    for _ in range(5):
        run_cycle(s, it, 30, fn)
        assert np.all(s.p > 0) and abs(s.p.sum() - 1) < 1e-9
        # eta * p0 is the floor, the clip bounds the rest.
        assert np.all(s.p >= 0.3 * s.p0 - 1e-12)


def test_cost_only_prefers_cheap_bucket_and_signal_needs_two_windows():
    s = TaskSampler(ROWS, NAMES, cfg(mode="cost_only"), seed=2)
    it = stream(s)
    run_cycle(s, it, 40, lambda b: (0.0, 200 if b == "easy_short" else 2000))
    # The warmup cycle froze C_ref; buckets with a full window are now cost-weighted.
    assert s.c_ref is not None
    for _ in range(3):
        run_cycle(s, it, 40, lambda b: (0.0, 200 if b == "easy_short" else 2000))
    assert s.p[0] / s.p0[0] > 1 > s.p[1] / s.p0[1]

    t = TaskSampler(ROWS, NAMES, cfg(mode="signal_only"), seed=2)
    it = stream(t)
    run_cycle(t, it, 40, lambda b: (0.0, 500))
    assert all(
        not row["statistics_ready"] for row in t.last
    )  # one window only: neutral
    np.testing.assert_allclose(t.p, t.p0)


def test_signal_formula():
    c = cfg(mode="signal_only", min_samples_per_window=4, exploration_mix=0.0)
    s = TaskSampler(["a"] * 10 + ["b"] * 10, ["a", "b"], c, seed=0)
    # Two finished cycles: bucket a goes 0/4 -> 4/4 successes, bucket b stays at 2/4.
    for version, succ in ((0, (0, 2)), (1, (4, 2))):
        cohort = s._cohort(version)
        cohort[[DISPATCHED, RESOLVED, COMPLETED]] = 4
        cohort[3] = succ
    s.version = 1
    s.refresh()
    p_new = np.array([5 / 6, 3 / 6])
    p_old = np.array([1 / 6, 3 / 6])
    L = np.abs(p_new - p_old)
    S = (0.05 + L) / (0.05 + (0.5 * L).sum())
    r = np.clip(S, 0.5, 2.0)
    np.testing.assert_allclose(s.p, 0.5 * r / (0.5 * r).sum())


def test_immature_cycle_is_excluded():
    s = TaskSampler(["a"] * 10, ["a"], cfg(min_samples_per_window=4), seed=0)
    cohort = s._cohort(0)
    cohort[DISPATCHED, 0], cohort[[RESOLVED, COMPLETED], 0] = 10, 9  # 90% < 95%
    new, _, mature = s._windows()
    assert mature == [] and new[0, 0] == 0
    cohort[[RESOLVED, COMPLETED], 0] = 10
    new, _, mature = s._windows()
    assert mature == [0] and new[0, 0] == 10


def test_each_trajectory_counts_once_and_infra_is_not_a_failure():
    s = TaskSampler(ROWS, NAMES, cfg(), seed=0)
    it = stream(s)
    ref_ok, ref_bad = dispatch(s, it, 2)
    s.complete(ref_ok, [(exp(1.0, 50, "r1"), None)])
    s.complete(ref_ok, [(exp(1.0, 50, "r1"), None)])  # duplicate delivery
    s.complete(ref_bad, [(None, "runner_error")])
    s.complete(object(), [(exp(1.0, 50, "eval"), None)])  # not ours (eval)
    total = sum(s.cohorts.values())
    assert total[DISPATCHED].sum() == 2 and total[RESOLVED].sum() == 2
    assert total[COMPLETED].sum() == 1 and total[INFRA].sum() == 1
    assert (
        s.budget["policy_tokens"] == 50
        and s.budget["rollouts"] == 1
        and s.budget["infra_failures"] == 1
    )


def test_empirical_frequencies_match_p():
    s = TaskSampler(ROWS, NAMES, cfg(), seed=3)
    s.p = np.array([0.2, 0.5, 0.3])
    counts = np.zeros(3)
    seen = {b: set() for b in range(3)}
    for _ in range(20000):
        b, idx = s._draw()
        counts[b] += 1
        seen[b].add(idx)
    np.testing.assert_allclose(counts / counts.sum(), s.p, atol=0.015)
    # Within a bucket the original scheme: a shuffled pass over all of its prompts.
    assert all(seen[b] == set(s.members[b].tolist()) for b in range(3))


def test_episode_draws_dataset_size_and_resume_keeps_remaining():
    s = TaskSampler(ROWS, NAMES, cfg(), seed=0)
    assert sum(1 for _ in s.episode(Dataset())) == len(ROWS)
    t = TaskSampler(ROWS, NAMES, cfg(), seed=0)
    it = t.episode(Dataset())
    dispatch(t, it, 30)
    u = TaskSampler(ROWS, NAMES, cfg(), seed=0)
    u.load_state_dict(t.state_dict())
    assert sum(1 for _ in u.episode(Dataset())) == len(ROWS) - 30


def test_state_round_trip():
    s = TaskSampler(ROWS, NAMES, cfg(), seed=4)
    it = stream(s)
    for _ in range(4):
        run_cycle(
            s, it, 30, lambda b: (float(b != "hard_long"), 300 if b == "mid" else 900)
        )
    dispatch(s, it, 5)  # in flight at the checkpoint
    state = s.state_dict()
    r = TaskSampler(ROWS, NAMES, cfg(), seed=999)
    r.load_state_dict(state)
    np.testing.assert_allclose(r.p, s.p)
    assert (r.c_ref, r.version, r.remaining, r.budget) == (
        s.c_ref,
        s.version,
        s.remaining,
        s.budget | {"infra_failures": s.budget["infra_failures"] + 5},
    )
    assert [r._draw() for _ in range(20)] == [s._draw() for _ in range(20)]
    # The lost in-flight rollouts resolve their cycle as infrastructure failures.
    lost = r.cohorts[s.version]
    assert lost[RESOLVED].sum() == lost[DISPATCHED].sum() and lost[INFRA].sum() == 5


def test_budget_eval_points():
    s = TaskSampler(
        ROWS,
        NAMES,
        cfg(mode="original", eval_every_tokens=1000, max_policy_tokens=2500),
        seed=0,
    )
    it = stream(s)
    assert not s.eval_due()
    run_cycle(s, it, 3, lambda b: (1.0, 400))
    assert s.eval_due()
    s.mark_eval(1, {}, 1.0)
    assert not s.eval_due() and s.next_eval_tokens == 2000
    run_cycle(s, it, 3, lambda b: (1.0, 400))
    assert s.eval_due()
    s.mark_eval(2, {}, 1.0)  # 2400 tokens: below the cap
    assert not s.final_eval_done and not s.eval_due()
    run_cycle(s, it, 1, lambda b: (1.0, 400))
    assert s.eval_due()  # the cap fires before the next 1000-token mark
    s.mark_eval(3, {}, 1.0)
    assert s.final_eval_done and not s.eval_due()


def test_bucket_tag_on_experience():
    s = TaskSampler(ROWS, NAMES, cfg(), seed=0)
    it = stream(s)
    (ref,) = dispatch(s, it, 1)
    e = exp(1.0, 10, "x")
    b = s.inflight[ref]["bucket"]
    s.complete(ref, [(e, None)])
    assert e.info["task_bucket"].tolist() == [b]
