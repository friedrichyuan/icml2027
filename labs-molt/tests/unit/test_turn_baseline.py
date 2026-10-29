# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FlashREINFORCE turn_linear baseline: turn segmentation, leak-free features, the
predict-before-fit order, the off switch, and checkpoint round-trips."""

import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from molt.trainer.algorithm.advantage import AdvantageContext, get_advantage_estimator
from molt.trainer.algorithm.experience import Experience
from molt.trainer.algorithm.turn_baseline import TurnLinearBaseline, turn_features, turn_metrics, turn_spans
from molt.trainer.rollout.experience_maker import RemoteExperienceMaker

# Step axis: prompt (positions 0), turn 0 at [1, 3), feedback at [3, 5), turn 1 at [5, 6),
# feedback at 6, turn 2 at [7, 9).
MASK = torch.tensor([[0, 1, 1, 0, 0, 1, 0, 1, 1]], dtype=torch.bool)


def test_turn_spans_are_contiguous_action_runs():
    assert turn_spans(MASK) == [[(1, 3), (5, 6), (7, 9)]]
    assert turn_spans(torch.ones(1, 4, dtype=torch.bool)) == [[(0, 4)]]  # pure CoT: one turn


def test_turn_features_use_only_the_history_before_each_turn():
    spans = turn_spans(MASK)[0]
    torch.testing.assert_close(
        turn_features(spans), torch.tensor([[0, 0, 2], [1, 2, 2], [2, 3, 2]], dtype=torch.float64)
    )
    # Cutting the trajectory right before turn j (no future tokens, turns, or lengths) must not
    # change the features of turns 0..j.
    for j, (start, _) in enumerate(spans):
        truncated = turn_spans(MASK[:, : start + 1])[0]
        torch.testing.assert_close(turn_features(truncated)[: j + 1], turn_features(spans)[: j + 1])


def _ctx(masks, baselines=None):
    return AdvantageContext(
        sample_to_rollout=torch.arange(len(masks)),
        exp_len=[1] * len(masks),
        action_masks=masks,
        kl_coef=0.0,
        gamma=1.0,
        lam=1.0,
        kls=[torch.zeros(m.shape) for m in masks],
        baselines=baselines,
    )


def test_flash_reinforce_subtracts_the_per_turn_baseline():
    b = torch.tensor([[0, 0.2, 0.2, 0, 0, 0.7, 0, 0.4, 0.4]])
    adv, ret = get_advantage_estimator("flash_reinforce")(torch.tensor([1.0]), [[0]], _ctx([MASK], [b]))
    expected = torch.tensor([[0, 0.8, 0.8, 0, 0, 0.3, 0, 0.6, 0.6]])
    torch.testing.assert_close(adv[0], expected)
    torch.testing.assert_close(ret[0], expected)


def test_flash_reinforce_without_baselines_is_the_batch_mean():
    masks = [torch.ones(1, 2, dtype=torch.bool), torch.ones(1, 3, dtype=torch.bool)]
    adv, _ = get_advantage_estimator("flash_reinforce")(torch.tensor([1.0, 0.0]), [[0], [1]], _ctx(masks))
    torch.testing.assert_close(adv[0], torch.full((1, 2), 0.5))
    torch.testing.assert_close(adv[1], torch.full((1, 3), -0.5))


def _maker(flash_baseline):
    args = SimpleNamespace(
        reward=SimpleNamespace(clip_range=None, overlong_buffer_len=None, stop_properly_penalty_coef=None),
        algo=SimpleNamespace(
            advantage=SimpleNamespace(
                estimator="flash_reinforce",
                gamma=1.0,
                lam=1.0,
                no_whiten=False,
                flash_baseline=flash_baseline,
                flash_baseline_lr=0.1,
            )
        ),
        rollout=SimpleNamespace(n_samples_per_prompt=1),
    )
    return RemoteExperienceMaker(None, None, SimpleNamespace(value=0.0), SimpleNamespace(args=args), None)


def _samples(rewards, prompt_lens=(3, 5, 2, 7)):
    """Failed rollouts take three turns, solved ones one, so turn features carry return signal."""
    out = []
    for i, (r, p) in enumerate(zip(rewards, prompt_lens)):
        turns = 1 if r else 3
        mask = torch.zeros(1, p - 1 + 4 * turns, dtype=torch.bool)
        for j in range(turns):  # two action tokens, then two feedback tokens
            mask[0, p - 1 + 4 * j : p + 1 + 4 * j] = True
        out.append(
            Experience(
                action_mask=mask,
                kl=torch.zeros(mask.shape),
                rewards=torch.tensor([float(r)]),
                index=[i],
                info={},
            )
        )
    return out


def test_batch_mean_switch_keeps_the_original_advantages():
    maker = _maker("batch_mean")
    assert maker.turn_baseline is None
    samples = maker.compute_advantages_and_returns(_samples([1, 0, 0, 1]))
    for s, r in zip(samples, [0.5, -0.5, -0.5, 0.5]):
        torch.testing.assert_close(s.advantages, r * s.action_mask.float())
    assert maker.advantage_metrics["baseline/pred_mean"] == pytest.approx(0.5)
    assert maker.advantage_metrics["baseline/adv_pos_frac"] == pytest.approx(2 / 8)  # solved: 1 turn each


def test_turn_linear_predicts_with_the_pre_update_regressor():
    maker = _maker("turn_linear")
    rewards = [1, 0, 0, 1]
    # Zero-initialized regressor: the first batch's baseline is 0 even though fitting on it would not be.
    first = maker.compute_advantages_and_returns(_samples(rewards))
    for s, r in zip(first, rewards):
        torch.testing.assert_close(s.advantages, r * s.action_mask.float())
        assert not s.advantages.requires_grad
    assert maker.turn_baseline.linear.bias.item() != 0  # fit ran after the advantages were fixed

    # Second batch: advantages come from the regressor as fit on batch 1 only, never on batch 2.
    snapshot = TurnLinearBaseline(lr=0.1)
    snapshot.load_state_dict(maker.turn_baseline.state_dict())
    second = _samples([0, 0, 1, 1])
    features = torch.cat([turn_features(turn_spans(s.action_mask)[0]) for s in second])
    snapshot.update_feature_stats(features)
    expected_b = snapshot.predict(features).float()
    maker.compute_advantages_and_returns(second)
    got_b = torch.cat(
        [s.rewards - s.advantages[0, start] for s in second for start, _ in turn_spans(s.action_mask)[0]]
    )
    torch.testing.assert_close(got_b, expected_b)
    # Distinct turns of one trajectory get distinct baselines (turn index / prior tokens differ).
    s = second[0]
    (a0, _), (a1, _), _ = turn_spans(s.action_mask)[0]
    assert s.advantages[0, a0] != s.advantages[0, a1]
    assert maker.advantage_metrics["baseline/turn0_return"] == pytest.approx(0.5)
    assert maker.advantage_metrics["baseline/turn1_return"] == pytest.approx(0.0)


def test_constant_features_standardize_to_zero():
    b = TurnLinearBaseline(lr=0.1)
    x = torch.tensor([[0.0, 0.0, 5.0], [0.0, 0.0, 9.0]], dtype=torch.float64)
    b.update_feature_stats(x)
    z = b._standardize(x)
    assert torch.isfinite(z).all()
    torch.testing.assert_close(z[:, :2], torch.zeros(2, 2, dtype=torch.float64))
    torch.testing.assert_close(z[:, 2], torch.tensor([-1.0, 1.0], dtype=torch.float64))


def test_feature_stats_merge_matches_full_batch_moments():
    b = TurnLinearBaseline(lr=0.1)
    x = torch.randn(50, 3, dtype=torch.float64) * 4 + 2
    for chunk in x.split([7, 20, 23]):
        b.update_feature_stats(chunk)
    torch.testing.assert_close(b.mean, x.mean(0))
    torch.testing.assert_close(b.m2 / b.count, x.var(0, unbiased=False))


def test_state_dict_round_trip_resumes_identically():
    x = torch.randn(16, 3, dtype=torch.float64)
    y = (torch.rand(16) > 0.5).double()
    a = TurnLinearBaseline(lr=0.05)
    a.update_feature_stats(x)
    a.fit(x, y)
    saved = a.state_dict()
    b = TurnLinearBaseline(lr=0.05)
    b.load_state_dict(saved)
    a.fit(x, y)  # continuing must not mutate the saved snapshot
    b.fit(x, y)
    torch.testing.assert_close(a.predict(x), b.predict(x))
    assert not torch.equal(saved["linear"]["bias"], a.linear.bias.detach())


def test_turn_metrics_report_held_out_predictions():
    features = torch.tensor([[0, 0, 3], [1, 4, 3]], dtype=torch.float64)
    m = turn_metrics(features, torch.tensor([1.0, 1.0]), torch.tensor([0.75, 0.5]))
    assert m["baseline/mse"] == pytest.approx((0.75**2 + 0.5**2) / 2)  # preds 0.25, 0.5 vs R = 1
    assert m["baseline/turn0_pred"] == pytest.approx(0.25)
    assert m["baseline/turn1_pred"] == pytest.approx(0.5)
    assert "baseline/turn2_pred" not in m


def test_trainer_restores_the_baseline_from_checkpoint_states():
    import molt.trainer.rl_trainer as rt

    source = TurnLinearBaseline(lr=0.1)
    x = torch.randn(8, 3, dtype=torch.float64)
    source.update_feature_stats(x)
    source.fit(x, torch.ones(8, dtype=torch.float64))
    target = SimpleNamespace(experience_maker=SimpleNamespace(turn_baseline=TurnLinearBaseline(lr=0.1)))
    rt.BaseRLTrainer.restore_turn_baseline(target, {"turn_baseline": source.state_dict()})
    torch.testing.assert_close(target.experience_maker.turn_baseline.predict(x), source.predict(x))


def test_cli_rejects_turn_linear_outside_flash_reinforce():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "molt.cli.train_rl_ray",
            "--actor.model_name_or_path",
            "unused",
            "--vllm.num_engines",
            "1",
            "--train.agent_path",
            "agent.py",
            "--algo.advantage.flash_baseline",
            "turn_linear",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode != 0
    assert "flash_baseline only applies" in result.stderr
