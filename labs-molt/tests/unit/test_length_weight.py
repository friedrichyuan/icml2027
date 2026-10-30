# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""--algo.length_weight: the frozen per-trajectory weight's direction, clipping and calibration,
the exact no-op at alpha = 1 / off, where it enters the loss, and the global-B reduction under
gradient accumulation."""

import json
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from molt.models import PolicyLoss
from molt.trainer.algorithm.experience import Experience
from molt.trainer.algorithm.length_weight import LengthWeight, length_metrics, reference_stats_id
from molt.trainer.rollout.experience_maker import RemoteExperienceMaker

REFERENCE = [100, 200, 300, 400, 800, 1600, 3200]  # median 400
LENGTHS = torch.tensor([50, 200, 400, 800, 1600, 4000, 12800])


def test_weight_direction_against_the_reference_median():
    up = LengthWeight.calibrate(REFERENCE, 0.75, 0.5, 2.0)
    down = LengthWeight.calibrate(REFERENCE, 1.25, 0.5, 2.0)
    assert up.reference_length == down.reference_length == 400
    assert (up(LENGTHS).diff() >= 0).all() and up(LENGTHS)[-1] > up(LENGTHS)[0]  # long up-weighted
    assert (down(LENGTHS).diff() <= 0).all() and down(LENGTHS)[-1] < down(LENGTHS)[0]  # long down-weighted
    torch.testing.assert_close(up.raw(torch.tensor([400])), torch.tensor([1.0], dtype=torch.float64))


def test_clipping_bounds_the_raw_weight():
    lw = LengthWeight(alpha=0.0, reference_length=400, raw_weight_min=0.5, raw_weight_max=2.0, calibration_scale=1.0)
    # alpha = 0: u = T / T0 before the clip.
    torch.testing.assert_close(
        lw.raw(torch.tensor([100, 200, 400, 800, 1600])),
        torch.tensor([0.5, 0.5, 1.0, 2.0, 2.0], dtype=torch.float64),
    )
    m = length_metrics(
        torch.tensor([100, 400, 1600]), torch.zeros(3), torch.zeros(3, dtype=torch.bool), torch.ones(3), lw
    )
    assert m["lw/at_min_frac"] == pytest.approx(1 / 3)
    assert m["lw/at_max_frac"] == pytest.approx(1 / 3)


@pytest.mark.parametrize("alpha", [0.5, 0.75, 1.25, 1.5])
def test_calibration_gives_unit_mean_weight_on_the_reference(alpha):
    lw = LengthWeight.calibrate(REFERENCE, alpha, 0.5, 2.0)
    assert lw(torch.tensor(REFERENCE)).double().mean().item() == pytest.approx(1.0, abs=1e-6)
    # Frozen: a different batch is not renormalized to mean 1.
    assert lw(torch.tensor([3200, 3200])).mean().item() != pytest.approx(1.0)
    assert lw.reference_stats_id == reference_stats_id(list(reversed(REFERENCE)))


def test_alpha_one_is_exactly_one():
    lw = LengthWeight.calibrate(REFERENCE, 1.0, 0.5, 2.0)
    assert lw.calibration_scale == 1.0
    assert torch.equal(lw(LENGTHS), torch.ones(len(LENGTHS)))


def _loss_fn():
    # The benchmark recipe: sequence-level binary_kl gate, mask mode, seq-mean-token-mean.
    return PolicyLoss(
        is_correction_level="seq",
        is_correction_mode="mask",
        is_correction_threshold=[float("-inf"), 3e-3],
        is_correction_gating="binary_kl",
        loss_agg_mode="seq-mean-token-mean",
    )


def _batch(seed=0, rows=6, width=12):
    g = torch.Generator().manual_seed(seed)
    lengths = torch.randint(2, width + 1, (rows,), generator=g)
    mask = torch.arange(width).expand(rows, width) < lengths.unsqueeze(-1)
    old = -torch.rand(rows, width, generator=g) - 0.1
    rollout = old.clone()
    rollout[0] -= 0.5  # row 0 fails the sequence gate (binary KL >> 3e-3)
    new = old + 0.05 * torch.randn(rows, width, generator=g)
    adv = torch.randn(rows, 1, generator=g).expand(rows, width) * mask
    return new, old, rollout, adv, mask


def _loss_and_grad(fn, new, old, rollout, adv, mask, seq_weights=None, global_batch_size=None):
    new = new.clone().requires_grad_(True)
    loss, reported, *_ = fn(
        new,
        old,
        adv,
        action_mask=mask,
        rollout_log_probs=rollout,
        global_batch_size=global_batch_size if global_batch_size is not None else mask.size(0),
        seq_weights=seq_weights,
    )
    loss.backward()
    return loss.detach(), reported, new.grad


def test_off_and_alpha_one_recover_the_original_loss_and_gradient():
    fn, (new, old, rollout, adv, mask) = _loss_fn(), _batch()
    base = _loss_and_grad(fn, new, old, rollout, adv, mask)
    lw = LengthWeight.calibrate(REFERENCE, 1.0, 0.5, 2.0)
    same = _loss_and_grad(fn, new, old, rollout, adv, mask, seq_weights=lw(mask.sum(-1)))
    for a, b in zip(base, same):
        assert torch.equal(a, b)


def test_weight_scales_each_trajectory_before_the_reduction():
    fn, (new, old, rollout, adv, mask) = _loss_fn(), _batch(seed=1)
    B = mask.size(0)
    w = torch.linspace(0.5, 2.0, B)
    loss, reported, grad = _loss_and_grad(fn, new, old, rollout, adv, mask, seq_weights=w)
    # L = sum_i w_i * L_i, with L_i the row's own (1/B)(1/T_i) sum_t term: the T_i and B
    # denominators are unchanged and the gated row 0 contributes nothing either way.
    rows = [_loss_and_grad(fn, new[i : i + 1], old[i : i + 1], rollout[i : i + 1], adv[i : i + 1], mask[i : i + 1],
                           global_batch_size=B) for i in range(B)]  # fmt: skip
    torch.testing.assert_close(loss, sum(w[i] * rows[i][0] for i in range(B)))
    torch.testing.assert_close(grad, torch.cat([w[i] * rows[i][2] for i in range(B)]))
    assert torch.equal(grad[0], torch.zeros_like(grad[0]))
    # The reported policy loss stays unweighted.
    assert torch.equal(reported, _loss_and_grad(fn, new, old, rollout, adv, mask)[1])


def test_gradient_accumulation_keeps_the_global_batch_reduction():
    fn, (new, old, rollout, adv, mask) = _loss_fn(), _batch(seed=2, rows=7)
    w = LengthWeight.calibrate(REFERENCE, 0.75, 0.5, 2.0)(mask.sum(-1) * 100)
    full_loss, _, full_grad = _loss_and_grad(fn, new, old, rollout, adv, mask, seq_weights=w)
    # Uneven microbatches, each normalized by the global B as policy_actor does.
    loss, grads = 0.0, []
    for idx in torch.tensor([0, 1, 2, 3, 4, 5, 6]).split([1, 4, 2]):
        part = _loss_and_grad(fn, new[idx], old[idx], rollout[idx], adv[idx], mask[idx], w[idx], mask.size(0))
        loss, grads = loss + part[0], grads + [part[2]]
    torch.testing.assert_close(loss, full_loss)
    torch.testing.assert_close(torch.cat(grads), full_grad)


def test_length_metrics_split_mass_by_length_and_outcome():
    lw = LengthWeight(alpha=0.0, reference_length=1000, raw_weight_min=0.5, raw_weight_max=2.0, calibration_scale=1.0)
    lengths = torch.tensor([300, 600, 2000, 4096])
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])
    truncated = torch.tensor([False, False, False, True])
    m = length_metrics(lengths, rewards, truncated, torch.ones(4), lw)
    assert m["length/success_mean"] == pytest.approx(1150)
    assert m["length/fail_truncated_rate"] == pytest.approx(0.5)
    assert m["length/success_truncated_rate"] == 0.0
    # Long (> T0 = 1000) failures: only the 4096 row. Weights 0.5, 0.6, 2.0, 2.0.
    assert m["lw/long_fail_share_unweighted"] == pytest.approx(1 / 4)
    assert m["lw/long_fail_share_weighted"] == pytest.approx(2.0 / 5.1)
    assert m["lw/mass_share_ge4k_weighted"] == pytest.approx(2.0 / 5.1)
    assert m["lw/truncated_w_mean"] == pytest.approx(2.0)
    for tag in ("weighted", "unweighted"):
        assert sum(v for k, v in m.items() if k.startswith("lw/mass_share") and k.endswith(f"_{tag}")) == pytest.approx(1)


def _maker(length_weight):
    args = SimpleNamespace(
        reward=SimpleNamespace(clip_range=None, overlong_buffer_len=None, stop_properly_penalty_coef=None),
        algo=SimpleNamespace(
            advantage=SimpleNamespace(
                estimator="flash_reinforce", gamma=1.0, lam=1.0, no_whiten=False, flash_baseline="batch_mean"
            ),
            length_weight=length_weight,
        ),
        rollout=SimpleNamespace(n_samples_per_prompt=1),
    )
    return RemoteExperienceMaker(None, None, SimpleNamespace(value=0.0), SimpleNamespace(args=args), None)


def _samples():
    out = []
    for i, (r, n) in enumerate([(1, 3), (0, 9), (0, 2), (1, 6)]):
        mask = torch.ones(1, n, dtype=torch.bool)
        out.append(Experience(action_mask=mask, kl=torch.zeros(mask.shape), rewards=torch.tensor([float(r)]),
                              truncated=torch.tensor([n == 9]), index=[i], info={}))  # fmt: skip
    return out


def test_experience_maker_logs_length_metrics_only_when_enabled():
    off = _maker(SimpleNamespace(enabled=False))
    assert off.length_weight is None
    off_samples = off.compute_advantages_and_returns(_samples())
    assert not any(k.startswith(("length/", "lw/")) for k in off.advantage_metrics)

    lw = SimpleNamespace(
        enabled=True, alpha=0.75, reference_length=4.0, raw_weight_min=0.5, raw_weight_max=2.0,
        calibration_scale=1.1, reference_stats_id="abc",
    )  # fmt: skip
    on = _maker(lw)
    on_samples = on.compute_advantages_and_returns(_samples())
    for a, b in zip(off_samples, on_samples):  # advantages are untouched; the actor applies w
        assert torch.equal(a.advantages, b.advantages)
    m = on.advantage_metrics
    assert m["length/fail_mean"] == pytest.approx(5.5)
    assert m["length/fail_truncated_rate"] == pytest.approx(0.5)
    want = on.length_weight(torch.tensor([3, 9, 2, 6])).double().mean().item()
    assert m["lw/w_mean"] == pytest.approx(want)
    assert "baseline/pred_mean" in m


def test_trainer_checkpoint_freezes_the_constants():
    import molt.trainer.rl_trainer as rt

    lw = LengthWeight.calibrate(REFERENCE, 0.75, 0.5, 2.0)
    trainer = SimpleNamespace(experience_maker=SimpleNamespace(length_weight=lw))
    rt.BaseRLTrainer.verify_length_weight(trainer, {"length_weight": lw.state_dict(), "global_step": 5})
    rt.BaseRLTrainer.verify_length_weight(trainer, {})  # fresh start
    other = LengthWeight.calibrate(REFERENCE, 1.25, 0.5, 2.0)
    with pytest.raises(ValueError, match="differs from the checkpoint"):
        rt.BaseRLTrainer.verify_length_weight(trainer, {"length_weight": other.state_dict(), "global_step": 5})
    off = SimpleNamespace(experience_maker=SimpleNamespace(length_weight=None))
    with pytest.raises(ValueError, match="differs from the checkpoint"):
        rt.BaseRLTrainer.verify_length_weight(off, {"length_weight": lw.state_dict(), "global_step": 5})
    rt.BaseRLTrainer.verify_length_weight(off, {"global_step": 5})  # plain run, plain checkpoint


def test_train_step_dumps_every_generated_length(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(SamplingParams=SimpleNamespace, __version__="0.21.0"))
    import molt.trainer.rl_trainer as rt
    from test_length_penalty import _maker as _lp_maker
    from test_length_penalty import _sample

    maker = _lp_maker()
    maker.args.logger = SimpleNamespace(rollout_lengths_file=str(tmp_path / "lengths.jsonl"))
    samples = [_sample("A", "g", 1, 100, 600, False, 0), _sample("B", "g", 0, 4000, 4500, True, 1)]
    trainer = SimpleNamespace(
        args=maker.args,
        experience_maker=SimpleNamespace(build_experiences=maker.compute_advantages_and_returns),
        tokenizer=SimpleNamespace(decode=lambda *a, **k: "sample"),
        actor_model_group=SimpleNamespace(async_run_method_batch=lambda **k: []),
        critic_model_group=None,
        freezing_actor_steps=0,
        vllm_engines=None,
        kl_ctl=SimpleNamespace(value=0.0),
        policy_train=lambda **k: {},
    )
    monkeypatch.setattr(rt.ray, "get", lambda refs: refs)
    rt.BaseRLTrainer.train_step(trainer, samples, 4)
    (record,) = [json.loads(line) for line in (tmp_path / "lengths.jsonl").read_text().splitlines()]
    assert record == {"step": 5, "lengths": [100, 4000], "rewards": [1.0, 0.0], "truncated": [0, 1]}


def _cli(*flags):
    return subprocess.run(
        [sys.executable, "-m", "molt.cli.train_rl_ray", "--actor.model_name_or_path", "unused",
         "--vllm.num_engines", "1", "--train.agent_path", "agent.py", *flags],
        capture_output=True, text=True, timeout=60,
    )  # fmt: skip


@pytest.mark.parametrize(
    "flags, message",
    [
        (["--actor.loss_agg_mode", "token-mean"], "seq-mean-token-mean"),
        (["--actor.loss_agg_mode", "seq-mean-token-mean"], "needs --reference_file"),
        (["--actor.loss_agg_mode", "seq-mean-token-mean", "--algo.length_weight.raw_weight_min", "3"], "raw_weight"),
    ],
)
def test_cli_rejects_incomplete_length_weight_configs(flags, message):
    result = _cli("--algo.length_weight.enabled", *flags)
    assert result.returncode != 0
    assert message in result.stderr


def test_cli_rejects_constants_that_disagree_with_the_reference(tmp_path):
    ref = tmp_path / "ref.json"
    ref.write_text(json.dumps({"lengths": REFERENCE}))
    result = _cli(
        "--algo.length_weight.enabled",
        "--actor.loss_agg_mode",
        "seq-mean-token-mean",
        "--algo.length_weight.reference_file",
        str(ref),
        "--algo.length_weight.reference_length",
        "500",
    )
    assert result.returncode != 0
    assert "disagrees with the reference file" in result.stderr
