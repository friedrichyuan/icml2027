# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""--train.replay / --train.reset: buffer eviction and sampling, frozen replay data, the K = 1
no-op, checkpoint round-trips, and a reset that touches only the selected params and their Adam
state."""

import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

import molt.trainer.algorithm.replay as replay_mod
from molt.trainer.algorithm.experience import Experience
from molt.trainer.algorithm.replay import TrajectoryReplayBuffer, materialize


def _exp(tag):
    return Experience(action_mask=torch.ones(1, 2, dtype=torch.bool), rollout_ids=[tag], info={})


def _ids(entries):
    return [e.experience.rollout_ids[0] for e in entries]


def test_eviction_drops_stale_entries_then_the_oldest():
    buf = TrajectoryReplayBuffer(capacity=4, max_uses=2, max_age=3)
    buf.add([_exp("a"), _exp("b")], insert_update=0)
    buf.add([_exp("c"), _exp("d"), _exp("e")], insert_update=2)
    buf.entries[0].uses = 2  # a used up
    assert buf.evict(learner_updates=3) == {"evicted_stale": 1, "evicted_fifo": 0}
    assert _ids(buf.entries) == ["b", "c", "d", "e"]
    buf.evict(learner_updates=4)  # b is now 4 updates old > max_age
    assert _ids(buf.entries) == ["c", "d", "e"]
    buf.add([_exp("f"), _exp("g")], insert_update=4)
    assert buf.evict(learner_updates=5) == {"evicted_stale": 0, "evicted_fifo": 1}
    assert _ids(buf.entries) == ["d", "e", "f", "g"]


def test_sampling_is_distinct_and_never_refills():
    buf = TrajectoryReplayBuffer(capacity=8, max_uses=4, max_age=32, seed=1)
    buf.add([_exp(str(i)) for i in range(5)], insert_update=0)
    picked = buf.sample(3)
    assert len(set(_ids(picked))) == 3
    assert buf.sample(6) is None


def test_buffer_state_round_trip_reproduces_sampling():
    a = TrajectoryReplayBuffer(capacity=8, max_uses=4, max_age=32, seed=3)
    a.add([_exp(str(i)) for i in range(8)], insert_update=0)
    a.sample(4)
    b = TrajectoryReplayBuffer(capacity=8, max_uses=4, max_age=32, seed=0)
    b.load_state_dict(a.state_dict())
    assert _ids(a.sample(4)) == _ids(b.sample(4))


def test_materialize_pulls_heavy_fields_into_a_detached_copy(monkeypatch):
    monkeypatch.setattr(replay_mod.ray, "get", lambda refs: refs)
    lp = torch.tensor([[-0.5, -1.0]])
    e = _exp("a")
    e.heavy_ref = {"rollout_log_probs": lp, "sequences": torch.tensor([[1, 2, 3]])}
    (c,) = materialize([e])
    assert c.heavy_ref is None and torch.equal(c.rollout_log_probs, lp)
    assert e.heavy_ref is not None and e.rollout_log_probs is None  # the original stays lazy
    c.info["x"] = 1
    assert "x" not in e.info


def _harness(monkeypatch, replay_ratio=1, **buffer_kwargs):
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(SamplingParams=SimpleNamespace, __version__="0.21.0"))
    import molt.trainer.rl_trainer as rt
    from test_length_penalty import _maker

    maker = _maker()
    maker.args.actor.num_gpus_per_node = 2
    appended, trained = [], []
    trainer = SimpleNamespace(
        args=maker.args,
        experience_maker=SimpleNamespace(build_experiences=maker.compute_advantages_and_returns),
        tokenizer=SimpleNamespace(decode=lambda *a, **k: "sample"),
        actor_model_group=SimpleNamespace(
            async_run_method_batch=lambda **k: appended.append(list(k["experience"])) or []
        ),
        critic_model_group=None,
        freezing_actor_steps=0,
        vllm_engines=None,
        kl_ctl=SimpleNamespace(value=0.0),
        policy_train=lambda **k: trained.append(1) or {"actor_grad_norm": float(len(trained)), "is_filter_ratio": 0.0},
    )
    if replay_ratio > 1:
        trainer.replay_ratio = replay_ratio
        trainer.learner_updates = 0
        trainer.replay_buffer = TrajectoryReplayBuffer(**{"capacity": 16, "max_uses": 4, "max_age": 32, **buffer_kwargs})
        trainer.replay_updates = lambda exps: rt.BaseRLTrainer.replay_updates(trainer, exps)
    monkeypatch.setattr(rt.ray, "get", lambda refs: refs)
    return rt, trainer, appended


def _batch(step, rewards=(1, 0, 0, 1)):
    from test_length_penalty import _sample

    out = []
    for i, r in enumerate(rewards):
        s = _sample(f"{step}-{i}", f"g{step}-{i // 2}", r, 100 + 10 * i, 600 + 10 * i, False, i)
        s.rollout_log_probs = torch.full((1, 4), -0.1 * (i + 1) - step)
        out.append(s)
    return out


def test_ratio_one_keeps_the_original_single_update(monkeypatch):
    rt, trainer, appended = _harness(monkeypatch)
    stats, _ = rt.BaseRLTrainer.train_step(trainer, _batch(0), 0)
    assert len(appended) == 1
    assert not any(k.startswith(("replay", "learner_updates", "reset")) for k in stats)


def test_replay_update_reuses_frozen_trajectories(monkeypatch):
    rt, trainer, appended = _harness(monkeypatch, replay_ratio=2)
    stats, _ = rt.BaseRLTrainer.train_step(trainer, _batch(0), 0)
    fresh, replayed = appended
    assert len(replayed) == len(fresh) == 4
    first_use = {e.rollout_ids[0]: (e.advantages.clone(), e.rollout_log_probs.clone()) for e in fresh}
    for e in replayed:  # same behavior log-probs, rewards and first-use advantages
        adv, lp = first_use[e.rollout_ids[0]]
        assert torch.equal(e.advantages, adv) and torch.equal(e.rollout_log_probs, lp)
    assert stats["replay/updates"] == 1 and stats["learner_updates"] == 2
    assert stats["replay/actor_grad_norm"] == 2.0  # the replay round's own metrics, prefixed
    assert stats["actor_grad_norm"] == 1.0
    assert all(e.uses == 2 for e in trainer.replay_buffer.entries)

    # Next batch: a different batch mean, but buffered trajectories keep their first-use advantage.
    appended.clear()
    rt.BaseRLTrainer.train_step(trainer, _batch(1, rewards=(1, 1, 1, 0)), 1)
    _, replayed = appended
    assert len({e.rollout_ids[0] for e in replayed}) == 4
    for e in replayed:
        if e.rollout_ids[0] in first_use:
            assert torch.equal(e.advantages, first_use[e.rollout_ids[0]][0])
    assert trainer.learner_updates == 4


def test_replay_skips_without_enough_eligible_trajectories(monkeypatch):
    rt, trainer, appended = _harness(monkeypatch, replay_ratio=2, max_uses=1)
    stats, _ = rt.BaseRLTrainer.train_step(trainer, _batch(0), 0)
    assert len(appended) == 1  # every trajectory is used up after its first update
    assert stats["replay/updates"] == 0 and stats["learner_updates"] == 1


def test_trainer_restores_the_replay_buffer():
    import molt.trainer.rl_trainer as rt

    source = TrajectoryReplayBuffer(capacity=8, max_uses=4, max_age=32)
    source.add([_exp("a"), _exp("b")], insert_update=3)
    target = SimpleNamespace(replay_buffer=TrajectoryReplayBuffer(capacity=8, max_uses=4, max_age=32))
    state = {"replay": {"buffer": source.state_dict(), "learner_updates": 7}, "global_step": 4}
    rt.BaseRLTrainer.restore_replay(target, state)
    assert _ids(target.replay_buffer.entries) == ["a", "b"] and target.learner_updates == 7


def test_reset_runs_on_the_new_batch_schedule(monkeypatch):
    import molt.trainer.rl_trainer as rt

    calls = []
    reset = SimpleNamespace(mode="shrink", eps=0.01, start=64, interval=64)
    trainer = SimpleNamespace(
        args=SimpleNamespace(train=SimpleNamespace(reset=reset)),
        actor_model_group=SimpleNamespace(async_run_method=lambda **k: calls.append(k) or [{"reset/applied": 1.0}]),
    )
    monkeypatch.setattr(rt.ray, "get", lambda refs: refs)
    fired = [n for n in range(1, 200) if rt.BaseRLTrainer.partial_reset(trainer, n)]
    assert fired == [64, 128, 192]
    assert calls[0] == {"method_name": "partial_reset", "eps": 0.01, "shrink": True}


def _reset_actor(mode):
    from molt.trainer.workers.policy_actor import PolicyModelActor

    model = torch.nn.ModuleDict(
        {"self_attn": torch.nn.ModuleDict({"o_proj": torch.nn.Linear(3, 3), "q_proj": torch.nn.Linear(3, 3)})}
    )
    params = dict(model.named_parameters())
    init = {name: p.detach().clone() for name, p in params.items()}
    optim = torch.optim.AdamW(model.parameters(), lr=0.1)
    for _ in range(2):
        model["self_attn"]["o_proj"](torch.randn(4, 3)).sum().backward()
        model["self_attn"]["q_proj"](torch.randn(4, 3)).sum().backward()
        optim.step()
    actor = object.__new__(getattr(PolicyModelActor, "__ray_actor_class__", PolicyModelActor))
    actor.actor_optim = optim
    actor._reset_params = {
        "self_attn.o_proj.weight": (params["self_attn.o_proj.weight"], init["self_attn.o_proj.weight"] if mode == "shrink" else None)
    }
    return actor, params, init, optim


@pytest.mark.parametrize("mode", ["shrink", "optim_only"])
def test_partial_reset_touches_only_the_selected_params(mode):
    actor, params, init, optim = _reset_actor(mode)
    o, q = params["self_attn.o_proj.weight"], params["self_attn.q_proj.weight"]
    before = {n: p.detach().clone() for n, p in params.items()}
    q_state = {k: v.clone() for k, v in optim.state[q].items()}
    out = actor.partial_reset(eps=0.25 if mode == "shrink" else 0.0, shrink=mode == "shrink")

    want = 0.75 * before["self_attn.o_proj.weight"] + 0.25 * init["self_attn.o_proj.weight"]
    torch.testing.assert_close(o.detach(), want if mode == "shrink" else before["self_attn.o_proj.weight"])
    for name in ("self_attn.q_proj.weight", "self_attn.o_proj.bias"):
        assert torch.equal(params[name].detach(), before[name])
    for key in ("exp_avg", "exp_avg_sq", "step"):
        assert not optim.state[o][key].any()
        assert torch.equal(optim.state[q][key], q_state[key])
    assert out["reset/applied"] == 1.0
    if mode == "shrink":
        drift = (before["self_attn.o_proj.weight"] - init["self_attn.o_proj.weight"]).norm().item()
        assert out["reset/drift_norm_before"] == pytest.approx(drift)


def _cli(*flags):
    return subprocess.run(
        [sys.executable, "-m", "molt.cli.train_rl_ray", "--actor.model_name_or_path", "unused",
         "--vllm.num_engines", "1", "--train.agent_path", "agent.py", *flags],
        capture_output=True, text=True, timeout=60,
    )  # fmt: skip


@pytest.mark.parametrize(
    "flags, message",
    [
        (["--train.replay.update_ratio", "2"], "--train.replay needs --train.force_on_policy"),
        (["--train.replay.update_ratio", "0"], "update_ratio must be >= 1"),
        (["--train.reset.mode", "shrink", "--train.reset.eps", "0"], "eps must be in (0, 1]"),
        (["--train.reset.mode", "optim_only", "--actor.lora_dim", "8"], "full-parameter FSDP"),
    ],
)
def test_cli_rejects_unsupported_replay_and_reset_configs(flags, message):
    result = _cli(*flags)
    assert result.returncode != 0
    assert message in result.stderr
