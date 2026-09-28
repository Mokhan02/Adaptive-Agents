import numpy as np
import pytest

torch = pytest.importorskip("torch")

from regime.agents.in_context import InContextAgent, ModelConfig, MoveTransformer, load_model, save_model
from regime.env import STRATEGY_A, STRATEGY_B, RegimeSwitchOpponent
from regime.runner import run_episode


def small_model():
    torch.manual_seed(0)
    return MoveTransformer(ModelConfig(context=10, d_model=16, n_heads=2, n_layers=2))


def test_model_is_causal():
    model = small_model().eval()
    x = torch.randint(0, 3, (1, 10))
    x2 = x.clone()
    x2[0, 7:] = (x2[0, 7:] + 1) % 3
    with torch.no_grad():
        a, _ = model(x)
        b, _ = model(x2)
    torch.testing.assert_close(a[0, :7], b[0, :7])


def test_agent_logs_primary_state_and_continuous_outputs(tmp_path):
    save_model(small_model(), tmp_path / "m.pt")
    agent = InContextAgent(load_model(tmp_path / "m.pt"))
    log = run_episode(agent, RegimeSwitchOpponent(STRATEGY_A, STRATEGY_B, switch_at=30), 60, seed=0)
    assert log.states.shape == (60, 16)
    assert log.outputs.shape == (60, 3)
    np.testing.assert_allclose(log.policies.sum(1), 1.0)
    assert len(agent.history) == 10  # rolling window, not full history
    assert not any(p.requires_grad for p in agent.model.parameters())


def test_state_and_outputs_do_not_depend_on_temperature():
    # The opponent ignores the agent, so only gameplay depends on temperature (AGENT_SPECS.md).
    model = small_model()
    opp = RegimeSwitchOpponent(STRATEGY_A, STRATEGY_B, switch_at=30)
    a = run_episode(InContextAgent(model, temperature=0.05), opp, 60, seed=0)
    b = run_episode(InContextAgent(model, temperature=1.0), opp, 60, seed=0)
    np.testing.assert_allclose(a.states, b.states)
    np.testing.assert_allclose(a.outputs, b.outputs)


def test_move_offsets_remove_latest_move_component():
    from regime.agents.in_context import estimate_move_offsets

    model = small_model()
    offsets = estimate_move_offsets(model, n_episodes=5, n_rounds=200)
    assert offsets.shape == (3, 3, 16)
    opp = RegimeSwitchOpponent.no_switch(STRATEGY_A, switch_at=30)
    raw = run_episode(InContextAgent(model), opp, 100, seed=0)
    fixed = run_episode(InContextAgent(model, move_offsets=offsets), opp, 100, seed=0)
    np.testing.assert_allclose(fixed.states, raw.states - offsets[1, raw.opp_actions])
    np.testing.assert_allclose(fixed.outputs, raw.outputs)  # behavior is untouched

    def r2(S, M):
        means = np.stack([S[M == k].mean(0) for k in range(3)])
        return 1 - ((S - means[M]) ** 2).sum() / ((S - S.mean(0)) ** 2).sum()

    assert r2(fixed.states[20:], fixed.opp_actions[20:]) < 0.5 * r2(raw.states[20:], raw.opp_actions[20:])


def test_frozen_bundle_ties_offsets_to_weights(tmp_path):
    from regime.agents.in_context import estimate_move_offsets, load_frozen, save_frozen

    model = small_model()
    offsets = estimate_move_offsets(model, n_episodes=2, n_rounds=200)
    save_frozen(model, offsets, {"step": 1}, tmp_path / "f.pt")
    loaded, loaded_offsets, meta = load_frozen(tmp_path / "f.pt")
    np.testing.assert_array_equal(loaded_offsets, offsets)
    assert meta == {"step": 1}

    bundle = torch.load(tmp_path / "f.pt", weights_only=False)
    bundle["state_dict"]["unembed.bias"] += 1e-3  # weights changed without refitting offsets
    torch.save(bundle, tmp_path / "g.pt")
    with pytest.raises(ValueError):
        load_frozen(tmp_path / "g.pt")


def test_fast_tuning_score_matches_episode_runner():
    from regime.tuning import score_episode, score_in_context_fast

    model = MoveTransformer(ModelConfig(context=50, d_model=16, n_heads=2, n_layers=2)).eval()
    for i, tau in [(0, 0.1), (3, 0.5)]:
        slow = score_episode(lambda: InContextAgent(model, temperature=tau), i)
        assert score_in_context_fast(model, tau, i) == pytest.approx(slow, rel=1e-5, abs=1e-6)
