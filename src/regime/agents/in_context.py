"""In-context agent: a frozen causal transformer over the last K opponent moves.

See AGENT_SPECS.md, section "1. In-context agent (primary)".
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from regime.agents.base import Agent
from regime.env import N_ACTIONS, PAYOFF


@dataclass
class ModelConfig:
    context: int = 50
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2


class MoveTransformer(nn.Module):
    """Causal transformer predicting the opponent's next move at every position.

    Positional embeddings index position within the window, never the round
    number, so the model cannot learn when switches happen in absolute time.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(N_ACTIONS, cfg.d_model)
        self.pos = nn.Embedding(cfg.context, cfg.d_model)
        self.layers = nn.ModuleList(
            nn.TransformerEncoderLayer(
                cfg.d_model, cfg.n_heads, dim_feedforward=4 * cfg.d_model, dropout=0.0, batch_first=True, norm_first=True
            )
            for _ in range(cfg.n_layers)
        )
        self.norm = nn.LayerNorm(cfg.d_model)
        self.unembed = nn.Linear(cfg.d_model, N_ACTIONS)

    def forward(self, moves: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """moves: (B, T) ints with T <= context.

        Returns next-move logits (B, T, 3) and the residual stream after the
        embeddings and after each layer, each (B, T, d_model). With pre-norm
        layers, each layer's output is the residual stream.
        """
        T = moves.shape[1]
        x = self.embed(moves) + self.pos(torch.arange(T, device=moves.device))
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=moves.device)
        resid = [x]
        for layer in self.layers:
            x = layer(x, src_mask=mask, is_causal=True)
            resid.append(x)
        return self.unembed(self.norm(x)), resid


def save_model(model: MoveTransformer, path) -> None:
    torch.save({"config": asdict(model.cfg), "state_dict": model.state_dict()}, path)


def load_model(path) -> MoveTransformer:
    ckpt = torch.load(path, map_location="cpu")
    if "model" in ckpt:  # a training checkpoint rather than an exported model
        ckpt = ckpt["model"]
    model = MoveTransformer(ModelConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()


def weights_digest(model: MoveTransformer) -> str:
    import hashlib

    h = hashlib.sha256()
    for name, t in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def save_frozen(model: MoveTransformer, move_offsets: np.ndarray, meta: dict, path) -> str:
    """Save weights and their move offsets as one unit, tied by a weights hash."""
    digest = weights_digest(model)
    torch.save(
        {"config": asdict(model.cfg), "state_dict": model.state_dict(), "move_offsets": move_offsets,
         "weights_sha256": digest, "meta": meta},
        path,
    )
    return digest


def load_frozen(path) -> tuple[MoveTransformer, np.ndarray, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = MoveTransformer(ModelConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["state_dict"])
    if weights_digest(model) != ckpt["weights_sha256"]:
        raise ValueError(f"{path}: weights do not match the hash their move offsets were fit to")
    return model.eval(), ckpt["move_offsets"], ckpt["meta"]


MOVE_OFFSET_SEED_BASE = 4_000_000  # pretraining-distribution episodes for estimating move offsets


@torch.no_grad()
def estimate_move_offsets(model: MoveTransformer, n_episodes: int = 200, n_rounds: int = 600) -> np.ndarray:
    """Mean final-position residual stream given the latest move, per layer.

    Returns (n_layers + 1, 3, d_model). Estimated on pretraining-distribution
    opponents only, never A or B (AGENT_SPECS.md, amendment of 2026-09-27),
    using full windows only (rounds >= context).
    """
    from regime.pretrain import sample_moves, sample_pretraining_opponent

    K = model.cfg.context
    sums = np.zeros((model.cfg.n_layers + 1, N_ACTIONS, model.cfg.d_model))
    counts = np.zeros(N_ACTIONS)
    for i in range(n_episodes):
        rng = np.random.default_rng(MOVE_OFFSET_SEED_BASE + i)
        moves = sample_moves(sample_pretraining_opponent(rng, n_rounds).distributions(n_rounds), rng)
        windows = torch.as_tensor(np.lib.stride_tricks.sliding_window_view(moves, K).copy())
        _, resid = model(windows)
        last = moves[K - 1 :]
        for layer, r in enumerate(resid):
            np.add.at(sums[layer], last, r[:, -1].double().numpy())
        counts += np.bincount(last, minlength=N_ACTIONS)
    return sums / counts[None, :, None]


class InContextAgent(Agent):
    """Best-responds to the transformer's predicted next opponent move.

    Weights are frozen; the agent adapts only through its context window.
    `state_layer` indexes the residual stream: 0 = embeddings, 1 = after
    layer 1 (the pre-registered primary state), 2 = after layer 2.

    With `move_offsets` (from `estimate_move_offsets`), the internal state
    has the mean state for the latest move subtracted. At the final position,
    that component explains 94-98% of the raw state's variance and changes at
    random every round (see the AGENT_SPECS.md amendment).
    """

    name = "in_context"

    def __init__(
        self,
        model: MoveTransformer,
        temperature: float = 0.1,
        state_layer: int = 1,
        move_offsets: np.ndarray | None = None,
    ):
        self.model = model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.temperature = temperature
        self.state_layer = state_layer
        self.move_offsets = move_offsets

    def reset(self, rng: np.random.Generator) -> None:
        super().reset(rng)
        self.history: deque[int] = deque(maxlen=self.model.cfg.context)
        self._update()

    @torch.no_grad()
    def _update(self) -> None:
        if not self.history:
            self._pred = np.full(N_ACTIONS, 1 / N_ACTIONS)
            self._resid = [np.zeros(self.model.cfg.d_model)] * (self.model.cfg.n_layers + 1)
            return
        logits, resid = self.model(torch.tensor([list(self.history)]))
        self._pred = torch.softmax(logits[0, -1], -1).double().numpy()
        self._resid = [r[0, -1].double().numpy() for r in resid]

    def predicted_opponent(self) -> np.ndarray:
        return self._pred

    def output_scores(self) -> np.ndarray:
        return PAYOFF @ self._pred

    def policy(self) -> np.ndarray:
        z = self.output_scores() / self.temperature
        z = np.exp(z - z.max())
        return z / z.sum()

    def observe(self, my_action: int, opp_action: int, reward: float) -> None:
        self.history.append(opp_action)
        self._update()

    def internal_state(self) -> np.ndarray:
        return self.residual_streams()[self.state_layer]

    def residual_streams(self, corrected: bool = True) -> list[np.ndarray]:
        """Residual-stream snapshots at every layer, latest-move component
        removed if offsets were given and `corrected` is True."""
        if not corrected or self.move_offsets is None or not self.history:
            return self._resid
        last = self.history[-1]
        return [r - self.move_offsets[layer, last] for layer, r in enumerate(self._resid)]
