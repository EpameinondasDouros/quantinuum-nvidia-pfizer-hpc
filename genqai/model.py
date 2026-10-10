"""Workstream C: Hamiltonian-conditioned generative transformer (ADAPT-GQE policy).

The policy emits a complete circuit -- operator sequence AND rotation angles -- for one
Hamiltonian, autoregressively:

    p(o_1..o_n, EOS, θ_1..θ_n | H) = prod_t p(o_t | H, o_<t, θ_<t) * p(θ_t | H, o_≤t, θ_<t)

* Conditioning: an MLP encoder of the standardised Pauli-coefficient vector (genqai.features;
  or the geometry parameters for the ablation) plus a linear projection of an optional budget
  vector [has_budget, max_ops/32, cnot/500]. The resulting embedding c is the prefix (BOS) token
  and is also added to every position.
* Decoder: causal pre-LN transformer. Input at step t = op_emb(o_{t-1}) + W [sin θ_{t-1}, cos θ_{t-1}]
  + learned position + c.
* Gauge: angles are modelled in the orbital-phase-fixed frame of genqai.features; ``Cond.op_sign``
  maps them to/from the record's own frame, so callers only ever see original-frame angles.
* Heads: categorical over pool ops + EOS (index ``policy.eos == n_ops``); angle head = mixture of
  K von Mises conditioned on the hidden state AND the chosen op's embedding. Angles are periodic
  everywhere (sin/cos inputs, von Mises likelihood, outputs wrapped to (-π, π]).

Policy API (stable; the GRPO workstream builds on it)
-----------------------------------------------------
``cond = policy.condition(records, max_ops=None, cnot_budget=None) -> Cond``
    Batch conditioning for a list of HamiltonianRecords (budget optional, same for all).
``policy.sample(cond, n=1, temperature=1.0, greedy=False) -> list[(ops, angles, logprob_total, valid)]``
    If ``len(cond) == 1`` draws ``n`` sequences for that Hamiltonian, otherwise one per row.
    ``greedy`` = argmax op and mixture-mode angle. ``temperature`` scales op logits and von Mises
    concentrations of the PROPOSAL only; ``logprob_total`` is always under the untempered policy
    (== ``log_prob`` of the returned sequence). ``valid`` is False iff no EOS within ``max_len``.
    Angles are wrapped to (-π, π]. Runs under no_grad.
``policy.log_prob(cond, ops_batch, angles_batch) -> Tensor[B]``
    Differentiable per-sequence log-probability (sum of op log-probs incl. EOS, and von Mises
    mixture log-densities of the angles). ``cond`` may have 1 row (broadcast) or B rows. A
    sequence of length ``max_len`` is treated as truncated (no EOS term).
``policy.save(path, **extra)`` / ``GQEPolicy.load(path)``
    Checkpoint holds config, featurizer (Pauli vocab + train normalisation stats), weights, extra.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from genqai.adapt import wrap_angle
from genqai.features import BUDGET_DIM, Featurizer, budget_vector
from genqai.schema import HamiltonianRecord

LOG2PI = math.log(2 * math.pi)
LOGK_MIN, LOGK_MAX = -2.0, 16.0  # von Mises concentration range: std ~ 2.7 rad .. 3e-4 rad


@dataclass
class ModelConfig:
    system: str
    n_ops: int
    cond_dim: int
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 512
    dropout: float = 0.1
    max_len: int = 32
    n_mix: int = 3
    enc_blocks: int = 2


@dataclass
class Cond:
    x: torch.Tensor  # [B, cond_dim] standardised (gauge-fixed) features
    budget: torch.Tensor  # [B, BUDGET_DIM]
    op_sign: torch.Tensor  # [B, n_ops] ±1: theta_original = op_sign[op] * theta_model

    def __len__(self) -> int:
        return self.x.shape[0]

    def expand(self, n: int) -> "Cond":
        return Cond(self.x.expand(n, -1), self.budget.expand(n, -1), self.op_sign.expand(n, -1))


def von_mises_mixture_logpdf(theta, logit_w, mu, log_kappa):
    """log p(θ) for a K-component von Mises mixture; params have a trailing K dim.

    Uses cos(x) - 1 = -2 sin^2(x/2) and log I0(k) = log i0e(k) + k, so it stays accurate in float32
    at the very large concentrations needed for mHa-level angles.
    """
    kappa = log_kappa.exp()
    half = 0.5 * (theta.unsqueeze(-1) - mu)
    comp = -2.0 * kappa * torch.sin(half) ** 2 - LOG2PI - torch.log(torch.special.i0e(kappa))
    return torch.logsumexp(F.log_softmax(logit_w, -1) + comp, -1)


class _ResBlock(nn.Module):
    def __init__(self, d: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.SiLU(), nn.Dropout(dropout), nn.Linear(2 * d, d))

    def forward(self, x):
        return x + self.net(x)


class GQEPolicy(nn.Module):
    def __init__(self, cfg: ModelConfig, featurizer: Featurizer):
        super().__init__()
        self.cfg, self.featurizer = cfg, featurizer
        d, K = cfg.d_model, cfg.n_mix
        self.encoder = nn.Sequential(
            nn.Linear(cfg.cond_dim, d), nn.LayerNorm(d), nn.SiLU(),
            *[_ResBlock(d, cfg.dropout) for _ in range(cfg.enc_blocks)], nn.LayerNorm(d),
        )
        self.budget_proj = nn.Linear(BUDGET_DIM, d)
        self.bos = nn.Parameter(0.02 * torch.randn(d))
        self.op_emb = nn.Embedding(cfg.n_ops, d)
        self.angle_proj = nn.Linear(2, d)
        self.pos = nn.Embedding(cfg.max_len, d)
        layer = nn.TransformerEncoderLayer(
            d, cfg.n_heads, cfg.d_ff, cfg.dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.blocks = nn.TransformerEncoder(layer, cfg.n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.op_head = nn.Linear(d, cfg.n_ops + 1)
        self.angle_head = nn.Sequential(nn.Linear(2 * d, d), nn.SiLU(), nn.Linear(d, 3 * K))
        with torch.no_grad():  # start with moderately concentrated components around 0
            self.angle_head[-1].bias[2 * K:].fill_(-1.0)
        mask = torch.triu(torch.full((cfg.max_len, cfg.max_len), float("-inf")), 1)
        self.register_buffer("causal_mask", mask, persistent=False)

    @property
    def eos(self) -> int:
        return self.cfg.n_ops

    @property
    def device(self) -> torch.device:
        return self.bos.device

    # ---------------------------------------------------------------- conditioning
    def condition(self, records: list[HamiltonianRecord], max_ops: int | None = None, cnot_budget: int | None = None) -> Cond:
        x = torch.from_numpy(self.featurizer(records)).to(self.device)
        b = torch.from_numpy(budget_vector(max_ops, cnot_budget)).to(self.device).expand(len(records), -1)
        sign = torch.from_numpy(self.featurizer.op_signs(records)).to(self.device)
        return Cond(x, b.clone(), sign)

    def _embed_cond(self, cond: Cond) -> torch.Tensor:
        return self.encoder(cond.x) + self.budget_proj(cond.budget)

    # ---------------------------------------------------------------- core network
    def _hidden(self, c: torch.Tensor, ops: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        """c [B,d], ops/angles [B,T] (T <= max_len-1) -> hidden states [B, T+1, d]."""
        steps = self.op_emb(ops) + self.angle_proj(torch.stack([angles.sin(), angles.cos()], -1))
        x = torch.cat([(self.bos + c).unsqueeze(1), steps], 1)
        P = x.shape[1]
        x = x + self.pos.weight[:P] + c.unsqueeze(1)
        return self.norm(self.blocks(x, mask=self.causal_mask[:P, :P], is_causal=True))

    def _angle_params(self, h: torch.Tensor, ops: torch.Tensor):
        K = self.cfg.n_mix
        z = self.angle_head(torch.cat([h, self.op_emb(ops)], -1))
        logit_w, mu, raw_k = z[..., :K], z[..., K:2 * K], z[..., 2 * K:]
        log_kappa = LOGK_MIN + (LOGK_MAX - LOGK_MIN) * torch.sigmoid(raw_k)
        return logit_w, mu, log_kappa

    def token_log_probs(self, cond: Cond, ops_batch, angles_batch):
        """Teacher-forced per-token log-probs.

        Returns (op_lp [B,P], op_mask [B,P], ang_lp [B,P], ang_mask [B,P]); position t scores token
        t (op or EOS) and, for t < len, the angle θ_t.
        """
        B, L = len(ops_batch), self.cfg.max_len
        if len(cond) == 1 and B > 1:
            cond = cond.expand(B)
        lens = [len(o) for o in ops_batch]
        if max(lens, default=0) > L:
            raise ValueError(f"sequence longer than max_len={L}")
        P = min(max(lens, default=0) + 1, L)  # number of scored positions
        dev = self.device
        tgt = torch.full((B, P), -1, dtype=torch.long, device=dev)
        ang = torch.zeros(B, P, device=dev)
        for b, (o, a) in enumerate(zip(ops_batch, angles_batch)):
            n = len(o)
            if n:
                tgt[b, :n] = torch.as_tensor(o, dtype=torch.long, device=dev)
                ang[b, :n] = torch.as_tensor(a, dtype=torch.float32, device=dev)
            if n < L:
                tgt[b, n] = self.eos
        op_mask = tgt >= 0
        ang_mask = op_mask & (tgt != self.eos)
        ang = ang * cond.op_sign.gather(1, tgt.clamp(0, self.cfg.n_ops - 1))  # -> model gauge
        op_in = tgt[:, : P - 1].clamp(0, self.cfg.n_ops - 1)
        h = self._hidden(self._embed_cond(cond), op_in, ang[:, : P - 1])
        logp = F.log_softmax(self.op_head(h), -1)
        op_lp = logp.gather(-1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1) * op_mask
        ang_lp = von_mises_mixture_logpdf(ang, *self._angle_params(h, tgt.clamp(0, self.cfg.n_ops - 1))) * ang_mask
        return op_lp, op_mask, ang_lp, ang_mask

    def log_prob(self, cond: Cond, ops_batch, angles_batch) -> torch.Tensor:
        op_lp, _, ang_lp, _ = self.token_log_probs(cond, ops_batch, angles_batch)
        return op_lp.sum(1) + ang_lp.sum(1)

    # ---------------------------------------------------------------- generation
    @torch.no_grad()
    def sample(self, cond: Cond, n: int = 1, temperature: float = 1.0, greedy: bool = False):
        was_training = self.training
        self.eval()
        if len(cond) == 1:
            cond = cond.expand(n)
        B, L, dev = len(cond), self.cfg.max_len, self.device
        c = self._embed_cond(cond)
        ops = torch.zeros(B, 0, dtype=torch.long, device=dev)
        angles = torch.zeros(B, 0, device=dev)
        done = torch.zeros(B, dtype=torch.bool, device=dev)
        length = torch.full((B,), L, dtype=torch.long, device=dev)
        lp = torch.zeros(B, device=dev)
        for t in range(L):
            h = self._hidden(c, ops, angles)[:, -1]
            logits = self.op_head(h)
            if greedy:
                tok = logits.argmax(-1)
            else:
                tok = torch.distributions.Categorical(logits=logits / temperature).sample()
            lp += torch.where(done, 0.0, F.log_softmax(logits, -1).gather(-1, tok[:, None]).squeeze(-1))
            stop = (tok == self.eos) & ~done
            length[stop] = t
            done |= stop
            op = tok.clamp(max=self.cfg.n_ops - 1)
            logit_w, mu, log_kappa = self._angle_params(h, op)
            if greedy:
                k = (F.log_softmax(logit_w, -1) + 0.5 * log_kappa).argmax(-1, keepdim=True)
                theta = mu.gather(-1, k).squeeze(-1)
            else:
                k = torch.distributions.Categorical(logits=logit_w / temperature).sample()[:, None]
                theta = torch.distributions.VonMises(
                    mu.gather(-1, k).squeeze(-1), log_kappa.gather(-1, k).squeeze(-1).exp() / temperature
                ).sample()
            lp += torch.where(done, 0.0, von_mises_mixture_logpdf(theta, logit_w, mu, log_kappa))
            if done.all():
                break
            if t < L - 1:
                ops = torch.cat([ops, torch.where(done, 0, op)[:, None]], 1)
                angles = torch.cat([angles, torch.where(done, 0.0, theta)[:, None]], 1)
            else:  # final position: keep the emitted op of truncated sequences
                ops = torch.cat([ops, op[:, None]], 1)
                angles = torch.cat([angles, theta[:, None]], 1)
        self.train(was_training)
        out = []
        for b in range(B):
            n_b = int(length[b])
            sign = cond.op_sign[b, ops[b, :n_b]]
            out.append((
                ops[b, :n_b].tolist(),
                [wrap_angle(a) for a in (sign * angles[b, :n_b]).tolist()],
                float(lp[b]),
                bool(n_b < L),
            ))
        return out

    # ---------------------------------------------------------------- checkpoints
    def save(self, path, **extra) -> None:
        torch.save(
            {"config": asdict(self.cfg), "featurizer": self.featurizer.to_dict(),
             "state_dict": self.state_dict(), "extra": extra},
            path,
        )

    @classmethod
    def load(cls, path, map_location="cpu") -> "GQEPolicy":
        ck = torch.load(path, map_location=map_location, weights_only=False)
        policy = cls(ModelConfig(**ck["config"]), Featurizer(**ck["featurizer"]))
        policy.load_state_dict(ck["state_dict"])
        policy.extra = ck.get("extra", {})
        return policy.eval()


def build_policy(system: str, featurizer: Featurizer, **overrides) -> GQEPolicy:
    from genqai.pool import get_pool

    cfg = ModelConfig(system=system, n_ops=len(get_pool(system)), cond_dim=featurizer.dim, **overrides)
    return GQEPolicy(cfg, featurizer)
