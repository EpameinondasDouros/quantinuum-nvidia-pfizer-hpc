"""Torch-free greedy (one-shot) inference of a trained GQEPolicy, for runtimes without PyTorch (Qollab/Pyodide).

    export_numpy("checkpoints/h2/sft.pt", "out/h2/sft.json")   # needs torch, run once offline
    policy = NumpyPolicy.load("out/h2/sft.json")                 # numpy + scipy only
    ops, angles, valid = policy.generate(record)

The export is a JSON manifest (model config, featurizer, tensor names/shapes) plus the float32
weights as raw little-endian ``.bin`` parts of at most ``part_bytes`` each (Qollab caps files at 2 MB).
Float32 is kept so the numpy forward pass reproduces ``GQEPolicy.sample(greedy=True)`` to float32
round-off. Conditioning (gauge-fixed standardised features, per-operator signs) reuses
genqai.features.Featurizer, so angles come back in the record's own orbital-sign convention.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.special import erf

from genqai.adapt import wrap_angle
from genqai.features import Featurizer, budget_vector
from genqai.schema import HamiltonianRecord

PART_BYTES = 1_900_000
LN_EPS = 1e-5
LOGK_MIN, LOGK_MAX = -2.0, 16.0  # must match genqai.model


def export_numpy(ckpt_path, out_path, part_bytes: int = PART_BYTES) -> list[Path]:
    """Write <out_path> (JSON manifest) and <stem>.part<i>.bin weight parts; returns all written paths."""
    import torch

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    names, shapes, chunks = [], [], []
    for k, v in ck["state_dict"].items():
        names.append(k)
        shapes.append(list(v.shape))
        chunks.append(v.detach().float().numpy().astype("<f4").ravel())
    blob = np.concatenate(chunks).tobytes()
    parts = []
    for i in range(0, len(blob), part_bytes):
        p = out_path.with_name(f"{out_path.stem}.part{len(parts)}.bin")
        p.write_bytes(blob[i:i + part_bytes])
        parts.append(p)
    manifest = {"config": ck["config"], "featurizer": ck["featurizer"], "tensors": names, "shapes": shapes,
                "parts": [p.name for p in parts], "source": "/".join(Path(ckpt_path).parts[-3:]),
                "extra": {"best": ck["extra"].get("best")}}
    out_path.write_text(json.dumps(manifest))
    return [out_path, *parts]


def _layer_norm(x, w, b):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + LN_EPS) * w + b


def _silu(x):
    return x / (1.0 + np.exp(-x))


def _gelu(x):  # exact (erf) GELU, as torch's activation="gelu"
    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))


def _log_softmax(z):
    z = z - z.max(-1, keepdims=True)
    return z - np.log(np.exp(z).sum(-1, keepdims=True))


class NumpyPolicy:
    """Greedy decoder with the GQEPolicy architecture (pre-LN causal transformer, von Mises angle head)."""

    def __init__(self, manifest: dict, weights: dict[str, np.ndarray]):
        self.cfg = manifest["config"]
        self.featurizer = Featurizer(**manifest["featurizer"])
        self.w = weights
        self.extra = manifest.get("extra", {})

    @classmethod
    def load(cls, path) -> "NumpyPolicy":
        path = Path(path)
        m = json.loads(path.read_text())
        flat = np.concatenate([np.fromfile(path.with_name(p), dtype="<f4") for p in m["parts"]])
        w, i = {}, 0
        for name, shape in zip(m["tensors"], m["shapes"]):
            n = int(np.prod(shape)) if shape else 1
            w[name] = flat[i:i + n].reshape(shape)
            i += n
        if i != flat.size:
            raise ValueError(f"{path}: weight parts hold {flat.size} floats, manifest expects {i}")
        return cls(m, w)

    @property
    def eos(self) -> int:
        return self.cfg["n_ops"]

    def _lin(self, name, x):
        return x @ self.w[f"{name}.weight"].T + self.w[f"{name}.bias"]

    def _ln(self, name, x):
        return _layer_norm(x, self.w[f"{name}.weight"], self.w[f"{name}.bias"])

    def _embed_cond(self, x: np.ndarray) -> np.ndarray:
        h = _silu(self._ln("encoder.1", self._lin("encoder.0", x)))
        for b in range(self.cfg["enc_blocks"]):
            p = f"encoder.{3 + b}.net"
            h = h + self._lin(f"{p}.4", _silu(self._lin(f"{p}.1", self._ln(f"{p}.0", h))))
        h = self._ln(f"encoder.{3 + self.cfg['enc_blocks']}", h)
        return h + self._lin("budget_proj", budget_vector())

    def _attn(self, p: str, x: np.ndarray) -> np.ndarray:
        P, d = x.shape
        H = self.cfg["n_heads"]
        qkv = x @ self.w[f"{p}.in_proj_weight"].T + self.w[f"{p}.in_proj_bias"]
        q, k, v = (qkv[:, i * d:(i + 1) * d].reshape(P, H, d // H).transpose(1, 0, 2) for i in range(3))
        s = q @ k.transpose(0, 2, 1) / np.sqrt(d // H) + np.triu(np.full((P, P), -np.inf), 1)
        a = np.exp(_log_softmax(s))
        return self._lin(f"{p}.out_proj", (a @ v).transpose(1, 0, 2).reshape(P, d))

    def _hidden_last(self, c: np.ndarray, ops: list[int], angles: list[float]) -> np.ndarray:
        t = np.asarray(angles, np.float32).reshape(-1, 1)
        steps = self.w["op_emb.weight"][ops] + self._lin("angle_proj", np.concatenate([np.sin(t), np.cos(t)], 1))
        x = np.concatenate([(self.w["bos"] + c)[None], steps], 0)
        x = x + self.w["pos.weight"][:len(x)] + c
        for li in range(self.cfg["n_layers"]):
            p = f"blocks.layers.{li}"
            x = x + self._attn(f"{p}.self_attn", self._ln(f"{p}.norm1", x))
            x = x + self._lin(f"{p}.linear2", _gelu(self._lin(f"{p}.linear1", self._ln(f"{p}.norm2", x))))
        return self._ln("norm", x[-1])

    def generate(self, record: HamiltonianRecord) -> tuple[list[int], list[float], bool]:
        """Greedy one-shot circuit for one Hamiltonian: (ops, angles in the record's frame, valid)."""
        x = self.featurizer([record])[0].astype(np.float32)
        sign = self.featurizer.op_signs([record])[0]
        c = self._embed_cond(x)
        K, L, n_ops = self.cfg["n_mix"], self.cfg["max_len"], self.cfg["n_ops"]
        ops, angles = [], []
        for _ in range(L):
            h = self._hidden_last(c, ops, angles)
            tok = int(np.argmax(self._lin("op_head", h)))
            if tok == self.eos:
                return ops, [wrap_angle(sign[o] * a) for o, a in zip(ops, angles)], True
            z = self._lin("angle_head.2", _silu(self._lin("angle_head.0", np.concatenate([h, self.w["op_emb.weight"][tok]]))))
            logit_w, mu, raw_k = z[:K], z[K:2 * K], z[2 * K:]
            log_kappa = LOGK_MIN + (LOGK_MAX - LOGK_MIN) / (1.0 + np.exp(-raw_k))
            ops.append(min(tok, n_ops - 1))
            angles.append(float(mu[int(np.argmax(_log_softmax(logit_w) + 0.5 * log_kappa))]))
        return ops, [wrap_angle(sign[o] * a) for o, a in zip(ops, angles)], False
