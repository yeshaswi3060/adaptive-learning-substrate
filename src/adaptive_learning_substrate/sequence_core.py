"""Independent NumPy reference for a causal gated sparse byte-sequence learner.

Local eligibility/feedback is an approximation, not exact temporal backpropagation.
There is no attention, cue latch, task routing or retained sequence history.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

H, V, K = 32, 256, 4
CORE = ("w", "g", "r", "a", "bc", "bg")
PARAMETERS = CORE + ("o", "bo")
SHAPES = {"w": (H, V), "g": (H, V), "r": (H, K), "a": (H,), "bc": (H,), "bg": (H,),
          "o": (V, H), "bo": (V,), "b": (H, V)}
SHAPES.update({"e_"+name: SHAPES[name] for name in CORE})
SHAPES.update(s=(H,), p=(V,))
OFFSETS: dict[str, int] = {}
SIZE = 0
for _name, _shape in SHAPES.items():
    OFFSETS[_name] = SIZE
    SIZE += int(np.prod(_shape))
CONFIG = {"version": "sparse-gated-byte-v1", "hidden": H, "vocabulary": V, "parents": K,
          "core_rate": .001, "output_rate": .05, "trace_clip": 4., "delta_clip": .02, "weight_clip": 3.}


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def write_record(path: Path, value: Any) -> str:
    content = canonical(value)+b"\n"
    digest = hashlib.sha256(content).hexdigest()
    with path.open("xb") as stream:
        stream.write(content)
    with path.with_suffix(path.suffix+".sha256").open("xb") as stream:
        stream.write(canonical({"algorithm": "sha256", "report_file": path.name, "report_sha256": digest})+b"\n")
    return digest


def read_record(path: Path) -> Any:
    content = path.read_bytes()
    sidecar = json.loads(path.with_suffix(path.suffix+".sha256").read_bytes())
    if sidecar != {"algorithm": "sha256", "report_file": path.name, "report_sha256": hashlib.sha256(content).hexdigest()}:
        raise ValueError("checkpoint/evidence sidecar mismatch")
    return json.loads(content)


def views(buffer: np.ndarray) -> dict[str, np.ndarray]:
    return {name: buffer[OFFSETS[name]:OFFSETS[name]+int(np.prod(shape))].reshape(shape)
            for name, shape in SHAPES.items()}


def initial(seed: int) -> tuple[np.ndarray, np.ndarray]:
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative built-in integer")
    rng = np.random.Generator(np.random.PCG64(seed))
    buffer = np.zeros(SIZE, dtype=np.float64)
    x = views(buffer)
    for name, scale in (("w", .2), ("g", .05), ("r", .04), ("b", 1/np.sqrt(H))):
        x[name][:] = rng.normal(0, scale, size=SHAPES[name])
    x["bg"][:] = np.linspace(0, 3, H)
    x["o"][:] = .01*x["b"].T
    x["p"][:] = 1/V
    parents = np.asarray([rng.choice([j for j in range(H) if j != i], K, replace=False)
                          for i in range(H)], dtype=np.int32)
    return buffer, parents


def token_check(token: int) -> None:
    if type(token) is not int or not 0 <= token < V:
        raise ValueError("token must be a built-in integer byte in 0..255")


class CpuSequence:
    def __init__(self, seed: int = 15000, *, core_learning: bool = True) -> None:
        if type(core_learning) is not bool:
            raise TypeError("core_learning must be bool")
        self._buffer, self._parents = initial(seed)
        self.seed, self.core_learning = seed, core_learning
        self.steps = self.updates = self.resets = 0
        self.pending = self.closed = False

    def _open(self) -> None:
        if self.closed:
            raise RuntimeError("sequence model is closed")

    def _export(self) -> tuple[np.ndarray, np.ndarray]:
        self._open()
        return self._buffer.copy(), self._parents.copy()

    def _install(self, buffer: np.ndarray, parents: np.ndarray) -> None:
        self._buffer, self._parents = buffer.copy(), parents.copy()

    def _probabilities(self) -> np.ndarray:
        return views(self._buffer)["p"].copy()

    def _forward(self, token: int) -> None:
        x = views(self._buffer)
        previous = x["s"].copy()
        c = np.tanh(x["w"][:, token] + np.sum(x["r"]*previous[self._parents], axis=1) + x["bc"])
        g = 1/(1+np.exp(-(x["g"][:, token]+x["a"]*previous+x["bg"])))
        dc, dg = (1-g)*(1-c*c), (previous-c)*g*(1-g)
        jacobian = g+dg*x["a"]
        for name in CORE:
            e = x["e_"+name]
            e *= jacobian.reshape((H,)+(1,)*(e.ndim-1))
        x["e_w"][:, token] += dc
        x["e_g"][:, token] += dg
        x["e_r"][:] += dc[:, None]*previous[self._parents]
        x["e_a"][:] += dg*previous
        x["e_bc"][:] += dc
        x["e_bg"][:] += dg
        for name in CORE:
            np.clip(x["e_"+name], -4, 4, out=x["e_"+name])
        x["s"][:] = g*previous+(1-g)*c
        logits = x["o"] @ x["s"]+x["bo"]
        probability = np.exp(logits-np.max(logits))
        x["p"][:] = probability/np.sum(probability)

    def _update(self, target: int) -> None:
        x = views(self._buffer)
        error = -x["p"].copy()
        error[target] += 1
        signal = np.clip(x["b"] @ error, -1, 1)
        changes = {"o": .05*np.outer(error, x["s"])/max(1., float(x["s"] @ x["s"])), "bo": .05*error}
        if self.core_learning:
            changes.update({name: .001*signal.reshape((H,)+(1,)*(x[name].ndim-1))*x["e_"+name] for name in CORE})
        staged = {name: np.clip(x[name]+np.clip(delta, -.02, .02), -3, 3) for name, delta in changes.items()}
        if not all(np.isfinite(value).all() for value in staged.values()):
            raise FloatingPointError("nonfinite staged parameter update")
        for name, value in staged.items():
            x[name][:] = value

    def step(self, token: int, *, training: bool = False) -> np.ndarray:
        self._open()
        token_check(token)
        if type(training) is not bool:
            raise TypeError("training must be bool")
        if self.pending:
            raise RuntimeError("reveal or explicitly cancel the pending target first")
        self._forward(token)
        self.steps += 1
        self.pending = training
        return self.predict_next()

    def predict_next(self) -> np.ndarray:
        self._open()
        result = self._probabilities()
        if result.shape != (V,) or not np.isfinite(result).all() or np.any(result < 0) or abs(float(result.sum())-1) > 1e-10:
            raise FloatingPointError("invalid next-token probabilities")
        return result

    def learn(self, target: int) -> None:
        self._open()
        token_check(target)
        if not self.pending:
            raise RuntimeError("learning requires a pending training prediction")
        self._update(target)
        self.updates += 1
        self.pending = False

    def cancel_prediction(self) -> None:
        self._open()
        if not self.pending:
            raise RuntimeError("no prediction to cancel")
        self.pending = False

    def reset(self) -> None:
        self._open()
        if self.pending:
            raise RuntimeError("pending target must be consumed or explicitly cancelled")
        buffer, parents = self._export()
        x = views(buffer)
        for name in ("s",)+tuple("e_"+name for name in CORE):
            x[name].fill(0)
        x["p"].fill(1/V)
        self._install(buffer, parents)
        self.resets += 1

    def snapshot(self) -> dict[str, Any]:
        buffer, parents = self._export()
        return {"config": dict(CONFIG), "seed": self.seed, "core_learning": self.core_learning, "steps": self.steps,
                "updates": self.updates, "resets": self.resets, "pending": self.pending,
                "buffer": buffer.tolist(), "parents": parents.tolist()}

    def save(self, path: Path) -> str:
        return write_record(path, self.snapshot())

    def restore(self, path: Path) -> None:
        self._open()
        if path.stat().st_size > 4*1024**2:
            raise ValueError("checkpoint exceeds bounded size")
        value = read_record(path)
        fields = {"config", "seed", "core_learning", "steps", "updates", "resets", "pending", "buffer", "parents"}
        if not isinstance(value, dict) or set(value) != fields or value["config"] != CONFIG:
            raise ValueError("incompatible checkpoint schema/configuration")
        if any(type(value[key]) is not int or value[key] < 0 for key in ("seed", "steps", "updates", "resets")):
            raise ValueError("invalid checkpoint counters")
        if any(type(value[key]) is not bool for key in ("core_learning", "pending")):
            raise ValueError("invalid checkpoint flags")
        if value["updates"]+int(value["pending"]) > value["steps"]:
            raise ValueError("invalid checkpoint chronology")
        buffer = np.asarray(value["buffer"], dtype=np.float64)
        raw_parents = np.asarray(value["parents"])
        if buffer.shape != (SIZE,) or not np.isfinite(buffer).all():
            raise ValueError("invalid checkpoint buffer")
        if raw_parents.shape != (H, K) or raw_parents.dtype.kind not in "iu" or np.any(raw_parents < 0) or np.any(raw_parents >= H):
            raise ValueError("invalid checkpoint parent indices")
        parents = raw_parents.astype(np.int32)
        if any(i in row or len(set(row)) != K for i, row in enumerate(parents)):
            raise ValueError("duplicate/self recurrent parent")
        x = views(buffer)
        if (any(np.max(np.abs(x[name])) > 3 for name in PARAMETERS)
                or any(np.max(np.abs(x["e_"+name])) > 4 for name in CORE)
                or np.max(np.abs(x["s"])) > 1 or np.any(x["p"] < 0) or abs(float(x["p"].sum())-1) > 1e-10):
            raise ValueError("checkpoint violates numeric bounds")
        if not np.array_equal(x["b"], views(initial(value["seed"])[0])["b"]):
            raise ValueError("fixed feedback matrix differs")
        self._install(buffer, parents)
        for key in ("seed", "core_learning", "steps", "updates", "resets", "pending"):
            setattr(self, key, value[key])

    def parameter_digest(self) -> str:
        x = views(self._export()[0])
        return hashlib.sha256(b"".join(x[name].tobytes() for name in PARAMETERS)).hexdigest()

    def close(self) -> None:
        self.closed = True
