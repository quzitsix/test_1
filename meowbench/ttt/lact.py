"""LaCT fast-weight memory, corrected and Windows-safe.

Derived from the official minimal implementation released with "Test-Time
Training Done Right" (arXiv:2505.23884), MIT licensed — see `LICENSE.LaCT` and
`_lact_upstream.py` for the untouched original.

Two deliberate divergences from that file, both load-bearing:

1. **A correctness fix.** Upstream's `bidirectional_lact_swiglu` writes

       if use_muon:
           w0 = zeropower_via_newtonschulz5(dw0)   # <- assigns to w0, not dw0
           ...
       w0 = w0 + dw0

   which overwrites the fast weight with the orthogonalised *gradient* before
   adding the gradient again. The prior state is destroyed, so the layer cannot
   carry memory at all: with `use_muon=True` (the default) and a learning rate
   of exactly zero — i.e. an update that provably must be a no-op — the layer
   returns all zeros instead of `f_w(q)`. Verified numerically before writing
   this file. Every other implementation in the same repository (the causal
   layer in `minimal_implementations/`, `lact_ar_video/.../ar_lact_swa_repeat.py`,
   and the Triton kernels) writes `dw0 = zeropower(dw0)`, so this is a typo
   confined to the bidirectional minimal file rather than the intended
   algorithm. We follow the majority form.

   This matters here more than it would elsewhere: the entire experiment asks
   whether fast weights *retain* anything, and the upstream bug guarantees the
   answer is "no" for reasons that have nothing to do with the science.

2. **`torch.compile` is opt-in.** Upstream decorates the hot functions with
   `@torch.compile()`. On Windows the inductor CPU backend shells out to MSVC
   and raises `RuntimeError: Compiler: cl is not found` — measured on this
   machine. Compilation is therefore off unless `MEOW_TTT_COMPILE=1`.

The update rule itself is unchanged from the paper: a SwiGLU MLP fast weight
`f_W(x) = W1 (silu(W0 x) * (W2 x))`, a negative-dot-product self-supervised
loss whose manual backward is written out explicitly, optional Muon
orthogonalisation of the update, and an L2 renormalisation that restores each
row's pre-update norm.
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


def _maybe_compile(fn):
    """Compile only when explicitly enabled; see note 2 in the module docstring."""
    if os.environ.get("MEOW_TTT_COMPILE") == "1":
        return torch.compile(fn)
    return fn


@_maybe_compile
def silu_backprop(dy: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """d/dx of silu, given the upstream gradient. Shapes [b, d, l]."""
    sigma = torch.sigmoid(x)
    return dy * sigma * (1 + x * (1 - sigma))


@_maybe_compile
def l2_norm(x: torch.Tensor) -> torch.Tensor:
    """Row-normalise the last dim, preserving dtype (norm upcasts to fp32)."""
    dtype = x.dtype
    return (x / (x.norm(dim=-1, keepdim=True) + 1e-5)).type(dtype)


@_maybe_compile
def zeropower_via_newtonschulz5(G: torch.Tensor) -> torch.Tensor:
    """Newton-Schulz orthogonalisation (Muon), batched over heads. G: [b, d, d']."""
    assert G.dim() == 3, f"expected [b, d, d'], got {tuple(G.shape)}"
    X = G.bfloat16()
    transposed = G.size(1) > G.size(2)
    if transposed:
        X = X.transpose(1, 2)
    X = X / (X.norm(dim=(1, 2), keepdim=True) + 1e-7)
    for a, b, c in [
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ]:
        A = X @ X.transpose(1, 2)
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.transpose(1, 2)
    return X


def lact_update(
    w0: torch.Tensor,  # [b, dh, dk]  fast weight
    w1: torch.Tensor,  # [b, dv, dh]
    w2: torch.Tensor,  # [b, dh, dk]
    k: torch.Tensor,   # [b, l, dk]   keys written this chunk
    v: torch.Tensor,   # [b, l, dv]   values written this chunk
    lr0: torch.Tensor,  # [b, l, 1]
    lr1: torch.Tensor,
    lr2: torch.Tensor,
    *,
    use_muon: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One large-chunk fast-weight update. Returns the new (w0, w1, w2).

    Split out from the upstream fused update+apply so that a caller can *write*
    to memory during ingestion and *read* from it later, which the MEOWBench
    two-phase protocol requires: the video is revoked between the two, so the
    read cannot re-derive the keys.

    The loss is `L = -f_W(k)ᵀ v` (negative dot product, not MSE) exactly as in
    the paper; the backward is hand-written below rather than autograd'd, which
    is what makes the update cheap enough to run per chunk.
    """
    # Pre-update row norms; the fast weights are rescaled back to these after
    # the step, so the update changes direction but not magnitude.
    w0_norm = w0.norm(dim=2, keepdim=True)
    w1_norm = w1.norm(dim=2, keepdim=True)
    w2_norm = w2.norm(dim=2, keepdim=True)

    kt = k.transpose(1, 2)  # [b, dk, l]
    vt = v.transpose(1, 2)  # [b, dv, l]

    # Forward through the fast weights with the keys.
    gate_before_act = torch.bmm(w0, kt)          # [b, dh, l]
    hidden_before_mul = torch.bmm(w2, kt)        # [b, dh, l]
    hidden = F.silu(gate_before_act) * hidden_before_mul

    # Manual backward of L = -f_W(k)ᵀ v.
    dhidden = torch.bmm(w1.transpose(1, 2), vt)  # [b, dh, l]
    dhidden_before_mul = dhidden * F.silu(gate_before_act)
    dgate = dhidden * hidden_before_mul
    dgate_before_act = silu_backprop(dgate, gate_before_act)

    dw1 = torch.bmm(vt, (hidden.transpose(1, 2) * lr1).type_as(vt))
    dw0 = torch.bmm(dgate_before_act, (k * lr0).type_as(dgate_before_act))
    dw2 = torch.bmm(dhidden_before_mul, (k * lr2).type_as(dhidden_before_mul))

    if use_muon:
        # NOTE: orthogonalise the UPDATE, not the weight. See docstring note 1.
        dw0 = zeropower_via_newtonschulz5(dw0).type_as(w0)
        dw1 = zeropower_via_newtonschulz5(dw1).type_as(w1)
        dw2 = zeropower_via_newtonschulz5(dw2).type_as(w2)

    w0 = w0 + dw0
    w1 = w1 + dw1
    w2 = w2 + dw2

    w0 = w0 / (w0.norm(dim=2, keepdim=True) + 1e-5) * w0_norm
    w1 = w1 / (w1.norm(dim=2, keepdim=True) + 1e-5) * w1_norm
    w2 = w2 / (w2.norm(dim=2, keepdim=True) + 1e-5) * w2_norm
    return w0, w1, w2


def lact_apply(
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    q: torch.Tensor,  # [b, l, dk]
) -> torch.Tensor:
    """Read the fast weights with queries. Returns [b, l, dv]."""
    qt = q.transpose(1, 2)
    h = torch.bmm(w2, qt)
    gate = F.silu(torch.bmm(w0, qt))
    return torch.bmm(w1, gate * h).transpose(1, 2)


def inv_softplus(x: float) -> float:
    return x + math.log(-math.expm1(-x))


class LaCTMemory(nn.Module):
    """A persistent LaCT fast-weight memory with an explicit write/read split.

    Unlike the upstream layer — which updates and applies inside a single
    forward pass over one sequence — this keeps the fast weights as buffers that
    survive across calls. That is the whole point here: `write()` is called once
    per ingested session, the video is then revoked by the harness, and `read()`
    is called per question with no access to the original keys.

    Args:
        dim: model width of the incoming token stream.
        head_dim: fast-weight head width; `num_heads = dim // head_dim`.
        inter_multi: SwiGLU hidden expansion.
        base_lr: initial test-time learning rate (softplus-parameterised).
        use_muon: orthogonalise updates. The paper recommends this when the
            chunk is longer than ~2x head_dim.
        learn_projections: when False the q/k/v/o projections are fixed (a
            deterministic, untrained encoder), so anything that survives from
            write to read survived in the fast weights rather than in a learned
            readout.
        projection: "identity" keeps the memory in the encoder's own embedding
            space — q/k/v/o are all the identity, so `read()` returns a vector
            directly comparable to the encoder's other embeddings. "random"
            uses fixed random projections, which is the upstream shape but puts
            the readout in an arbitrarily rotated space; a caller that compares
            `read()` output against encoder embeddings must use "identity" or
            the comparison is meaningless.
    """

    def __init__(
        self,
        dim: int,
        head_dim: int = 64,
        inter_multi: float = 1.0,
        base_lr: float = 1e-2,
        use_muon: bool = True,
        qk_l2_norm: bool = True,
        learn_projections: bool = False,
        projection: str = "identity",
        seed: int | None = 0,
    ) -> None:
        super().__init__()
        if dim % head_dim != 0:
            raise ValueError(f"dim={dim} must be divisible by head_dim={head_dim}")
        if projection not in {"identity", "random"}:
            raise ValueError(f"projection must be 'identity' or 'random', got {projection!r}")
        self.dim = dim
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        self.use_muon = use_muon
        self.qk_l2_norm = qk_l2_norm
        self.base_lr = base_lr
        self.projection = projection

        gen = None
        if seed is not None:
            gen = torch.Generator().manual_seed(seed)

        def _lin(o: int, i: int) -> nn.Parameter:
            if projection == "identity" and o == i:
                w = torch.eye(i)
            else:
                w = torch.randn(o, i, generator=gen) / math.sqrt(i)
            return nn.Parameter(w, requires_grad=learn_projections)

        self.wq = _lin(dim, dim)
        self.wk = _lin(dim, dim)
        self.wv = _lin(dim, dim)
        self.wo = _lin(dim, dim)
        # The learning-rate head is always a real projection: an identity here
        # would make the rate depend on a single coordinate of the input.
        w_lr = torch.randn(3 * self.num_heads, dim, generator=gen) / math.sqrt(dim)
        self.w_lr = nn.Parameter(w_lr, requires_grad=learn_projections)
        self.base_lr_inv = inv_softplus(base_lr)

        d_h = int(head_dim * inter_multi)
        self._init0 = torch.randn(self.num_heads, d_h, head_dim, generator=gen) / math.sqrt(head_dim)
        self._init1 = torch.randn(self.num_heads, head_dim, d_h, generator=gen) / math.sqrt(d_h)
        self._init2 = torch.randn(self.num_heads, d_h, head_dim, generator=gen) / math.sqrt(head_dim)

        self.register_buffer("fw0", self._init0.clone(), persistent=False)
        self.register_buffer("fw1", self._init1.clone(), persistent=False)
        self.register_buffer("fw2", self._init2.clone(), persistent=False)
        self.n_writes = 0

    # -- state ---------------------------------------------------------------

    def reset(self) -> None:
        """Return the fast weights to their initial state.

        Called at `env_begin`: one household must not inherit another's memory,
        which would silently turn a cross-environment leak into apparent recall.
        """
        device, dtype = self.fw0.device, self.fw0.dtype
        self.fw0 = self._init0.clone().to(device=device, dtype=dtype)
        self.fw1 = self._init1.clone().to(device=device, dtype=dtype)
        self.fw2 = self._init2.clone().to(device=device, dtype=dtype)
        self.n_writes = 0

    def state_norm(self) -> float:
        """Frobenius norm of the memory; a cheap drift/saturation diagnostic."""
        with torch.no_grad():
            total = sum(float(w.float().pow(2).sum()) for w in (self.fw0, self.fw1, self.fw2))
        return math.sqrt(total)

    def memory_bytes(self) -> int:
        return sum(w.numel() * w.element_size() for w in (self.fw0, self.fw1, self.fw2))

    # -- projections ---------------------------------------------------------

    def _heads(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """[l, dim] -> [num_heads, l, head_dim]."""
        proj = F.linear(x, w)                        # [l, dim]
        return proj.view(-1, self.num_heads, self.head_dim).transpose(0, 1)

    def _rates(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw = F.linear(x, self.w_lr)                 # [l, 3*heads]
        rates = F.softplus(raw + self.base_lr_inv)
        rates = rates.view(-1, 3, self.num_heads).permute(1, 2, 0).unsqueeze(-1)
        return rates[0], rates[1], rates[2]          # each [heads, l, 1]

    # -- write / read --------------------------------------------------------

    @torch.no_grad()
    def write(self, tokens: torch.Tensor) -> dict:
        """Fold one chunk of tokens into the fast weights.

        `tokens` is [l, dim] — one session's worth of frame embeddings. The
        whole session is a single large chunk, which is LaCT's central claim:
        chunk boundaries should follow the data's own structure (here, a visit
        to the home) rather than a fixed token count.
        """
        if tokens.dim() != 2 or tokens.size(-1) != self.dim:
            raise ValueError(f"expected [l, {self.dim}], got {tuple(tokens.shape)}")
        tokens = tokens.to(self.fw0.device, self.fw0.dtype)

        k = self._heads(tokens, self.wk)
        v = self._heads(tokens, self.wv)
        if self.qk_l2_norm:
            k = l2_norm(k)
        lr0, lr1, lr2 = self._rates(tokens)

        before = self.state_norm()
        self.fw0, self.fw1, self.fw2 = lact_update(
            self.fw0, self.fw1, self.fw2, k, v, lr0, lr1, lr2, use_muon=self.use_muon
        )
        self.n_writes += 1
        return {
            "tokens": int(tokens.size(0)),
            "state_norm_before": before,
            "state_norm_after": self.state_norm(),
            "n_writes": self.n_writes,
        }

    @torch.no_grad()
    def read(self, tokens: torch.Tensor) -> torch.Tensor:
        """Query the memory. `tokens` [l, dim] -> [l, dim]."""
        if tokens.dim() != 2 or tokens.size(-1) != self.dim:
            raise ValueError(f"expected [l, {self.dim}], got {tuple(tokens.shape)}")
        tokens = tokens.to(self.fw0.device, self.fw0.dtype)
        q = self._heads(tokens, self.wq)
        if self.qk_l2_norm:
            q = l2_norm(q)
        out = lact_apply(self.fw0, self.fw1, self.fw2, q)   # [heads, l, head_dim]
        merged = out.transpose(0, 1).reshape(-1, self.dim)
        return F.linear(merged, self.wo)


__all__ = [
    "LaCTMemory",
    "lact_apply",
    "lact_update",
    "l2_norm",
    "zeropower_via_newtonschulz5",
]
