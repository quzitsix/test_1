# Extracted from a1600012888/LaCT (MIT). Pure PyTorch, no Triton, no flash-attn, no fla, no einops.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def silu_backprop(dy, x):
    s = torch.sigmoid(x); return dy * s * (1 + x * (1 - s))

def l2_norm(x):
    t = x.dtype; return (x / (x.norm(dim=-1, keepdim=True) + 1e-5)).type(t)

def zeropower_via_newtonschulz5(G):
    X = G.bfloat16()
    tr = G.size(1) > G.size(2)
    if tr: X = X.transpose(1, 2)
    X = X / (X.norm(dim=(1, 2), keepdim=True) + 1e-7)
    for a, b, c in [(4.0848,-6.8946,2.9270),(3.9505,-6.3029,2.6377),
                    (3.7418,-5.5913,2.3037),(2.8769,-3.1427,1.2046),
                    (2.8366,-3.0525,1.2012)]:
        A = X @ X.transpose(1, 2); B = b * A + c * A @ A; X = a * X + B @ X
    if tr: X = X.transpose(1, 2)
    return X.type_as(G)

def inv_softplus(x): return x + math.log(-math.expm1(-x))

def lact_update(w0, w1, w2, k, v, lr0, lr1, lr2, w_norms,
                chunk_size=2048, use_muon=True, momentum=None, prenorm=True):
    """Fast-weight UPDATE only. Returns (w0,w1,w2) after consuming k/v."""
    w0n, w1n, w2n = w_norms
    w0m, w1m, w2m = w0, w1, w2
    if momentum is not None:
        m0 = torch.zeros_like(w0); m1 = torch.zeros_like(w1); m2 = torch.zeros_like(w2)
    L = k.shape[1]
    for s in range(0, L, chunk_size):
        e = min(s + chunk_size, L)
        ki = k[:, s:e, :]; vi = v[:, s:e, :].transpose(1, 2)
        l0 = lr0[:, s:e, :]; l1 = lr1[:, s:e, :]; l2 = lr2[:, s:e, :]
        gba = torch.bmm(w0, ki.transpose(1, 2))
        hbm = torch.bmm(w2, ki.transpose(1, 2))
        hid = F.silu(gba) * hbm
        dhid = torch.bmm(w1.transpose(1, 2), vi)
        dhbm = dhid * F.silu(gba)
        dgba = silu_backprop(dhid * hbm, gba)
        dw1 = torch.bmm(vi, (hid.transpose(1, 2) * l1).type_as(vi))
        dw0 = torch.bmm(dgba, (ki * l0).type_as(dgba))
        dw2 = torch.bmm(dhbm, (ki * l2).type_as(dhbm))
        if momentum is not None:
            mi = momentum[:, s:e, :].mean(dim=1, keepdim=True)
            dw0 = dw0 + m0 * mi; dw1 = dw1 + m1 * mi; dw2 = dw2 + m2 * mi
            m0, m1, m2 = dw0, dw1, dw2
        if use_muon:
            dw0 = zeropower_via_newtonschulz5(dw0)
            dw1 = zeropower_via_newtonschulz5(dw1)
            dw2 = zeropower_via_newtonschulz5(dw2)
        w0m = w0m + dw0; w1m = w1m + dw1; w2m = w2m + dw2
        w0 = w0m / (w0m.norm(dim=2, keepdim=True) + 1e-5) * w0n
        w1 = w1m / (w1m.norm(dim=2, keepdim=True) + 1e-5) * w1n
        w2 = w2m / (w2m.norm(dim=2, keepdim=True) + 1e-5) * w2n
        if not prenorm: w0m, w1m, w2m = w0, w1, w2
    return w0, w1, w2

def lact_apply(w0, w1, w2, q):
    """READ-ONLY apply. q: [b,l,dk] -> [b,l,dv]. No fast-weight change."""
    qT = q.transpose(1, 2)
    h = torch.bmm(w2, qT); g = F.silu(torch.bmm(w0, qT))
    return torch.bmm(w1, g * h).transpose(1, 2)

class LaCTMemory(nn.Module):
    """LaCT fast-weight memory with an EXPLICIT ingest/read boundary."""
    def __init__(self, dim, head_dim, inter_multi=1.0, base_lr=1e-2,
                 chunk_size=2048, use_muon=True, use_momentum=True, prenorm=True):
        super().__init__()
        assert dim % head_dim == 0
        self.dim, self.head_dim = dim, head_dim
        self.nh = dim // head_dim
        d_in = d_out = head_dim; d_h = int(head_dim * inter_multi)
        self.to_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.lr_proj = nn.Linear(dim, 3 * self.nh, bias=False)
        self.base_lr_inv = inv_softplus(base_lr)
        self.w0 = nn.Parameter(torch.randn(self.nh, d_h, d_in) / math.sqrt(d_in))
        self.w1 = nn.Parameter(torch.randn(self.nh, d_out, d_h) / math.sqrt(d_h))
        self.w2 = nn.Parameter(torch.randn(self.nh, d_h, d_in) / math.sqrt(d_in))
        self.o_norm = nn.RMSNorm(head_dim, eps=1e-5)
        self.use_momentum = use_momentum
        if use_momentum:
            self.momentum_proj = nn.Sequential(nn.Linear(dim, self.nh), nn.Sigmoid())
        self.chunk_size, self.use_muon, self.prenorm = chunk_size, use_muon, prenorm

    def _heads(self, t, b):
        return t.reshape(b, -1, self.nh, self.head_dim).permute(0, 2, 1, 3).reshape(b * self.nh, -1, self.head_dim)

    def _feats(self, x):
        b = x.shape[0]
        qkv = F.silu(self.to_qkv(x))
        q, k, v = qkv.chunk(3, dim=-1)
        q = l2_norm(self._heads(q, b)); k = l2_norm(self._heads(k, b)); v = self._heads(v, b)
        lr = F.softplus(self.lr_proj(x).float() + self.base_lr_inv)  # [b,l,3*nh]
        lr = lr.reshape(b, -1, 3, self.nh).permute(0, 3, 1, 2).reshape(b * self.nh, -1, 3)
        mom = None
        if self.use_momentum:
            m = self.momentum_proj(x).float()
            mom = m.permute(0, 2, 1).reshape(b * self.nh, -1, 1)
        return q, k, v, lr[..., 0:1], lr[..., 1:2], lr[..., 2:3], mom

    def init_state(self, b):
        f = lambda w: w.repeat(b, 1, 1).float()
        w0, w1, w2 = f(self.w0), f(self.w1), f(self.w2)
        norms = (w0.norm(dim=2, keepdim=True), w1.norm(dim=2, keepdim=True), w2.norm(dim=2, keepdim=True))
        return {"w": (w0, w1, w2), "norms": norms}

    @torch.no_grad()
    def ingest(self, x, state):
        _, k, v, l0, l1, l2, mom = self._feats(x)
        w0, w1, w2 = lact_update(*state["w"], k, v, l0, l1, l2, state["norms"],
                                 self.chunk_size, self.use_muon, mom, self.prenorm)
        return {"w": (w0, w1, w2), "norms": state["norms"]}

    @torch.no_grad()
    def read(self, x, state):
        b = x.shape[0]
        q, *_ = self._feats(x)
        o = lact_apply(*state["w"], q)
        o = self.o_norm(o)
        o = o.reshape(b, self.nh, -1, self.head_dim).permute(0, 2, 1, 3).reshape(b, -1, self.dim)
        return self.o_proj(o)

    def state_bytes(self, state):
        return sum(w.numel() * w.element_size() for w in state["w"])
