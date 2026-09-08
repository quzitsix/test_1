"""Tests for the LaCT fast-weight memory.

These pin two separate things:

* that the corrected update is *mathematically* an update — the upstream
  bidirectional file fails `test_zero_lr_is_a_noop` outright, which is how the
  bug was found;
* that the memory demonstrably *retains* written content, because an experiment
  asking "does TTT retain bindings" is meaningless if the substrate cannot
  retain anything at all.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from meowbench.ttt.lact import LaCTMemory, lact_apply, lact_update


def _fw(dim: int = 8, heads: int = 1, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    w0 = torch.randn(heads, dim, dim, generator=g)
    w1 = torch.randn(heads, dim, dim, generator=g)
    w2 = torch.randn(heads, dim, dim, generator=g)
    return w0, w1, w2


def test_zero_lr_is_a_noop():
    """A zero learning rate must leave the fast weights untouched.

    This is the regression test for the upstream bug: `w0 = zeropower(dw0)`
    overwrites the weight with the orthogonalised gradient, so with lr=0 the
    layer returns zeros instead of `f_w(q)`. It fails on the original file for
    both muon settings and passes here.
    """
    torch.manual_seed(0)
    heads, length, dim = 2, 8, 4
    w0, w1, w2 = _fw(dim, heads)
    k = torch.randn(heads, length, dim)
    v = torch.randn(heads, length, dim)
    zero = [torch.zeros(heads, length, 1) for _ in range(3)]

    for use_muon in (False, True):
        n0, n1, n2 = lact_update(
            w0.clone(), w1.clone(), w2.clone(), k, v, *zero, use_muon=use_muon
        )
        assert torch.allclose(n0, w0, atol=1e-4), f"w0 moved with lr=0 (muon={use_muon})"
        assert torch.allclose(n1, w1, atol=1e-4), f"w1 moved with lr=0 (muon={use_muon})"
        assert torch.allclose(n2, w2, atol=1e-4), f"w2 moved with lr=0 (muon={use_muon})"


def test_zero_lr_preserves_the_readout():
    """With lr=0 the read must equal f_w(q) under the original weights."""
    torch.manual_seed(0)
    heads, length, dim = 1, 6, 4
    w0, w1, w2 = _fw(dim, heads)
    q = torch.randn(heads, length, dim)
    k = torch.randn(heads, length, dim)
    v = torch.randn(heads, length, dim)
    zero = [torch.zeros(heads, length, 1) for _ in range(3)]

    qt = q.transpose(1, 2)
    reference = torch.bmm(w1, F.silu(torch.bmm(w0, qt)) * torch.bmm(w2, qt)).transpose(1, 2)

    for use_muon in (False, True):
        n0, n1, n2 = lact_update(
            w0.clone(), w1.clone(), w2.clone(), k, v, *zero, use_muon=use_muon
        )
        got = lact_apply(n0, n1, n2, q)
        assert torch.allclose(got, reference, atol=1e-3), f"readout changed (muon={use_muon})"


@pytest.mark.parametrize("use_muon", [False, True])
def test_update_preserves_row_norms(use_muon: bool):
    """The L2 renormalisation restores each row's pre-update norm."""
    torch.manual_seed(0)
    heads, length, dim = 2, 16, 8
    w0, w1, w2 = _fw(dim, heads)
    k = torch.randn(heads, length, dim)
    v = torch.randn(heads, length, dim)
    lr = [torch.full((heads, length, 1), 0.05) for _ in range(3)]

    n0, n1, n2 = lact_update(w0.clone(), w1.clone(), w2.clone(), k, v, *lr, use_muon=use_muon)
    for before, after in ((w0, n0), (w1, n1), (w2, n2)):
        assert torch.allclose(
            before.norm(dim=2), after.norm(dim=2), rtol=1e-3, atol=1e-3
        )


@pytest.mark.parametrize("use_muon", [False, True])
def test_a_write_actually_changes_the_state(use_muon: bool):
    """A nonzero learning rate must move the memory."""
    mem = LaCTMemory(dim=64, head_dim=16, use_muon=use_muon, seed=0)
    before = mem.fw0.clone()
    mem.write(torch.randn(24, 64))
    assert not torch.allclose(before, mem.fw0), "write() left the fast weights unchanged"
    assert mem.n_writes == 1


@pytest.mark.parametrize("use_muon", [False, True])
def test_memory_retains_what_was_written(use_muon: bool):
    """The central sanity check: reading a written key beats reading a novel one.

    LaCT's fast weight is an associative memory trained by one gradient step on
    `-f_W(k)ᵀv`, so after writing (k, v) the readout at k should align with v
    more than at an unrelated query. If this fails there is no memory to study
    and every downstream retention number would be noise.
    """
    torch.manual_seed(0)
    dim = 64
    mem = LaCTMemory(dim=dim, head_dim=16, base_lr=0.05, use_muon=use_muon, seed=0)

    written = torch.randn(32, dim)
    mem.write(written)

    # Read at the written tokens vs. at unrelated tokens drawn the same way.
    novel = torch.randn(32, dim)
    out_written = mem.read(written)
    out_novel = mem.read(novel)

    # The value the memory was asked to associate with `written`.
    target = F.linear(written.to(mem.fw0.dtype), mem.wv)

    sim_written = F.cosine_similarity(out_written.flatten(), target.flatten(), dim=0)
    sim_novel = F.cosine_similarity(out_novel.flatten(), target.flatten(), dim=0)
    assert sim_written > sim_novel, (
        f"memory does not discriminate written from novel content "
        f"(written={sim_written:.4f} novel={sim_novel:.4f}, muon={use_muon})"
    )


def test_reset_restores_the_initial_state():
    """`env_begin` must not inherit the previous household's memory."""
    mem = LaCTMemory(dim=64, head_dim=16, seed=0)
    pristine = mem.fw0.clone()
    mem.write(torch.randn(16, 64))
    assert not torch.allclose(pristine, mem.fw0)
    mem.reset()
    assert torch.allclose(pristine, mem.fw0)
    assert mem.n_writes == 0


def test_state_is_fixed_size_regardless_of_input_length():
    """Memory footprint must not grow with the stream — the O(1) claim."""
    mem = LaCTMemory(dim=64, head_dim=16, seed=0)
    size = mem.memory_bytes()
    for length in (8, 64, 512):
        mem.write(torch.randn(length, 64))
        assert mem.memory_bytes() == size


def test_write_rejects_wrong_width():
    mem = LaCTMemory(dim=64, head_dim=16, seed=0)
    with pytest.raises(ValueError):
        mem.write(torch.randn(8, 32))


def test_dim_must_divide_by_head_dim():
    with pytest.raises(ValueError):
        LaCTMemory(dim=100, head_dim=32)


def test_state_norm_is_finite_after_many_writes():
    """Long streams must not blow up or collapse the state.

    Renormalisation bounds the magnitude, so this is really a NaN/inf guard on
    the update path — the failure mode that would silently poison a long run.
    """
    mem = LaCTMemory(dim=64, head_dim=16, base_lr=0.05, seed=0)
    for _ in range(50):
        mem.write(torch.randn(32, 64))
    norm = mem.state_norm()
    assert math.isfinite(norm) and norm > 0
    assert torch.isfinite(mem.fw0).all()
