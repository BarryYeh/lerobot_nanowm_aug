"""CountSketch projection of per-sample gradients -- stands in for TRAK's CudaProjector.

TRAK's dense Rademacher JL needs the `fast_jl` CUDA extension (no nvcc on this box); its pure
torch fallback materializes a (P x k) matrix, 1e12 entries at P = 260.8M. CountSketch is also a
JL-type sketch: coordinate p goes to one bucket h(p) with a random sign s(p), so
E[<Sx, Sy>] = <x, y> and the projection is one index_add per parameter tensor, O(P).

Buckets/signs are derived per parameter NAME (not flat offset), so restricting `param_prefixes`
does not reshuffle the sketch of the parameters that remain.

`n_sketches > 1` applies that many INDEPENDENT sketches to the same gradients. Their outputs live
in different random subspaces and must never be averaged with each other -- score_pairs.py instead
computes a score per sketch and averages the SCORES, which cuts the sketch's own variance ~n-fold
(Spearman-Brown) for the cost of extra index_adds, the gradient being computed only once.

Buckets are int16 (valid while proj_dim <= 32767) to halve the resident table: each sketch costs
P * 3 bytes on the GPU, ~0.78 GB at P = 260.8M.
"""

from __future__ import annotations

import zlib

import torch


class CountSketch:
    def __init__(
        self,
        shapes: dict[str, torch.Size],
        proj_dim: int,
        seed: int,
        device: str = "cuda",
        n_sketches: int = 1,
    ):
        assert proj_dim <= 32767, "int16 buckets require proj_dim <= 32767"
        self.proj_dim = proj_dim
        self.n_sketches = n_sketches
        self.buckets: list[dict[str, torch.Tensor]] = []
        self.signs: list[dict[str, torch.Tensor]] = []
        for m in range(n_sketches):
            bk, sg = {}, {}
            for name, shape in shapes.items():
                n = int(torch.Size(shape).numel())
                g = torch.Generator(device=device).manual_seed(
                    (seed * 1_000_003 + m * 7_919 + zlib.crc32(name.encode())) % (2**63)
                )
                bk[name] = torch.randint(0, proj_dim, (n,), generator=g, device=device, dtype=torch.int16)
                sg[name] = torch.randint(0, 2, (n,), generator=g, device=device, dtype=torch.int8)
            self.buckets.append(bk)
            self.signs.append(sg)

    @torch.no_grad()
    def project(self, grads: dict[str, torch.Tensor]) -> torch.Tensor:
        """{name: (B, *shape)} -> (n_sketches, B, proj_dim). Consumes `grads` to free memory early."""
        out = None
        for name in list(grads):
            g = grads.pop(name)
            g = g.reshape(g.shape[0], -1).float()
            if out is None:
                out = torch.zeros(self.n_sketches, g.shape[0], self.proj_dim, device=g.device)
            for m in range(self.n_sketches):
                signed = torch.where(self.signs[m][name].bool(), g, -g)
                out[m].index_add_(1, self.buckets[m][name].int(), signed)
                del signed
            del g
        return out
