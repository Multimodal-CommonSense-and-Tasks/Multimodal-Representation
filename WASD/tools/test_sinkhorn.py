"""Sanity checks for the WASD scaling iterations in distillm/losses.py.

Usage (from the repository root): python tools/test_sinkhorn.py [--eps 0.5] [--iters 500]

1. Algorithm 1: with a dense (full-support) kernel the scaling vectors returned by
   sparse_sinkhorn_sequence reproduce the marginals of the entropic transport plan
   between p and r. With the nearest-k sparse kernel a feasible plan between two
   arbitrary distributions does not exist in general, so the residual is only reported.
2. Algorithm 2: the potential returned by sparse_sinkhorn_self_sequence equals
   eps * log(u) with u * (K u) = r.
3. Theorem 3: at p == r the WASD weights are constant and the gradient of the
   loss w.r.t. the student logits vanishes.
4. For p != r the sink / wass losses are finite and the gradient is non-zero.
"""
import argparse
import math
import sys
import types

import torch

sys.path.insert(0, ".")
from distillm.losses import sparse_sinkhorn_sequence, sparse_sinkhorn_self_sequence, amid  # noqa: E402


def make_kernel(d, eps, k, device):
    emb = torch.nn.functional.normalize(torch.randn(d, 16, device=device), dim=1)
    cost = 1 - emb @ emb.T
    cost.fill_diagonal_(0.0)
    _, idx = torch.topk(cost, k + 1, largest=False)
    rows = torch.arange(d, device=device).repeat_interleave(k)
    cols = idx[:, 1:].reshape(-1)
    rows_s = torch.cat([rows, cols, torch.arange(d, device=device)])
    cols_s = torch.cat([cols, rows, torch.arange(d, device=device)])
    K_dense = torch.zeros(d, d, device=device)
    K_dense[rows_s, cols_s] = torch.exp(-cost[rows_s, cols_s] / eps)
    K = K_dense.to_sparse_coo().coalesce()
    return K, K.transpose(0, 1).coalesce(), K_dense


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eps", type=float, default=0.5)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--vocab", type=int, default=64)
    parser.add_argument("--knn", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d, eps, B, T = args.vocab, args.eps, 2, 5

    K, Kt, K_dense = make_kernel(d, eps, args.knn, device)
    t_logits = torch.randn(B, T, d, device=device)
    s_logits = torch.randn(B, T, d, device=device)
    p = torch.softmax(t_logits, -1)
    r = torch.softmax(s_logits, -1)
    starts = torch.zeros(B, dtype=torch.long, device=device)
    ends = torch.full((B,), T, dtype=torch.long, device=device)
    ok = True

    # 1. Algorithm 1 marginals: u * (K v) = r and v * (K^T u) = p
    pf, rf = p.reshape(-1, d), r.reshape(-1, d)
    emb = torch.nn.functional.normalize(torch.randn(d, 16, device=device), dim=1)
    K_full = torch.exp(-(1 - emb @ emb.T).fill_diagonal_(0.0) / eps)      # dense kernel: always feasible
    K_full_sp = K_full.to_sparse_coo().coalesce()
    f_pr = sparse_sinkhorn_sequence(p, r, K_full_sp, K_full_sp.transpose(0, 1).coalesce(), starts, ends,
                                    epsilon=eps, max_iter=args.iters, tol=1e-12)
    u = torch.exp(f_pr / eps).reshape(-1, d)
    v = pf / (u @ K_full)             # v = p / (K^T u)
    err1 = (u * (v @ K_full.T) - rf).abs().max().item()    # u * (K v) == r
    print(f"[1] Algorithm 1 marginal error (dense K)  : {err1:.2e}")
    ok &= err1 < 1e-4
    f_pr = sparse_sinkhorn_sequence(p, r, K, Kt, starts, ends, epsilon=eps, max_iter=args.iters, tol=1e-12)
    u = torch.exp(f_pr / eps).reshape(-1, d)
    v = pf / (u @ K_dense)
    err1s = (u * (v @ K_dense.T) - rf).abs().max().item()
    print(f"[1] Algorithm 1 residual (sparse k-NN K)  : {err1s:.2e}  (info only: no feasible plan in general)")

    # 2. Algorithm 2: u * (K u) = r with u = exp(f_rr / eps)
    f_rr = sparse_sinkhorn_self_sequence(r, K, starts, ends, epsilon=eps, max_iter=args.iters, tol=1e-12)
    u_self = torch.exp(f_rr / eps).reshape(-1, d)
    err2 = (u_self * (u_self @ K_dense.T) - rf).abs().max().item()   # u * (K u) == r
    print(f"[2] Algorithm 2 marginal error            : {err2:.2e}")
    ok &= err2 < 1e-4

    # 3. Theorem 3 at p == r: weights constant per position, zero gradient
    wd_args = types.SimpleNamespace(amid_alpha=-1.0, amid_lam=0.0, amid_div_name="sink", amid_div_order="pr",
                                    wd_epsilon=eps, wd_max_iter=args.iters, wd_tol=1e-12, wd_sinkhorn_chunk=1024,
                                    wd_lambda=1.0)
    f_pp = sparse_sinkhorn_sequence(p, p, K, Kt, starts, ends, epsilon=eps, max_iter=args.iters, tol=1e-12)
    f_pp_self = sparse_sinkhorn_self_sequence(p, K, starts, ends, epsilon=eps, max_iter=args.iters, tol=1e-12)
    w = (f_pp - f_pp_self).reshape(-1, d)
    spread = (w.max(dim=1).values - w.min(dim=1).values).max().item()
    print(f"[3] p==r : max spread of weights          : {spread:.2e}")
    logits = t_logits.clone().requires_grad_(True)
    labels = torch.zeros(B, T, dtype=torch.long, device=device)
    loss = amid(logits, t_logits, {"label": labels}, wd_args, mat_K=K, mat_Kt=Kt)
    loss.backward()
    gnorm = logits.grad.norm().item()
    print(f"[3] p==r : grad norm of WASD loss          : {gnorm:.2e}")
    ok &= spread < 1e-3 and gnorm < 1e-3

    # 4. p != r : finite loss, non-zero gradient (sink and wass)
    for name in ("sink", "wass"):
        wd_args.amid_div_name = name
        logits = s_logits.clone().requires_grad_(True)
        loss = amid(logits, t_logits, {"label": labels}, wd_args, mat_K=K, mat_Kt=Kt)
        loss.backward()
        g = logits.grad.norm().item()
        print(f"[4] {name:4s} : loss {loss.item():+.4f} | grad norm {g:.2e} | finite {math.isfinite(loss.item())}")
        ok &= math.isfinite(loss.item()) and g > 1e-6

    print("ALL TESTS PASSED" if ok else "SOME TESTS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
