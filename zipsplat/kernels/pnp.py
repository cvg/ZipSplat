"""Fused Triton kernel of TokenBA's PnP camera update (zipsplat/pnp.py).

The whole robust solve runs as one launch, one program per view hypothesis: Cauchy IRLS under a
graduated non-convexity scale ladder, per-view Marquardt damping with accept / reject, and
behind-camera points masked out. The first call compiles the kernel, which is cached on disk.
`_pnp_gnc_torch` is the pure-torch twin, used without Triton.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import torch

from zipsplat.pose import so3exp_map

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover
    _HAS_TRITON = False

_Z_EPS = 1e-4


if _HAS_TRITON:

    @triton.jit
    def _pnp_kernel(
        R_ptr,  # [V, 9] w2c rotation, row-major
        t_ptr,  # [V, 3]
        f_ptr,  # [V, 2]
        c_ptr,  # [V, 2]
        pts_ptr,  # [V, M, 3]
        obs_ptr,  # [M, 2] shared patch centers
        scales_ptr,  # [NS]
        M: tl.constexpr,
        NS: tl.constexpr,
        ITERS: tl.constexpr,
        BLOCK: tl.constexpr,
        damping: tl.constexpr,
        EPS: tl.constexpr,
    ):
        v = tl.program_id(0)

        # pose + intrinsics in registers
        r00 = tl.load(R_ptr + v * 9 + 0)
        r01 = tl.load(R_ptr + v * 9 + 1)
        r02 = tl.load(R_ptr + v * 9 + 2)
        r10 = tl.load(R_ptr + v * 9 + 3)
        r11 = tl.load(R_ptr + v * 9 + 4)
        r12 = tl.load(R_ptr + v * 9 + 5)
        r20 = tl.load(R_ptr + v * 9 + 6)
        r21 = tl.load(R_ptr + v * 9 + 7)
        r22 = tl.load(R_ptr + v * 9 + 8)
        tx = tl.load(t_ptr + v * 3 + 0)
        ty = tl.load(t_ptr + v * 3 + 1)
        tz = tl.load(t_ptr + v * 3 + 2)
        fx = tl.load(f_ptr + v * 2 + 0)
        fy = tl.load(f_ptr + v * 2 + 1)
        cx = tl.load(c_ptr + v * 2 + 0)
        cy = tl.load(c_ptr + v * 2 + 1)

        lam = damping * 1.0

        for s in range(NS):
            scale = tl.load(scales_ptr + s)
            s2 = scale * scale

            # ---- cost at the current pose ------------------------------------
            cost = 0.0
            for m0 in range(0, M, BLOCK):
                idx = m0 + tl.arange(0, BLOCK)
                mask = idx < M
                px = tl.load(pts_ptr + v * M * 3 + idx * 3 + 0, mask=mask, other=0.0)
                py = tl.load(pts_ptr + v * M * 3 + idx * 3 + 1, mask=mask, other=0.0)
                pz = tl.load(pts_ptr + v * M * 3 + idx * 3 + 2, mask=mask, other=0.0)
                ox = tl.load(obs_ptr + idx * 2 + 0, mask=mask, other=0.0)
                oy = tl.load(obs_ptr + idx * 2 + 1, mask=mask, other=0.0)
                X = r00 * px + r01 * py + r02 * pz + tx
                Y = r10 * px + r11 * py + r12 * pz + ty
                Zr = r20 * px + r21 * py + r22 * pz + tz
                Z = tl.maximum(Zr, EPS)
                u = fx * X / Z + cx
                vv = fy * Y / Z + cy
                ru = u - ox
                rv = vv - oy
                r2 = ru * ru + rv * rv
                val = tl.where(mask & (Zr > EPS), 0.5 * s2 * tl.log(1.0 + r2 / s2), 0.0)
                cost += tl.sum(val)

            for _ in range(ITERS):
                # ---- accumulate H (21 sym entries), g (6) ---------------------
                h00 = 0.0
                h01 = 0.0
                h02 = 0.0
                h03 = 0.0
                h04 = 0.0
                h05 = 0.0
                h11 = 0.0
                h12 = 0.0
                h13 = 0.0
                h14 = 0.0
                h15 = 0.0
                h22 = 0.0
                h23 = 0.0
                h24 = 0.0
                h25 = 0.0
                h33 = 0.0
                h34 = 0.0
                h35 = 0.0
                h44 = 0.0
                h45 = 0.0
                h55 = 0.0
                g0 = 0.0
                g1 = 0.0
                g2 = 0.0
                g3 = 0.0
                g4 = 0.0
                g5 = 0.0
                for m0 in range(0, M, BLOCK):
                    idx = m0 + tl.arange(0, BLOCK)
                    mask = idx < M
                    px = tl.load(pts_ptr + v * M * 3 + idx * 3 + 0, mask=mask, other=0.0)
                    py = tl.load(pts_ptr + v * M * 3 + idx * 3 + 1, mask=mask, other=0.0)
                    pz = tl.load(pts_ptr + v * M * 3 + idx * 3 + 2, mask=mask, other=0.0)
                    ox = tl.load(obs_ptr + idx * 2 + 0, mask=mask, other=0.0)
                    oy = tl.load(obs_ptr + idx * 2 + 1, mask=mask, other=0.0)
                    X = r00 * px + r01 * py + r02 * pz + tx
                    Y = r10 * px + r11 * py + r12 * pz + ty
                    Zr = r20 * px + r21 * py + r22 * pz + tz
                    Z = tl.maximum(Zr, EPS)
                    zi = 1.0 / Z
                    u = fx * X * zi + cx
                    vv = fy * Y * zi + cy
                    ru = u - ox
                    rv = vv - oy
                    r2 = ru * ru + rv * rv
                    w = tl.where(mask & (Zr > EPS), 1.0 / (1.0 + r2 / s2), 0.0)
                    x = X * zi
                    y = Y * zi
                    # Jacobian rows of the (v, omega) twist, closed form
                    ju0 = fx * zi
                    ju2 = -fx * x * zi
                    ju3 = -fx * x * y
                    ju4 = fx * (1.0 + x * x)
                    ju5 = -fx * y
                    jv1 = fy * zi
                    jv2 = -fy * y * zi
                    jv3 = -fy * (1.0 + y * y)
                    jv4 = fy * x * y
                    jv5 = fy * x
                    # ju1 = jv0 = 0
                    h00 += tl.sum(w * ju0 * ju0)
                    h01 += 0.0
                    h02 += tl.sum(w * ju0 * ju2)
                    h03 += tl.sum(w * ju0 * ju3)
                    h04 += tl.sum(w * ju0 * ju4)
                    h05 += tl.sum(w * ju0 * ju5)
                    h11 += tl.sum(w * jv1 * jv1)
                    h12 += tl.sum(w * jv1 * jv2)
                    h13 += tl.sum(w * jv1 * jv3)
                    h14 += tl.sum(w * jv1 * jv4)
                    h15 += tl.sum(w * jv1 * jv5)
                    h22 += tl.sum(w * (ju2 * ju2 + jv2 * jv2))
                    h23 += tl.sum(w * (ju2 * ju3 + jv2 * jv3))
                    h24 += tl.sum(w * (ju2 * ju4 + jv2 * jv4))
                    h25 += tl.sum(w * (ju2 * ju5 + jv2 * jv5))
                    h33 += tl.sum(w * (ju3 * ju3 + jv3 * jv3))
                    h34 += tl.sum(w * (ju3 * ju4 + jv3 * jv4))
                    h35 += tl.sum(w * (ju3 * ju5 + jv3 * jv5))
                    h44 += tl.sum(w * (ju4 * ju4 + jv4 * jv4))
                    h45 += tl.sum(w * (ju4 * ju5 + jv4 * jv5))
                    h55 += tl.sum(w * (ju5 * ju5 + jv5 * jv5))
                    g0 += tl.sum(w * ju0 * ru)
                    g1 += tl.sum(w * jv1 * rv)
                    g2 += tl.sum(w * (ju2 * ru + jv2 * rv))
                    g3 += tl.sum(w * (ju3 * ru + jv3 * rv))
                    g4 += tl.sum(w * (ju4 * ru + jv4 * rv))
                    g5 += tl.sum(w * (ju5 * ru + jv5 * rv))
                # h01 stays 0 since ju1 = jv0 = 0

                # ---- solve (H + lam I) dx = -g, 6x6 Cholesky, unrolled --------
                a00 = h00 + lam
                a11 = h11 + lam
                a22 = h22 + lam
                a33 = h33 + lam
                a44 = h44 + lam
                a55 = h55 + lam
                l00 = tl.sqrt(a00)
                l10 = h01 / l00
                l20 = h02 / l00
                l30 = h03 / l00
                l40 = h04 / l00
                l50 = h05 / l00
                l11 = tl.sqrt(a11 - l10 * l10)
                l21 = (h12 - l20 * l10) / l11
                l31 = (h13 - l30 * l10) / l11
                l41 = (h14 - l40 * l10) / l11
                l51 = (h15 - l50 * l10) / l11
                l22 = tl.sqrt(a22 - l20 * l20 - l21 * l21)
                l32 = (h23 - l30 * l20 - l31 * l21) / l22
                l42 = (h24 - l40 * l20 - l41 * l21) / l22
                l52 = (h25 - l50 * l20 - l51 * l21) / l22
                l33 = tl.sqrt(a33 - l30 * l30 - l31 * l31 - l32 * l32)
                l43 = (h34 - l40 * l30 - l41 * l31 - l42 * l32) / l33
                l53 = (h35 - l50 * l30 - l51 * l31 - l52 * l32) / l33
                l44 = tl.sqrt(a44 - l40 * l40 - l41 * l41 - l42 * l42 - l43 * l43)
                l54 = (h45 - l50 * l40 - l51 * l41 - l52 * l42 - l53 * l43) / l44
                l55 = tl.sqrt(a55 - l50 * l50 - l51 * l51 - l52 * l52 - l53 * l53 - l54 * l54)
                # forward: L z = -g
                z0 = -g0 / l00
                z1 = (-g1 - l10 * z0) / l11
                z2 = (-g2 - l20 * z0 - l21 * z1) / l22
                z3 = (-g3 - l30 * z0 - l31 * z1 - l32 * z2) / l33
                z4 = (-g4 - l40 * z0 - l41 * z1 - l42 * z2 - l43 * z3) / l44
                z5 = (-g5 - l50 * z0 - l51 * z1 - l52 * z2 - l53 * z3 - l54 * z4) / l55
                # backward: Lt dx = z
                d5 = z5 / l55
                d4 = (z4 - l54 * d5) / l44
                d3 = (z3 - l43 * d4 - l53 * d5) / l33
                d2 = (z2 - l32 * d3 - l42 * d4 - l52 * d5) / l22
                d1 = (z1 - l21 * d2 - l31 * d3 - l41 * d4 - l51 * d5) / l11
                d0 = (z0 - l10 * d1 - l20 * d2 - l30 * d3 - l40 * d4 - l50 * d5) / l00

                # ---- Rodrigues: dR = exp([w]x), w = (d3, d4, d5) --------------
                th = tl.sqrt(d3 * d3 + d4 * d4 + d5 * d5)
                small = th < 1e-7
                div = tl.where(small, 1.0, th)
                kx = d3 / div
                ky = d4 / div
                kz = d5 / div
                st = tl.sin(th)
                ct = 1.0 - tl.cos(th)
                # exp = I + sin(th) K + (1 - cos(th)) K^2. For tiny th, K = skew(d) stays
                # unnormalized and st = 1 gives the first-order I + skew(d).
                st = tl.where(small, 1.0, st)
                ct = tl.where(small, 0.0, ct)
                e00 = 1.0 + ct * (-ky * ky - kz * kz)
                e01 = -st * kz + ct * kx * ky
                e02 = st * ky + ct * kx * kz
                e10 = st * kz + ct * kx * ky
                e11 = 1.0 + ct * (-kx * kx - kz * kz)
                e12 = -st * kx + ct * ky * kz
                e20 = -st * ky + ct * kx * kz
                e21 = st * kx + ct * ky * kz
                e22 = 1.0 + ct * (-kx * kx - ky * ky)
                n00 = e00 * r00 + e01 * r10 + e02 * r20
                n01 = e00 * r01 + e01 * r11 + e02 * r21
                n02 = e00 * r02 + e01 * r12 + e02 * r22
                n10 = e10 * r00 + e11 * r10 + e12 * r20
                n11 = e10 * r01 + e11 * r11 + e12 * r21
                n12 = e10 * r02 + e11 * r12 + e12 * r22
                n20 = e20 * r00 + e21 * r10 + e22 * r20
                n21 = e20 * r01 + e21 * r11 + e22 * r21
                n22 = e20 * r02 + e21 * r12 + e22 * r22
                ntx = e00 * tx + e01 * ty + e02 * tz + d0
                nty = e10 * tx + e11 * ty + e12 * tz + d1
                ntz = e20 * tx + e21 * ty + e22 * tz + d2

                # ---- candidate cost + accept/reject ---------------------------
                cost_new = 0.0
                for m0 in range(0, M, BLOCK):
                    idx = m0 + tl.arange(0, BLOCK)
                    mask = idx < M
                    px = tl.load(pts_ptr + v * M * 3 + idx * 3 + 0, mask=mask, other=0.0)
                    py = tl.load(pts_ptr + v * M * 3 + idx * 3 + 1, mask=mask, other=0.0)
                    pz = tl.load(pts_ptr + v * M * 3 + idx * 3 + 2, mask=mask, other=0.0)
                    ox = tl.load(obs_ptr + idx * 2 + 0, mask=mask, other=0.0)
                    oy = tl.load(obs_ptr + idx * 2 + 1, mask=mask, other=0.0)
                    X = n00 * px + n01 * py + n02 * pz + ntx
                    Y = n10 * px + n11 * py + n12 * pz + nty
                    Zr = n20 * px + n21 * py + n22 * pz + ntz
                    Z = tl.maximum(Zr, EPS)
                    u = fx * X / Z + cx
                    vv = fy * Y / Z + cy
                    ru = u - ox
                    rv = vv - oy
                    r2 = ru * ru + rv * rv
                    val = tl.where(mask & (Zr > EPS), 0.5 * s2 * tl.log(1.0 + r2 / s2), 0.0)
                    cost_new += tl.sum(val)

                ok = cost_new < cost
                r00 = tl.where(ok, n00, r00)
                r01 = tl.where(ok, n01, r01)
                r02 = tl.where(ok, n02, r02)
                r10 = tl.where(ok, n10, r10)
                r11 = tl.where(ok, n11, r11)
                r12 = tl.where(ok, n12, r12)
                r20 = tl.where(ok, n20, r20)
                r21 = tl.where(ok, n21, r21)
                r22 = tl.where(ok, n22, r22)
                tx = tl.where(ok, ntx, tx)
                ty = tl.where(ok, nty, ty)
                tz = tl.where(ok, ntz, tz)
                cost = tl.where(ok, cost_new, cost)
                lam = tl.minimum(tl.maximum(tl.where(ok, lam * 0.3, lam * 5.0), 1e-6), 1e4)

        tl.store(R_ptr + v * 9 + 0, r00)
        tl.store(R_ptr + v * 9 + 1, r01)
        tl.store(R_ptr + v * 9 + 2, r02)
        tl.store(R_ptr + v * 9 + 3, r10)
        tl.store(R_ptr + v * 9 + 4, r11)
        tl.store(R_ptr + v * 9 + 5, r12)
        tl.store(R_ptr + v * 9 + 6, r20)
        tl.store(R_ptr + v * 9 + 7, r21)
        tl.store(R_ptr + v * 9 + 8, r22)
        tl.store(t_ptr + v * 3 + 0, tx)
        tl.store(t_ptr + v * 3 + 1, ty)
        tl.store(t_ptr + v * 3 + 2, tz)


def _pnp_gnc_torch(R, t, f, c, pts, obs, scales, iters, damping):
    """Pure-torch twin of the kernel, with the same accept / reject and damping schedule."""
    V = pts.shape[0]
    eye = torch.eye(6, device=pts.device, dtype=pts.dtype)
    lam = torch.full((V,), damping, device=pts.device, dtype=pts.dtype)

    def project(R, t):
        P = pts @ R.transpose(-1, -2) + t[:, None]  # [V, M, 3]
        Z = P[..., 2].clamp(min=_Z_EPS)
        uv = P[..., :2] / Z[..., None] * f[:, None] + c[:, None]
        return uv, P, P[..., 2] > _Z_EPS

    def cost_of(res, valid, scale):
        d2 = res.square().sum(-1)
        return (0.5 * scale**2 * torch.log1p(d2 / scale**2) * valid).sum(-1)

    for scale in scales:
        uv, P, valid = project(R, t)
        cost = cost_of(uv - obs, valid, scale)
        for _ in range(iters):
            res = uv - obs
            w = valid / (1.0 + res.square().sum(-1) / scale**2)
            Z = P[..., 2].clamp(min=_Z_EPS)
            zi = 1.0 / Z
            x, y = P[..., 0] * zi, P[..., 1] * zi
            fx, fy = f[:, None, 0], f[:, None, 1]
            zero = torch.zeros_like(Z)
            # closed-form interaction-matrix rows for the (v, omega) twist on w2c
            cols_u = [fx * zi, zero, -fx * x * zi, -fx * x * y, fx * (1 + x * x), -fx * y]
            cols_v = [zero, fy * zi, -fy * y * zi, -fy * (1 + y * y), fy * x * y, fy * x]
            J = torch.stack([torch.stack(cols_u, -1), torch.stack(cols_v, -1)], -2)  # [V,M,2,6]
            # outer-product + segmented sum: the einsum route lowers to a tiny-output GEMM
            A = J * w[..., None, None]
            H = (A.unsqueeze(-1) * J.unsqueeze(-2)).sum(dim=(1, 2))  # [V, 6, 6]
            g = (A * res[..., None]).sum(dim=(1, 2))  # [V, 6]
            L, _ = torch.linalg.cholesky_ex(H + lam[:, None, None] * eye)
            step = torch.cholesky_solve(-g[..., None], L)[..., 0]
            R_delta = so3exp_map(step[:, 3:])
            R_new = R_delta @ R
            t_new = (R_delta @ t[..., None]).squeeze(-1) + step[:, :3]
            uv_new, P_new, valid_new = project(R_new, t_new)
            cost_new = cost_of(uv_new - obs, valid_new, scale)
            ok = cost_new < cost  # per-view accept
            R = torch.where(ok[:, None, None], R_new, R)
            t = torch.where(ok[:, None], t_new, t)
            uv = torch.where(ok[:, None, None], uv_new, uv)
            P = torch.where(ok[:, None, None], P_new, P)
            valid = torch.where(ok[:, None], valid_new, valid)
            cost = torch.where(ok, cost_new, cost)
            lam = torch.where(ok, lam * 0.3, lam * 5.0).clamp(1e-6, 1e4)
        # no early stop: its host sync costs more than the iterations it would save
    return R, t


def pnp_gnc(
    R: torch.Tensor,
    t: torch.Tensor,
    f: torch.Tensor,
    c: torch.Tensor,
    pts: torch.Tensor,
    obs: torch.Tensor,
    scales: Sequence[float] = (64.0, 12.0, 4.0),
    iters: int = 4,
    damping: float = 1e-3,
    backend: str = "auto",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Robust per-view PnP in the world-to-camera frame. Returns the refined (R, t).

    Args:
        R: Initial w2c rotations [V, 3, 3].
        t: Initial w2c translations [V, 3].
        f: Focal lengths [V, 2].
        c: Principal points [V, 2].
        pts: World points [V, M, 3].
        obs: Pixel each point should project to [M, 2], shared across views.
        scales: Robust-scale ladder in pixels, coarse to fine.
        iters: IRLS iterations per scale.
        damping: Initial Marquardt damping.
        backend: "auto" (Triton on CUDA, else torch), "triton" or "torch".
    """
    use_triton = backend == "triton" or (backend == "auto" and _HAS_TRITON and pts.is_cuda)
    if backend == "triton" and not _HAS_TRITON:
        raise RuntimeError("backend='triton' but Triton is unavailable")
    if not use_triton:
        return _pnp_gnc_torch(R, t, f, c, pts, obs, scales, iters, damping)

    V, M = pts.shape[:2]
    # the kernel refines R/t IN PLACE, so hand it private contiguous copies
    R_ = R.reshape(V, 9).float().contiguous().clone()
    t_ = t.float().contiguous().clone()
    _pnp_kernel[(V,)](
        R_,
        t_,
        f.contiguous().float(),
        c.contiguous().float(),
        pts.contiguous().float(),
        obs.contiguous().float(),
        torch.tensor(tuple(scales), device=pts.device, dtype=torch.float32),
        M=M,
        NS=len(scales),
        ITERS=iters,
        BLOCK=512,
        damping=damping,
        EPS=_Z_EPS,
    )
    return R_.reshape(V, 3, 3), t_
