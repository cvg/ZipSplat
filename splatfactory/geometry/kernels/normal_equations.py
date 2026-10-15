"""Fused Triton kernel of the photometric aligner's Gauss-Newton normal equations.

Builds each pixel's Jacobian row on the fly and reduces it to H = JtJ and g = Jtr across many
blocks, without materialising J. Deterministic (partial sums, no atomics). Without Triton,
`analytic_normal_eq` is None and the aligner uses its torch path (pose_aligner/photometric.py).
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _ane_kernel(
        gx_ptr,
        gy_ptr,
        r_ptr,
        dep_ptr,
        fx_ptr,
        fy_ptr,
        cx_ptr,
        cy_ptr,
        Hp_ptr,
        gp_ptr,
        M,
        W,
        NB,
        BLOCK: tl.constexpr,
    ):
        """Partial H and g of one view and pixel block.

        gx, gy and r are [N, 3, M] with the mask applied, dep is [N, M]. Written with explicit
        scalars, since Triton has no list comprehensions.
        """
        n = tl.program_id(0)
        blk = tl.program_id(1)
        offs = blk * BLOCK + tl.arange(0, BLOCK)
        msk = offs < M
        fx = tl.load(fx_ptr + n)
        fy = tl.load(fy_ptr + n)
        cx = tl.load(cx_ptr + n)
        cy = tl.load(cy_ptr + n)
        Z = tl.load(dep_ptr + n * M + offs, mask=msk, other=1.0)
        Z = tl.where(Z < 1e-4, 1e-4, Z)
        u = (offs % W).to(tl.float32)
        v = (offs // W).to(tl.float32)
        iZ = 1.0 / Z
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        pu0 = fx * iZ
        pu2 = -fx * X * iZ * iZ
        pv1 = fy * iZ
        pv2 = -fy * Y * iZ * iZ
        zz = u * 0.0
        a0 = -pu2 * Y  # Ju
        a1 = -pu0 * Z + pu2 * X
        a2 = pu0 * Y
        a3 = -pu0
        a4 = zz
        a5 = -pu2
        b0 = pv1 * Z - pv2 * Y  # Jv
        b1 = pv2 * X
        b2 = -pv1 * X
        b3 = zz
        b4 = -pv1
        b5 = -pv2
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
        gg0 = 0.0
        gg1 = 0.0
        gg2 = 0.0
        gg3 = 0.0
        gg4 = 0.0
        gg5 = 0.0
        cbase = n * 3 * M
        for c in range(3):
            gxc = tl.load(gx_ptr + cbase + c * M + offs, mask=msk, other=0.0)
            gyc = tl.load(gy_ptr + cbase + c * M + offs, mask=msk, other=0.0)
            rc = tl.load(r_ptr + cbase + c * M + offs, mask=msk, other=0.0)
            j0 = -(gxc * a0 + gyc * b0)
            j1 = -(gxc * a1 + gyc * b1)
            j2 = -(gxc * a2 + gyc * b2)
            j3 = -(gxc * a3 + gyc * b3)
            j4 = -(gxc * a4 + gyc * b4)
            j5 = -(gxc * a5 + gyc * b5)
            h00 += tl.sum(j0 * j0)
            h01 += tl.sum(j0 * j1)
            h02 += tl.sum(j0 * j2)
            h03 += tl.sum(j0 * j3)
            h04 += tl.sum(j0 * j4)
            h05 += tl.sum(j0 * j5)
            h11 += tl.sum(j1 * j1)
            h12 += tl.sum(j1 * j2)
            h13 += tl.sum(j1 * j3)
            h14 += tl.sum(j1 * j4)
            h15 += tl.sum(j1 * j5)
            h22 += tl.sum(j2 * j2)
            h23 += tl.sum(j2 * j3)
            h24 += tl.sum(j2 * j4)
            h25 += tl.sum(j2 * j5)
            h33 += tl.sum(j3 * j3)
            h34 += tl.sum(j3 * j4)
            h35 += tl.sum(j3 * j5)
            h44 += tl.sum(j4 * j4)
            h45 += tl.sum(j4 * j5)
            h55 += tl.sum(j5 * j5)
            gg0 += tl.sum(j0 * rc)
            gg1 += tl.sum(j1 * rc)
            gg2 += tl.sum(j2 * rc)
            gg3 += tl.sum(j3 * rc)
            gg4 += tl.sum(j4 * rc)
            gg5 += tl.sum(j5 * rc)
        hb = (n * NB + blk) * 36
        gb = (n * NB + blk) * 6
        tl.store(Hp_ptr + hb + 0, h00)
        tl.store(Hp_ptr + hb + 1, h01)
        tl.store(Hp_ptr + hb + 2, h02)
        tl.store(Hp_ptr + hb + 3, h03)
        tl.store(Hp_ptr + hb + 4, h04)
        tl.store(Hp_ptr + hb + 5, h05)
        tl.store(Hp_ptr + hb + 6, h01)
        tl.store(Hp_ptr + hb + 7, h11)
        tl.store(Hp_ptr + hb + 8, h12)
        tl.store(Hp_ptr + hb + 9, h13)
        tl.store(Hp_ptr + hb + 10, h14)
        tl.store(Hp_ptr + hb + 11, h15)
        tl.store(Hp_ptr + hb + 12, h02)
        tl.store(Hp_ptr + hb + 13, h12)
        tl.store(Hp_ptr + hb + 14, h22)
        tl.store(Hp_ptr + hb + 15, h23)
        tl.store(Hp_ptr + hb + 16, h24)
        tl.store(Hp_ptr + hb + 17, h25)
        tl.store(Hp_ptr + hb + 18, h03)
        tl.store(Hp_ptr + hb + 19, h13)
        tl.store(Hp_ptr + hb + 20, h23)
        tl.store(Hp_ptr + hb + 21, h33)
        tl.store(Hp_ptr + hb + 22, h34)
        tl.store(Hp_ptr + hb + 23, h35)
        tl.store(Hp_ptr + hb + 24, h04)
        tl.store(Hp_ptr + hb + 25, h14)
        tl.store(Hp_ptr + hb + 26, h24)
        tl.store(Hp_ptr + hb + 27, h34)
        tl.store(Hp_ptr + hb + 28, h44)
        tl.store(Hp_ptr + hb + 29, h45)
        tl.store(Hp_ptr + hb + 30, h05)
        tl.store(Hp_ptr + hb + 31, h15)
        tl.store(Hp_ptr + hb + 32, h25)
        tl.store(Hp_ptr + hb + 33, h35)
        tl.store(Hp_ptr + hb + 34, h45)
        tl.store(Hp_ptr + hb + 35, h55)
        tl.store(gp_ptr + gb + 0, gg0)
        tl.store(gp_ptr + gb + 1, gg1)
        tl.store(gp_ptr + gb + 2, gg2)
        tl.store(gp_ptr + gb + 3, gg3)
        tl.store(gp_ptr + gb + 4, gg4)
        tl.store(gp_ptr + gb + 5, gg5)

    def analytic_normal_eq(gx, gy, r, depth, fxv, fyv, cxv, cyv, Hh, Ww, block: int = 1024):
        """H [B, V, 6, 6] and g [B, V, 6] of the analytic photometric Jacobian.

        gx, gy and r are [B, V, 3, H, W] with the mask applied, depth is [B, V, H, W] and the
        intrinsics are [B, V].
        """
        b, vv = gx.shape[:2]
        N, M = b * vv, Hh * Ww
        gxf = gx.reshape(N, 3, M).contiguous()
        gyf = gy.reshape(N, 3, M).contiguous()
        rf = r.reshape(N, 3, M).contiguous()
        depf = depth.reshape(N, M).contiguous()
        fxf = fxv.reshape(N).contiguous().float()
        fyf = fyv.reshape(N).contiguous().float()
        cxf = cxv.reshape(N).contiguous().float()
        cyf = cyv.reshape(N).contiguous().float()
        NB = triton.cdiv(M, block)
        Hp = torch.empty(N, NB, 36, device=gx.device, dtype=torch.float32)
        gp = torch.empty(N, NB, 6, device=gx.device, dtype=torch.float32)
        _ane_kernel[(N, NB)](gxf, gyf, rf, depf, fxf, fyf, cxf, cyf, Hp, gp, M, Ww, NB, BLOCK=block)
        return Hp.sum(1).reshape(b, vv, 6, 6), gp.sum(1).reshape(b, vv, 6)

else:  # pragma: no cover, the aligner falls back to its torch path
    analytic_normal_eq = None
