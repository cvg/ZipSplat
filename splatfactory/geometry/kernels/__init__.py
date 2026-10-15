"""Fused Triton kernels of the geometry solvers, each with a pure-torch fallback.

    normal_equations  JtJ and Jtr of the photometric aligner, None without Triton.
    pnp               The robust PnP solve of TokenBA's camera update.
"""

from splatfactory.geometry.kernels.normal_equations import analytic_normal_eq
from splatfactory.geometry.kernels.pnp import pnp_gnc

__all__ = ["analytic_normal_eq", "pnp_gnc"]
