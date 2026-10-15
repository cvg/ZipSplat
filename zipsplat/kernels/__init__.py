"""Fused Triton kernels for TokenBA's solvers.

A kernel is tensor-in/tensor-out and ships its own pure-torch fallback, so importing this
package never requires Triton:

  * ``normal_equations`` -- JtJ / Jtr reduction of the photometric pose aligner
    (``pose_aligner.normal_equations``); ``analytic_normal_eq`` is None without Triton.
  * ``pnp`` -- whole-solver GNC-IRLS-LM PnP for TokenBA's camera update
    (``zipsplat.pnp.robust_pnp``); ``pnp_gnc`` picks the Triton or torch solver.
"""

from zipsplat.kernels.normal_equations import analytic_normal_eq
from zipsplat.kernels.pnp import pnp_gnc

__all__ = ["analytic_normal_eq", "pnp_gnc"]
