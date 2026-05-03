# May-2026, Claude and Pat Welch, pat@mousebrains.com
"""Profile-bound cleanup that runs on top of `get_profiles`.

These algorithms only know about ``(depth, channels_dict, fs)`` arrays —
the caller decides which sensors to feed in. That keeps a VMP's
``{Ax, Ay}`` adapter separate from a glider's ``{Ax, Ay, Az, pitch_rate}``
adapter without forking the algorithm.

- :func:`top_trim.compute_trim_depth` finds where shear / acceleration
  variance settles below a quantile threshold (prop-wash exit).
- :func:`bottom.detect_bottom_crash` finds where vibration spikes near
  the deepest bin (seafloor impact).
"""

from odas_tpw.processing.bottom import detect_bottom_crash
from odas_tpw.processing.top_trim import compute_trim_depth, compute_trim_depths

__all__ = ["compute_trim_depth", "compute_trim_depths", "detect_bottom_crash"]
