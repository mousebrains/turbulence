# Aug-2026, Claude and Pat Welch, pat@mousebrains.com
"""Per-file instrument clock synchronization against a reference pressure record.

A MicroRider's clock jumps between ``.p`` files and runs at the wrong rate.
When it is deployed alongside an instrument whose clock is trusted, the wave
band in a shared pressure signal recovers the offset and the drift for each
file to a few milliseconds.

See ``docs/clocksync/runbook.md``.
"""

from odas_tpw.clocksync.fit import FileFit, fit_file
from odas_tpw.clocksync.lag import LagResult, bandpass, estimate_lag, shift
from odas_tpw.clocksync.qc import check_continuity, summarize
from odas_tpw.clocksync.reference import load_reference
from odas_tpw.clocksync.series import PressureSeries, epoch_from_datenum

__all__ = [
    "FileFit",
    "LagResult",
    "PressureSeries",
    "bandpass",
    "check_continuity",
    "epoch_from_datenum",
    "estimate_lag",
    "fit_file",
    "load_reference",
    "shift",
    "summarize",
]
