# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Bottom crash detection for VMP profiles.

Detects where the VMP impacts the seafloor using accelerometer signals
and returns the bottom depth for trimming.

Reference: Code/bottom_crash_profiles.m (currently a stub in Matlab)
"""


import numpy as np
import numpy.typing as npt


def detect_bottom_crash(
    depth_fast: npt.ArrayLike,
    Ax: npt.ArrayLike,
    Ay: npt.ArrayLike,
    fs: float,
    *,
    depth_window: float = 4.0,
    depth_minimum: float = 10.0,
    speed_factor: float = 0.3,
    median_factor: float = 1.0,
    vibration_frequency: int = 16,
    vibration_factor: float = 4.0,
) -> float | None:
    """Detect bottom crash from accelerometer signals.

    Bins acceleration variance by depth and looks for a spike in variance
    near the maximum depth, indicating seafloor impact.

    Parameters
    ----------
    depth_fast : array_like
        Depth array (positive downward, fast rate) [m].
    Ax, Ay : array_like
        Accelerometer channels (fast rate) [m/s^2].
    fs : float
        Fast sampling rate [Hz].
    depth_window : float
        Depth window for variance computation [m].
    depth_minimum : float
        Minimum depth to start searching [m].
    speed_factor : float
        Speed reduction factor for crash identification.
    median_factor : float
        Acceleration std dev filter factor.
    vibration_frequency : int
        Frequency for vibration binning [Hz].
    vibration_factor : float
        Vibration std dev acceptance factor.

    Returns
    -------
    float or None
        Bottom depth [m], or None if no crash detected.
    """
    depth = np.asarray(depth_fast, dtype=np.float64)
    Ax = np.asarray(Ax, dtype=np.float64)
    Ay = np.asarray(Ay, dtype=np.float64)

    max_depth = np.nanmax(depth)
    if max_depth < depth_minimum:
        return None

    # Bin acceleration variance by depth
    bin_size = depth_window
    bins = np.arange(depth_minimum, max_depth + bin_size, bin_size)
    if len(bins) < 2:
        return None

    accel_mag = np.sqrt(Ax**2 + Ay**2)
    bin_std = np.full(len(bins) - 1, np.nan)
    for i in range(len(bins) - 1):
        mask = (depth >= bins[i]) & (depth < bins[i + 1])
        vals = accel_mag[mask]
        if len(vals) > 1:
            bin_std[i] = np.nanstd(vals)

    valid = np.isfinite(bin_std)
    if np.sum(valid) < 3:
        return None

    med_std = np.nanmedian(bin_std[valid])
    threshold = med_std * vibration_factor

    # Search from deepest bin upward for spike
    for i in range(len(bin_std) - 1, -1, -1):
        if np.isfinite(bin_std[i]) and bin_std[i] > threshold:
            bottom_depth = bins[i]
            return float(bottom_depth)

    return None
