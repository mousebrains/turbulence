# May-2026, Claude and Pat Welch, pat@mousebrains.com
"""Per-segment QC gate.

For each entry along a dissipation / chi dataset's ``time`` dimension we
sample each named hotel-injected QC channel over that segment's time
window and OR-reduce. Nonzero output = "drop this segment." The flag is
always written; whether the corresponding epsilon / chi values are
NaN'd is governed by ``qc.drop_action``.

The hotel-injected channels are loaded as slow-rate numpy arrays on
``pf.channels``. They are expected to be uint8 CF bitfields (or
boolean), so a bitwise OR across all named channels gives one uint8
``qc_drop`` per segment, preserving the source ``flag_meanings`` /
``flag_masks`` attrs.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def compute_segment_drop(
    seg_times: np.ndarray,
    diss_length_seconds: float,
    pf: Any,
    drop_from: Sequence[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    """OR-reduce the named slow-rate hotel channels over each segment.

    Parameters
    ----------
    seg_times : (n_seg,) float64
        Segment center times in seconds since ``pf.start_time``. (Matches
        the ``time`` coord on the diss / chi xr.Dataset returned by
        ``_compute_epsilon`` / ``_compute_chi``.)
    diss_length_seconds : float
        Length of one dissipation window in seconds. The drop test
        covers ``[t_center - L/2, t_center + L/2]``.
    pf : PFile
        The merged PFile holding ``channels``, ``channel_info``, and
        ``t_slow``. The named drop_from channels must exist on
        ``pf.channels`` at slow rate.
    drop_from : list of str
        Hotel channel names to OR together. Channels missing from
        ``pf.channels`` are skipped silently — that lets a config name
        a channel that an instrument's hotel file simply doesn't have
        (e.g. a VMP run without a thruster).

    Returns
    -------
    qc_drop : (n_seg,) uint8
        Bitwise-OR of the source channel bits over each segment window.
    attrs : dict
        ``flag_meanings`` / ``flag_masks`` aggregated across the source
        channels. Empty when no source channel had those attrs.
    """
    n_seg = len(seg_times)
    qc = np.zeros(n_seg, dtype=np.uint8)

    if not drop_from:
        return qc, {}

    t_slow = np.asarray(pf.t_slow, dtype=np.float64)
    half = float(diss_length_seconds) / 2.0
    starts = np.asarray(seg_times, dtype=np.float64) - half
    ends = np.asarray(seg_times, dtype=np.float64) + half

    # searchsorted gives O(log n) lookup of segment boundaries on the
    # slow grid; for each segment, OR-reduce the slice.
    lo = np.searchsorted(t_slow, starts, side="left")
    hi = np.searchsorted(t_slow, ends, side="right")

    flag_meanings: list[str] = []
    flag_masks: list[int] = []
    seen_meanings: set[str] = set()

    for name in drop_from:
        if name not in pf.channels:
            continue
        arr = np.asarray(pf.channels[name])
        # Uint8 cast covers bool/int sources; anything truthy contributes.
        arr_u = arr.astype(np.uint8, copy=False)
        for i in range(n_seg):
            a, b = lo[i], hi[i]
            if a >= b:
                continue
            qc[i] |= np.bitwise_or.reduce(arr_u[a:b])

        # Lift CF flag attrs through if the source channel carries them.
        info = pf.channel_info.get(name, {}) if hasattr(pf, "channel_info") else {}
        meanings = info.get("flag_meanings")
        masks = info.get("flag_masks")
        if meanings and masks is not None:
            mlist = (
                meanings.split() if isinstance(meanings, str) else list(meanings)
            )
            mklist: list[int] = (
                np.asarray(masks).ravel().astype(np.int64).tolist()
                if not np.isscalar(masks)
                else [int(masks)]  # type: ignore[arg-type]
            )
            for word, mask in zip(mlist, mklist):
                if word not in seen_meanings:
                    seen_meanings.add(word)
                    flag_meanings.append(word)
                    flag_masks.append(int(mask))

    attrs: dict[str, Any] = {}
    if flag_meanings:
        attrs["flag_meanings"] = " ".join(flag_meanings)
        attrs["flag_masks"] = np.array(flag_masks, dtype=np.uint8)
    return qc, attrs


def apply_qc_to_dataset(
    ds,
    pf: Any,
    drop_from: Sequence[str],
    diss_length_seconds: float,
    flag_var_name: str,
    value_vars: Sequence[str],
    drop_action: str = "nan",
) -> Any:
    """Add ``flag_var_name`` to *ds* and (optionally) NaN the value vars.

    Returns the (mutated) dataset. ``ds`` must have a ``time`` coord
    holding segment center times in seconds since ``pf.start_time``.
    """
    import xarray as xr

    seg_times = np.asarray(ds["time"].values, dtype=np.float64)
    qc, attrs = compute_segment_drop(
        seg_times, diss_length_seconds, pf, drop_from
    )
    ds[flag_var_name] = xr.DataArray(qc, dims=["time"], attrs=attrs)
    if drop_action == "nan":
        mask = qc > 0
        if np.any(mask):
            for v in value_vars:
                if v not in ds.data_vars:
                    continue
                # Float-only fields get NaN'd; integer fields are left as is.
                if np.issubdtype(ds[v].dtype, np.floating):
                    arr = ds[v].values.copy()
                    if arr.ndim == 1:
                        arr[mask] = np.nan
                    else:
                        # 2-D (probe, time) — broadcast the mask.
                        arr[..., mask] = np.nan
                    ds[v].values[...] = arr
    return ds
