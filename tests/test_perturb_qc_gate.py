# May-2026, Claude and Pat Welch, pat@mousebrains.com
"""Tests for perturb.qc_gate — segment-level drop machinery."""

from __future__ import annotations

import numpy as np
import xarray as xr

from odas_tpw.perturb.qc_gate import apply_qc_to_dataset, compute_segment_drop


class _StubPF:
    """Minimal PFile-like stub: t_slow, channels, channel_info."""

    def __init__(self, t_slow, channels=None, channel_info=None):
        self.t_slow = np.asarray(t_slow, dtype=np.float64)
        self.channels = channels or {}
        self.channel_info = channel_info or {}


def _make_diss_ds(seg_times, n_probes=2):
    n = len(seg_times)
    return xr.Dataset(
        {
            "epsilonMean": (["time"], np.full(n, 1e-8)),
            "epsilonLnSigma": (["time"], np.full(n, 0.5)),
            "e_1": (["time"], np.full(n, 1e-8)),
            "e_2": (["time"], np.full(n, 2e-8)),
            "epsilon": (["probe", "time"], np.full((n_probes, n), 1e-8)),
        },
        coords={"time": np.asarray(seg_times, dtype=np.float64)},
    )


# ---------------------------------------------------------------------------
# compute_segment_drop
# ---------------------------------------------------------------------------


class TestComputeSegmentDrop:
    def test_no_drop_from_returns_zeros(self):
        seg_times = np.array([1.0, 2.0, 3.0])
        pf = _StubPF(t_slow=np.arange(0, 5))
        qc, attrs = compute_segment_drop(seg_times, 1.0, pf, [])
        assert qc.dtype == np.uint8
        np.testing.assert_array_equal(qc, [0, 0, 0])
        assert attrs == {}

    def test_missing_channel_silently_skipped(self):
        """A channel listed in drop_from but not on pf is fine — the same
        config can name a thruster channel that a VMP run doesn't have."""
        seg_times = np.array([1.0, 2.0])
        pf = _StubPF(t_slow=np.arange(0, 5))
        qc, _ = compute_segment_drop(seg_times, 1.0, pf, ["q_drop_epsilon"])
        np.testing.assert_array_equal(qc, [0, 0])

    def test_or_reduces_within_segment(self):
        # 5-second slow grid; q_drop=1 only at t=2.5.
        t_slow = np.arange(0.0, 5.0, 0.5)
        q = np.zeros_like(t_slow, dtype=np.uint8)
        q[5] = 1  # t=2.5
        pf = _StubPF(t_slow=t_slow, channels={"q_drop": q})

        # Segment centered at t=2.5 with diss_length=2 covers [1.5, 3.5] → drop.
        # Segment centered at t=4.0 with same length covers [3.0, 5.0] → no drop.
        seg_times = np.array([2.5, 4.0])
        qc, _ = compute_segment_drop(seg_times, 2.0, pf, ["q_drop"])
        np.testing.assert_array_equal(qc, [1, 0])

    def test_bitwise_or_across_channels(self):
        t_slow = np.array([0.0, 1.0, 2.0])
        thr = np.array([0, 1, 0], dtype=np.uint8)   # bit 1
        mod = np.array([0, 0, 2], dtype=np.uint8)   # bit 2
        pf = _StubPF(t_slow=t_slow, channels={"thr": thr, "mod": mod})
        seg_times = np.array([1.0, 2.0])
        qc, _ = compute_segment_drop(seg_times, 0.5, pf, ["thr", "mod"])
        # Segment 1 (0.75-1.25) sees thr=1 → bit 1.
        # Segment 2 (1.75-2.25) sees mod=2 → bit 2.
        np.testing.assert_array_equal(qc, [1, 2])

    def test_flag_meanings_propagated(self):
        t_slow = np.array([0.0, 1.0, 2.0])
        q = np.array([0, 3, 0], dtype=np.uint8)
        pf = _StubPF(
            t_slow=t_slow,
            channels={"q_drop": q},
            channel_info={
                "q_drop": {
                    "flag_meanings": "thruster modem",
                    "flag_masks": [1, 2],
                },
            },
        )
        _qc, attrs = compute_segment_drop(
            np.array([1.0]), 0.5, pf, ["q_drop"]
        )
        assert attrs["flag_meanings"] == "thruster modem"
        np.testing.assert_array_equal(attrs["flag_masks"], [1, 2])

    def test_segment_below_grid_returns_zero(self):
        t_slow = np.array([10.0, 11.0])
        pf = _StubPF(t_slow=t_slow, channels={"q": np.array([1, 1], dtype=np.uint8)})
        qc, _ = compute_segment_drop(np.array([0.0]), 0.5, pf, ["q"])
        np.testing.assert_array_equal(qc, [0])


# ---------------------------------------------------------------------------
# apply_qc_to_dataset
# ---------------------------------------------------------------------------


class TestApplyQCToDataset:
    def test_writes_flag_var_and_nans_floats(self):
        seg_times = np.array([1.0, 2.0, 3.0])
        ds = _make_diss_ds(seg_times)
        t_slow = np.arange(0.0, 5.0, 0.5)
        q = np.zeros_like(t_slow, dtype=np.uint8)
        q[2] = 1  # t=1.0 → only segment 0 (centered 1.0, [0.5, 1.5]) catches it
        pf = _StubPF(t_slow=t_slow, channels={"q_drop": q})

        apply_qc_to_dataset(
            ds, pf, ["q_drop"], 1.0,
            flag_var_name="qc_drop_epsilon",
            value_vars=["epsilonMean", "e_1", "e_2", "epsilon"],
        )

        assert "qc_drop_epsilon" in ds.data_vars
        assert ds["qc_drop_epsilon"].dtype == np.uint8
        # First segment got NaN'd; others untouched.
        assert np.isnan(ds["epsilonMean"].values[0])
        assert np.isfinite(ds["epsilonMean"].values[1])
        # 2-D probe-time array also masked along time.
        assert np.all(np.isnan(ds["epsilon"].values[:, 0]))
        assert np.all(np.isfinite(ds["epsilon"].values[:, 1]))

    def test_flag_only_keeps_values(self):
        seg_times = np.array([1.0, 2.0])
        ds = _make_diss_ds(seg_times)
        t_slow = np.array([1.0, 2.0])
        pf = _StubPF(t_slow=t_slow,
                     channels={"q": np.array([1, 1], dtype=np.uint8)})

        apply_qc_to_dataset(
            ds, pf, ["q"], 1.0,
            flag_var_name="qc_drop_epsilon",
            value_vars=["epsilonMean"],
            drop_action="flag_only",
        )
        # Flag is set everywhere but values are unchanged.
        assert np.all(ds["qc_drop_epsilon"].values > 0)
        np.testing.assert_array_equal(ds["epsilonMean"].values, [1e-8, 1e-8])

    def test_uses_t_coord_when_present(self):
        """Diss / chi NetCDFs index segment time via a ``t`` variable in
        seconds-since-pf.start_time. The dim coord ``time`` is just integer
        indices [0..N-1] and must not be used."""
        n = 3
        ds = xr.Dataset(
            {
                "epsilonMean": (["time"], np.full(n, 1e-8)),
                "t": (["time"], np.array([100.0, 101.0, 102.0])),
            },
            coords={"time": np.arange(n)},  # integer indices — wrong source
        )
        # q_drop hits exactly t=101 → only the middle segment should drop.
        t_slow = np.arange(0.0, 200.0, 0.5)
        q = np.zeros_like(t_slow, dtype=np.uint8)
        q[202] = 1  # t=101.0
        pf = _StubPF(t_slow=t_slow, channels={"q": q})
        apply_qc_to_dataset(
            ds, pf, ["q"], 1.0,
            flag_var_name="qc_drop_epsilon",
            value_vars=["epsilonMean"],
        )
        np.testing.assert_array_equal(ds["qc_drop_epsilon"].values, [0, 1, 0])
        assert np.isfinite(ds["epsilonMean"].values[0])
        assert np.isnan(ds["epsilonMean"].values[1])
        assert np.isfinite(ds["epsilonMean"].values[2])

    def test_no_drop_from_only_writes_flag(self):
        seg_times = np.array([1.0, 2.0])
        ds = _make_diss_ds(seg_times)
        pf = _StubPF(t_slow=np.array([0.0, 5.0]))
        apply_qc_to_dataset(
            ds, pf, [], 1.0,
            flag_var_name="qc_drop_epsilon",
            value_vars=["epsilonMean"],
        )
        np.testing.assert_array_equal(ds["qc_drop_epsilon"].values, [0, 0])
        # Nothing was NaN'd.
        np.testing.assert_array_equal(ds["epsilonMean"].values, [1e-8, 1e-8])
