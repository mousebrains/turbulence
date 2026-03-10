# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Validate Python deconvolution against MATLAB for ALL .mat/.p file pairs.

Runs exact-match comparisons on P, T1, T2 (slow-rate) and T1_dT1, T2_dT2
(fast-rate) for every file in VMP/.

Handles both MATLAB v5 (.mat via scipy.io) and v7.3 (.mat via h5py) formats.
"""

from pathlib import Path

import numpy as np
import pytest
import scipy.io

h5py = pytest.importorskip("h5py")

VMP_DIR = Path(__file__).parents[1] / "VMP"

# Discover all matched .mat/.p pairs
_pairs = []
if VMP_DIR.exists():
    for mat_path in sorted(VMP_DIR.glob("*.mat")):
        p_path = mat_path.with_suffix(".p")
        if p_path.exists():
            _pairs.append((p_path, mat_path))


def _file_id(pair):
    return pair[0].stem.split("_")[-1]


def _load_mat(mat_path: Path) -> dict:
    """Load a .mat file, handling both v5 and v7.3 (HDF5) formats.

    Returns a dict mapping variable names to squeezed numpy arrays.
    """
    try:
        return scipy.io.loadmat(str(mat_path), squeeze_me=True)
    except NotImplementedError:
        pass

    # v7.3 / HDF5 format
    out = {}
    with h5py.File(str(mat_path), "r") as f:
        for key in f.keys():
            if key.startswith("#"):
                continue
            obj = f[key]
            if isinstance(obj, h5py.Dataset):
                arr = obj[()]
                out[key] = np.squeeze(arr)
            # Skip groups (structs) — we only need numeric arrays
    return out


@pytest.fixture(params=_pairs, ids=[_file_id(p) for p in _pairs])
def file_pair(request):
    """Yield (PFile, mat_dict) for each matched file pair."""
    p_path, mat_path = request.param
    from rsi_python.p_file import PFile

    pf = PFile(p_path)
    mat = _load_mat(mat_path)
    return pf, mat


class TestAllFilesExactMatch:
    """Every .p/.mat pair should produce exact channel matches."""

    def test_sampling_rates(self, file_pair):
        pf, mat = file_pair
        np.testing.assert_allclose(pf.fs_fast, float(mat["fs_fast"]), rtol=1e-10)
        np.testing.assert_allclose(pf.fs_slow, float(mat["fs_slow"]), rtol=1e-10)

    def test_array_lengths(self, file_pair):
        pf, mat = file_pair
        assert pf.t_fast.shape == mat["t_fast"].shape, (
            f"t_fast: Python {pf.t_fast.shape} vs MATLAB {mat['t_fast'].shape}"
        )
        assert pf.t_slow.shape == mat["t_slow"].shape, (
            f"t_slow: Python {pf.t_slow.shape} vs MATLAB {mat['t_slow'].shape}"
        )

    def test_pressure_slow(self, file_pair):
        pf, mat = file_pair
        np.testing.assert_allclose(
            pf.channels["P"], mat["P_slow"], rtol=1e-10, atol=1e-10
        )

    def test_temperature_T1_slow(self, file_pair):
        pf, mat = file_pair
        np.testing.assert_allclose(
            pf.channels["T1"], mat["T1_slow"], rtol=1e-10, atol=1e-10
        )

    def test_temperature_T2_slow(self, file_pair):
        pf, mat = file_pair
        np.testing.assert_allclose(
            pf.channels["T2"], mat["T2_slow"], rtol=1e-10, atol=1e-10
        )

    def test_temperature_T1_fast(self, file_pair):
        pf, mat = file_pair
        np.testing.assert_allclose(
            pf.channels["T1_dT1"], mat["T1_fast"], rtol=1e-10, atol=1e-10
        )

    def test_temperature_T2_fast(self, file_pair):
        pf, mat = file_pair
        np.testing.assert_allclose(
            pf.channels["T2_dT2"], mat["T2_fast"], rtol=1e-10, atol=1e-10
        )

    def test_shear_speed_consistency(self, file_pair):
        """Python shear * speed^-2 should equal MATLAB shear (du/dz)."""
        pf, mat = file_pair
        py_sh1 = pf.channels["sh1"]
        ml_sh1 = mat["sh1"]
        ml_speed = mat["speed_fast"]

        reconstructed = ml_sh1 * ml_speed**2
        mask = ml_speed > 0.1
        if np.any(mask):
            np.testing.assert_allclose(
                py_sh1[mask], reconstructed[mask], rtol=1e-6, atol=1e-6
            )
