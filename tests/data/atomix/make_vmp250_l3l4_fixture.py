#!/usr/bin/env python3
"""Generate the committed VMP250 Haro Strait L3->L4 CI fixture.

The full ATOMIX benchmark file ``VMP250_TidalChannel_024.nc`` is ~15 MB, almost
all of it the L1/L2 time series (TIME = 102400 samples). The reference-spectra
epsilon gate (``tests/test_atomix_l3l4_gate.py``) only needs the *reference*
L3 wavenumber spectra and the *reference* L4 dissipation -- 32 spectra each --
which are tiny. This script strips the file down to just the ``L3_spectra`` and
``L4_dissipation`` groups (plus the root global attributes that carry the
provenance/DOI/citation), producing a sub-MB fixture that can be committed and
run in CI without the gitignored ``AtomixData`` symlink.

Source: ATOMIX shear-probe benchmark, VMP250 Haro Strait (a.k.a. Tidal Channel)
dataset. See ``docs/atomix_benchmark.md``. Dataset DOI (from the file's own
``citation`` attribute): 10.5285/0ec16a65-abdf-2822-e063-6c86abc06533 (Lueck
2024, BODC); benchmark-description paper: doi:10.1038/s41597-024-03323-y (Fer et
al. 2024). The committed fixture is a verbatim subset of the reference L3/L4
groups -- no values are altered.

Usage (run once when the fixture needs regenerating; the AtomixData symlink must
be present):

    python tests/data/atomix/make_vmp250_l3l4_fixture.py
"""

from __future__ import annotations

from pathlib import Path

import netCDF4

# Groups to carry over verbatim. L1_converted / L2_cleaned (the bulk of the
# file) are intentionally dropped: the reference-spectra gate reads these two
# groups directly and never touches the raw/cleaned time series.
_KEEP_GROUPS = ("L3_spectra", "L4_dissipation")

_SRC = Path(__file__).resolve().parents[3] / "AtomixData" / "VMP250_TidalChannel_024.nc"
_DST = Path(__file__).resolve().parent / "VMP250_HaroStrait_L3L4.nc"


def _copy_var(src_var: netCDF4.Variable, dst_grp: netCDF4.Group) -> None:
    """Copy one variable (data + attributes) into *dst_grp*."""
    fill = getattr(src_var, "_FillValue", None)
    dst_var = dst_grp.createVariable(
        src_var.name,
        src_var.dtype,
        src_var.dimensions,
        fill_value=fill,
    )
    # Attributes first (skip _FillValue -- already set at creation).
    for attr in src_var.ncattrs():
        if attr == "_FillValue":
            continue
        dst_var.setncattr(attr, src_var.getncattr(attr))
    dst_var[:] = src_var[:]


def main() -> None:
    if not _SRC.exists():
        raise SystemExit(
            f"Source benchmark file not found: {_SRC}\n"
            "The gitignored AtomixData symlink must be present to regenerate the fixture."
        )

    src = netCDF4.Dataset(str(_SRC), "r")
    dst = netCDF4.Dataset(str(_DST), "w", format="NETCDF4")
    try:
        # Root global attributes (provenance: citation, references, DOI, ...).
        for attr in src.ncattrs():
            dst.setncattr(attr, src.getncattr(attr))
        dst.setncattr(
            "history",
            "Downsampled L3/L4 subset of VMP250_TidalChannel_024.nc for the "
            "rsi-tpw ATOMIX epsilon CI gate (make_vmp250_l3l4_fixture.py); "
            "L1_converted/L2_cleaned dropped, L3/L4 values verbatim.",
        )

        for gname in _KEEP_GROUPS:
            src_grp = src.groups[gname]
            dst_grp = dst.createGroup(gname)
            # Recreate every dimension the group's variables reference. L3/L4
            # vars use root-scope dims (TIME_SPECTRA, N_WAVENUMBER, ...), so
            # create them at the dest root as needed, matching source sizes.
            for var in src_grp.variables.values():
                for dim in var.dimensions:
                    if dim not in dst.dimensions:
                        src_dim = src.dimensions[dim]
                        dst.createDimension(
                            dim, None if src_dim.isunlimited() else len(src_dim)
                        )
            for var in src_grp.variables.values():
                _copy_var(var, dst_grp)
            for attr in src_grp.ncattrs():
                dst_grp.setncattr(attr, src_grp.getncattr(attr))
    finally:
        src.close()
        dst.close()

    size_kb = _DST.stat().st_size / 1024
    print(f"Wrote {_DST} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
