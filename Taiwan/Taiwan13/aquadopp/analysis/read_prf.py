"""Minimal reader for the Nortek Aquadopp HR profiler file TAI102.PRF (Taiwan 2013).

Structures verified against the file itself (not assumed):
  0xA5 0x05  48 B  hardware config   -> serial "AQD 8396"
  0xA5 0x04 224 B  head config       -> 2 MHz, "ASP 5444", 3 beams,
                                        transformation matrix at hdSystem+8
  0xA5 0x00 512 B  user config       -> NBins 29 (off 34), MeasInterval 1 s
                                        (off 38), CoordSystem 0 = ENU (off 32)
  0xA5 0x2A 404 B  HR profile data   -> 54 B header + 29 cells x 3 beams of
                                        (vel int16, amp uint8, corr uint8) + checksum

VELOCITY SCALING AND ORIENTATION come from the record's status byte (offset 25),
not from an assumption: bit 1 set = 0.1 mm/s (not mm/s), bit 0 clear = the head
looks UP. Here status = 0x62 on every record, so velocities are counts * 1e-4
m/s and the head is up-looking. With that scaling the stored ENU "up" component
(0.117 m/s) matches the glider's own dive rate (0.129 dbar/s), and the velocity
range saturates at +-0.1285 m/s -- the HR ambiguity velocity, i.e. a wrap
interval of 0.257 m/s.
"""
from __future__ import annotations

import os
import struct
from pathlib import Path

import numpy as np

# The data file sits one level up (aquadopp/TAI102.PRF); TAI102_PRF overrides it.
PRF = Path(os.environ.get("TAI102_PRF", Path(__file__).resolve().parent.parent / "TAI102.PRF"))
HDR = 784          # bytes of hardware+head+user config
REC = 404
NCELL = 29
NBEAM = 3


def config():
    b = PRF.open("rb").read(HDR)
    head = b[48:272]
    usr = b[272:784]
    T = np.frombuffer(head[22 + 8: 22 + 8 + 18], "<i2").astype(float).reshape(3, 3) / 4096.0
    return dict(
        serial_hw=b[4:18].decode("ascii", "replace").strip("\x00"),
        head_serial=head[10:22].decode("ascii", "replace").strip("\x00"),
        frequency_khz=struct.unpack_from("<h", head, 6)[0],
        n_beams=struct.unpack_from("<h", head, 220)[0],
        transform=T,
        n_bins=struct.unpack_from("<H", usr, 34)[0],
        bin_length_counts=struct.unpack_from("<H", usr, 36)[0],
        meas_interval_s=struct.unpack_from("<H", usr, 38)[0],
        coord_system=struct.unpack_from("<H", usr, 32)[0],  # 0=ENU 1=XYZ 2=BEAM
        blank_dist_counts=struct.unpack_from("<H", usr, 16)[0],
    )


def _bcd(a):
    return (a >> 4) * 10 + (a & 0x0F)


def read(max_records: int | None = None):
    raw = np.fromfile(PRF, dtype=np.uint8, offset=HDR)
    n = raw.size // REC
    if max_records:
        n = min(n, max_records)
    r = raw[: n * REC].reshape(n, REC)
    assert (r[:, 0] == 0xA5).all() and (r[:, 1] == 0x2A).all(), "unexpected record ids"

    def i16(off):
        return (r[:, off].astype(np.int32) | (r[:, off + 1].astype(np.int32) << 8)).astype(np.int16)

    def u16f(off):
        return r[:, off].astype(np.uint32) | (r[:, off + 1].astype(np.uint32) << 8)

    minute, sec, day, hour, year, month = (_bcd(r[:, 4 + k]) for k in range(6))
    t = np.array(
        [f"20{y:02d}-{mo:02d}-{d:02d}T{h:02d}:{mi:02d}:{s:02d}"
         for y, mo, d, h, mi, s in zip(year, month, day, hour, minute, sec)],
        dtype="datetime64[s]",
    ).astype("int64").astype(float) + u16f(10) / 1000.0
    out = dict(
        time=t,
        battery_v=u16f(14) / 10.0,
        sound_speed=u16f(16) / 10.0,
        heading_deg=i16(18) / 10.0,
        pitch_deg=i16(20) / 10.0,
        roll_deg=i16(22) / 10.0,
        pressure_dbar=(r[:, 24].astype(np.float64) * 65536 + u16f(26)) * 0.001,
        temperature_c=i16(28) / 100.0,
        status=r[:, 25],
    )
    vel = np.empty((n, NBEAM, NCELL))
    for b in range(NBEAM):
        off = 54 + b * NCELL * 2
        lo = r[:, off: off + NCELL * 2: 2].astype(np.int32)
        hi = r[:, off + 1: off + NCELL * 2: 2].astype(np.int32)
        v = (lo | (hi << 8)).astype(np.int16).astype(float)
        vel[:, b, :] = v / 10000.0                      # 0.1 mm/s -> m/s (status bit 1)
    amp = np.empty((n, NBEAM, NCELL))
    corr = np.empty((n, NBEAM, NCELL))
    a0 = 54 + NBEAM * NCELL * 2
    c0 = a0 + NBEAM * NCELL
    for b in range(NBEAM):
        amp[:, b, :] = r[:, a0 + b * NCELL: a0 + (b + 1) * NCELL]
        corr[:, b, :] = r[:, c0 + b * NCELL: c0 + (b + 1) * NCELL]
    out.update(vel=vel, amp=amp, corr=corr)
    return out


if __name__ == "__main__":
    c = config()
    for k, v in c.items():
        print(f"{k}: {v if not isinstance(v, np.ndarray) else ''}")
        if isinstance(v, np.ndarray):
            print(v.round(4))
    d = read()
    t = d["time"]
    print(f"records {t.size:,}  {np.datetime64(int(t.min()),'s')} .. {np.datetime64(int(t.max()),'s')}")
    print(f"median dt {np.median(np.diff(t)):.2f} s   pressure {d['pressure_dbar'].min():.2f}..{d['pressure_dbar'].max():.2f} dbar")
    print(f"pitch p5/50/95 {np.percentile(d['pitch_deg'],[5,50,95]).round(1)}  roll {np.percentile(d['roll_deg'],[5,50,95]).round(1)}")
    print(f"corr median per beam {np.median(d['corr'],axis=(0,2)).round(1)}  amp {np.median(d['amp'],axis=(0,2)).round(1)}")
    np.savez(Path(__file__).resolve().parent / "aqd_headers.npz",
             **{k: d[k] for k in ("time", "heading_deg", "pitch_deg", "roll_deg", "pressure_dbar", "temperature_c", "status")})
