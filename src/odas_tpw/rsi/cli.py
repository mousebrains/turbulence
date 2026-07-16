# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""
CLI for rsi-tpw.

Main command:
    rsi-tpw <subcommand> [options]

Subcommands:
    info     — Print summary of .p file(s)
    config   — Print a .p file's raw embedded configuration (INI) record
    cutp     — Copy a short record range from a .p file for debugging
    v1to6    — Translate legacy header-v1 .p files to v6 (issue #141)
    nc       — Convert .p files to NetCDF
    patch-config   — Edit config fields in .p file(s), writing new files
    patch-template — Scaffold a patch-config edit spec from a .p file
    prof     — Extract profiles from .p or full-record .nc files
    eps      — Compute epsilon (TKE dissipation) from any pipeline stage
    chi      — Compute chi (thermal variance dissipation) from any pipeline stage
    pipeline — Run full processing pipeline (.p → profiles → epsilon → chi)
    sensors  — Inventory shear/FP07 sensors across a .p file tree
    ql       — Interactive quick-look viewer
    dl       — Interactive dissipation quality viewer
    ml       — Interactive mixing viewer
    bench    — Bench-test diagnostic (quick_bench + auto checklist)
    init     — Generate a template configuration file
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any


def _resolve_p_files(patterns: list[str]) -> list[Path]:
    """Expand glob patterns and return list of .p file Paths."""
    import glob as globmod

    files = []
    for pattern in patterns:
        expanded = sorted(globmod.glob(pattern))
        if not expanded:
            print(f"Warning: no files match '{pattern}'", file=sys.stderr)
        for f in expanded:
            p = Path(f)
            if p.is_file() and p.suffix.lower() == ".p":
                files.append(p)
    if not files:
        print("Error: no .p files found", file=sys.stderr)
        sys.exit(1)
    return files


def _resolve_files(patterns: list[str], extensions: set[str] | None = None) -> list[Path]:
    """Expand glob patterns and return list of matching file Paths.

    Parameters
    ----------
    patterns : list of str
        File paths or glob patterns.
    extensions : set of str or None
        Allowed file extensions (e.g. {'.p', '.nc'}). If None, accept all.
    """
    import glob as globmod

    files = []
    for pattern in patterns:
        expanded = sorted(globmod.glob(pattern))
        if not expanded:
            print(f"Warning: no files match '{pattern}'", file=sys.stderr)
        for f in expanded:
            p = Path(f)
            if p.is_file() and (extensions is None or p.suffix.lower() in extensions):
                files.append(p)
    if not files:
        ext_str = ", ".join(extensions) if extensions else "any"
        print(f"Error: no matching files found ({ext_str})", file=sys.stderr)
        sys.exit(1)
    return files


# ---------------------------------------------------------------------------
# Argument-type parsers
# ---------------------------------------------------------------------------


def _parse_salinity(value: str) -> float | str:
    """--salinity: a PSU number, or 'measured' (from the C/T channels)."""
    v = value.strip()
    if v.lower() == "measured":
        return "measured"
    try:
        return float(v)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected a salinity in PSU or 'measured', got {value!r}"
        ) from None


def _parse_temperature(value: str) -> float | str:
    """--temperature: a channel name, 'auto', or a fixed value [degC]."""
    try:
        return float(value)
    except ValueError:
        return value


def _parse_sens(value: str) -> dict[str, float]:
    """--sens: 'sh1=0.0893,sh2=0.0558' -> {'sh1': 0.0893, 'sh2': 0.0558}."""
    out: dict[str, float] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        name, eq, num = item.partition("=")
        name = name.strip()
        try:
            if not eq or not name:
                raise ValueError
            out[name] = float(num)
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"expected NAME=VALUE[,NAME=VALUE...] (e.g. sh1=0.0893,sh2=0.0558), "
                f"got {value!r}"
            ) from None
    if not out:
        raise argparse.ArgumentTypeError("empty --sens value")
    return out


# ---------------------------------------------------------------------------
# Config integration helpers
# ---------------------------------------------------------------------------


def _load_file_config(args: argparse.Namespace) -> dict[str, Any]:
    """Load config from the -c/--config flag, or return empty dict."""
    if getattr(args, "config", None) is not None:
        from odas_tpw.rsi.config import load_config

        return load_config(args.config)
    return {}


def _extract_cli_overrides(args: argparse.Namespace, section: str) -> dict[str, Any]:
    """Extract CLI-provided overrides for a config section.

    Only includes values that were explicitly specified on the command line
    (i.e., not None).  Handles arg-name -> config-key mapping and the
    --no-goodman inversion.
    """
    if section == "profiles":
        mapping = {
            "P_min": "P_min",
            "W_min": "W_min",
            "direction": "direction",
            "vehicle": "vehicle",
            "min_duration": "min_duration",
        }
    elif section == "epsilon":
        mapping = {
            "fft_length": "fft_length",
            "diss_length": "diss_length",
            "overlap": "overlap",
            "speed": "speed",
            "direction": "direction",
            "vehicle": "vehicle",
            "f_AA": "f_AA",
            "salinity": "salinity",
            "temperature": "temperature",
            "conductivity": "conductivity",
        }
    elif section == "chi":
        mapping = {
            "fft_length": "fft_length",
            "diss_length": "diss_length",
            "overlap": "overlap",
            "speed": "speed",
            "direction": "direction",
            "vehicle": "vehicle",
            "fp07_model": "fp07_model",
            "f_AA": "f_AA",
            "fit_method": "fit_method",
            "spectrum_model": "spectrum_model",
            "salinity": "salinity",
            "temperature": "temperature",
            "conductivity": "conductivity",
        }
    elif section == "epsilon_pipeline":
        mapping = {
            "eps_fft_length": "fft_length",
            "direction": "direction",
            "vehicle": "vehicle",
            "f_AA": "f_AA",
            "speed": "speed",
            "salinity": "salinity",
            "temperature": "temperature",
            "conductivity": "conductivity",
        }
    elif section == "chi_pipeline":
        mapping = {
            "chi_fft_length": "fft_length",
            "direction": "direction",
            "vehicle": "vehicle",
            "f_AA": "f_AA",
            "fp07_model": "fp07_model",
            "spectrum_model": "spectrum_model",
            "speed": "speed",
            "salinity": "salinity",
            "temperature": "temperature",
            "conductivity": "conductivity",
        }
    else:
        return {}

    overrides = {}
    for attr, key in mapping.items():
        val = getattr(args, attr, None)
        if val is not None:
            overrides[key] = val

    # Handle --no-goodman (store_const: None=not specified, True=specified)
    if section in ("epsilon", "epsilon_pipeline", "chi", "chi_pipeline"):
        no_goodman = getattr(args, "no_goodman", None)
        if no_goodman is True:
            overrides["goodman"] = False

    return overrides


def _merge_for_section(args: argparse.Namespace, section: str) -> dict[str, Any]:
    """Load config file + CLI overrides and merge for the given section.

    Returns the merged kwargs dict (None values stripped).
    """
    from odas_tpw.rsi.config import merge_config

    file_config = _load_file_config(args)
    # For pipeline pseudo-sections, map to the real config section name
    real_section = section.replace("_pipeline", "")
    file_values = file_config.get(real_section, {})
    cli_overrides = _extract_cli_overrides(args, section)
    return merge_config(real_section, file_values, cli_overrides)


def _setup_output_dir(
    args: argparse.Namespace,
    prefix: str,
    section: str,
    params: dict[str, Any],
    upstream: list[tuple[str, dict[str, Any]]] | None = None,
) -> Path:
    """Resolve the sequential output directory, write signature file and config.yaml.

    Parameters
    ----------
    upstream : list of (section, params) tuples, optional
        Upstream sections to include in the hash for cumulative tracking.
    """
    from odas_tpw.rsi.config import resolve_output_dir, write_resolved_config

    real_section = section.replace("_pipeline", "")
    output_dir = resolve_output_dir(args.output, prefix, real_section, params, upstream=upstream)
    write_resolved_config(output_dir, real_section, params, upstream=upstream)
    return output_dir


# ---------------------------------------------------------------------------
# Subcommand implementations
# ---------------------------------------------------------------------------


def _cmd_nc(args: argparse.Namespace) -> None:
    """Convert Rockland .p files to NetCDF4."""
    from odas_tpw.rsi.convert import convert_all, p_to_L1

    p_files = _resolve_p_files(args.files)

    if len(p_files) == 1 and args.output and not Path(args.output).is_dir():
        _, nc_path = p_to_L1(p_files[0], args.output)
        print(f"Wrote {nc_path} ({nc_path.stat().st_size / 1e6:.1f} MB)")
    else:
        output_dir = Path(args.output) if args.output else None
        convert_all(p_files, output_dir, jobs=args.jobs)


def _cmd_info(args: argparse.Namespace) -> None:
    """Print summary information about .p file(s)."""
    from odas_tpw.rsi.p_file import PFile

    p_files = _resolve_p_files(args.files)
    for i, pf_path in enumerate(p_files):
        if i > 0:
            print("\n" + "=" * 60 + "\n")
        pf = PFile(pf_path)
        pf.summary()


def _cmd_config(args: argparse.Namespace) -> None:
    """Print the raw embedded configuration (INI) record of .p file(s)."""
    from odas_tpw.rsi.p_file import read_config_string

    p_files = _resolve_p_files(args.files)
    multi = len(p_files) > 1
    failures = 0
    printed = 0
    for pf_path in p_files:
        try:
            cfg = read_config_string(pf_path)
        except (OSError, ValueError) as e:
            print(f"Error: {e}", file=sys.stderr)
            failures += 1
            continue
        if multi:
            if printed:
                print()
            print(f"# ===== {pf_path} =====")
        print(cfg, end="" if cfg.endswith("\n") else "\n")
        printed += 1
    if failures:
        sys.exit(1)


def _cmd_cutp(args: argparse.Namespace) -> None:
    """Copy a contiguous data-record range from a .p file."""
    from odas_tpw.rsi.p_file import extract_pfile_segment

    try:
        out = extract_pfile_segment(
            args.file,
            args.output,
            start_record=args.start,
            n_records=args.n_records,
            overwrite=args.overwrite,
        )
    except FileExistsError:
        print(f"Error: {args.output} exists; pass --force to replace it", file=sys.stderr)
        sys.exit(1)
    except (FileNotFoundError, OSError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Wrote {args.n_records} record(s) to {out}")


def _cmd_v1to6(args: argparse.Namespace) -> None:
    """Translate legacy ODAS header-v1 .p files to v6 (issue #141)."""
    from odas_tpw.rsi.v1_translate import translate_v1_to_v6

    p_files = _resolve_p_files(args.files)
    out_dir = Path(args.output)
    failures = 0
    for f in p_files:
        dst = out_dir / f.name
        try:
            dst_path, meta = translate_v1_to_v6(
                f,
                dst,
                setup_file=args.setup_file,
                sens=args.sens,
                overwrite=args.force,
            )
        except (OSError, ValueError, FileExistsError) as e:
            print(f"  {f.name}: ERROR: {e}", file=sys.stderr)
            failures += 1
            continue
        print(
            f"  {f.name} -> {dst_path} "
            f"({meta['n_records']} data records; setup: {meta['setup_file']} "
            f"md5 {meta['setup_md5']}; sens: {meta['sens_source']})"
        )
    if failures:
        sys.exit(1)


def _cmd_init(args: argparse.Namespace) -> None:
    """Generate a template configuration file."""
    from odas_tpw.rsi.config import generate_template

    path = Path(args.path)
    if path.exists() and not args.force:
        print(f"Error: {path} already exists (use --force to overwrite)", file=sys.stderr)
        sys.exit(1)
    generate_template(path)
    print(f"Wrote template config to {path}")


def _cmd_patch_config(args: argparse.Namespace) -> None:
    """Patch config fields in .p file(s), writing new files."""
    from datetime import datetime

    from ruamel.yaml import YAMLError

    from odas_tpw.rsi.config_patch import load_edit_spec, patch_files

    files = _resolve_p_files(args.files)
    when = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        spec = load_edit_spec(args.edits)
        patch_files(
            files,
            args.out,
            spec,
            dry_run=args.dry_run,
            add_keys=args.add_keys,
            batch_cal=args.batch_cal,
            when=when,
        )
    # YAMLError: load_edit_spec parses with ruamel, whose ParserError/ScannerError
    # subclass YAMLError (not ValueError/OSError) and otherwise escape as a traceback.
    except (FileNotFoundError, FileExistsError, ValueError, OSError, YAMLError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def _cmd_patch_template(args: argparse.Namespace) -> None:
    """Scaffold a patch-config edit-spec template from a .p file."""
    from odas_tpw.rsi.config_patch import scaffold_yaml

    src = Path(args.file)
    if not src.is_file():
        print(f"Error: {src} not found", file=sys.stderr)
        sys.exit(1)
    try:
        text = scaffold_yaml(src)
    except (OSError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    if args.output:
        out = Path(args.output)
        if out.exists() and not args.force:
            print(f"Error: {out} already exists (use --force to overwrite)", file=sys.stderr)
            sys.exit(1)
        out.write_text(text)
        print(f"Wrote template to {out}")
    else:
        print(text, end="")


def _cmd_prof(args: argparse.Namespace) -> None:
    """Extract profiles from .p or full-record .nc files."""
    from odas_tpw.rsi.profile import extract_profiles

    files = _resolve_files(args.files, {".p", ".nc"})

    merged = _merge_for_section(args, "profiles")
    output_dir = _setup_output_dir(args, "prof", "profiles", merged)
    print(f"Output directory: {output_dir}")

    for f in files:
        print(f"{f.name}:")
        extract_profiles(f, output_dir, **merged)


def _cmd_eps(args: argparse.Namespace) -> None:
    """Compute epsilon (TKE dissipation rate) from any pipeline stage."""
    from odas_tpw.rsi.dissipation import _compute_diss_one, compute_diss_file

    files = _resolve_files(args.files, {".p", ".nc"})

    merged = _merge_for_section(args, "epsilon")
    output_dir = _setup_output_dir(args, "eps", "epsilon", merged)
    print(f"Output directory: {output_dir}")

    jobs = args.jobs
    if jobs == 0:
        jobs = os.cpu_count() or 1

    if jobs == 1:
        for f in files:
            print(f"{f.name}:")
            try:
                compute_diss_file(f, output_dir, **merged)
            except (OSError, ValueError, RuntimeError) as e:
                print(f"  ERROR: {e}")
    else:
        work = []
        for f in files:
            work.append((f, output_dir, merged))
        print(f"Processing {len(work)} files with {jobs} workers")
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = {pool.submit(_compute_diss_one, w): w for w in work}
            for future in as_completed(futures):
                src, _, _ = futures[future]
                try:
                    name, n_profiles = future.result()
                    print(f"  {Path(name).name}: {n_profiles} profile(s)")
                except (OSError, ValueError, RuntimeError) as e:
                    print(f"  {src.name}: ERROR: {e}")


def _cmd_chi(args: argparse.Namespace) -> None:
    """Compute chi (thermal variance dissipation rate) from any pipeline stage."""
    from odas_tpw.rsi.chi_io import _compute_chi_one, compute_chi_file

    files = _resolve_files(args.files, {".p", ".nc"})

    merged = _merge_for_section(args, "chi")
    output_dir = _setup_output_dir(args, "chi", "chi", merged)
    print(f"Output directory: {output_dir}")

    # Load epsilon datasets if Method 1
    epsilon_dir = getattr(args, "epsilon_dir", None)

    jobs = args.jobs
    if jobs == 0:
        jobs = os.cpu_count() or 1

    if jobs == 1:
        for f in files:
            print(f"{f.name}:")

            kw = dict(merged)
            eps_ds = None
            if epsilon_dir is not None:
                from odas_tpw.rsi.chi_io import load_epsilon_dataset

                eps_ds = load_epsilon_dataset(f, epsilon_dir)
                if eps_ds is not None:
                    kw["epsilon_ds"] = eps_ds
                else:
                    print(
                        f"  Warning: no epsilon files for {f.stem} under "
                        f"{epsilon_dir} (searched eps_* subdirectories too); "
                        "using Method 2"
                    )

            try:
                compute_chi_file(f, output_dir, **kw)
            except (OSError, ValueError, RuntimeError) as e:
                print(f"  ERROR: {e}")
            finally:
                if eps_ds is not None:
                    eps_ds.close()
    else:
        work = []
        for f in files:
            work.append((f, output_dir, merged, epsilon_dir))
        print(f"Processing {len(work)} files with {jobs} workers")
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = {pool.submit(_compute_chi_one, w): w for w in work}
            for future in as_completed(futures):
                src = futures[future][0]
                try:
                    name, n_profiles = future.result()
                    print(f"  {Path(name).name}: {n_profiles} profile(s)")
                except (OSError, ValueError, RuntimeError) as e:
                    print(f"  {Path(src).name}: ERROR: {e}")


def _cmd_pipeline(args: argparse.Namespace) -> None:
    """Run full processing pipeline: .p -> L2 -> L3 -> L4 -> chi -> binning."""
    from odas_tpw.rsi.pipeline import run_pipeline

    p_files = _resolve_p_files(args.files)
    output_dir = Path(args.output)

    # Gather parameters from CLI/config
    eps_merged = _merge_for_section(args, "epsilon_pipeline")
    chi_merged = _merge_for_section(args, "chi_pipeline")

    # --salinity measured maps to None: run_pipeline already auto-prefers
    # measured JAC C/T salinity (rsi/pipeline._resolve_salinity); the string
    # must never reach the numeric salinity paths.
    salinity = eps_merged.get("salinity")
    if isinstance(salinity, str):
        if salinity.strip().lower() == "measured":
            print(
                "Note: salinity 'measured' is the pipeline's automatic behavior "
                "(measured JAC C/T salinity is preferred when present); "
                "proceeding with the automatic path"
            )
            salinity = None
        else:
            print(
                f"Error: salinity={salinity!r} is not valid for the pipeline; "
                "use a number or 'measured'",
                file=sys.stderr,
            )
            sys.exit(1)

    # The pipeline resolves ONE temperature/conductivity for both epsilon and
    # chi (they share the per-file L1 load); a differing [chi] value in the
    # config would otherwise be silently ignored.
    for key in ("temperature", "conductivity"):
        eps_val, chi_val = eps_merged.get(key, "auto"), chi_merged.get(key, "auto")
        if chi_val != eps_val:
            print(
                f"Warning: pipeline uses a single {key} for epsilon and chi; "
                f"[chi] {key}={chi_val!r} is ignored in favor of [epsilon] "
                f"{key}={eps_val!r}",
                file=sys.stderr,
            )

    kwargs = {
        "direction": eps_merged.get("direction", "auto"),
        "vehicle": eps_merged.get("vehicle"),
        "speed": eps_merged.get("speed"),
        "fft_length": eps_merged.get("fft_length", 1024),
        "f_AA": eps_merged.get("f_AA", 98.0),
        "salinity": salinity,
        "temperature": eps_merged.get("temperature", "auto"),
        "conductivity": eps_merged.get("conductivity", "auto"),
        "goodman": eps_merged.get("goodman", True),
        "chi_fft_length": chi_merged.get("fft_length", 1024),
        "fp07_model": chi_merged.get("fp07_model", "single_pole"),
        "spectrum_model": chi_merged.get("spectrum_model", "kraichnan"),
    }
    # Remove None values
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    # Speed model (pipeline-level; default pressure / |dP/dt|).
    if getattr(args, "speed_method", None):
        kwargs["speed_method"] = args.speed_method
    if getattr(args, "aoa", None) is not None:
        kwargs["aoa_deg"] = args.aoa

    run_pipeline(p_files, output_dir, **kwargs)


def _cmd_ql(args: argparse.Namespace) -> None:
    """Interactive quick-look viewer."""
    from odas_tpw.rsi.quick_look import quick_look

    spec_P_range = None
    if args.spec_P_range is not None:
        spec_P_range = tuple(args.spec_P_range)

    # Merge config file + CLI overrides for epsilon section
    merged = _merge_for_section(args, "epsilon")
    fft_length = args.fft_length or merged.get("fft_length", 1024)
    diss_length = args.diss_length or merged.get("diss_length")
    f_AA = args.f_AA or merged.get("f_AA", 98.0)
    goodman = merged.get("goodman", True) if not args.no_goodman else False

    W_min = getattr(args, "W_min", None)  # None -> direction-aware default in the viewer

    p_files = _resolve_p_files(args.files)
    for pf_path in p_files:
        quick_look(
            pf_path,
            fft_length=fft_length,
            diss_length=diss_length,
            f_AA=f_AA,
            goodman=goodman,
            direction=args.direction or "auto",
            vehicle=getattr(args, "vehicle", None),
            W_min=W_min,
            spec_P_range=spec_P_range,
            chi_method=args.chi_method,
            spectrum_model=args.spectrum_model,
        )


def _cmd_dl(args: argparse.Namespace) -> None:
    """Interactive dissipation quality viewer."""
    from odas_tpw.rsi.diss_look import diss_look

    spec_P_range = None
    if args.spec_P_range is not None:
        spec_P_range = tuple(args.spec_P_range)

    # Merge config file + CLI overrides for epsilon section
    merged = _merge_for_section(args, "epsilon")
    fft_length = args.fft_length or merged.get("fft_length", 1024)
    diss_length = args.diss_length or merged.get("diss_length")
    f_AA = args.f_AA or merged.get("f_AA", 98.0)
    goodman = merged.get("goodman", True) if not args.no_goodman else False

    W_min = getattr(args, "W_min", None)  # None -> direction-aware default in the viewer

    p_files = _resolve_p_files(args.files)
    for pf_path in p_files:
        diss_look(
            pf_path,
            fft_length=fft_length,
            diss_length=diss_length,
            f_AA=f_AA,
            goodman=goodman,
            direction=args.direction or "auto",
            vehicle=getattr(args, "vehicle", None),
            W_min=W_min,
            spec_P_range=spec_P_range,
        )


def _cmd_bench(args: argparse.Namespace) -> None:
    """Bench-test diagnostic: raw-count figures + auto-evaluated checklist."""
    # Force a non-interactive backend unless the user wants windows, so batch /
    # headless runs need no display. Must precede the bench import, which loads
    # pyplot at module level.
    if not args.show:
        import matplotlib

        matplotlib.use("Agg")
    from odas_tpw.rsi.bench import run_bench

    p_files = _resolve_p_files(args.files)

    # Save by default (to ./bench/) unless the user only wants an interactive
    # look (--show with no -o), in which case display without writing files.
    out_dir = args.output
    if out_dir is None and not args.show:
        out_dir = "bench"

    for i, pf_path in enumerate(p_files):
        if i > 0:
            print("\n" + "=" * 60 + "\n")
        try:
            run_bench(
                pf_path,
                out_dir=out_dir,
                show=args.show,
                sn=args.sn,
                fft_sec=args.fft_sec,
                dpi=args.dpi,
                fmt=args.format,
            )
        except (OSError, ValueError) as e:
            print(f"  ERROR: {e}", file=sys.stderr)


def _cmd_sensors(args: argparse.Namespace) -> None:
    """Inventory microstructure sensors across a tree of .p files."""
    from odas_tpw.rsi.sensor_inventory import resolve_kinds, run

    kinds = resolve_kinds(args.shear, args.fp07, args.want_all)
    code = run(
        [Path(p) for p in args.paths],
        kinds,
        csv_out=Path(args.csv) if args.csv else None,
        verbose=args.verbose,
        compact=args.compact,
        cal_dir=Path(args.cal_dir) if args.cal_dir else None,
        cal_tol=args.cal_tol,
    )
    if code != 0:
        sys.exit(code)


def _cmd_ml(args: argparse.Namespace) -> None:
    """Interactive mixing viewer."""
    from odas_tpw.rsi.mixing_look import mixing_look

    spec_P_range = None
    if args.spec_P_range is not None:
        spec_P_range = tuple(args.spec_P_range)

    # Merge config file + CLI overrides for epsilon section
    merged = _merge_for_section(args, "epsilon")
    fft_length = args.fft_length or merged.get("fft_length", 1024)
    diss_length = args.diss_length or merged.get("diss_length")
    f_AA = args.f_AA or merged.get("f_AA", 98.0)
    goodman = merged.get("goodman", True) if not args.no_goodman else False

    W_min = getattr(args, "W_min", None)  # None -> direction-aware default in the viewer

    # The viewer already prefers measured JAC C/T salinity when no fixed value
    # is given, so 'measured' maps to None (the automatic path).
    salinity = args.salinity
    if isinstance(salinity, str):
        salinity = None

    p_files = _resolve_p_files(args.files)
    for pf_path in p_files:
        mixing_look(
            pf_path,
            fft_length=fft_length,
            diss_length=diss_length,
            f_AA=f_AA,
            goodman=goodman,
            direction=args.direction or "auto",
            vehicle=getattr(args, "vehicle", None),
            W_min=W_min,
            spec_P_range=spec_P_range,
            salinity=salinity,
        )


# ---------------------------------------------------------------------------
# Subcommand parsers
# ---------------------------------------------------------------------------


def _add_nc_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "nc",
        help="Convert .p files to NetCDF",
        description="Convert Rockland Scientific .p data files to NetCDF4.",
    )
    p.add_argument("files", nargs="+", metavar="FILE", help=".p file(s) or glob pattern(s)")
    p.add_argument(
        "-o",
        "--output",
        metavar="PATH",
        default=None,
        help="Output file (single input) or directory (multiple inputs)",
    )
    p.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        metavar="N",
        help="Parallel workers (0 = all cores, default: 1)",
    )
    p.set_defaults(func=_cmd_nc)


def _add_info_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "info",
        help="Print .p file summary",
        description="Print summary of Rockland Scientific .p data files.",
    )
    p.add_argument("files", nargs="+", metavar="FILE", help=".p file(s) or glob pattern(s)")
    p.set_defaults(func=_cmd_info)


def _add_config_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "config",
        help="Print a .p file's embedded config record",
        description=(
            "Print the raw embedded configuration (INI) record from Rockland .p "
            "file(s) to stdout — the setup.cfg-style text with the address matrix "
            "and per-channel calibration coefficients. Reads only the header and "
            "config record, so it also works on startup/truncated files that carry "
            "a config but no data records. With multiple files, each is preceded by "
            "a '# ===== <path> =====' banner."
        ),
    )
    p.add_argument("files", nargs="+", metavar="FILE", help=".p file(s) or glob pattern(s)")
    p.set_defaults(func=_cmd_config)


def _add_cutp_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "cutp",
        help="Copy .p records for debugging",
        description=(
            "Create a valid .p file containing a contiguous range of complete "
            "data records. This is a byte-level debugging utility, not a "
            "pressure- or profile-aware scientific extraction. Absolute time "
            "is correct only when --start is 0 because the header is copied "
            "unchanged."
        ),
    )
    p.add_argument("file", metavar="FILE", help="Input .p file")
    p.add_argument(
        "-o",
        "--output",
        metavar="FILE",
        required=True,
        help="Output .p file",
    )
    p.add_argument(
        "-s",
        "--start",
        type=int,
        default=0,
        metavar="N",
        help="First data record to copy, 0-based after the config record (default: 0)",
    )
    p.add_argument(
        "-n",
        "--n-records",
        type=int,
        default=60,
        metavar="N",
        help="Number of complete data records to copy (default: 60)",
    )
    p.add_argument(
        "-f",
        "--force",
        "--overwrite",
        dest="overwrite",
        action="store_true",
        help="Overwrite output file if it exists",
    )
    p.set_defaults(func=_cmd_cutp)


def _add_v1to6_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "v1to6",
        help="Translate legacy header-v1 .p files to v6",
        description=(
            "Translate ODAS header-version-1 .p files (pre-2015 instruments; "
            "record 0 holds a binary address matrix and the configuration lives "
            "in an external setup file) into standard v6 files with an embedded "
            "INI configuration synthesized from the setup file. Data records "
            "are copied verbatim (lossless). The translated files work with "
            "every rsi-tpw / perturb tool unchanged. Shear-probe sensitivities "
            "are NOT in v1 setup files: supply them with --sens, with "
            "sh1_sens:/sh2_sens: keys added to the setup file, or afterwards "
            "with 'rsi-tpw patch-config --add-keys' on the translated files "
            "(processing errors loudly until a sens exists). See GitHub issue "
            "#141."
        ),
    )
    p.add_argument("files", nargs="+", metavar="FILE", help="v1 .p file(s) or glob pattern(s)")
    p.add_argument(
        "-o",
        "--output",
        metavar="DIR",
        required=True,
        help="Output directory for the translated v6 .p files (same basenames)",
    )
    p.add_argument(
        "--setup-file",
        metavar="PATH",
        default=None,
        help="Setup file to use (old 'key: values' dialect or INI). Default: "
        "auto-detect next to each .p file, then one level up (setup.txt first, "
        "then setup*.txt, setup*.cfg; case-insensitive)",
    )
    p.add_argument(
        "--sens",
        type=_parse_sens,
        default=None,
        metavar="NAME=VAL[,...]",
        help="Shear-probe sensitivities, e.g. sh1=0.0893,sh2=0.0558 "
        "(overrides <name>_sens: keys in the setup file)",
    )
    p.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Overwrite existing output files",
    )
    p.set_defaults(func=_cmd_v1to6)


def _add_init_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "init",
        help="Generate a template configuration file",
        description="Write a fully-commented template config.yaml with all default values.",
    )
    p.add_argument(
        "path",
        nargs="?",
        default="config.yaml",
        help="Output path (default: config.yaml)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing file",
    )
    p.set_defaults(func=_cmd_init)


def _add_prof_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "prof",
        help="Extract profiles",
        description="Extract profiles from .p or full-record .nc files.",
    )
    p.add_argument("files", nargs="+", metavar="FILE", help=".p or .nc file(s) or glob pattern(s)")
    p.add_argument(
        "-o",
        "--output",
        metavar="DIR",
        required=True,
        help="Output directory for per-profile .nc files",
    )
    p.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        metavar="N",
        help="Parallel workers (0 = all cores, default: 1)",
    )
    p.add_argument(
        "--P-min", type=float, default=None, help="Minimum pressure [dbar] (default: 0.5)"
    )
    p.add_argument(
        "--W-min", type=float, default=None, help="Minimum fall rate [dbar/s] (default: 0.3)"
    )
    p.add_argument(
        "--direction",
        default=None,
        choices=["auto", "up", "down", "glide", "horizontal"],
        help="Profile direction (default: auto, from vehicle)",
    )
    p.add_argument(
        "--vehicle",
        default=None,
        help="Vehicle type override (e.g. slocum_glider, vmp)",
    )
    p.add_argument(
        "--min-duration", type=float, default=None, help="Minimum profile duration [s] (default: 7)"
    )
    p.set_defaults(func=_cmd_prof)


def _add_eps_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "eps",
        help="Compute epsilon (TKE dissipation)",
        description="Compute TKE dissipation rate (epsilon) from VMP data.",
    )
    p.add_argument(
        "files", nargs="+", metavar="FILE", help=".p, full-record .nc, or per-profile .nc file(s)"
    )
    p.add_argument(
        "-o",
        "--output",
        metavar="DIR",
        required=True,
        help="Output directory for epsilon .nc files",
    )
    p.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        metavar="N",
        help="Parallel workers (0 = all cores, default: 1)",
    )
    p.add_argument(
        "--fft-length", type=int, default=None, help="FFT segment length [samples] (default: 1024)"
    )
    p.add_argument(
        "--diss-length",
        type=int,
        default=None,
        help="Dissipation window [samples] (default: 4*fft-length)",
    )
    p.add_argument(
        "--overlap",
        type=int,
        default=None,
        help="Window overlap [samples] (default: diss-length//2)",
    )
    p.add_argument(
        "--speed",
        type=float,
        default=None,
        help="Fixed profiling speed [m/s] (default: from dP/dt)",
    )
    p.add_argument(
        "--direction",
        default=None,
        choices=["auto", "up", "down", "glide", "horizontal"],
        help="Profile direction (default: auto, from vehicle)",
    )
    p.add_argument(
        "--vehicle",
        default=None,
        help="Vehicle type override (e.g. slocum_glider, vmp)",
    )
    p.add_argument(
        "--no-goodman",
        action="store_const",
        const=True,
        default=None,
        help="Disable Goodman coherent noise removal",
    )
    p.add_argument(
        "--f-AA", type=float, default=None, help="Anti-aliasing filter cutoff [Hz] (default: 98)"
    )
    p.add_argument(
        "--salinity",
        type=_parse_salinity,
        default=None,
        metavar="PSU|measured",
        help="Salinity for viscosity: a PSU value, or 'measured' to compute it "
        "from the conductivity/temperature channels (default: 35, fixed S)",
    )
    p.add_argument(
        "--temperature",
        type=_parse_temperature,
        default=None,
        metavar="NAME|degC",
        help="Reference temperature for viscosity: a channel name (e.g. T2, "
        "JAC_T), a fixed value [degC], or 'auto' = first plausible of "
        "T1..Tn, T, JAC_T (default: auto)",
    )
    p.add_argument(
        "--conductivity",
        default=None,
        metavar="NAME",
        help="Conductivity channel for --salinity measured "
        "(default: auto = JAC_C when present)",
    )
    p.set_defaults(func=_cmd_eps)


def _add_chi_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "chi",
        help="Compute chi (thermal dissipation)",
        description="Compute thermal variance dissipation rate (chi) from VMP data.",
    )
    p.add_argument(
        "files", nargs="+", metavar="FILE", help=".p, full-record .nc, or per-profile .nc file(s)"
    )
    p.add_argument(
        "-o",
        "--output",
        metavar="DIR",
        required=True,
        help="Output directory for chi .nc files",
    )
    p.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        metavar="N",
        help="Parallel workers (0 = all cores, default: 1)",
    )
    p.add_argument(
        "--fft-length", type=int, default=None, help="FFT segment length [samples] (default: 1024)"
    )
    p.add_argument(
        "--diss-length",
        type=int,
        default=None,
        help="Dissipation window [samples] (default: 4*fft-length)",
    )
    p.add_argument(
        "--overlap",
        type=int,
        default=None,
        help="Window overlap [samples] (default: diss-length//2)",
    )
    p.add_argument(
        "--speed",
        type=float,
        default=None,
        help="Fixed profiling speed [m/s] (default: from dP/dt)",
    )
    p.add_argument(
        "--direction",
        default=None,
        choices=["auto", "up", "down", "glide", "horizontal"],
        help="Profile direction (default: auto, from vehicle)",
    )
    p.add_argument(
        "--vehicle",
        default=None,
        help="Vehicle type override (e.g. slocum_glider, vmp)",
    )
    p.add_argument(
        "--no-goodman",
        action="store_const",
        const=True,
        default=None,
        help="Disable Goodman coherent noise removal",
    )
    p.add_argument(
        "--fp07-model",
        default=None,
        choices=["single_pole", "double_pole"],
        help="FP07 transfer function model (default: single_pole)",
    )
    p.add_argument(
        "--epsilon-dir",
        metavar="DIR",
        default=None,
        help="Directory with epsilon .nc files from 'rsi-tpw eps' (Method 1). "
        "If omitted, uses Method 2 (spectral fitting).",
    )
    p.add_argument(
        "--fit-method",
        default=None,
        choices=["mle", "iterative"],
        help="Method 2 fitting: mle or iterative (default: iterative)",
    )
    p.add_argument(
        "--spectrum-model",
        default=None,
        choices=["batchelor", "kraichnan"],
        help="Theoretical spectrum model (default: kraichnan)",
    )
    p.add_argument(
        "--f-AA", type=float, default=None, help="Anti-aliasing filter cutoff [Hz] (default: 98)"
    )
    p.add_argument(
        "--salinity",
        type=_parse_salinity,
        default=None,
        metavar="PSU|measured",
        help="Salinity for viscosity: a PSU value, or 'measured' to compute it "
        "from the conductivity/temperature channels (default: 35, fixed S)",
    )
    p.add_argument(
        "--temperature",
        type=_parse_temperature,
        default=None,
        metavar="NAME|degC",
        help="Reference temperature for viscosity/kappa_T: a channel name "
        "(e.g. T2, JAC_T), a fixed value [degC], or 'auto' = first plausible "
        "of T1..Tn, T, JAC_T (default: auto)",
    )
    p.add_argument(
        "--conductivity",
        default=None,
        metavar="NAME",
        help="Conductivity channel for --salinity measured "
        "(default: auto = JAC_C when present)",
    )
    p.set_defaults(func=_cmd_chi)


def _add_pipeline_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "pipeline",
        help="Run full pipeline (.p -> epsilon -> chi)",
        description="Run the full processing pipeline from raw .p files through "
        "epsilon (TKE dissipation) and chi (thermal dissipation). "
        "Profiles are detected automatically. Chi is computed using "
        "Method 1 (from shear-probe epsilon).",
    )
    p.add_argument("files", nargs="+", metavar="FILE", help=".p file(s) or glob pattern(s)")
    p.add_argument(
        "-o",
        "--output",
        metavar="DIR",
        required=True,
        help="Base output directory",
    )
    p.add_argument(
        "--direction",
        default=None,
        choices=["auto", "up", "down", "glide", "horizontal"],
        help="Profile direction (default: auto, from vehicle)",
    )
    p.add_argument(
        "--vehicle",
        default=None,
        help="Vehicle type override (e.g. slocum_glider, vmp)",
    )
    p.add_argument(
        "--speed",
        type=float,
        default=None,
        help="Fixed profiling speed [m/s] (default: from dP/dt)",
    )
    p.add_argument(
        "--speed-method",
        choices=["pressure", "em", "flight"],
        default=None,
        help=(
            "Through-water speed model (default: pressure = |dP/dt|). "
            "'em' uses the U_EM flowmeter channel; 'flight' uses the inviscid "
            "glider flight model |W|/sin(|pitch|-aoa) from the inclinometers."
        ),
    )
    p.add_argument(
        "--aoa",
        type=float,
        default=None,
        help="Angle of attack [deg] for --speed-method flight (default: 3.0)",
    )
    p.add_argument(
        "--eps-fft-length",
        type=int,
        default=None,
        help="FFT length for epsilon [samples] (default: 1024)",
    )
    p.add_argument(
        "--chi-fft-length",
        type=int,
        default=None,
        help="FFT length for chi [samples] (default: 1024)",
    )
    p.add_argument(
        "--no-goodman",
        action="store_const",
        const=True,
        default=None,
        help="Disable Goodman coherent noise removal for epsilon and chi",
    )
    p.add_argument(
        "--fp07-model",
        default=None,
        choices=["single_pole", "double_pole"],
        help="FP07 transfer function model (default: single_pole)",
    )
    p.add_argument(
        "--spectrum-model",
        default=None,
        choices=["batchelor", "kraichnan"],
        help="Theoretical spectrum model for chi (default: kraichnan)",
    )
    p.add_argument(
        "--f-AA", type=float, default=None, help="Anti-aliasing filter cutoff [Hz] (default: 98)"
    )
    p.add_argument(
        "--salinity",
        type=_parse_salinity,
        default=None,
        metavar="PSU|measured",
        help="Fixed salinity [PSU] fallback for viscosity (default: 35). "
        "'measured' maps to the pipeline's automatic behavior — measured JAC "
        "C/T salinity is already preferred when conductivity is present",
    )
    p.add_argument(
        "--temperature",
        type=_parse_temperature,
        default=None,
        metavar="NAME|degC",
        help="Reference temperature for viscosity: a channel name (e.g. T2, "
        "JAC_T), a fixed value [degC], or 'auto' = first plausible of "
        "T1..Tn, T, JAC_T (default: auto)",
    )
    p.add_argument(
        "--conductivity",
        default=None,
        metavar="NAME",
        help="Conductivity channel for the measured practical salinity "
        "(default: auto = JAC_C when present)",
    )
    p.set_defaults(func=_cmd_pipeline)


def _add_ql_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "ql",
        help="Interactive quick-look viewer",
        description="Open an interactive multi-panel viewer with profile navigation.",
    )
    p.add_argument("files", nargs="+", metavar="FILE", help=".p file(s) or glob pattern(s)")
    p.add_argument(
        "--fft-length", type=int, default=None, help="FFT segment length [samples] (default: 1024)"
    )
    p.add_argument(
        "--diss-length",
        type=int,
        default=None,
        help="Dissipation window [samples] (default: 4*fft-length)",
    )
    p.add_argument(
        "--f-AA", type=float, default=None, help="Anti-aliasing filter cutoff [Hz] (default: 98)"
    )
    p.add_argument(
        "--no-goodman",
        action="store_true",
        default=False,
        help="Disable Goodman coherent noise removal",
    )
    p.add_argument(
        "--direction",
        default=None,
        choices=["auto", "up", "down", "glide", "horizontal"],
        help="Profile direction (default: auto, from vehicle)",
    )
    p.add_argument(
        "--vehicle",
        default=None,
        help="Vehicle type override (e.g. slocum_glider, vmp)",
    )
    p.add_argument(
        "--W-min",
        type=float,
        default=None,
        help="Minimum fall rate [dbar/s] (default: 0.3, or 0.05 for glide/horizontal)",
    )
    p.add_argument(
        "--spec-P-range",
        type=float,
        nargs=2,
        metavar=("P_MIN", "P_MAX"),
        default=None,
        help="Pressure range [dbar] for spectral calculations (default: full profile)",
    )
    p.add_argument(
        "--chi-method",
        type=int,
        default=1,
        choices=[1, 2],
        help="Chi method for profile estimates: 1 = from epsilon, "
        "2 = iterative spectral fit (default: 1)",
    )
    p.add_argument(
        "--spectrum-model",
        default="kraichnan",
        choices=["batchelor", "kraichnan"],
        help="Theoretical spectrum model (default: kraichnan)",
    )
    p.set_defaults(func=_cmd_ql)


def _add_dl_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "dl",
        help="Interactive dissipation quality viewer",
        description="Open an interactive viewer comparing epsilon, chi (Batchelor vs "
        "Kraichnan), and Lueck (2022) figure of merit (FM) with profile navigation.",
    )
    p.add_argument("files", nargs="+", metavar="FILE", help=".p file(s) or glob pattern(s)")
    p.add_argument(
        "--fft-length", type=int, default=None, help="FFT segment length [samples] (default: 1024)"
    )
    p.add_argument(
        "--diss-length",
        type=int,
        default=None,
        help="Dissipation window [samples] (default: 4*fft-length)",
    )
    p.add_argument(
        "--f-AA", type=float, default=None, help="Anti-aliasing filter cutoff [Hz] (default: 98)"
    )
    p.add_argument(
        "--no-goodman",
        action="store_true",
        default=False,
        help="Disable Goodman coherent noise removal",
    )
    p.add_argument(
        "--direction",
        default=None,
        choices=["auto", "up", "down", "glide", "horizontal"],
        help="Profile direction (default: auto, from vehicle)",
    )
    p.add_argument(
        "--vehicle",
        default=None,
        help="Vehicle type override (e.g. slocum_glider, vmp)",
    )
    p.add_argument(
        "--W-min",
        type=float,
        default=None,
        help="Minimum fall rate [dbar/s] (default: 0.3, or 0.05 for glide/horizontal)",
    )
    p.add_argument(
        "--spec-P-range",
        type=float,
        nargs=2,
        metavar=("P_MIN", "P_MAX"),
        default=None,
        help="Pressure range [dbar] for spectral calculations (default: full profile)",
    )
    p.set_defaults(func=_cmd_dl)


def _add_sensors_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "sensors",
        help="Inventory shear/FP07 sensors across a .p file tree",
        description=(
            "Walk a directory tree of Rockland .p files and summarize, per sensor "
            "serial number, the date range of use, the number of files, the "
            "platform(s) it was mounted on, and whether its calibration parameters "
            "changed. Reads only each file's header + config block (not the data), "
            "so it is fast over large trees. Select sensor kinds with --shear / "
            "--fp07 / --all (default: all)."
        ),
    )
    p.add_argument(
        "paths",
        nargs="+",
        metavar="PATH",
        help="Directories (scanned recursively), .p files, or glob patterns",
    )
    p.add_argument("--shear", action="store_true", help="Inventory shear probes")
    p.add_argument("--fp07", action="store_true", help="Inventory FP07 thermistors")
    p.add_argument(
        "--all",
        dest="want_all",
        action="store_true",
        help="Inventory every sensor kind (shear + fp07; the default if none is given)",
    )
    p.add_argument(
        "--csv",
        metavar="PATH",
        default=None,
        help="Write a per-(file,channel) CSV table (overwritten if it exists)",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="List the individual files behind each changed parameter value",
    )
    p.add_argument(
        "--compact",
        action="store_true",
        help="One line per probe: SN, file count, calibration, and date range",
    )
    p.add_argument(
        "--cal-dir",
        metavar="DIR",
        default=None,
        help="Directory of Rockland shear-probe calibration PDFs. Check each shear "
        "probe's configured sensitivity against the calibration in effect at its "
        "observation time and report mismatches. Needs the 'cal' extra "
        "(pip install 'microstructure-tpw[cal]').",
    )
    p.add_argument(
        "--cal-tol",
        type=float,
        default=0.00005,
        metavar="SENS",
        help="Sensitivity-mismatch threshold for --cal-dir, in absolute sensitivity "
        "units (default: 0.00005, half the sheets' 4th-decimal resolution).",
    )
    p.set_defaults(func=_cmd_sensors)


def _add_ml_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "ml",
        help="Interactive mixing viewer",
        description="Open an interactive viewer of the background stratification "
        "(N², dT/dz) and derived diapycnal-mixing quantities (K_T, Γ, K_rho) with "
        "profile navigation.",
    )
    p.add_argument("files", nargs="+", metavar="FILE", help=".p file(s) or glob pattern(s)")
    p.add_argument(
        "--fft-length", type=int, default=None, help="FFT segment length [samples] (default: 1024)"
    )
    p.add_argument(
        "--diss-length",
        type=int,
        default=None,
        help="Dissipation window [samples] (default: 4*fft-length)",
    )
    p.add_argument(
        "--f-AA", type=float, default=None, help="Anti-aliasing filter cutoff [Hz] (default: 98)"
    )
    p.add_argument(
        "--no-goodman",
        action="store_true",
        default=False,
        help="Disable Goodman coherent noise removal",
    )
    p.add_argument(
        "--direction",
        default=None,
        choices=["auto", "up", "down", "glide", "horizontal"],
        help="Profile direction (default: auto, from vehicle)",
    )
    p.add_argument(
        "--vehicle",
        default=None,
        help="Vehicle type override (e.g. slocum_glider, vmp)",
    )
    p.add_argument(
        "--W-min",
        type=float,
        default=None,
        help="Minimum fall rate [dbar/s] (default: 0.3, or 0.05 for glide/horizontal)",
    )
    p.add_argument(
        "--spec-P-range",
        type=float,
        nargs=2,
        metavar=("P_MIN", "P_MAX"),
        default=None,
        help="Pressure range [dbar] to highlight on the profiles (default: none)",
    )
    p.add_argument(
        "--salinity",
        type=_parse_salinity,
        default=None,
        metavar="PSU|measured",
        help="Fixed practical salinity [PSU] for stratification "
        "(default: measured from JAC C/T, else 35; 'measured' selects "
        "that automatic behavior explicitly)",
    )
    p.set_defaults(func=_cmd_ml)


def _add_bench_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "bench",
        help="Bench-test diagnostic (quick_bench + checklist)",
        description=(
            "Evaluate a bench-test recording (dummy probes, instrument at rest on "
            "foam). Produces the raw-count time-series and counts^2/Hz spectra "
            "figures of ODAS quick_bench.m, plus an automatic PASS/FAIL evaluation "
            "of the Rockland Bench Test Review Checklist (subjective items are "
            "flagged REVIEW). Figures and the checklist text are written to the "
            "output directory (default: ./bench/); --show also opens them "
            "interactively. Compare the spectra against the instrument's RSI "
            "calibration report."
        ),
    )
    p.add_argument("files", nargs="+", metavar="FILE", help=".p file(s) or glob pattern(s)")
    p.add_argument(
        "-o",
        "--output",
        metavar="DIR",
        default=None,
        help="Output directory for figures + checklist (default: ./bench/, unless "
        "--show is given without -o, which only displays)",
    )
    p.add_argument(
        "--show", action="store_true", help="Open the figures in interactive windows"
    )
    p.add_argument(
        "--sn",
        default=None,
        metavar="SN",
        help="Serial number for figure titles/filenames (default: from config)",
    )
    p.add_argument(
        "--fft-sec",
        type=float,
        default=2.0,
        help="FFT segment length for spectra [seconds] (default: 2.0)",
    )
    p.add_argument(
        "--dpi", type=int, default=150, help="Figure resolution when saving (default: 150)"
    )
    p.add_argument(
        "--format",
        choices=["png", "pdf", "both", "pdf-bundle"],
        default="png",
        help="Saved figure format: one file per figure (png/pdf/both), or "
        "'pdf-bundle' for a single multi-page PDF (default: png)",
    )
    p.set_defaults(func=_cmd_bench)


def _add_patchconfig_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "patch-config",
        help="Patch config fields in .p file(s) (writes new files)",
        description=(
            "Edit selected configuration fields (instrument/cruise info, per-channel "
            "calibration) in Rockland .p files, driven by a YAML edit spec. Originals "
            "are never modified; patched copies are written to --out, with the change "
            "annotated and the full original configuration embedded for recovery. If a "
            "file's targeted values already match, that file is left unwritten. "
            "Scaffold the YAML with 'rsi-tpw patch-template'."
        ),
    )
    p.add_argument("files", nargs="+", metavar="FILE", help=".p file(s) or glob pattern(s)")
    p.add_argument(
        "--edits", required=True, metavar="YAML", help="YAML edit spec (see 'patch-template')"
    )
    p.add_argument(
        "-o", "--out", required=True, metavar="DIR", help="Output directory for patched .p files"
    )
    p.add_argument(
        "--dry-run", action="store_true", help="Show the configuration diff; write nothing"
    )
    p.add_argument(
        "--add-keys", action="store_true", help="Allow adding keys that do not already exist"
    )
    p.add_argument(
        "--batch-cal",
        action="store_true",
        help="Allow per-channel calibration edits across multiple files",
    )
    p.set_defaults(func=_cmd_patch_config)


def _add_patchtemplate_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "patch-template",
        help="Scaffold a patch-config edit spec from a .p file",
        description=(
            "Write a commented YAML edit-spec template pre-filled with the file's "
            "current editable values and channel names. Edit it, then apply it with "
            "'rsi-tpw patch-config'."
        ),
    )
    p.add_argument("file", metavar="FILE", help="Input .p file")
    p.add_argument("-o", "--output", metavar="YAML", help="Output path (default: print to stdout)")
    p.add_argument("--force", action="store_true", help="Overwrite existing output file")
    p.set_defaults(func=_cmd_patch_template)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Main CLI entry point for rsi-tpw."""
    import logging

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from odas_tpw.rsi import __version__

    parser = argparse.ArgumentParser(
        prog="rsi-tpw",
        description="microstructure-tpw: Read Rockland Scientific .P files and convert to NetCDF.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "-c",
        "--config",
        metavar="YAML",
        default=None,
        help="Configuration file (YAML). Generate a template with 'rsi-tpw init'.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    _add_info_parser(subparsers)
    _add_config_parser(subparsers)
    _add_cutp_parser(subparsers)
    _add_v1to6_parser(subparsers)
    _add_nc_parser(subparsers)
    _add_init_parser(subparsers)
    _add_patchconfig_parser(subparsers)
    _add_patchtemplate_parser(subparsers)
    _add_prof_parser(subparsers)
    _add_eps_parser(subparsers)
    _add_chi_parser(subparsers)
    _add_pipeline_parser(subparsers)
    _add_ql_parser(subparsers)
    _add_dl_parser(subparsers)
    _add_ml_parser(subparsers)
    _add_bench_parser(subparsers)
    _add_sensors_parser(subparsers)

    from odas_tpw._completion import enable_argcomplete

    enable_argcomplete(parser)
    args = parser.parse_args()
    args.func(args)
