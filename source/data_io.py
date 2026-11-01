"""Input compatibility and complete-scan selection for Makima Fitter.

The EC-Lab and CH Instruments parsers are inherited unchanged from SD-Fitter
CV 2.02.  All readers return the same chronological E/I/Time schema so the
fitting layer never needs instrument-specific branches.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from chi_to_sd_fitter import ChiConversionError, parse_chi_cv
from ec_lab_to_sd_fitter import ConversionError, find_complete_scans, parse_mpt


FORMAT_TEMPLATE = "Template CSV file"
FORMAT_ECLAB = "EC-Lab ASCII export (.mpt/.txt)"
FORMAT_CHI = "CH Instruments text file"
FORMAT_NOVA = "Nova ASCII export"
FORMAT_PSTRACE = "PSTrace CSV export"
SUPPORTED_FORMATS = [
    FORMAT_TEMPLATE,
    FORMAT_ECLAB,
    FORMAT_CHI,
    FORMAT_NOVA,
    FORMAT_PSTRACE,
]


@dataclass
class LoadedScan:
    dataframe: pd.DataFrame
    source_path: Path
    source_format: str
    scan_number: int = 1
    scan_count: int = 1
    scan_rate_v_s: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def detect_input_format(path: str | Path) -> str | None:
    """Identify formats supported by SD-Fitter 2.02 from content first."""
    path = Path(path)
    previews: list[str] = []
    for encoding in ("utf-8-sig", "utf-16"):
        try:
            previews.append(path.read_text(encoding=encoding, errors="strict")[:12000])
        except (OSError, UnicodeError):
            continue
    preview = "\n".join(previews)
    first_line = preview.lstrip().splitlines()[0] if preview.strip() else ""
    if path.suffix.lower() == ".mpt" or first_line.startswith("EC-Lab ASCII FILE"):
        return FORMAT_ECLAB
    if "Instrument Model:" in preview and "Scan Rate (V/s)" in preview and "Potential/V" in preview:
        return FORMAT_CHI
    if "Potential applied (V)" in preview and "WE(1).Current (A)" in preview:
        return FORMAT_NOVA
    if path.suffix.lower() == ".csv" and any(
        marker in preview for marker in ("PSTrace", "PalmSens", "Current (µA)")
    ):
        return FORMAT_PSTRACE
    if path.suffix.lower() == ".csv":
        return FORMAT_TEMPLATE
    return None


def scan_count(path: str | Path, source_format: str) -> int:
    if source_format == FORMAT_ECLAB:
        return len(_cached_eclab_scans(Path(path)))
    if source_format == FORMAT_CHI:
        return len(_cached_chi(Path(path)).scans)
    return 1


def _file_cache_key(path: Path) -> tuple[str, int, int]:
    """Include file identity in parser caches without retaining stale edits."""
    resolved = path.resolve()
    stat = resolved.stat()
    return str(resolved), int(stat.st_size), int(stat.st_mtime_ns)


@lru_cache(maxsize=8)
def _parse_eclab_cached(path: str, _size: int, _mtime_ns: int):
    return parse_mpt(Path(path))


@lru_cache(maxsize=8)
def _parse_chi_cached(path: str, _size: int, _mtime_ns: int):
    return parse_chi_cv(Path(path))


@lru_cache(maxsize=8)
def _eclab_scans_cached(path: str, size: int, mtime_ns: int):
    return tuple(find_complete_scans(_parse_eclab_cached(path, size, mtime_ns)))


def _cached_eclab(path: Path):
    return _parse_eclab_cached(*_file_cache_key(path))


def _cached_eclab_scans(path: Path):
    return _eclab_scans_cached(*_file_cache_key(path))


def _cached_chi(path: Path):
    return _parse_chi_cached(*_file_cache_key(path))


def _numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame[["E", "I", "Time"]].copy()
    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    if len(frame) < 12:
        raise ValueError("The selected scan contains fewer than 12 valid samples")
    frame["Time"] -= float(frame["Time"].iloc[0])
    time = frame["Time"].to_numpy(float)
    if np.any(np.diff(time) <= 0):
        raise ValueError("Elapsed time must be strictly increasing within a scan")
    return frame


def _estimate_time(potential: np.ndarray, scan_rate_v_s: float) -> np.ndarray:
    rate = float(scan_rate_v_s)
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError("A positive scan rate is required when timestamps are absent")
    steps = np.abs(np.diff(potential))
    positive = steps[steps > np.finfo(float).eps]
    fallback = float(np.median(positive)) if len(positive) else 1e-3
    steps = np.where(steps > np.finfo(float).eps, steps, fallback)
    return np.r_[0.0, np.cumsum(steps / rate)]


def _read_template(path: Path, scan_rate_v_s: float | None) -> tuple[pd.DataFrame, float | None]:
    first_line = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()[0]
    embedded_rate: float | None = None
    skiprows = 0
    if first_line.lower().startswith("scan rate"):
        try:
            embedded_rate = float(first_line.split(",", 1)[1])
            skiprows = 1
        except (IndexError, ValueError):
            pass
    frame = pd.read_csv(path, skiprows=skiprows)
    frame = frame.dropna(axis=0, how="all").dropna(axis=1, how="all")
    normalized = {str(column).strip().casefold(): column for column in frame.columns}
    aliases = {
        "E": ("e", "potential", "potential (v)", "potential/v"),
        "I": ("i", "current", "current (a)", "current/a"),
        "Time": ("time", "time (s)", "time/s", "elapsed time (s)"),
    }
    selected: dict[str, Any] = {}
    for target, candidates in aliases.items():
        for candidate in candidates:
            if candidate in normalized:
                selected[target] = normalized[candidate]
                break
    if "E" not in selected or "I" not in selected:
        if frame.shape[1] not in (2, 3):
            raise ValueError("Template CSV must contain E and I columns, with optional Time")
        selected = {"E": frame.columns[0], "I": frame.columns[1]}
        if frame.shape[1] == 3:
            selected["Time"] = frame.columns[2]
    output = pd.DataFrame({"E": frame[selected["E"]], "I": frame[selected["I"]]})
    resolved_rate = scan_rate_v_s or embedded_rate
    if "Time" in selected:
        output["Time"] = frame[selected["Time"]]
    else:
        output["Time"] = _estimate_time(output["E"].to_numpy(float), float(resolved_rate or 0))
    return output, resolved_rate


def _read_nova(path: Path) -> tuple[pd.DataFrame, float | None]:
    frame = pd.read_csv(path)
    output = pd.DataFrame(
        {
            "E": frame["Potential applied (V)"],
            "I": frame["WE(1).Current (A)"],
            "Time": frame["Time (s)"],
        }
    )
    rate = None
    if "Scan rate (V/s)" in frame:
        values = pd.to_numeric(frame["Scan rate (V/s)"], errors="coerce").dropna()
        rate = float(values.iloc[0]) if len(values) else None
    return output, rate


def _read_pstrace(path: Path, scan_rate_v_s: float | None) -> tuple[pd.DataFrame, float]:
    if not scan_rate_v_s:
        raise ValueError("PSTrace input requires a positive scan rate")
    frame = pd.read_csv(path, header=4, encoding="utf-16_le")
    frame = frame.iloc[:, :2].copy()
    frame.columns = ["E", "I"]
    frame = frame[frame["E"].astype(str) != "\ufeff"].astype(float)
    lines = path.read_text(encoding="utf-16", errors="replace").splitlines()
    header = lines[5] if len(lines) > 5 else ""
    factors = {"m": 1e-3, "µ": 1e-6, "u": 1e-6, "n": 1e-9, "p": 1e-12}
    for marker, factor in factors.items():
        if marker + "A" in header:
            frame["I"] *= factor
            break
    frame["Time"] = _estimate_time(frame["E"].to_numpy(float), float(scan_rate_v_s))
    return frame, float(scan_rate_v_s)


def _chronological_scan_frames(
    path: Path,
    source_format: str,
) -> tuple[list[pd.DataFrame], float | None, list[dict[str, Any]]]:
    """Return complete scans with source-chronological timestamps.

    Frames deliberately retain the shared endpoint between neighboring scans;
    the window combiner removes that duplicate after slicing.
    """
    frames: list[pd.DataFrame] = []
    details: list[dict[str, Any]] = []
    if source_format == FORMAT_ECLAB:
        parsed = _cached_eclab(path)
        scans = _cached_eclab_scans(path)
        for selected in scans:
            frames.append(
                pd.DataFrame(
                    {
                        "E": [point.control_v for point in selected.points],
                        "I": [point.current_a for point in selected.points],
                        "Time": [point.time_s for point in selected.points],
                    }
                )
            )
            details.append(
                {
                    "source_start_row": selected.source_start_row,
                    "source_end_row": selected.source_end_row,
                    "raw_cycles": tuple(selected.raw_cycles),
                }
            )
        return frames, float(parsed.scan_rate_v_s), details
    if source_format == FORMAT_CHI:
        parsed = _cached_chi(path)
        for selected in parsed.scans:
            frame = parsed.scan_dataframe(selected.number)
            frame["Time"] += float(selected.source_start_time_s)
            frames.append(frame)
            details.append(
                {
                    "source_start_row": selected.source_start_row,
                    "source_end_row": selected.source_end_row,
                    "chi_segments": tuple(selected.segments),
                }
            )
        return frames, float(parsed.scan_rate_v_s), details
    raise ValueError(
        "Custom chronology windows are available for multi-scan EC-Lab and CH Instruments files"
    )


def _combine_scan_window_frames(
    frames: list[pd.DataFrame],
    start_scan: float,
    end_scan: float,
) -> pd.DataFrame:
    """Slice an arbitrary source-chronology interval using scan coordinates.

    Complete scan N occupies coordinates N.00 through N+1.00, so 3.75 to
    4.75 selects the last quarter of scan 3 followed by the first three
    quarters of scan 4. Fractions are resolved against elapsed acquisition
    time, which remains stable when neighboring scans contain unequal sample
    counts.
    """
    count = len(frames)
    start_scan = float(start_scan)
    end_scan = float(end_scan)
    if not (np.isfinite(start_scan) and np.isfinite(end_scan)):
        raise ValueError("Chronology-window limits must be finite")
    if start_scan < 1.0 or end_scan > count + 1.0 or end_scan <= start_scan:
        raise ValueError(
            f"Choose a chronology window inside 1.00 to {count + 1:.2f}, with the end after the start"
        )

    pieces: list[pd.DataFrame] = []
    first_scan = max(1, int(math.floor(start_scan)))
    last_scan = min(count, int(math.ceil(end_scan) - 1))
    for number in range(first_scan, last_scan + 1):
        local_start = max(0.0, start_scan - number)
        local_end = min(1.0, end_scan - number)
        if local_end < 0.0 or local_start > 1.0 or local_end <= local_start:
            continue
        frame = frames[number - 1]
        times = frame["Time"].to_numpy(float)
        duration = float(times[-1] - times[0])
        if duration <= 0:
            continue
        fractions = (times - times[0]) / duration
        start_index = int(np.searchsorted(fractions, local_start, side="left"))
        end_index = int(np.searchsorted(fractions, local_end, side="right"))
        start_index = min(max(start_index, 0), len(frame) - 1)
        end_index = min(max(end_index, start_index + 1), len(frame))
        pieces.append(frame.iloc[start_index:end_index].copy())

    if not pieces:
        raise ValueError("The chronology window does not contain any recorded samples")
    combined = pd.concat(pieces, ignore_index=True)
    combined = combined.drop_duplicates(subset="Time", keep="first")
    return combined.reset_index(drop=True)


def load_scan_window(
    path: str | Path,
    source_format: str | None,
    start_scan: float,
    end_scan: float,
    scan_rate_v_s: float | None = None,
) -> LoadedScan:
    """Load any chronological portion of a multi-cycle CV.

    Scan coordinates are one-based and continuous. A conventional complete
    scan N is therefore the window N.00 to N+1.00.
    """
    path = Path(path).resolve()
    source_format = source_format or detect_input_format(path)
    if source_format not in SUPPORTED_FORMATS:
        raise ValueError("The input format was not recognized")
    frames, detected_rate, details = _chronological_scan_frames(path, source_format)
    frame = _numeric_frame(_combine_scan_window_frames(frames, start_scan, end_scan))
    resolved_rate = scan_rate_v_s or detected_rate
    count = len(frames)
    first = max(1, int(math.floor(float(start_scan))))
    last = min(count, int(math.ceil(float(end_scan)) - 1))
    covered = details[first - 1 : last]
    metadata: dict[str, Any] = {
        "chronology_window": f"{float(start_scan):.6g}-{float(end_scan):.6g}",
        "chronology_window_custom": True,
    }
    if covered:
        metadata["source_rows"] = (
            f"{covered[0].get('source_start_row', '?')}-"
            f"{covered[-1].get('source_end_row', '?')}"
        )
    frame.attrs["source_format"] = source_format
    frame.attrs["scan_rate_v_s"] = resolved_rate
    return LoadedScan(
        frame,
        path,
        source_format,
        first,
        count,
        resolved_rate,
        metadata,
    )


def load_scan(
    path: str | Path,
    source_format: str | None = None,
    scan_number: int = 1,
    scan_rate_v_s: float | None = None,
) -> LoadedScan:
    """Read one chronological complete scan without startup or previous-tail augmentation."""
    path = Path(path).resolve()
    source_format = source_format or detect_input_format(path)
    if source_format not in SUPPORTED_FORMATS:
        raise ValueError("The input format was not recognized")
    scan_number = int(scan_number)
    if scan_number < 1:
        raise ValueError("Complete scan number must be at least 1")
    metadata: dict[str, Any] = {}

    if source_format == FORMAT_ECLAB:
        parsed = _cached_eclab(path)
        scans = _cached_eclab_scans(path)
        if scan_number > len(scans):
            raise ValueError(f"Requested scan {scan_number}; only {len(scans)} complete scan(s) were found")
        selected = scans[scan_number - 1]
        frame = pd.DataFrame(
            {
                "E": [point.control_v for point in selected.points],
                "I": [point.current_a for point in selected.points],
                "Time": [point.time_s for point in selected.points],
            }
        )
        scan_rate_v_s = float(parsed.scan_rate_v_s)
        metadata.update(
            {
                "raw_cycles": ", ".join(str(value) for value in selected.raw_cycles),
                "source_rows": f"{selected.source_start_row}-{selected.source_end_row}",
                "input_time_range_s": f"{selected.points[0].time_s:.12g}-{selected.points[-1].time_s:.12g}",
            }
        )
        count = len(scans)
    elif source_format == FORMAT_CHI:
        parsed = _cached_chi(path)
        selected = parsed.scan(scan_number)
        frame = parsed.scan_dataframe(scan_number)
        scan_rate_v_s = float(parsed.scan_rate_v_s)
        metadata.update(
            {
                "chi_segments": f"{selected.segments[0]}-{selected.segments[1]}",
                "source_rows": f"{selected.source_start_row}-{selected.source_end_row}",
                "input_time_range_s": f"{selected.source_start_time_s:.12g}-{selected.source_end_time_s:.12g}",
            }
        )
        count = len(parsed.scans)
    elif source_format == FORMAT_NOVA:
        frame, detected_rate = _read_nova(path)
        scan_rate_v_s = scan_rate_v_s or detected_rate
        count = 1
    elif source_format == FORMAT_PSTRACE:
        frame, scan_rate_v_s = _read_pstrace(path, scan_rate_v_s)
        count = 1
    else:
        frame, embedded_rate = _read_template(path, scan_rate_v_s)
        scan_rate_v_s = scan_rate_v_s or embedded_rate
        count = 1

    frame = _numeric_frame(frame)
    if scan_rate_v_s is None:
        delta_t = np.diff(frame["Time"].to_numpy(float))
        delta_e = np.abs(np.diff(frame["E"].to_numpy(float)))
        moving = delta_e > np.finfo(float).eps
        if np.any(moving):
            scan_rate_v_s = float(np.median(delta_e[moving] / delta_t[moving]))
    frame.attrs["source_format"] = source_format
    frame.attrs["scan_rate_v_s"] = scan_rate_v_s
    return LoadedScan(frame, path, source_format, scan_number, count, scan_rate_v_s, metadata)


__all__ = [
    "ChiConversionError",
    "ConversionError",
    "LoadedScan",
    "SUPPORTED_FORMATS",
    "detect_input_format",
    "load_scan",
    "load_scan_window",
    "scan_count",
]
