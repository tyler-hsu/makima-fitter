"""Validated chronological scan extraction for CH Instruments CV text exports."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd


class ChiConversionError(ValueError):
    """Raised when a CH Instruments export cannot be segmented safely."""


@dataclass(frozen=True)
class ChiScan:
    number: int
    segments: tuple[int, int]
    start_index: int
    end_index: int
    source_start_row: int
    source_end_row: int
    source_start_time_s: float
    source_end_time_s: float


@dataclass
class ParsedChiCV:
    path: Path
    scan_rate_v_s: float
    sample_interval_v: float
    declared_segments: int
    data_header_index: int
    dataframe: pd.DataFrame
    segment_boundaries: tuple[int, ...]
    scans: tuple[ChiScan, ...]

    def scan(self, number: int) -> ChiScan:
        if number < 1 or number > len(self.scans):
            raise ChiConversionError(
                f"Requested CHI scan {number}, but this file contains "
                f"{len(self.scans)} complete scan(s)."
            )
        return self.scans[number - 1]

    def scan_dataframe(self, number: int) -> pd.DataFrame:
        selected = self.scan(number)
        frame = self.dataframe.iloc[
            selected.start_index : selected.end_index + 1
        ].copy()
        frame.reset_index(drop=True, inplace=True)
        frame["Time"] = (
            np.arange(len(frame), dtype=float)
            * self.sample_interval_v
            / self.scan_rate_v_s
        )
        frame.attrs["time_source"] = (
            "reconstructed from CH Instruments scan rate and sample interval; "
            f"complete scan {number}, segments "
            f"{selected.segments[0]}-{selected.segments[1]}"
        )
        return frame


def _header_float(lines: list[str], label: str) -> float:
    pattern = re.compile(
        rf"^\s*{re.escape(label)}\s*=\s*"
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*$"
    )
    for line in lines:
        match = pattern.match(line)
        if match:
            return float(match.group(1))
    raise ChiConversionError(f"CHI header is missing '{label} = ...'.")


def _header_int(lines: list[str], label: str) -> int:
    value = _header_float(lines, label)
    if not float(value).is_integer() or value < 1:
        raise ChiConversionError(f"CHI header has an invalid {label}: {value}")
    return int(value)


def _read_lines(path: Path) -> list[str]:
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return path.read_text(encoding=encoding, errors="strict").splitlines(True)
        except UnicodeError:
            continue
        except OSError as exc:
            raise ChiConversionError(f"Could not read CHI file: {exc}") from exc
    raise ChiConversionError("CHI file text encoding was not recognized.")


def _segment_boundaries(
    potential: np.ndarray,
    sample_interval_v: float,
    declared_segments: int,
) -> tuple[int, ...]:
    if len(potential) < 5:
        raise ChiConversionError("CHI file contains too few potential points.")

    differences = np.diff(potential)
    threshold = max(
        abs(sample_interval_v) * 0.20,
        float(np.ptp(potential)) * 1e-10,
        np.finfo(float).eps * 100,
    )
    active_edges = np.flatnonzero(np.abs(differences) > threshold)
    if len(active_edges) < 2:
        raise ChiConversionError("CHI potential waveform has no resolvable sweeps.")

    directions = np.sign(differences[active_edges]).astype(int)
    reversals: list[int] = []
    previous_direction = int(directions[0])
    for edge, direction in zip(active_edges[1:], directions[1:]):
        direction = int(direction)
        if direction != previous_direction:
            reversals.append(int(edge))
            previous_direction = direction

    detected_segments = len(reversals) + 1
    if detected_segments != declared_segments:
        raise ChiConversionError(
            "CHI segment header and potential waveform disagree: "
            f"header declares {declared_segments}, but {detected_segments} "
            "monotonic segment(s) were detected. No scan was selected."
        )

    boundaries = (0, *reversals, len(potential) - 1)
    if any(right <= left for left, right in zip(boundaries, boundaries[1:])):
        raise ChiConversionError("CHI segment boundaries are not chronological.")
    return tuple(int(index) for index in boundaries)


def parse_chi_cv(path: str | Path) -> ParsedChiCV:
    path = Path(path).resolve()
    lines = _read_lines(path)
    scan_rate_v_s = _header_float(lines, "Scan Rate (V/s)")
    sample_interval_v = _header_float(lines, "Sample Interval (V)")
    declared_segments = _header_int(lines, "Segment")
    if scan_rate_v_s <= 0 or sample_interval_v <= 0:
        raise ChiConversionError(
            "CHI scan rate and sample interval must both be positive."
        )

    data_header_index = next(
        (
            index
            for index, line in enumerate(lines)
            if "Potential/V" in line and "Current/A" in line
        ),
        None,
    )
    if data_header_index is None:
        raise ChiConversionError(
            "CHI file is missing the 'Potential/V, Current/A' data header."
        )
    header_line = lines[data_header_index]
    separator = "\t" if "\t" in header_line else ","

    try:
        frame = pd.read_csv(path, skiprows=data_header_index, sep=separator)
    except (OSError, pd.errors.ParserError, UnicodeError) as exc:
        raise ChiConversionError(f"Could not parse CHI potential/current data: {exc}") from exc
    frame = frame.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if frame.shape[1] != 2:
        raise ChiConversionError(
            f"Expected two CHI data columns, but found {frame.shape[1]}."
        )
    frame.columns = ["E", "I"]
    try:
        frame = frame.apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ChiConversionError("CHI potential/current data are not numeric.") from exc
    frame.reset_index(drop=True, inplace=True)

    potential = frame["E"].to_numpy(dtype=float)
    if not np.all(np.isfinite(potential)):
        raise ChiConversionError("CHI potential data contain non-finite values.")
    boundaries = _segment_boundaries(
        potential, sample_interval_v, declared_segments
    )

    complete_scan_count = declared_segments // 2
    if complete_scan_count < 1:
        raise ChiConversionError(
            "CHI file does not contain a complete two-segment scan."
        )
    scans: list[ChiScan] = []
    seconds_per_point = sample_interval_v / scan_rate_v_s
    first_data_row = data_header_index + 2  # one-based file row number
    for scan_index in range(complete_scan_count):
        start = boundaries[2 * scan_index]
        end = boundaries[2 * scan_index + 2]
        first_segment = 2 * scan_index + 1
        scans.append(
            ChiScan(
                number=scan_index + 1,
                segments=(first_segment, first_segment + 1),
                start_index=start,
                end_index=end,
                source_start_row=first_data_row + start,
                source_end_row=first_data_row + end,
                source_start_time_s=start * seconds_per_point,
                source_end_time_s=end * seconds_per_point,
            )
        )

    return ParsedChiCV(
        path=path,
        scan_rate_v_s=scan_rate_v_s,
        sample_interval_v=sample_interval_v,
        declared_segments=declared_segments,
        data_header_index=data_header_index,
        dataframe=frame,
        segment_boundaries=boundaries,
        scans=tuple(scans),
    )


def write_chi_scan_csv(parsed: ParsedChiCV, scan_number: int, output: str | Path) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = parsed.scan_dataframe(scan_number)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("scan rate (V/s)", f"{parsed.scan_rate_v_s:.15g}"))
        writer.writerow(("E (V)", "I (A)", "Time (s)"))
        for potential, current, time_s in frame[["E", "I", "Time"]].itertuples(
            index=False, name=None
        ):
            writer.writerow(
                (
                    f"{float(potential):.15g}",
                    f"{float(current):.15g}",
                    f"{float(time_s):.15g}",
                )
            )
    return output
