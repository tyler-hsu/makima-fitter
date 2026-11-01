#!/usr/bin/env python3
"""Convert chronological EC-Lab CV scans to SD-fitter template CSV files.

Complete scans are reconstructed from the continuous potential trajectory,
not from EC-Lab's numeric cycle groups.  This is necessary because EC-Lab can
place a cycle-number boundary at a switching potential, splitting one
experimental CV scan across two raw groups.  Point membership and order are
never circularly rotated.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


class ConversionError(Exception):
    """Raised when an input cannot be converted safely."""


@dataclass
class Point:
    raw_cycle: int
    control_v: float
    potential_v: float
    current_a: float
    time_s: float | None = None
    source_row: int = -1
    compensated_resistance_ohm: float | None = None


@dataclass
class CompleteScan:
    raw_cycle: int
    points: list[Point]
    raw_cycles: tuple[int, ...] = ()
    source_start_row: int = -1
    source_end_row: int = -1


@dataclass
class ParsedMpt:
    path: Path
    scan_rate_v_s: float
    switching_limits_v: tuple[float, float]
    points: list[Point]
    compensated_resistance_ohm: float | None = None


@dataclass
class PreviousTailResult:
    """A selected scan with an optional, automatically screened history tail."""

    scan: CompleteScan
    analysis_start_index: int = 0
    used: bool = False
    reason: str = "not requested"
    source_start_row: int = -1
    source_end_row: int = -1
    duration_s: float = 0.0
    potential_span_v: float = 0.0
    current_range_fraction: float = float("nan")
    detrended_rmse_fraction: float = float("nan")


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            pass
    raise ConversionError(f"Could not decode {path}")


def _number_after_label(lines: Sequence[str], label_pattern: str) -> float | None:
    pattern = re.compile(
        rf"^\s*{label_pattern}\s+([-+]?\d+(?:[.,]\d+)?(?:[Ee][-+]?\d+)?)\s*$",
        re.IGNORECASE,
    )
    for line in lines:
        match = pattern.match(line)
        if match:
            return float(match.group(1).replace(",", "."))
    return None


def _parse_scan_rate(lines: Sequence[str]) -> float:
    value = _number_after_label(lines, r"dE/dt")
    if value is None:
        raise ConversionError("Could not find the EC-Lab 'dE/dt' scan rate")

    unit_pattern = re.compile(r"^\s*dE/dt unit\s+(.+?)\s*$", re.IGNORECASE)
    unit = next(
        (match.group(1) for line in lines if (match := unit_pattern.match(line))),
        None,
    )
    if unit is None:
        raise ConversionError("Could not find the EC-Lab 'dE/dt unit'")

    normalized = unit.strip().replace("µ", "u").replace("μ", "u").lower()
    factors = {"v/s": 1.0, "mv/s": 1e-3, "uv/s": 1e-6}
    if normalized not in factors:
        raise ConversionError(f"Unsupported scan-rate unit: {unit!r}")
    return value * factors[normalized]


def _find_column(fieldnames: Sequence[str], candidates: Sequence[str]) -> str:
    normalized = {name.strip().lower(): name for name in fieldnames if name}
    for candidate in candidates:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]
    raise ConversionError(
        f"Missing required column; expected one of: {', '.join(candidates)}"
    )


def _find_optional_column(
    fieldnames: Sequence[str], candidates: Sequence[str]
) -> str | None:
    normalized = {name.strip().lower(): name for name in fieldnames if name}
    return next(
        (normalized[candidate.lower()] for candidate in candidates if candidate.lower() in normalized),
        None,
    )


def _current_factor(column: str) -> float:
    unit = column.rsplit("/", 1)[-1].strip().lower()
    factors = {"a": 1.0, "ma": 1e-3, "ua": 1e-6, "µa": 1e-6, "na": 1e-9}
    if unit not in factors:
        raise ConversionError(f"Unsupported current unit in column {column!r}")
    return factors[unit]


def parse_mpt(path: Path) -> ParsedMpt:
    text = _read_text(path)
    lines = text.splitlines()
    if not lines or not lines[0].lstrip().startswith("EC-Lab ASCII FILE"):
        raise ConversionError("Not an EC-Lab ASCII file")

    header_match = next(
        (re.search(r"Nb header lines\s*:\s*(\d+)", line, re.IGNORECASE) for line in lines[:20] if "Nb header lines" in line),
        None,
    )
    if not header_match:
        raise ConversionError("Could not find 'Nb header lines' metadata")
    header_count = int(header_match.group(1))
    if header_count < 2 or header_count > len(lines):
        raise ConversionError(f"Invalid header-line count: {header_count}")

    header = lines[: header_count - 1]
    table_lines = lines[header_count - 1 :]
    scan_rate = _parse_scan_rate(header)
    e1 = _number_after_label(header, r"E1\s*\(V\)")
    e2 = _number_after_label(header, r"E2\s*\(V\)")

    reader = csv.DictReader(table_lines, delimiter="\t")
    if not reader.fieldnames:
        raise ConversionError("The EC-Lab data table has no column names")
    cycle_col = _find_column(reader.fieldnames, ("cycle number", "cycle"))
    time_col = _find_column(reader.fieldnames, ("time/s", "time"))
    potential_col = _find_column(reader.fieldnames, ("Ewe/V", "Ewe/V vs.", "E/V"))
    control_col = _find_column(reader.fieldnames, ("control/V", potential_col))
    current_col = _find_column(
        reader.fieldnames,
        ("<I>/mA", "I/mA", "<I>/A", "I/A", "<I>/uA", "I/uA", "<I>/µA", "I/µA"),
    )
    resistance_col = _find_optional_column(
        reader.fieldnames, ("Rcmp/Ohm", "Rcmp / Ohm", "Rcmp")
    )
    factor = _current_factor(current_col)

    points: list[Point] = []
    for line_number, row in enumerate(reader, start=header_count + 1):
        try:
            cycle = int(round(float(row[cycle_col].replace(",", "."))))
            time_s = float(row[time_col].replace(",", "."))
            control = float(row[control_col].replace(",", "."))
            potential = float(row[potential_col].replace(",", "."))
            current = float(row[current_col].replace(",", ".")) * factor
            compensated_resistance = (
                float(row[resistance_col].replace(",", "."))
                if resistance_col and row.get(resistance_col)
                else None
            )
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise ConversionError(f"Invalid numeric data on line {line_number}") from exc
        if all(
            math.isfinite(value)
            for value in (time_s, control, potential, current)
        ):
            points.append(
                Point(
                    cycle,
                    control,
                    potential,
                    current,
                    time_s=time_s,
                    source_row=line_number - header_count - 1,
                    compensated_resistance_ohm=(
                        compensated_resistance
                        if compensated_resistance is not None
                        and math.isfinite(compensated_resistance)
                        and compensated_resistance >= 0
                        else None
                    ),
                )
            )

    if len(points) < 4:
        raise ConversionError("Not enough numeric CV data")

    active = [point for point in points if point.raw_cycle > 0]
    if not active:
        raise ConversionError("No positive EC-Lab cycle numbers were found")
    if e1 is None or e2 is None or math.isclose(e1, e2):
        limits = (
            min(point.control_v for point in active),
            max(point.control_v for point in active),
        )
    else:
        limits = (min(e1, e2), max(e1, e2))

    resistance_values = [
        point.compensated_resistance_ohm
        for point in active
        if point.compensated_resistance_ohm is not None
    ]
    compensated_resistance_ohm = (
        float(statistics.median(resistance_values)) if resistance_values else None
    )
    return ParsedMpt(
        path,
        scan_rate,
        limits,
        active,
        compensated_resistance_ohm=compensated_resistance_ohm,
    )


def _direction_from(
    points: Sequence[Point], start: int, direction_excursion: float
) -> int:
    """Return the first sustained potential direction after ``start``."""
    start_v = points[start].control_v
    for point in points[start + 1 :]:
        displacement = point.control_v - start_v
        if abs(displacement) >= direction_excursion:
            return 1 if displacement > 0 else -1
    return 0


def _find_vertex(
    points: Sequence[Point],
    start: int,
    direction: int,
    low: float,
    high: float,
    limit_tolerance: float,
    direction_excursion: float,
) -> int | None:
    """Find the true extremum after the trajectory enters a limit region."""
    if direction > 0:
        entered = next(
            (
                index
                for index in range(start, len(points))
                if points[index].control_v >= high - limit_tolerance
            ),
            None,
        )
    else:
        entered = next(
            (
                index
                for index in range(start, len(points))
                if points[index].control_v <= low + limit_tolerance
            ),
            None,
        )
    if entered is None:
        return None

    best = entered
    for index in range(entered + 1, len(points)):
        value = points[index].control_v
        best_value = points[best].control_v
        if (direction > 0 and value > best_value) or (
            direction < 0 and value < best_value
        ):
            best = index
            continue
        reversed_distance = (
            points[best].control_v - value
            if direction > 0
            else value - points[best].control_v
        )
        if reversed_distance >= direction_excursion:
            return best
    return None


def _find_return_to_start(
    points: Sequence[Point],
    start: int,
    start_v: float,
    direction: int,
) -> int | None:
    """Find the chronological crossing back through an interior start potential."""
    for index in range(start + 1, len(points)):
        before = points[index - 1].control_v
        after = points[index].control_v
        crossed = (
            before <= start_v <= after
            if direction > 0
            else before >= start_v >= after
        )
        if crossed:
            return min(
                (index - 1, index),
                key=lambda candidate: abs(points[candidate].control_v - start_v),
            )
    return None


def _scan_from_slice(points: Sequence[Point], start: int, end: int) -> CompleteScan:
    scan_points = list(points[start : end + 1])
    raw_cycles = tuple(dict.fromkeys(point.raw_cycle for point in scan_points))
    return CompleteScan(
        raw_cycle=raw_cycles[0],
        points=scan_points,
        raw_cycles=raw_cycles,
        source_start_row=scan_points[0].source_row,
        source_end_row=scan_points[-1].source_row,
    )


def find_complete_scans(parsed: ParsedMpt) -> list[CompleteScan]:
    """Return experimental CV scans in exact acquisition order.

    The first scan begins at the first active EC-Lab data point.  Each scan
    follows the uninterrupted trajectory through both switching limits and
    ends either at its interior starting potential or at the second switching
    vertex.  Raw EC-Lab cycle-number boundaries are retained as metadata but
    never used to cut or reorder the voltammogram.
    """
    low, high = parsed.switching_limits_v
    span = abs(high - low)
    if span <= 0:
        raise ConversionError("Could not determine distinct CV switching potentials")
    limit_tolerance = max(span * 0.01, 2e-3)
    boundary_tolerance = max(span * 0.025, 5e-3)
    direction_excursion = max(span * 0.002, 1e-5)
    points = parsed.points
    complete: list[CompleteScan] = []
    scan_start = 0
    while scan_start < len(points) - 3:
        start_v = points[scan_start].control_v
        initial_direction = _direction_from(
            points, scan_start, direction_excursion
        )
        if initial_direction == 0:
            break

        first_vertex = _find_vertex(
            points,
            scan_start,
            initial_direction,
            low,
            high,
            limit_tolerance,
            direction_excursion,
        )
        if first_vertex is None:
            break
        second_vertex = _find_vertex(
            points,
            first_vertex + 1,
            -initial_direction,
            low,
            high,
            limit_tolerance,
            direction_excursion,
        )
        if second_vertex is None:
            break

        starts_at_boundary = (
            start_v <= low + boundary_tolerance
            or start_v >= high - boundary_tolerance
        )
        if starts_at_boundary:
            scan_end = second_vertex
        else:
            scan_end = _find_return_to_start(
                points, second_vertex, start_v, initial_direction
            )
            if scan_end is None:
                break

        if scan_end <= scan_start:
            raise ConversionError("Detected a non-increasing CV scan boundary")
        complete.append(_scan_from_slice(points, scan_start, scan_end))
        scan_start = scan_end

    return complete


def _linear_tail_metrics(points: Sequence[Point], current_scale: float) -> tuple[float, float]:
    """Return range and detrended-RMSE fractions for a candidate history tail."""
    if len(points) < 3 or current_scale <= 0:
        return float("inf"), float("inf")
    times = [float(point.time_s) for point in points]
    currents = [float(point.current_a) for point in points]
    origin = times[0]
    x = [value - origin for value in times]
    x_mean = statistics.fmean(x)
    y_mean = statistics.fmean(currents)
    denominator = sum((value - x_mean) ** 2 for value in x)
    slope = (
        sum((x_value - x_mean) * (y_value - y_mean) for x_value, y_value in zip(x, currents))
        / denominator
        if denominator > 0
        else 0.0
    )
    intercept = y_mean - slope * x_mean
    residuals = [
        y_value - (slope * x_value + intercept)
        for x_value, y_value in zip(x, currents)
    ]
    rmse = math.sqrt(statistics.fmean(value * value for value in residuals))
    current_range = max(currents) - min(currents)
    return current_range / current_scale, rmse / current_scale


def add_previous_tail(
    parsed: ParsedMpt,
    scans: Sequence[CompleteScan],
    scan_number: int,
    *,
    potential_span_v: float = 0.50,
    maximum_current_range_fraction: float = 0.08,
    maximum_detrended_rmse_fraction: float = 0.015,
) -> PreviousTailResult:
    """Prepend a flat terminal branch from the preceding scan when defensible.

    The tail is numerical history, not another fitted scan.  It is accepted
    only for an interior-start scan that continues in the same potential
    direction and whose final 0.50 V from the preceding scan is peak-free by
    simple current-range and detrended-residual screens.
    """
    scan_number = int(scan_number)
    if scan_number < 1 or scan_number > len(scans):
        raise ConversionError(f"Complete scan {scan_number} is not available")
    selected = scans[scan_number - 1]
    result = PreviousTailResult(scan=selected)
    if scan_number == 1:
        result.reason = "first scan has no preceding history tail"
        return result

    low, high = parsed.switching_limits_v
    waveform_span = abs(high - low)
    start_v = float(selected.points[0].control_v)
    boundary_tolerance = max(0.025 * waveform_span, 5e-3)
    if min(abs(start_v - low), abs(start_v - high)) <= boundary_tolerance:
        result.reason = "selected scan begins at a switching limit"
        return result

    direction_excursion = max(0.002 * waveform_span, 1e-5)
    direction = _direction_from(selected.points, 0, direction_excursion)
    if direction == 0:
        result.reason = "selected scan has no sustained initial direction"
        return result

    previous = scans[scan_number - 2]
    shared_previous = previous.points[-1]
    shared_selected = selected.points[0]
    if (
        shared_previous.time_s is None
        or shared_selected.time_s is None
        or not math.isclose(
            float(shared_previous.time_s),
            float(shared_selected.time_s),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        result.reason = "preceding scan does not share the selected scan boundary"
        return result

    previous_direction = _direction_from(
        list(reversed(previous.points)), 0, direction_excursion
    )
    # Reversing the points reverses the sign returned by _direction_from.
    if previous_direction != -direction:
        result.reason = "preceding and selected scan directions are not continuous"
        return result

    if direction > 0:
        branch_start = min(
            range(len(previous.points)),
            key=lambda index: previous.points[index].control_v,
        )
    else:
        branch_start = max(
            range(len(previous.points)),
            key=lambda index: previous.points[index].control_v,
        )
    branch = previous.points[branch_start:]
    end_control = float(branch[-1].control_v)
    candidates = [
        point
        for point in branch
        if -1e-9
        <= direction * (end_control - float(point.control_v))
        <= float(potential_span_v) + 1e-9
    ]
    if len(candidates) < 20:
        result.reason = "fewer than 20 points are available in the previous tail"
        return result

    actual_span = abs(float(candidates[-1].control_v) - float(candidates[0].control_v))
    if actual_span < min(0.30, 0.8 * float(potential_span_v)):
        result.reason = "previous tail has insufficient potential span"
        return result

    selected_currents = [float(point.current_a) for point in selected.points]
    current_scale = max(
        max(selected_currents) - min(selected_currents),
        1e-15,
    )
    range_fraction, rmse_fraction = _linear_tail_metrics(candidates, current_scale)
    if range_fraction > float(maximum_current_range_fraction):
        result.reason = "previous tail current range is too large"
        return result
    if rmse_fraction > float(maximum_detrended_rmse_fraction):
        result.reason = "previous tail contains structured current"
        return result

    # The shared endpoint is already the first selected-scan sample.  Exclude
    # its previous-scan copy so the written timestamps remain strictly increasing.
    prefix = list(candidates[:-1])
    combined = CompleteScan(
        raw_cycle=selected.raw_cycle,
        points=prefix + list(selected.points),
        raw_cycles=selected.raw_cycles,
        source_start_row=prefix[0].source_row,
        source_end_row=selected.source_end_row,
    )
    start_time = float(prefix[0].time_s)
    analysis_time = float(selected.points[0].time_s)
    return PreviousTailResult(
        scan=combined,
        analysis_start_index=len(prefix),
        used=True,
        reason="accepted flat previous-scan terminal branch",
        source_start_row=prefix[0].source_row,
        source_end_row=prefix[-1].source_row,
        duration_s=analysis_time - start_time,
        potential_span_v=actual_span,
        current_range_fraction=range_fraction,
        detrended_rmse_fraction=rmse_fraction,
    )


def rotate_scan(
    scan: CompleteScan,
    switching_limits_v: tuple[float, float],
    start_mode: str,
) -> CompleteScan:
    """Compatibility wrapper that now enforces chronological point order."""
    if start_mode in ("chronological", "switching-limit"):
        return scan
    raise ConversionError(
        "Circular scan rotation was removed in version 1.5.0; "
        "use chronological conversion."
    )


def write_sd_fitter_csv(scan: CompleteScan, scan_rate_v_s: float, output: Path) -> None:
    timestamps = [point.time_s for point in scan.points]
    if any(time_s is None or not math.isfinite(time_s) for time_s in timestamps):
        raise ConversionError(
            "Cannot preserve EC-Lab timing because this scan contains a missing "
            "or non-finite timestamp"
        )
    elapsed = [float(time_s) - float(timestamps[0]) for time_s in timestamps]
    if any(later <= earlier for earlier, later in zip(elapsed, elapsed[1:])):
        raise ConversionError(
            "EC-Lab timestamps must increase strictly within the selected scan"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("scan rate (V/s)", format(scan_rate_v_s, ".12g")))
        writer.writerow(("E (V)", "I (A)", "Time (s)"))
        for point, elapsed_s in zip(scan.points, elapsed):
            writer.writerow(
                (
                    format(point.potential_v, ".12g"),
                    format(point.current_a, ".12g"),
                    format(elapsed_s, ".12g"),
                )
            )


def _collect_inputs(items: Iterable[str], recursive: bool) -> list[Path]:
    found: list[Path] = []
    for item in items:
        path = Path(item).expanduser()
        if path.is_dir():
            iterator = path.rglob if recursive else path.glob
            found.extend(iterator("*.mpt"))
            found.extend(iterator("*.txt"))
        elif path.is_file():
            found.append(path)
        else:
            raise ConversionError(f"Input does not exist: {path}")
    unique = sorted({path.resolve() for path in found})
    if not unique:
        raise ConversionError("No EC-Lab .mpt or .txt input files were found")
    return unique


def convert_file(
    path: Path,
    output_dir: Path | None,
    scan_number: int,
    start_mode: str,
) -> Path:
    parsed = parse_mpt(path)
    scans = find_complete_scans(parsed)
    if len(scans) < scan_number:
        raw = "; ".join(
            "+".join(str(cycle) for cycle in scan.raw_cycles) for scan in scans
        ) or "none"
        raise ConversionError(
            f"requested complete scan {scan_number}, but found {len(scans)} "
            f"(EC-Lab cycle numbers: {raw})"
        )
    scan = rotate_scan(scans[scan_number - 1], parsed.switching_limits_v, start_mode)
    destination = output_dir if output_dir is not None else path.parent
    output = destination / f"{path.stem}_scan{scan_number}_chronological_sd-fitter.csv"
    write_sd_fitter_csv(scan, parsed.scan_rate_v_s, output)
    raw_cycles = ", ".join(str(cycle) for cycle in scan.raw_cycles)
    print(
        f"Converted {path.name}: complete scan {scan_number} spans EC-Lab cycle "
        f"group(s) {raw_cycles}, source rows {scan.source_start_row}-"
        f"{scan.source_end_row}, {len(scan.points)} points, "
        f"{parsed.scan_rate_v_s:g} V/s, chronological order\n"
        f"  -> {output}"
    )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert EC-Lab CV ASCII .mpt files to SD-fitter template CSV files. "
            "The default is the first complete chronological CV scan."
        )
    )
    parser.add_argument(
        "inputs", nargs="+", help="One or more EC-Lab .mpt/.txt files or folders"
    )
    parser.add_argument(
        "--scan", type=int, default=1, help="Complete CV scan to export (default: 1)"
    )
    parser.add_argument(
        "-o", "--output-dir", type=Path, help="Output folder (default: beside each input)"
    )
    parser.add_argument(
        "-r", "--recursive", action="store_true", help="Search input folders recursively"
    )
    parser.add_argument(
        "--start-mode",
        choices=("chronological", "switching-limit"),
        default="chronological",
        help=(
            "'chronological' preserves the original EC-Lab acquisition order "
            "(default); 'switching-limit' is a compatibility alias. Circular "
            "rotation is not supported."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.scan < 1:
        print("error: --scan must be at least 1", file=sys.stderr)
        return 2
    try:
        inputs = _collect_inputs(args.inputs, args.recursive)
        for path in inputs:
            convert_file(path, args.output_dir, args.scan, args.start_mode)
    except (ConversionError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
