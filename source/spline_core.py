"""Numerical core for the spline-background semi-derivative fitter."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
import textwrap
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from lmfit import Parameters
from lmfit.models import Pearson4Model, SplineModel
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from scipy.interpolate import Akima1DInterpolator, PchipInterpolator, make_interp_spline
from scipy.optimize import least_squares
from scipy.signal import (
    butter,
    buttord,
    filtfilt,
    find_peaks,
    fftconvolve,
    freqz,
    iirnotch,
    savgol_coeffs,
    savgol_filter,
    sosfiltfilt,
    sosfreqz,
)
from scipy.special import betaln, gamma, loggamma

from data_io import LoadedScan


PACKAGE_VERSION = "2.0.0"
PEARSON4_MATHEMATICAL_MIN = 0.5 + 1e-9


@dataclass(frozen=True)
class FilterStage:
    """One independently configured stage in an ordered signal-processing chain."""

    kind: str = "savgol"
    domain: str = "current"
    scope: str = "global"
    range: tuple[int, int] | None = None
    frequency_hz: float = 5.0
    bandwidth_hz: float = 1.0
    attenuation_db: float = 10.0
    window_mv: float = 10.0
    polynomial_order: int = 4
    hampel_sigma: float = 3.5
    harmonics: int = 1
    feather_mv: float = 10.0
    envelope_prominence_pct: float = 10.0


@dataclass
class FitConfig:
    pearson4_exponent_min: float = 0.5
    peak_center_window_v: float = 0.25
    preview_max_nfev: int = 700
    final_max_nfev: int = 5000
    background_kind: str = "makima"
    background_scope: str = "independent"
    knots_locked_to_data: bool = True
    filter_kind: str = "none"
    filter_domain: str = "current"
    filter_scope: str = "global"
    filter_range: tuple[int, int] | None = None
    filter_frequency_hz: float = 5.0
    filter_bandwidth_hz: float = 1.0
    filter_attenuation_db: float = 10.0
    filter_window_mv: float = 10.0
    filter_polynomial_order: int = 4
    filter_hampel_sigma: float = 3.5
    filter_harmonics: int = 1
    filter_feather_mv: float = 10.0
    filter_envelope_prominence_pct: float = 10.0
    filter_secondary_kind: str = "none"
    filter_secondary_range: tuple[int, int] | None = None
    filter_secondary_frequency_hz: float = 5.0
    filter_secondary_feather_mv: float = 10.0
    filter_secondary_envelope_prominence_pct: float = 10.0
    filter_stages: tuple[FilterStage, ...] = ()


@dataclass
class AnalysisData:
    scan: LoadedScan
    time: np.ndarray
    potential: np.ndarray
    current_a: np.ndarray
    current: np.ndarray
    transform: np.ndarray
    scaler: float
    scale_prefix: str
    report_scale_prefix: str
    scan_rate_v_s: float
    switching_indices: list[int]
    segment_bounds: list[tuple[int, int]]
    excluded_intervals: list[tuple[int, int]]
    excluded_mask: np.ndarray
    fit_weights: np.ndarray
    filter_diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class PeakResult:
    number: int
    selected_index: int
    fitted_index: int
    current_index: int
    potential_v: float
    time_s: float
    current_scaled: float
    charge_scaled: float
    sweep_direction: str
    exponent: float
    skew: float
    baseline_current_scaled: float
    center_time_s: float


@dataclass
class FitOutcome:
    data: AnalysisData
    config: FitConfig
    knot_indices: list[int]
    knot_values: dict[int, float]
    peak_indices: list[int]
    lmfit_result: Any
    components: dict[str, np.ndarray]
    current_components: dict[str, np.ndarray]
    spline_background: np.ndarray
    polynomial_background: np.ndarray
    reconstructed_current: np.ndarray
    background_current: np.ndarray
    segment_baseline_current: np.ndarray
    peak_baseline_currents: dict[str, np.ndarray]
    inverse_boundary_correction: np.ndarray
    current_offset_scaled: float
    normalized_signal: np.ndarray
    peak_results: list[PeakResult]
    background_parameters: dict[str, float] = field(default_factory=dict)
    background_diagnostics: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class BackgroundConstruction:
    """Complete result of either interpolation or anchor-guided fitting."""

    background: np.ndarray
    polynomial: np.ndarray
    components: dict[str, np.ndarray]
    anchors: dict[int, float]
    parameters: dict[str, float] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class IndependentFitResult:
    """Small lmfit-compatible result assembled from independent sweep fits."""

    best_fit: np.ndarray
    components: dict[str, np.ndarray]
    params: Parameters
    segment_results: list[Any]
    success: bool
    message: str
    chisqr: float
    redchi: float
    aic: float
    bic: float
    nfev: int
    ndata: int
    nvarys: int

    def eval_components(self) -> dict[str, np.ndarray]:
        return self.components

    def fit_report(self) -> str:
        lines = [
            "[[Independent Sweep Fits]]",
            "Forward and reverse sweep parameters were optimized separately.",
        ]
        for number, result in enumerate(self.segment_results, start=1):
            lines.extend(["", f"[[Sweep {number}]]"])
            if result is None:
                lines.append("No Pearson-IV peaks selected on this sweep.")
            else:
                lines.append(result.fit_report())
        return "\n".join(lines)


class SegmentSplineModel(SplineModel):
    """An lmfit spline confined to one sweep with polynomial extrapolation.

    The spline is used only between the first and last user-selected knot on
    its sweep segment.  A fixed polynomial through that segment's selected
    knot samples supplies the background before and after that interval.  Each
    polynomial tail is shifted to meet its fitted spline boundary continuously.
    Each chronological sweep owns a separate instance, so no continuity
    constraint is imposed at a switching potential.
    """

    def __init__(
        self,
        xknots: Sequence[float],
        segment_start_time: float,
        segment_end_time: float,
        polynomial_coefficients: Sequence[float],
        polynomial_center: float,
        polynomial_scale: float,
        prefix: str,
    ):
        self.segment_start_time = float(segment_start_time)
        self.segment_end_time = float(segment_end_time)
        self.polynomial_coefficients = np.asarray(polynomial_coefficients, dtype=float)
        self.polynomial_center = float(polynomial_center)
        self.polynomial_scale = float(polynomial_scale)
        self.first_knot_time = float(np.min(xknots))
        self.last_knot_time = float(np.max(xknots))
        super().__init__(xknots=xknots, prefix=prefix)

    def polynomial(self, x: Sequence[float]) -> np.ndarray:
        normalized = (np.asarray(x, dtype=float) - self.polynomial_center) / self.polynomial_scale
        return np.polyval(self.polynomial_coefficients, normalized)

    def eval(self, params=None, **kwargs):
        x = np.asarray(kwargs.get("x"), dtype=float)
        spline_values = np.asarray(super().eval(params=params, **kwargs), dtype=float)
        boundary_kwargs = dict(kwargs)
        boundary_kwargs["x"] = np.asarray(
            [self.first_knot_time, self.last_knot_time], dtype=float
        )
        boundary_values = np.asarray(
            super().eval(params=params, **boundary_kwargs), dtype=float
        )
        output = np.zeros_like(x, dtype=float)
        segment_mask = (x >= self.segment_start_time) & (x <= self.segment_end_time)
        spline_mask = segment_mask & (x >= self.first_knot_time) & (x <= self.last_knot_time)
        left_mask = segment_mask & (x < self.first_knot_time)
        right_mask = segment_mask & (x > self.last_knot_time)
        output[spline_mask] = spline_values[spline_mask]
        output[left_mask] = self.polynomial(x[left_mask]) + (
            boundary_values[0] - self.polynomial([self.first_knot_time])[0]
        )
        output[right_mask] = self.polynomial(x[right_mask]) + (
            boundary_values[1] - self.polynomial([self.last_knot_time])[0]
        )
        return output


def fractional_differintegral(x: Sequence[float], y: Sequence[float], q: float) -> np.ndarray:
    """Oldham-Spanier Riemann-Liouville operator used by SD-Fitter.

    This convolution form is numerically equivalent to the historical loop and
    is fast enough for interactive updates.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or len(x) < 2:
        raise ValueError("Fractional differintegration requires matching 1-D arrays")
    elapsed = x - float(x[0])
    if np.any(np.diff(elapsed) <= 0):
        raise ValueError("Time must be strictly increasing")
    n = len(x)
    j = np.arange(n, dtype=float)
    weights = np.zeros(n, dtype=float)
    weights[0] = 1.0 / gamma(2.0 - q)
    weights[1:] = (
        (j[1:] + 1.0) ** (1.0 - q)
        - 2.0 * j[1:] ** (1.0 - q)
        + (j[1:] - 1.0) ** (1.0 - q)
    ) / gamma(2.0 - q)
    quotient = np.zeros(n, dtype=float)
    quotient[1:] = j[1:] ** q / elapsed[1:] ** q
    # Direct convolution preserves the established numerical path for ordinary
    # scans. FFT convolution avoids quadratic scaling on exceptionally dense
    # acquisitions while producing the same full-resolution fitted signal.
    convolution = (
        np.convolve(y, weights, mode="full")[:n]
        if n < 4096
        else fftconvolve(y, weights, mode="full")[:n]
    )
    return quotient * convolution


def current_scale(current_a: Sequence[float]) -> tuple[float, str, str]:
    maximum = float(np.nanmax(np.abs(np.asarray(current_a, dtype=float))))
    if maximum >= 1e-3:
        return 1e3, "m", "m"
    if maximum >= 1e-6:
        return 1e6, "μ", "u"
    if maximum >= 1e-9:
        return 1e9, "n", "n"
    return 1e12, "p", "p"


def display_time_axis(time_s: Sequence[float]) -> tuple[np.ndarray, str, float]:
    """Return scan time in a readable unit without changing fitting units."""
    values = np.asarray(time_s, dtype=float)
    finite = values[np.isfinite(values)]
    span = float(np.ptp(finite)) if len(finite) else 0.0
    if 0 < span < 1e-3:
        scale, unit = 1e6, "μs"
    elif 0 < span < 1.0:
        scale, unit = 1e3, "ms"
    elif span >= 120.0:
        scale, unit = 1.0 / 60.0, "min"
    else:
        scale, unit = 1.0, "s"
    return values * scale, unit, scale


def experimental_decimal_places(
    values: Sequence[float],
    *,
    smallest_step: bool,
    maximum: int = 6,
) -> int:
    """Infer useful fixed decimals from recorded sampling resolution."""
    values = np.asarray(values, dtype=float)
    differences = np.abs(np.diff(values))
    scale = max(float(np.nanmax(np.abs(values))), 1.0)
    differences = differences[
        np.isfinite(differences)
        & (differences > np.finfo(float).eps * scale * 16.0)
    ]
    if not len(differences):
        return 0
    resolution = float(np.min(differences) if smallest_step else np.median(differences))
    resolution = float(f"{resolution:.8g}")
    tolerance = max(abs(resolution) * 1e-6, np.finfo(float).eps)
    for decimals in range(maximum + 1):
        if abs(round(resolution, decimals) - resolution) <= tolerance:
            return decimals
    return maximum


def format_significant(value: float, significant_digits: int = 4) -> str:
    """Format a value with visible trailing zeroes when they carry precision."""
    value = float(value)
    if not np.isfinite(value):
        return str(value)
    if value == 0:
        return "0." + "0" * (significant_digits - 1)
    exponent = int(np.floor(np.log10(abs(value))))
    decimals = significant_digits - exponent - 1
    if decimals >= 0 and exponent < significant_digits:
        return f"{value:.{decimals}f}"
    return f"{value:.{significant_digits - 1}e}"


def normalize_exclusion_intervals(
    length: int, intervals: Sequence[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Clamp, sort, and merge inclusive transient-exclusion intervals."""
    normalized: list[tuple[int, int]] = []
    for first, second in intervals:
        start, end = sorted((int(first), int(second)))
        start = max(0, start)
        end = min(length - 1, end)
        if start <= end:
            normalized.append((start, end))
    merged: list[tuple[int, int]] = []
    for start, end in sorted(normalized):
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def build_exclusion_mask(
    length: int, intervals: Sequence[tuple[int, int]]
) -> np.ndarray:
    mask = np.zeros(length, dtype=bool)
    for start, end in normalize_exclusion_intervals(length, intervals):
        mask[start : end + 1] = True
    return mask


def apply_transient_exclusions(
    data: AnalysisData, intervals: Sequence[tuple[int, int]]
) -> AnalysisData:
    normalized = normalize_exclusion_intervals(len(data.time), intervals)
    mask = build_exclusion_mask(len(data.time), normalized)
    return replace(
        data,
        excluded_intervals=normalized,
        excluded_mask=mask,
        fit_weights=np.where(mask, 0.0, np.asarray(data.fit_weights, dtype=float)),
    )


def _fill_zero_signs(signs: np.ndarray) -> np.ndarray:
    signs = signs.astype(float, copy=True)
    for index in range(1, len(signs)):
        if signs[index] == 0:
            signs[index] = signs[index - 1]
    for index in range(len(signs) - 2, -1, -1):
        if signs[index] == 0:
            signs[index] = signs[index + 1]
    return signs


def detect_switching_points(potential: Sequence[float]) -> list[int]:
    """Detect physical turning points by monotonic-run changes.

    Duplicate-potential plateaus are collapsed into one reversal and short
    noise runs are ignored.  Endpoints are segment boundaries, not internal
    switching events.
    """
    values = np.asarray(potential, dtype=float)
    if len(values) < 12 or not np.all(np.isfinite(values)):
        raise ValueError("Switching detection requires at least 12 finite potentials")
    span = float(np.ptp(values))
    if span <= np.finfo(float).eps:
        raise ValueError("The selected scan has no potential excursion")
    delta = np.diff(values)
    nonzero = np.abs(delta[np.abs(delta) > np.finfo(float).eps])
    typical_step = float(np.median(nonzero)) if len(nonzero) else span / len(values)
    tolerance = max(typical_step * 0.12, span * 1e-8)
    signs = np.sign(np.where(np.abs(delta) <= tolerance, 0.0, delta))
    signs = _fill_zero_signs(signs)
    transitions = np.flatnonzero(signs[:-1] * signs[1:] < 0) + 1
    if not len(transitions):
        return []
    clustered: list[list[int]] = [[int(transitions[0])]]
    cluster_gap = max(3, int(round(len(values) * 0.004)))
    for raw in transitions[1:]:
        index = int(raw)
        if index - clustered[-1][-1] <= cluster_gap:
            clustered[-1].append(index)
        else:
            clustered.append([index])
    margin = max(3, int(round(len(values) * 0.01)))
    resolved: list[int] = []
    for cluster in clustered:
        center = int(round(float(np.mean(cluster))))
        start = max(0, center - cluster_gap)
        end = min(len(values), center + cluster_gap + 1)
        increasing_to_decreasing = signs[max(0, center - 1)] > 0
        local = values[start:end]
        index = start + int(np.argmax(local) if increasing_to_decreasing else np.argmin(local))
        if margin <= index < len(values) - margin:
            resolved.append(index)
    return sorted(set(resolved))


def segment_bounds(length: int, switching_indices: Sequence[int]) -> list[tuple[int, int]]:
    cuts = [0] + sorted({int(index) for index in switching_indices if 0 < index < length - 1}) + [length - 1]
    return [
        (cuts[index] + (1 if index else 0), cuts[index + 1])
        for index in range(len(cuts) - 1)
    ]


def prepare_analysis(scan: LoadedScan) -> AnalysisData:
    frame = scan.dataframe
    time = frame["Time"].to_numpy(float)
    potential = frame["E"].to_numpy(float)
    current_a = frame["I"].to_numpy(float)
    scaler, prefix, report_prefix = current_scale(current_a)
    current = current_a * scaler
    transform = fractional_differintegral(time, current, 0.5)
    switches = detect_switching_points(potential)
    rate = scan.scan_rate_v_s
    if not rate or not np.isfinite(rate) or rate <= 0:
        step_e = np.abs(np.diff(potential))
        step_t = np.diff(time)
        moving = step_e > np.finfo(float).eps
        rate = float(np.median(step_e[moving] / step_t[moving]))
    return AnalysisData(
        scan=scan,
        time=time,
        potential=potential,
        current_a=current_a,
        current=current,
        transform=transform,
        scaler=scaler,
        scale_prefix=prefix,
        report_scale_prefix=report_prefix,
        scan_rate_v_s=float(rate),
        switching_indices=switches,
        segment_bounds=segment_bounds(len(time), switches),
        excluded_intervals=[],
        excluded_mask=np.zeros(len(time), dtype=bool),
        fit_weights=np.ones(len(time), dtype=float),
    )


FILTER_LABELS = {
    "none": "None",
    "modified_sinc": "Modified sinc",
    "savgol": "Savitzky-Golay",
    "lowpass": "Butterworth low-pass (zero phase)",
    "notch": "IIR notch (zero phase)",
    "fft_bandstop": "FFT band-stop",
    "hampel": "Hampel filter",
    "upper_envelope": "Current-maxima Makima envelope",
}


def _odd_window_from_mv(
    potential: np.ndarray,
    requested_mv: float,
    minimum: int,
) -> int:
    steps = np.abs(np.diff(np.asarray(potential, dtype=float)))
    steps = steps[np.isfinite(steps) & (steps > np.finfo(float).eps)]
    if not len(steps):
        return 0
    points = max(minimum, int(round((requested_mv / 1000.0) / float(np.median(steps)))))
    if points % 2 == 0:
        points += 1
    maximum = len(potential) if len(potential) % 2 else len(potential) - 1
    return min(points, maximum)


def _safe_sosfiltfilt(sos: np.ndarray, values: np.ndarray) -> np.ndarray:
    if len(values) < 5:
        return values.copy()
    padlen = min(len(values) - 1, max(3, 3 * (2 * len(sos) + 1)))
    return np.asarray(sosfiltfilt(sos, values, padlen=padlen), dtype=float)


def _safe_filtfilt(b: np.ndarray, a: np.ndarray, values: np.ndarray) -> np.ndarray:
    if len(values) < 5:
        return values.copy()
    padlen = min(len(values) - 1, max(3, 3 * max(len(a), len(b))))
    return np.asarray(filtfilt(b, a, values, padlen=padlen), dtype=float)


def _hampel_filter(values: np.ndarray, window: int, sigma: float) -> tuple[np.ndarray, int]:
    output = np.asarray(values, dtype=float).copy()
    replaced = 0
    half = max(1, window // 2)
    for index, value in enumerate(values):
        local = values[max(0, index - half) : min(len(values), index + half + 1)]
        median = float(np.median(local))
        mad = float(np.median(np.abs(local - median)))
        robust_sigma = 1.4826 * mad
        if robust_sigma > np.finfo(float).eps and abs(float(value) - median) > sigma * robust_sigma:
            output[index] = median
            replaced += 1
    return output, replaced


_MODIFIED_SINC_CORRECTIONS = {
    2: (),
    4: (),
    6: ((0.001717576, 0.02437382, 1.64375),),
    8: (
        (0.0043993373, 0.088211164, 2.359375),
        (0.006146815, 0.024715371, 3.6359375),
    ),
    10: (
        (0.0011840032, 0.04219344, 2.746875),
        (0.0036718843, 0.12780383, 2.7703125),
    ),
}


def _modified_sinc_kernel(degree: int, half_width: int) -> np.ndarray:
    """Return the standard modified-sinc smoothing kernel.

    This is a Python port of the reference algorithm accompanying Schmid,
    Rath, and Diebold, *Why and How Savitzky-Golay Filters Should Be
    Replaced*, ACS Measurement Science Au 2 (2022) 185-196.
    """
    if degree not in _MODIFIED_SINC_CORRECTIONS:
        raise ValueError("Modified sinc degree must be 2, 4, 6, 8, or 10")
    if half_width < degree // 2 + 2:
        raise ValueError(
            f"Modified sinc degree {degree} needs at least "
            f"{2 * (degree // 2 + 2) + 1} points"
        )
    x = np.arange(half_width + 1, dtype=float) / float(half_width + 1)
    sinc_argument = np.pi * 0.5 * (degree + 4) * x
    side = np.ones(half_width + 1, dtype=float)
    side[1:] = np.sin(sinc_argument[1:]) / sinc_argument[1:]
    start_harmonic = 2 if (degree // 2) % 2 == 0 else 1
    for number, (a, b, c) in enumerate(_MODIFIED_SINC_CORRECTIONS[degree]):
        kappa = a + b / (c - half_width) ** 3
        side += kappa * x * np.sin((2 * number + start_harmonic) * np.pi * x)
    alpha = 4.0
    window = (
        np.exp(-alpha * x**2)
        + np.exp(-alpha * (x - 2.0) ** 2)
        + np.exp(-alpha * (x + 2.0) ** 2)
        - 2.0 * np.exp(-alpha)
        - np.exp(-9.0 * alpha)
    )
    side *= window
    kernel = np.concatenate((side[:0:-1], side))
    return kernel / np.sum(kernel)


def _weighted_line(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    sum_weights = float(np.sum(weights))
    sum_x = float(np.sum(x * weights))
    sum_y = float(np.sum(y * weights))
    denominator = float(np.sum(x * x * weights)) * sum_weights - sum_x * sum_x
    slope = 0.0 if abs(denominator) <= np.finfo(float).eps else (
        float(np.sum(x * y * weights)) * sum_weights - sum_x * sum_y
    ) / denominator
    return (sum_y - slope * sum_x) / sum_weights, slope


def _modified_sinc_filter(values: np.ndarray, degree: int, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Smooth with a modified-sinc kernel and weighted linear edge extension."""
    half_width = window // 2
    kernel = _modified_sinc_kernel(degree, half_width)
    first_zero = (half_width + 1) / (1.5 + 0.5 * degree)
    beta = 0.70 + 0.14 * np.exp(-0.60 * (degree - 4))
    fit_length = min(len(values), max(1, int(np.ceil(first_zero * beta))))
    x = np.arange(fit_length, dtype=float)
    weights = np.cos(0.5 * np.pi / (first_zero * beta) * x) ** 2
    extended = np.empty(len(values) + 2 * half_width, dtype=float)
    extended[half_width : half_width + len(values)] = values
    offset, slope = _weighted_line(x, np.asarray(values[:fit_length], dtype=float), weights)
    positions = np.arange(-half_width, 0, dtype=float)
    extended[:half_width] = offset + slope * positions
    offset, slope = _weighted_line(
        x,
        np.asarray(values[-fit_length:][::-1], dtype=float),
        weights,
    )
    extended[half_width + len(values) :] = offset + slope * positions[::-1]
    convolved = np.convolve(extended, kernel, mode="same")
    return np.asarray(convolved[half_width : half_width + len(values)], dtype=float), kernel


def _local_maxima_upper_envelope(
    data: AnalysisData,
    current: np.ndarray,
    config: FitConfig,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], list[int]]:
    """Build regional Makima current envelopes supported only by maxima.

    Maxima are detected on a locally detrended current trace. Their original
    current values are the only interpolation anchors; the resulting Makima
    envelope is evaluated at every regional sample before semi-differentiation.
    """
    if config.filter_scope != "regional" or config.filter_range is None:
        raise ValueError("Current-maxima Makima envelope requires a selected regional range")
    first, second = sorted(map(int, config.filter_range))
    first = max(0, first)
    second = min(len(data.time) - 1, second)
    frequency = float(config.filter_frequency_hz)
    prominence_pct = float(config.filter_envelope_prominence_pct)
    if prominence_pct < 0 or prominence_pct > 100:
        raise ValueError("Envelope prominence must be between 0 and 100%")

    envelope = np.asarray(current, dtype=float).copy()
    original_weights = np.asarray(data.fit_weights, dtype=float)
    fit_weights = original_weights.copy()
    segment_details: list[dict[str, Any]] = []
    maxima_indices: list[int] = []
    for segment_start, segment_end in data.segment_bounds:
        start = max(first, segment_start)
        end = min(second, segment_end)
        if end - start < 4:
            continue
        local_time = np.asarray(data.time[start : end + 1], dtype=float)
        local_current = np.asarray(current[start : end + 1], dtype=float)
        dt = float(np.median(np.diff(local_time)))
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Envelope detection requires strictly increasing acquisition time")
        sample_rate = 1.0 / dt
        nyquist = sample_rate * 0.5
        if frequency <= 0 or frequency >= nyquist:
            raise ValueError(
                f"Envelope frequency must be between 0 and the Nyquist frequency ({nyquist:.4g} Hz)"
            )
        period_points = sample_rate / frequency
        minimum_distance = max(1, int(round(0.60 * period_points)))
        trend_window = max(5, int(round(3.0 * period_points)))
        if trend_window % 2 == 0:
            trend_window += 1
        maximum_window = len(local_current) if len(local_current) % 2 else len(local_current) - 1
        trend_window = min(trend_window, maximum_window)
        if trend_window >= 5:
            trend = savgol_filter(local_current, trend_window, 2, mode="interp")
        else:
            coordinate = np.arange(len(local_current), dtype=float)
            trend = np.polyval(np.polyfit(coordinate, local_current, 1), coordinate)
        oscillation = local_current - trend
        oscillation_span = float(np.percentile(oscillation, 95) - np.percentile(oscillation, 5))
        prominence = max(0.0, prominence_pct / 100.0 * oscillation_span)
        peaks, properties = find_peaks(
            oscillation,
            distance=minimum_distance,
            prominence=prominence,
        )
        if len(peaks) < 3:
            segment_details.append(
                {
                    "start_index": start,
                    "end_index": end,
                    "maxima_count": int(len(peaks)),
                    "frequency_hz": frequency,
                    "minimum_distance_points": minimum_distance,
                    "prominence": prominence,
                    "processed": False,
                }
            )
            continue
        maxima = np.asarray(peaks + start, dtype=int)
        interpolator = Akima1DInterpolator(
            data.time[maxima],
            current[maxima],
            method="makima",
            extrapolate=False,
        )
        interpolated = interpolator(local_time)
        envelope[start : end + 1] = _linear_tails(
            local_time,
            data.time[maxima],
            current[maxima],
            interpolated,
        )
        maxima_indices.extend(maxima.tolist())
        segment_details.append(
            {
                "start_index": start,
                "end_index": end,
                "maxima_count": int(len(maxima)),
                "maxima_indices": maxima.tolist(),
                "frequency_hz": frequency,
                "minimum_distance_points": minimum_distance,
                "prominence": prominence,
                "median_prominence": float(np.median(properties["prominences"])),
                "processed": True,
            }
        )
    if len(maxima_indices) < 3:
        raise ValueError(
            "Fewer than three oscillation maxima were found in the selected range. "
            "Check the frequency, lower the prominence, or select a longer range."
        )
    return envelope, fit_weights, segment_details, maxima_indices


def _fft_bandstop(
    values: np.ndarray,
    sample_rate_hz: float,
    center_hz: float,
    bandwidth_hz: float,
    attenuation_db: float,
    harmonics: int,
) -> np.ndarray:
    """Apply a reflect-padded FFT band-stop with cosine-tapered edges."""
    count = len(values)
    if count < 8:
        return values.copy()
    pad = min(count - 1, max(8, count // 4))
    padded = np.pad(np.asarray(values, dtype=float), pad, mode="reflect")
    coordinate = np.linspace(-1.0, 1.0, len(padded))
    trend = np.polyval(np.polyfit(coordinate, padded, 1), coordinate)
    spectrum = np.fft.rfft(padded - trend)
    frequencies = np.fft.rfftfreq(len(padded), d=1.0 / sample_rate_hz)
    gain = np.ones_like(frequencies)
    floor = 10.0 ** (-attenuation_db / 20.0)
    half_width = max(bandwidth_hz * 0.5, sample_rate_hz / len(padded))
    transition = half_width
    for harmonic in range(1, harmonics + 1):
        center = center_hz * harmonic
        if center >= sample_rate_hz * 0.5:
            break
        distance = np.abs(frequencies - center)
        local_gain = np.ones_like(frequencies)
        local_gain[distance <= half_width] = floor
        shoulder = (distance > half_width) & (distance < half_width + transition)
        phase = (distance[shoulder] - half_width) / transition
        local_gain[shoulder] = floor + (1.0 - floor) * 0.5 * (1.0 - np.cos(np.pi * phase))
        gain *= local_gain
    filtered = np.fft.irfft(spectrum * gain, n=len(padded)) + trend
    return np.asarray(filtered[pad : pad + count], dtype=float)


def _filter_segment(
    time: np.ndarray,
    potential: np.ndarray,
    values: np.ndarray,
    config: FitConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    kind = config.filter_kind
    dt = float(np.median(np.diff(time)))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Filtering requires strictly increasing acquisition time")
    sample_rate = 1.0 / dt
    nyquist = sample_rate * 0.5
    details: dict[str, Any] = {"sample_rate_hz": sample_rate, "nyquist_hz": nyquist}

    if kind == "modified_sinc":
        degree = int(config.filter_polynomial_order)
        minimum_window = 2 * (degree // 2 + 2) + 1
        window = _odd_window_from_mv(potential, config.filter_window_mv, minimum_window)
        if window < minimum_window:
            raise ValueError(
                f"This sweep is too short for a degree-{degree} modified sinc filter"
            )
        output, coefficients = _modified_sinc_filter(values, degree, window)
        frequencies, response = freqz(coefficients, worN=4096, fs=sample_rate)
        magnitude_db = 20.0 * np.log10(np.maximum(np.abs(response), np.finfo(float).tiny))
        crossing = np.flatnonzero(magnitude_db <= -3.0)
        noise_gain = float(np.sqrt(np.sum(coefficients**2)))
        details.update(
            degree=degree,
            window_points=window,
            effective_window_mv=float(config.filter_window_mv),
            cutoff_3db_hz=float(frequencies[crossing[0]]) if len(crossing) else np.nan,
            white_noise_attenuation_db=float(-20.0 * np.log10(noise_gain)),
        )
        return output, details

    if kind == "savgol":
        polynomial = int(config.filter_polynomial_order)
        window = _odd_window_from_mv(potential, config.filter_window_mv, polynomial + 2)
        if window <= polynomial or window < 3:
            raise ValueError("The smoothing window is too short for this sweep")
        output = np.asarray(savgol_filter(values, window, polynomial, mode="interp"), dtype=float)
        coefficients = savgol_coeffs(window, polynomial)
        frequencies, response = freqz(coefficients, worN=4096, fs=sample_rate)
        magnitude_db = 20.0 * np.log10(np.maximum(np.abs(response), np.finfo(float).tiny))
        crossing = np.flatnonzero(magnitude_db <= -3.0)
        details.update(
            window_points=window,
            effective_window_mv=float(config.filter_window_mv),
            cutoff_3db_hz=float(frequencies[crossing[0]]) if len(crossing) else np.nan,
            white_noise_attenuation_db=float(
                -20.0 * np.log10(np.sqrt(np.sum(coefficients**2)))
            ),
        )
        return output, details

    if kind == "hampel":
        window = _odd_window_from_mv(potential, config.filter_window_mv, 3)
        if window < 3:
            raise ValueError("The Hampel window is too short for this sweep")
        output, replaced = _hampel_filter(values, window, float(config.filter_hampel_sigma))
        details.update(window_points=window, replaced_samples=replaced)
        return output, details

    frequency = float(config.filter_frequency_hz)
    bandwidth = float(config.filter_bandwidth_hz)
    attenuation = float(config.filter_attenuation_db)
    if frequency <= 0 or frequency >= nyquist:
        raise ValueError(f"Filter frequency must be between 0 and the Nyquist frequency ({nyquist:.4g} Hz)")
    if bandwidth <= 0:
        raise ValueError("Filter bandwidth/transition width must be positive")

    if kind == "lowpass":
        stop = frequency + bandwidth
        if stop >= nyquist:
            raise ValueError(f"Low-pass stopband edge must be below {nyquist:.4g} Hz")
        order, natural = buttord(
            frequency,
            stop,
            0.5,
            max(1.0, attenuation * 0.5),
            fs=sample_rate,
        )
        requested_order = int(order)
        order = min(requested_order, 12)
        sos = butter(order, natural, btype="lowpass", fs=sample_rate, output="sos")
        output = _safe_sosfiltfilt(sos, values)
        _, response = sosfreqz(sos, worN=np.asarray([stop]), fs=sample_rate)
        applied_db = -20.0 * np.log10(max(float(np.abs(response[0]) ** 2), np.finfo(float).tiny))
        details.update(
            order=order,
            requested_order=requested_order,
            stopband_edge_hz=stop,
            attenuation_at_stopband_db=applied_db,
            order_limited=requested_order > order,
        )
        return output, details

    if kind == "notch":
        output = np.asarray(values, dtype=float).copy()
        line_gain = 10.0 ** (-attenuation / 20.0)
        applied_harmonics = 0
        for harmonic in range(1, int(config.filter_harmonics) + 1):
            center = frequency * harmonic
            if center >= nyquist:
                break
            quality = max(center / bandwidth, 0.5001)
            b, a = iirnotch(center, quality, fs=sample_rate)
            fully_notched = _safe_filtfilt(b, a, output)
            output = line_gain * output + (1.0 - line_gain) * fully_notched
            applied_harmonics += 1
        details.update(
            center_hz=frequency,
            bandwidth_hz=bandwidth,
            attenuation_at_center_db=attenuation,
            harmonics_applied=applied_harmonics,
        )
        return output, details

    if kind == "fft_bandstop":
        output = _fft_bandstop(
            values,
            sample_rate,
            frequency,
            bandwidth,
            attenuation,
            int(config.filter_harmonics),
        )
        details.update(
            center_hz=frequency,
            bandwidth_hz=bandwidth,
            attenuation_at_center_db=attenuation,
            harmonics_applied=sum(
                frequency * harmonic < nyquist
                for harmonic in range(1, int(config.filter_harmonics) + 1)
            ),
        )
        return output, details

    raise ValueError(f"Unknown signal filter: {kind}")


def _regional_weight(data: AnalysisData, config: FitConfig) -> np.ndarray:
    if config.filter_scope == "global":
        return np.ones(len(data.time), dtype=float)
    if config.filter_scope != "regional":
        raise ValueError(f"Unknown filter scope: {config.filter_scope}")
    if config.filter_range is None:
        return np.zeros(len(data.time), dtype=float)
    first, second = sorted(map(int, config.filter_range))
    first = max(0, first)
    second = min(len(data.time) - 1, second)
    weight = np.zeros(len(data.time), dtype=float)
    feather_v = max(0.0, float(config.filter_feather_mv) / 1000.0)
    for segment_start, segment_end in data.segment_bounds:
        start = max(first, segment_start)
        end = min(second, segment_end)
        if start > end:
            continue
        weight[start : end + 1] = 1.0
        if feather_v <= 0 or start == end:
            continue
        local_potential = data.potential[start : end + 1]
        left_distance = np.abs(local_potential - local_potential[0])
        right_distance = np.abs(local_potential - local_potential[-1])
        distance = np.minimum(left_distance, right_distance)
        ramp = np.sin(0.5 * np.pi * np.clip(distance / feather_v, 0.0, 1.0)) ** 2
        weight[start : end + 1] = ramp
    return weight


def _apply_signal_filter_stage(data: AnalysisData, config: FitConfig) -> AnalysisData:
    """Return a filtered analysis without mutating the imported measurement.

    Every chronological sweep is filtered independently. Regional filtering is
    produced from the full-sweep filtered result and softly blended into the
    selected interval, avoiding hard FFT/filter boundaries.
    """
    if config.filter_kind == "none":
        return replace(data, filter_diagnostics={"summary": "Filtering off"})
    if config.filter_kind not in FILTER_LABELS:
        raise ValueError(f"Unknown signal filter: {config.filter_kind}")
    if config.filter_domain not in {"current", "semiderivative"}:
        raise ValueError(f"Unknown filter domain: {config.filter_domain}")

    raw_current = np.asarray(data.current, dtype=float)
    raw_transform = np.asarray(data.transform, dtype=float)
    source = raw_current if config.filter_domain == "current" else raw_transform
    fit_weights = np.asarray(data.fit_weights, dtype=float).copy()
    maxima_indices: list[int] = []
    if config.filter_kind == "upper_envelope":
        if config.filter_domain != "current":
            raise ValueError("Current-maxima Makima envelope is defined only in the base-current domain")
        complete_filtered, fit_weights, segment_details, maxima_indices = (
            _local_maxima_upper_envelope(data, raw_current, config)
        )
    else:
        complete_filtered = source.copy()
        segment_details = []
        for start, end in data.segment_bounds:
            filtered, details = _filter_segment(
                data.time[start : end + 1],
                data.potential[start : end + 1],
                source[start : end + 1],
                config,
            )
            complete_filtered[start : end + 1] = filtered
            details.update(start_index=start, end_index=end)
            segment_details.append(details)

    weight = _regional_weight(data, config)
    working_signal = source + weight * (complete_filtered - source)
    if config.filter_domain == "current":
        filtered_current = working_signal
        filtered_transform = fractional_differintegral(data.time, filtered_current, 0.5)
    else:
        filtered_transform = working_signal
        raw_roundtrip = fractional_differintegral(data.time, raw_transform, -0.5)
        filtered_current = (
            fractional_differintegral(data.time, filtered_transform, -0.5)
            + raw_current
            - raw_roundtrip
        )

    difference = working_signal - source
    rms_removed = float(np.sqrt(np.mean(difference**2)))
    domain_label = "base current" if config.filter_domain == "current" else "semi-derivative"
    scope_label = "each sweep" if config.filter_scope == "global" else "selected range"
    first_details = segment_details[0] if segment_details else {}
    if config.filter_kind == "upper_envelope":
        response = (
            f"{len(maxima_indices)} current maxima used as Makima knots; "
            f"expected oscillation frequency {config.filter_frequency_hz:.4g} Hz; "
            "envelope semi-differentiated for fitting"
        )
    elif config.filter_kind in {"modified_sinc", "savgol"}:
        response = (
            f"{first_details.get('window_points', 0)}-point window; "
            f"~−3 dB at {first_details.get('cutoff_3db_hz', np.nan):.4g} Hz; "
            f"white-noise attenuation "
            f"{first_details.get('white_noise_attenuation_db', np.nan):.3g} dB"
        )
    elif config.filter_kind == "lowpass":
        response = f"−{first_details.get('attenuation_at_stopband_db', np.nan):.3g} dB at the stopband edge"
    elif config.filter_kind in {"notch", "fft_bandstop"}:
        response = f"−{config.filter_attenuation_db:.3g} dB at {config.filter_frequency_hz:.4g} Hz"
    else:
        replaced = sum(int(item.get("replaced_samples", 0)) for item in segment_details)
        response = f"{replaced} sample(s) replaced"
    summary = f"{FILTER_LABELS[config.filter_kind]} on {domain_label}, {scope_label}; {response}"
    diagnostics = {
        "summary": summary,
        "kind": config.filter_kind,
        "domain": config.filter_domain,
        "scope": config.filter_scope,
        "range": config.filter_range,
        "rms_removed": rms_removed,
        "segment_details": segment_details,
        "blend_weight": weight,
        "maxima_indices": maxima_indices,
        "fit_weight_count": int(np.count_nonzero(fit_weights)),
    }
    return replace(
        data,
        current=np.asarray(filtered_current, dtype=float),
        transform=np.asarray(filtered_transform, dtype=float),
        fit_weights=np.asarray(fit_weights, dtype=float),
        filter_diagnostics=diagnostics,
    )


def _stage_config(config: FitConfig, stage: FilterStage) -> FitConfig:
    return replace(
        config,
        filter_kind=stage.kind,
        filter_domain=stage.domain,
        filter_scope=stage.scope,
        filter_range=stage.range,
        filter_frequency_hz=stage.frequency_hz,
        filter_bandwidth_hz=stage.bandwidth_hz,
        filter_attenuation_db=stage.attenuation_db,
        filter_window_mv=stage.window_mv,
        filter_polynomial_order=stage.polynomial_order,
        filter_hampel_sigma=stage.hampel_sigma,
        filter_harmonics=stage.harmonics,
        filter_feather_mv=stage.feather_mv,
        filter_envelope_prominence_pct=stage.envelope_prominence_pct,
        filter_secondary_kind="none",
        filter_stages=(),
    )


def _combine_filter_stages(filtered: AnalysisData, stages: list[dict[str, Any]]) -> AnalysisData:
    if len(stages) == 1:
        return filtered
    maxima = [
        int(index)
        for details in stages
        for index in details.get("maxima_indices", [])
    ]
    diagnostics = {
        "summary": " → ".join(
            f"{number}. {details.get('summary', '')}"
            for number, details in enumerate(stages, start=1)
        ),
        "kind": "chain",
        "domain": "per-stage",
        "scope": "per-stage",
        "range": None,
        "stages": stages,
        "rms_removed": [details.get("rms_removed", np.nan) for details in stages],
        "segment_details": [details.get("segment_details", []) for details in stages],
        "blend_weight": [details.get("blend_weight") for details in stages],
        "maxima_indices": sorted(set(maxima)),
        "fit_weight_count": int(np.count_nonzero(filtered.fit_weights)),
    }
    return replace(filtered, filter_diagnostics=diagnostics)


def apply_signal_filter(data: AnalysisData, config: FitConfig) -> AnalysisData:
    """Apply any configured signal-processing stages in their displayed order."""
    if config.filter_stages:
        filtered = data
        diagnostics: list[dict[str, Any]] = []
        for stage in config.filter_stages:
            if stage.kind == "none":
                continue
            filtered = _apply_signal_filter_stage(filtered, _stage_config(config, stage))
            diagnostics.append(dict(filtered.filter_diagnostics))
        if not diagnostics:
            return replace(data, filter_diagnostics={"summary": "Filtering off"})
        return _combine_filter_stages(filtered, diagnostics)

    # Legacy configuration support for releases through 1.6.
    secondary_kind = str(config.filter_secondary_kind)
    if secondary_kind == "none":
        return _apply_signal_filter_stage(data, config)
    if secondary_kind != "upper_envelope":
        raise ValueError(f"Unknown secondary signal filter: {secondary_kind}")
    if config.filter_kind == "upper_envelope":
        raise ValueError("The current-maxima Makima envelope cannot be applied twice")

    envelope_config = replace(
        config,
        filter_kind="upper_envelope",
        filter_domain="current",
        filter_scope="regional",
        filter_range=config.filter_secondary_range,
        filter_frequency_hz=config.filter_secondary_frequency_hz,
        filter_feather_mv=config.filter_secondary_feather_mv,
        filter_envelope_prominence_pct=config.filter_secondary_envelope_prominence_pct,
        filter_secondary_kind="none",
    )
    filtered = _apply_signal_filter_stage(data, envelope_config)
    envelope_diagnostics = dict(filtered.filter_diagnostics)
    stage_diagnostics = [envelope_diagnostics]
    if config.filter_kind != "none":
        filtered = _apply_signal_filter_stage(filtered, config)
        primary_diagnostics = dict(filtered.filter_diagnostics)
        stage_diagnostics.append(primary_diagnostics)
    return _combine_filter_stages(filtered, stage_diagnostics)


def suggested_knot_indices(data: AnalysisData) -> list[int]:
    """Return no knots: every background anchor must be selected deliberately."""
    return []


def _segment_for_index(data: AnalysisData, index: int) -> tuple[int, int]:
    for start, end in data.segment_bounds:
        if start <= index <= end:
            return start, end
    return 0, len(data.time) - 1


def _segment_number_for_index(data: AnalysisData, index: int) -> int:
    for number, (start, end) in enumerate(data.segment_bounds, start=1):
        if start <= index <= end:
            return number
    return 0


def background_source_labels(
    data: AnalysisData,
    knot_indices: Sequence[int],
    background_kind: str = "makima",
    background_scope: str = "independent",
) -> np.ndarray:
    knots = validate_background_selection(
        data, knot_indices, background_kind, background_scope
    )
    labels = np.full(len(data.time), "", dtype=object)
    if background_kind in FITTED_BACKGROUNDS:
        if background_scope == "continuous":
            labels[:] = f"continuous anchor-fitted {background_kind} background"
        else:
            for number, (start, end) in enumerate(data.segment_bounds, start=1):
                labels[start : end + 1] = (
                    f"segment {number} anchor-fitted {background_kind} background"
                )
        return labels
    if background_scope == "continuous":
        first, last = min(knots), max(knots)
        outside = "polynomial extrapolation" if background_kind == "cubic" else "linear extrapolation"
        labels[:] = f"continuous {outside}"
        labels[first : last + 1] = f"continuous {background_kind} background"
        return labels
    for number, (start, end) in enumerate(data.segment_bounds, start=1):
        segment_knots = [index for index in knots if start <= index <= end]
        first, last = min(segment_knots), max(segment_knots)
        outside = (
            "polynomial extrapolation"
            if background_kind == "cubic"
            else "linear extrapolation"
        )
        labels[start : end + 1] = f"segment {number} {outside}"
        labels[first : last + 1] = f"segment {number} {background_kind} background"
    return labels


def _sanitize_selections(data: AnalysisData, indices: Sequence[int], minimum_gap: int = 2) -> list[int]:
    selected = sorted({int(np.clip(index, 0, len(data.time) - 1)) for index in indices})
    output: list[int] = []
    for index in selected:
        if not output or index - output[-1] >= minimum_gap:
            output.append(index)
    return output


def validate_knots(
    data: AnalysisData,
    knot_indices: Sequence[int],
    minimum_per_segment: int = 4,
) -> list[int]:
    knots = _sanitize_selections(data, knot_indices)
    excluded_knots = [index for index in knots if data.excluded_mask[index]]
    if excluded_knots:
        raise ValueError(
            "Background knots cannot lie inside excluded transient intervals: "
            + ", ".join(map(str, excluded_knots))
        )
    missing = []
    for number, (start, end) in enumerate(data.segment_bounds, start=1):
        count = sum(start <= index <= end for index in knots)
        if count < minimum_per_segment:
            missing.append(f"{number} ({count}/{minimum_per_segment})")
    if missing:
        raise ValueError(
            f"Each independent background needs {minimum_per_segment} manually selected knots. "
            "Add knots to segment(s): " + ", ".join(missing)
        )
    return knots


def validate_background_knots(
    data: AnalysisData,
    knot_indices: Sequence[int],
    minimum: int = 1,
    background_scope: str = "independent",
) -> list[int]:
    """Validate either sweep-local anchors or one scan-wide anchor set."""
    if background_scope == "independent":
        return validate_knots(data, knot_indices, minimum)
    if background_scope != "continuous":
        raise ValueError("Background scope must be 'independent' or 'continuous'")
    knots = _sanitize_selections(data, knot_indices)
    excluded_knots = [index for index in knots if data.excluded_mask[index]]
    if excluded_knots:
        raise ValueError(
            "Background knots cannot lie inside excluded transient intervals: "
            + ", ".join(map(str, excluded_knots))
        )
    if len(knots) < minimum:
        raise ValueError(f"The continuous background needs at least {minimum} manually selected knot(s)")
    return knots


def _segment_polynomial(
    data: AnalysisData,
    indices: Sequence[int],
    start: int,
    end: int,
    knot_values: Mapping[int, float] | None = None,
) -> tuple[np.ndarray, float, float, np.ndarray]:
    selected = np.asarray(indices, dtype=int)
    center = float(np.mean(data.time[selected]))
    scale = max(float(np.ptp(data.time[selected])), np.finfo(float).eps)
    normalized_knots = (data.time[selected] - center) / scale
    anchors = np.asarray(
        [
            (knot_values or {}).get(int(index), data.transform[int(index)])
            for index in selected
        ],
        dtype=float,
    )
    coefficients = np.polyfit(normalized_knots, anchors, min(3, len(selected) - 1))
    segment_x = (data.time[start : end + 1] - center) / scale
    return coefficients, center, scale, np.polyval(coefficients, segment_x)


def polynomial_reference(
    data: AnalysisData,
    knot_indices: Sequence[int],
    require_complete: bool = True,
    knot_values: Mapping[int, float] | None = None,
    minimum_per_segment: int = 1,
    background_scope: str = "independent",
) -> np.ndarray:
    """Evaluate each segment polynomial, including during partial selection.

    A provisional constant reference appears from the first anchor, then gains
    polynomial order as more anchors are selected.
    """
    knots = (
        validate_background_knots(data, knot_indices, minimum_per_segment, background_scope)
        if require_complete
        else _sanitize_selections(data, knot_indices)
    )
    reference = np.full(len(data.time), np.nan, dtype=float)
    if background_scope == "continuous":
        if knots:
            _, _, _, values = _segment_polynomial(
                data, knots, 0, len(data.time) - 1, knot_values
            )
            reference[:] = values
        return reference
    for start, end in data.segment_bounds:
        segment_knots = [index for index in knots if start <= index <= end]
        if not segment_knots:
            continue
        _, _, _, values = _segment_polynomial(
            data, segment_knots, start, end, knot_values
        )
        reference[start : end + 1] = values
    return reference


def fit_report_without_r_squared(result: Any) -> str:
    """Return lmfit's report without its nonlinear-model R-squared line."""
    return "\n".join(
        line for line in result.fit_report().splitlines()
        if "R-squared" not in line
    )


def _pearson_amplitude(height: float, sigma: float, exponent: float, skew: float = 0.0) -> float:
    denominator = np.exp(
        2 * (np.real(loggamma(exponent + skew * 0.5j)) - loggamma(exponent))
        - betaln(exponent - 0.5, 0.5)
        - exponent * np.log1p(np.square(skew / (2 * exponent)))
        - skew * np.arctan(-skew / (2 * exponent))
    )
    return float(sigma * height / denominator)


def _lag1(values: Sequence[float]) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) < 4 or np.std(values) <= np.finfo(float).eps:
        return float("nan")
    return float(np.corrcoef(values[:-1], values[1:])[0, 1])


INTERPOLATING_BACKGROUNDS = {"makima", "pchip", "cubic", "linear"}
FITTED_BACKGROUNDS = {"cpe", "constant"}
BACKGROUND_KINDS = INTERPOLATING_BACKGROUNDS | FITTED_BACKGROUNDS


def minimum_background_anchors(background_kind: str) -> int:
    """Return the deliberate-anchor minimum for one fitted scope segment."""
    return 2 if background_kind == "cpe" else 1


def validate_background_selection(
    data: AnalysisData,
    knot_indices: Sequence[int],
    background_kind: str,
    background_scope: str,
) -> list[int]:
    """Validate model-specific anchor requirements.

    Every CPE sweep owns Q and alpha, so even a joint continuous fit needs at
    least two deliberate anchors on every sweep. Other continuous models use
    one scan-wide anchor collection.
    """
    if background_kind == "cpe":
        return validate_knots(data, knot_indices, 2)
    return validate_background_knots(
        data,
        knot_indices,
        minimum_background_anchors(background_kind),
        background_scope,
    )


def background_treatment_description(background_kind: str, background_scope: str) -> str:
    scope = "scan-wide" if background_scope == "continuous" else "segment-specific"
    if background_kind == "cubic":
        return f"{scope} cubic interpolation with shifted polynomial tails"
    if background_kind in {"makima", "pchip", "linear"}:
        return f"{scope} {background_kind} interpolation with linear secant tails"
    if background_kind == "constant":
        return f"{scope} constant level fitted to manual knots"
    if background_kind == "cpe":
        coupling = "joint" if background_scope == "continuous" else "chronological sequential"
        return f"{scope} piecewise CPE fitted to manual knots ({coupling} fit)"
    return f"{scope} {background_kind} background"


def _linear_tails(
    segment_time: np.ndarray,
    knot_times: np.ndarray,
    knot_heights: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    """Use transparent secant-line extrapolation outside manual anchors."""
    values = np.asarray(values, dtype=float).copy()
    if len(knot_times) < 2:
        values[:] = knot_heights[0]
        return values
    left = segment_time < knot_times[0]
    right = segment_time > knot_times[-1]
    left_slope = (knot_heights[1] - knot_heights[0]) / (
        knot_times[1] - knot_times[0]
    )
    right_slope = (knot_heights[-1] - knot_heights[-2]) / (
        knot_times[-1] - knot_times[-2]
    )
    values[left] = knot_heights[0] + left_slope * (
        segment_time[left] - knot_times[0]
    )
    values[right] = knot_heights[-1] + right_slope * (
        segment_time[right] - knot_times[-1]
    )
    return values


def _cpe_current(
    data: AnalysisData,
    q_values: Sequence[float],
    alpha_values: Sequence[float],
) -> np.ndarray:
    """Evaluate the chronological piecewise-CPE current used by SD-Fitter."""
    time = np.asarray(data.time, dtype=float)
    q_values = np.asarray(q_values, dtype=float)
    alpha_values = np.asarray(alpha_values, dtype=float)
    if len(q_values) != len(data.segment_bounds) or len(alpha_values) != len(data.segment_bounds):
        raise ValueError("CPE requires one Q and alpha pair per chronological sweep")
    current = np.zeros_like(time)
    segment_number = np.zeros(len(time), dtype=int)
    for number, (start, end) in enumerate(data.segment_bounds):
        segment_number[start : end + 1] = number
    q_t = q_values[segment_number]
    alpha_t = alpha_values[segment_number]
    rate = abs(float(data.scan_rate_v_s))
    for start, end in data.segment_bounds:
        direction = np.sign(data.potential[end] - data.potential[start])
        if direction == 0:
            direction = 1.0
        start_time = float(time[start])
        end_time = float(time[end])
        with np.errstate(invalid="ignore", divide="ignore"):
            onset = np.where(time >= start_time, (time - start_time) ** (1.0 - alpha_t), 0.0)
            offset = np.where(time >= end_time, (time - end_time) ** (1.0 - alpha_t), 0.0)
        current += direction * (onset - offset)
    return current * q_t * rate / gamma(2.0 - alpha_t)


def _cpe_sd_background(
    data: AnalysisData,
    q_values: Sequence[float],
    alpha_values: Sequence[float],
) -> np.ndarray:
    return fractional_differintegral(
        data.time, _cpe_current(data, q_values, alpha_values), 0.5
    )


def _fit_cpe_background(
    data: AnalysisData,
    knots: Sequence[int],
    anchors: Mapping[int, float],
    background_scope: str,
) -> BackgroundConstruction:
    """Fit CPE parameters only to deliberate background anchors.

    Independent scope is chronological: each sweep is fitted to its own
    anchors while earlier Q/alpha values remain frozen. Continuous scope fits
    every sweep parameter jointly to every selected anchor.
    """
    count = len(data.segment_bounds)
    q_guess = max(
        float(np.median(np.abs(data.current[np.asarray(knots, dtype=int)])))
        / max(abs(float(data.scan_rate_v_s)), np.finfo(float).eps),
        np.finfo(float).eps,
    )
    q_values = np.full(count, q_guess, dtype=float)
    alpha_values = np.full(count, 0.9, dtype=float)
    lower = np.tile([0.0, 0.6], count)
    upper = np.tile([np.inf, 1.0], count)
    initial = np.ravel(np.column_stack([q_values, alpha_values]))

    def unpack(vector: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
        matrix = np.asarray(vector, dtype=float).reshape(count, 2)
        return matrix[:, 0], matrix[:, 1]

    def residual(vector: Sequence[float], selected: Sequence[int]) -> np.ndarray:
        q_trial, alpha_trial = unpack(vector)
        prediction = _cpe_sd_background(data, q_trial, alpha_trial)
        indices = np.asarray(selected, dtype=int)
        target = np.asarray([anchors[int(index)] for index in indices], dtype=float)
        scale = max(float(np.std(target)), float(np.std(data.transform[indices])), 1.0)
        return (prediction[indices] - target) / scale

    evaluations = 0
    success = True
    messages: list[str] = []
    condition_numbers: list[float] = []

    def record_condition(fit: Any) -> None:
        jacobian = np.asarray(getattr(fit, "jac", np.empty((0, 0))), dtype=float)
        if jacobian.ndim == 2 and min(jacobian.shape) > 0:
            try:
                condition_numbers.append(float(np.linalg.cond(jacobian)))
            except np.linalg.LinAlgError:
                condition_numbers.append(float("inf"))

    if background_scope == "continuous":
        fit = least_squares(
            residual,
            initial,
            bounds=(lower, upper),
            args=(knots,),
            max_nfev=3000,
            x_scale="jac",
        )
        q_values, alpha_values = unpack(fit.x)
        evaluations += int(fit.nfev)
        success = bool(fit.success)
        messages.append(str(fit.message))
        record_condition(fit)
    else:
        fitted = initial.copy()
        for number, (start, end) in enumerate(data.segment_bounds):
            selected = [index for index in knots if start <= index <= end]
            active = np.zeros_like(fitted, dtype=bool)
            active[2 * number : 2 * number + 2] = True

            def local_residual(pair: Sequence[float]) -> np.ndarray:
                trial = fitted.copy()
                trial[active] = pair
                return residual(trial, selected)

            fit = least_squares(
                local_residual,
                fitted[active],
                bounds=(lower[active], upper[active]),
                max_nfev=1500,
                x_scale="jac",
            )
            fitted[active] = fit.x
            evaluations += int(fit.nfev)
            success = success and bool(fit.success)
            messages.append(f"Sweep {number + 1}: {fit.message}")
            record_condition(fit)
        q_values, alpha_values = unpack(fitted)

    background = _cpe_sd_background(data, q_values, alpha_values)
    parameters: dict[str, float] = {}
    bound_names: list[str] = []
    for number, (q_value, alpha_value) in enumerate(zip(q_values, alpha_values), start=1):
        parameters[f"Q{number}"] = float(q_value)
        parameters[f"alpha{number}"] = float(alpha_value)
        if alpha_value <= 0.6001 or alpha_value >= 0.9999:
            bound_names.append(f"alpha{number}")
    component_name = "cpe_background_"
    anchor_indices = np.asarray(knots, dtype=int)
    anchor_targets = np.asarray([anchors[int(index)] for index in anchor_indices])
    anchor_rmse = float(
        np.sqrt(np.mean((background[anchor_indices] - anchor_targets) ** 2))
    )
    return BackgroundConstruction(
        background=background,
        polynomial=np.full(len(data.time), np.nan, dtype=float),
        components={component_name: background.copy()},
        anchors=dict(anchors),
        parameters=parameters,
        diagnostics={
            "fit_success": success,
            "fit_message": "; ".join(messages),
            "function_evaluations": evaluations,
            "parameters_at_bounds": bound_names,
            "anchor_rmse": anchor_rmse,
            "jacobian_condition_max": (
                float(np.max(condition_numbers)) if condition_numbers else float("nan")
            ),
            "fit_scope": background_scope,
        },
    )


def _construct_background_details(
    data: AnalysisData,
    knot_indices: Sequence[int],
    knot_values: Mapping[int, float] | None,
    background_kind: str,
    background_scope: str = "independent",
) -> BackgroundConstruction:
    """Construct an interpolated or anchor-guided fitted background."""
    if background_kind not in BACKGROUND_KINDS:
        raise ValueError("Unknown background kind: " + str(background_kind))
    knots = validate_background_selection(
        data, knot_indices, background_kind, background_scope
    )
    anchors = {
        index: float((knot_values or {}).get(index, data.transform[index]))
        for index in knots
    }
    if not all(np.isfinite(value) for value in anchors.values()):
        raise ValueError("Background-knot heights must be finite")
    if background_kind == "cpe":
        return _fit_cpe_background(data, knots, anchors, background_scope)

    background = np.zeros(len(data.time), dtype=float)
    polynomial = np.full(len(data.time), np.nan, dtype=float)
    components: dict[str, np.ndarray] = {}
    parameters: dict[str, float] = {}
    groups = (
        [(1, 0, len(data.time) - 1, "continuous_")]
        if background_scope == "continuous"
        else [
            (number, start, end, f"segment{number}_")
            for number, (start, end) in enumerate(data.segment_bounds, start=1)
        ]
    )
    for segment_number, start, end, component_name in groups:
        segment_knots = [index for index in knots if start <= index <= end]
        knot_times = data.time[np.asarray(segment_knots, dtype=int)]
        knot_heights = np.asarray([anchors[index] for index in segment_knots])
        component = np.zeros(len(data.time), dtype=float)
        segment_time = data.time[start : end + 1]

        if background_kind == "constant":
            level = float(np.mean(knot_heights))
            component[start : end + 1] = level
            parameters[f"background{segment_number}_level"] = level
        else:
            coefficients, center, scale, polynomial_values = _segment_polynomial(
                data, segment_knots, start, end, anchors
            )
            if background_kind == "cubic":
                polynomial[start : end + 1] = polynomial_values
            if background_kind == "cubic" and len(segment_knots) >= 4:
                model = SegmentSplineModel(
                    xknots=knot_times,
                    segment_start_time=float(data.time[start]),
                    segment_end_time=float(data.time[end]),
                    polynomial_coefficients=coefficients,
                    polynomial_center=center,
                    polynomial_scale=scale,
                    prefix=component_name,
                )
                params = model.make_params()
                parameter_names = list(model.param_names)
                basis = np.zeros((len(segment_knots), len(segment_knots)), dtype=float)
                for column, parameter_name in enumerate(parameter_names):
                    trial = params.copy()
                    for name in parameter_names:
                        trial[name].set(value=0.0, vary=False)
                    trial[parameter_name].set(value=1.0, vary=False)
                    basis[:, column] = model.eval(trial, x=knot_times)
                coefficients_for_anchors = np.linalg.lstsq(
                    basis, knot_heights, rcond=None
                )[0]
                for name, value in zip(parameter_names, coefficients_for_anchors):
                    params[name].set(value=float(value), vary=False)
                component = np.asarray(model.eval(params, x=data.time), dtype=float)
            else:
                if len(segment_knots) == 1:
                    segment_values = np.full(len(segment_time), knot_heights[0], dtype=float)
                elif background_kind == "cubic":
                    interpolator = make_interp_spline(
                        knot_times,
                        knot_heights,
                        k=min(3, len(segment_knots) - 1),
                    )
                    segment_values = np.asarray(interpolator(segment_time), dtype=float)
                elif background_kind == "pchip":
                    segment_values = np.asarray(
                        PchipInterpolator(knot_times, knot_heights, extrapolate=False)(segment_time),
                        dtype=float,
                    )
                elif background_kind == "makima":
                    segment_values = np.asarray(
                        Akima1DInterpolator(
                            knot_times, knot_heights, method="makima", extrapolate=False
                        )(segment_time),
                        dtype=float,
                    )
                else:
                    segment_values = np.interp(segment_time, knot_times, knot_heights)

                if background_kind in {"linear", "pchip", "makima"}:
                    segment_values = _linear_tails(
                        segment_time, knot_times, knot_heights, segment_values
                    )
                else:
                    left = segment_time < knot_times[0]
                    right = segment_time > knot_times[-1]
                    if np.any(left):
                        segment_values[left] = polynomial_values[left] + (
                            knot_heights[0]
                            - np.polyval(coefficients, (knot_times[0] - center) / scale)
                        )
                    if np.any(right):
                        segment_values[right] = polynomial_values[right] + (
                            knot_heights[-1]
                            - np.polyval(coefficients, (knot_times[-1] - center) / scale)
                        )
                component[start : end + 1] = segment_values

        background += component
        components[component_name] = component
    diagnostics: dict[str, Any] = {"fit_scope": background_scope}
    if background_kind == "constant":
        selected = np.asarray(knots, dtype=int)
        targets = np.asarray([anchors[int(index)] for index in selected])
        diagnostics.update(
            {
                "fit_success": True,
                "anchor_rmse": float(
                    np.sqrt(np.mean((background[selected] - targets) ** 2))
                ),
            }
        )
    return BackgroundConstruction(
        background=background,
        polynomial=polynomial,
        components=components,
        anchors=anchors,
        parameters=parameters,
        diagnostics=diagnostics,
    )


def construct_background(
    data: AnalysisData,
    knot_indices: Sequence[int],
    knot_values: Mapping[int, float] | None,
    background_kind: str,
    background_scope: str = "independent",
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[int, float]]:
    """Compatibility wrapper returning the historical four background values."""
    construction = _construct_background_details(
        data, knot_indices, knot_values, background_kind, background_scope
    )
    return (
        construction.background,
        construction.polynomial,
        construction.components,
        construction.anchors,
    )


def preview_interpolated_background(
    data: AnalysisData,
    knot_indices: Sequence[int],
    knot_values: Mapping[int, float] | None,
    background_kind: str,
    background_scope: str = "independent",
) -> np.ndarray:
    """Build a live interpolation through every knot selected so far.

    Unlike the fit-time constructor, this preview permits unfinished
    independent sweeps. Segments without a knot remain blank; one-knot
    segments show a provisional constant line. It never relaxes fit-time
    validation and is not used in any numerical result or export.
    """
    if background_kind not in INTERPOLATING_BACKGROUNDS:
        return np.full(len(data.time), np.nan, dtype=float)
    if background_scope not in {"independent", "continuous"}:
        return np.full(len(data.time), np.nan, dtype=float)
    knots = _sanitize_selections(data, knot_indices)
    knots = [index for index in knots if not data.excluded_mask[index]]
    preview = np.full(len(data.time), np.nan, dtype=float)
    groups = (
        [(0, len(data.time) - 1)]
        if background_scope == "continuous"
        else list(data.segment_bounds)
    )
    for start, end in groups:
        selected = [index for index in knots if start <= index <= end]
        if not selected:
            continue
        knot_times = data.time[np.asarray(selected, dtype=int)]
        heights = np.asarray(
            [float((knot_values or {}).get(index, data.transform[index])) for index in selected],
            dtype=float,
        )
        segment_time = data.time[start : end + 1]
        if len(selected) == 1:
            values = np.full(len(segment_time), heights[0], dtype=float)
        elif background_kind == "linear":
            values = np.interp(segment_time, knot_times, heights)
        elif background_kind == "pchip":
            values = np.asarray(
                PchipInterpolator(knot_times, heights, extrapolate=False)(segment_time),
                dtype=float,
            )
        elif background_kind == "makima":
            values = np.asarray(
                Akima1DInterpolator(
                    knot_times, heights, method="makima", extrapolate=False
                )(segment_time),
                dtype=float,
            )
        else:
            values = np.asarray(
                make_interp_spline(
                    knot_times, heights, k=min(3, len(selected) - 1)
                )(segment_time),
                dtype=float,
            )
        if background_kind in {"linear", "pchip", "makima"}:
            values = _linear_tails(segment_time, knot_times, heights, values)
        elif len(selected) > 1:
            coefficients, center, scale, polynomial_values = _segment_polynomial(
                data, selected, start, end, knot_values
            )
            left = segment_time < knot_times[0]
            right = segment_time > knot_times[-1]
            if np.any(left):
                values[left] = polynomial_values[left] + (
                    heights[0]
                    - np.polyval(coefficients, (knot_times[0] - center) / scale)
                )
            if np.any(right):
                values[right] = polynomial_values[right] + (
                    heights[-1]
                    - np.polyval(coefficients, (knot_times[-1] - center) / scale)
                )
        preview[start : end + 1] = values
    return preview


def fit_spline_peaks(
    data: AnalysisData,
    knot_indices: Sequence[int],
    peak_indices: Sequence[int],
    config: FitConfig | None = None,
    preview: bool = True,
    knot_values: Mapping[int, float] | None = None,
) -> FitOutcome:
    config = config or FitConfig()
    requested_m = float(config.pearson4_exponent_min)
    if not 0.5 <= requested_m <= 100 or not np.isfinite(requested_m):
        raise ValueError("Pearson-IV exponent minimum must be between 0.5 and 100")
    exponent_min = max(requested_m, PEARSON4_MATHEMATICAL_MIN)
    knots = validate_background_selection(
        data, knot_indices, config.background_kind, config.background_scope
    )
    peaks = _sanitize_selections(data, peak_indices, minimum_gap=3)
    if not peaks:
        raise ValueError("Select at least one semi-derivative peak")
    excluded_peaks = [index for index in peaks if data.excluded_mask[index]]
    if excluded_peaks:
        raise ValueError(
            "Peak selections cannot lie inside excluded transient intervals: "
            + ", ".join(map(str, excluded_peaks))
        )
    fit_weights = np.asarray(data.fit_weights, dtype=float)
    if fit_weights.shape != data.time.shape or np.any(~np.isfinite(fit_weights)) or np.any(fit_weights < 0):
        raise ValueError("Fit weights must be finite, nonnegative, and match the scan length")
    included_mask = (fit_weights > 0) & ~np.asarray(data.excluded_mask, dtype=bool)
    if np.count_nonzero(included_mask) <= 5 * len(peaks):
        raise ValueError("The current exclusions and fit weights leave too few samples for fitting")

    background_construction = _construct_background_details(
        data, knots, knot_values, config.background_kind, config.background_scope
    )
    spline_background = background_construction.background
    polynomial = background_construction.polynomial
    components = background_construction.components
    anchors = background_construction.anchors
    residual_target = data.transform - spline_background
    median_dt = float(np.median(np.diff(data.time)))
    fwhm_seconds = max(0.075 / data.scan_rate_v_s, median_dt * 5)
    exponent_start = max(1.5, exponent_min + 0.1)
    sigma_start = fwhm_seconds / (2.0 * np.sqrt(2.0 ** (1.0 / exponent_start) - 1.0))

    max_nfev = int(config.preview_max_nfev if preview else config.final_max_nfev)
    best_fit = spline_background.copy()
    combined_params = Parameters()
    segment_results: list[Any] = []
    function_evaluations = 0
    varying_parameters = 0
    success = True
    messages: list[str] = []

    for segment_number, (start, end) in enumerate(data.segment_bounds, start=1):
        selected_peaks = [
            (number, index)
            for number, index in enumerate(peaks, start=1)
            if start <= index <= end
        ]
        if not selected_peaks:
            segment_results.append(None)
            continue
        model = None
        params = Parameters()
        for number, index in selected_peaks:
            prefix = f"peak{number}_"
            peak_model = Pearson4Model(prefix=prefix)
            model = peak_model if model is None else model + peak_model
            height = float(residual_target[index])
            if abs(height) <= np.finfo(float).eps:
                height = float(np.std(residual_target[start : end + 1])) or 1.0
            amplitude = _pearson_amplitude(height, sigma_start, exponent_start)
            peak_params = peak_model.make_params(
                amplitude=dict(value=amplitude),
                center=dict(value=float(data.time[index])),
                sigma=dict(value=sigma_start),
                expon=dict(value=exponent_start),
                skew=dict(value=0.0),
            )
            center_window_s = float(config.peak_center_window_v) / data.scan_rate_v_s
            center_min = max(float(data.time[start]), float(data.time[index]) - center_window_s)
            center_max = min(float(data.time[end]), float(data.time[index]) + center_window_s)
            peak_params[f"{prefix}center"].set(min=center_min, max=center_max)
            peak_params[f"{prefix}sigma"].set(
                min=median_dt * 0.5,
                max=max(float(data.time[end] - data.time[start]) * 0.5, median_dt),
            )
            peak_params[f"{prefix}expon"].set(min=exponent_min, max=100.0)
            peak_params[f"{prefix}skew"].set(min=-50.0, max=50.0)
            params.update(peak_params)

        segment_result = model.fit(
            residual_target[start : end + 1],
            params,
            x=data.time[start : end + 1],
            weights=fit_weights[start : end + 1],
            method="least_squares",
            max_nfev=max_nfev,
        )
        segment_results.append(segment_result)
        best_fit[start : end + 1] += np.asarray(segment_result.best_fit, dtype=float)
        for name, values in segment_result.eval_components().items():
            expanded = np.zeros(len(data.time), dtype=float)
            expanded[start : end + 1] = np.asarray(values, dtype=float)
            components[name] = expanded
        combined_params.update(segment_result.params)
        function_evaluations += int(segment_result.nfev)
        varying_parameters += int(segment_result.nvarys)
        success = success and bool(segment_result.success)
        messages.append(f"Sweep {segment_number}: {segment_result.message}")

    residual = data.transform - best_fit
    included_residual = residual[included_mask]
    chi_square = float(np.sum(included_residual**2))
    ndata = int(np.count_nonzero(included_mask))
    degrees_of_freedom = max(1, ndata - varying_parameters)
    normalized_chi = max(chi_square / max(ndata, 1), np.finfo(float).tiny)
    result = IndependentFitResult(
        best_fit=best_fit,
        components=components,
        params=combined_params,
        segment_results=segment_results,
        success=success,
        message="; ".join(messages),
        chisqr=chi_square,
        redchi=chi_square / degrees_of_freedom,
        aic=float(ndata * np.log(normalized_chi) + 2 * varying_parameters),
        bic=float(ndata * np.log(normalized_chi) + np.log(max(ndata, 1)) * varying_parameters),
        nfev=function_evaluations,
        ndata=ndata,
        nvarys=varying_parameters,
    )

    current_components = {
        name: fractional_differintegral(data.time, values, -0.5)
        for name, values in components.items()
    }
    raw_measured_roundtrip = fractional_differintegral(
        data.time, data.transform, -0.5
    )
    inverse_boundary_correction = data.current - raw_measured_roundtrip
    current_offset = float(inverse_boundary_correction[0])
    raw_reconstruction = fractional_differintegral(data.time, best_fit, -0.5)
    reconstructed_current = raw_reconstruction + inverse_boundary_correction
    background_current = (
        fractional_differintegral(data.time, spline_background, -0.5)
        + inverse_boundary_correction
    )
    peak_sd = best_fit - spline_background
    normalizer = max(float(np.nanmax(np.abs(peak_sd))), np.finfo(float).eps)
    normalized = peak_sd / normalizer
    normalized[~included_mask] = np.nan
    metrics: dict[str, Any] = {
        "success": bool(result.success),
        "message": str(result.message),
        "chi_square": float(result.chisqr),
        "reduced_chi_square": float(result.redchi),
        "aic": float(result.aic),
        "bic": float(result.bic),
        "sd_rmse": float(np.sqrt(np.mean(included_residual**2))),
        "sd_residual_lag1": _lag1(included_residual),
        "current_rmse": float(np.sqrt(np.mean((data.current[included_mask] - reconstructed_current[included_mask]) ** 2))),
        "excluded_samples": int(np.count_nonzero(data.excluded_mask)),
        "zero_weight_samples": int(np.count_nonzero(~included_mask)),
        "current_offset": current_offset,
        "function_evaluations": int(result.nfev),
        "preview": bool(preview),
        "scale_prefix": data.scale_prefix,
    }
    metrics["optimizer_failure_regions"] = _optimizer_failure_regions(
        data,
        residual,
        fit_weights,
        peaks,
        segment_results,
        config,
    )
    background_steps = np.diff(spline_background)
    switch_positions = [index for index in data.switching_indices if index < len(background_steps)]
    ordinary_mask = np.ones(len(background_steps), dtype=bool)
    ordinary_mask[switch_positions] = False
    ordinary_steps = np.abs(background_steps[ordinary_mask])
    ordinary_reference = float(np.median(ordinary_steps)) if len(ordinary_steps) else 0.0
    ordinary_reference = max(ordinary_reference, np.finfo(float).eps)
    switch_steps = np.asarray([background_steps[index] for index in switch_positions], dtype=float)
    metrics["switch_background_step_max"] = (
        float(np.max(np.abs(switch_steps))) if len(switch_steps) else 0.0
    )
    metrics["switch_background_step_ratio_max"] = (
        float(np.max(np.abs(switch_steps)) / ordinary_reference) if len(switch_steps) else 0.0
    )
    metrics["switch_neighborhood_excluded"] = int(
        sum(
            np.any(data.excluded_mask[max(0, index - 2) : min(len(data.time), index + 3)])
            for index in data.switching_indices
        )
    )
    metrics["background_scope"] = config.background_scope
    metrics["background_kind"] = config.background_kind
    metrics["background_fit_success"] = bool(
        background_construction.diagnostics.get("fit_success", True)
    )
    metrics["background_parameters_at_bounds"] = list(
        background_construction.diagnostics.get("parameters_at_bounds", [])
    )
    metrics["background_jacobian_condition_max"] = float(
        background_construction.diagnostics.get("jacobian_condition_max", np.nan)
    )

    peak_baseline_currents = {
        name: fractional_differintegral(data.time, best_fit - component, -0.5)
        + inverse_boundary_correction
        for name, component in components.items()
        if name.startswith("peak")
    }
    segment_baseline_current = np.empty(len(data.time), dtype=float)
    for start, end in data.segment_bounds:
        local_names = [
            f"peak{number}_"
            for number, index in enumerate(peaks, start=1)
            if start <= index <= end
        ]
        baseline_sd = best_fit.copy()
        for name in local_names:
            baseline_sd -= components[name]
        baseline_history = (
            fractional_differintegral(data.time, baseline_sd, -0.5)
            + inverse_boundary_correction
        )
        segment_baseline_current[start : end + 1] = baseline_history[start : end + 1]

    peak_results: list[PeakResult] = []
    for number, selected_index in enumerate(peaks, start=1):
        prefix = f"peak{number}_"
        component = components[prefix]
        center_time = float(result.params[f"{prefix}center"].value)
        fitted_index = int(np.argmin(np.abs(data.time - center_time)))
        current_component = current_components[prefix]
        start, end = _segment_for_index(data, selected_index)
        charge = float(
            np.trapezoid(
                current_component[start : end + 1],
                data.time[start : end + 1],
            )
        )
        potential_window = np.abs(data.potential - data.potential[selected_index]) <= 0.15
        local_mask = potential_window & included_mask & (np.arange(len(data.time)) >= start) & (np.arange(len(data.time)) <= end)
        candidate = np.flatnonzero(local_mask)
        if not len(candidate):
            candidate = np.arange(max(start, selected_index - 5), min(end + 1, selected_index + 6))
        current_index = int(candidate[np.argmax(np.abs(current_component[candidate]))])
        # Peak stems and reported heights must use the same all-peaks-removed
        # segment baseline that is drawn in the current-domain CV.  The older
        # per-peak baseline retained every neighbouring peak, so stems appeared
        # detached whenever two fitted peaks occupied the same sweep.
        baseline = segment_baseline_current
        reported_current = float(data.current[current_index] - baseline[current_index])
        direction_start = max(start, current_index - 3)
        direction_end = min(end, current_index + 3)
        direction_delta = data.potential[direction_end] - data.potential[direction_start]
        direction = "increasing potential" if direction_delta > 0 else "decreasing potential" if direction_delta < 0 else "undetermined"
        peak_results.append(
            PeakResult(
                number=number,
                selected_index=selected_index,
                fitted_index=fitted_index,
                current_index=current_index,
                potential_v=float(data.potential[current_index]),
                time_s=float(data.time[current_index]),
                current_scaled=reported_current,
                charge_scaled=charge,
                sweep_direction=direction,
                exponent=float(result.params[f"{prefix}expon"].value),
                skew=float(result.params[f"{prefix}skew"].value),
                baseline_current_scaled=float(baseline[current_index]),
                center_time_s=center_time,
            )
        )
    metrics["warnings"] = quality_warnings(metrics, peak_results)
    return FitOutcome(
        data=data,
        config=config,
        knot_indices=knots,
        knot_values=anchors,
        peak_indices=peaks,
        lmfit_result=result,
        components=components,
        current_components=current_components,
        spline_background=spline_background,
        polynomial_background=polynomial,
        reconstructed_current=reconstructed_current,
        background_current=background_current,
        segment_baseline_current=segment_baseline_current,
        peak_baseline_currents=peak_baseline_currents,
        inverse_boundary_correction=inverse_boundary_correction,
        current_offset_scaled=current_offset,
        normalized_signal=normalized,
        peak_results=peak_results,
        background_parameters=background_construction.parameters,
        background_diagnostics=background_construction.diagnostics,
        metrics=metrics,
    )


def _optimizer_failure_regions(
    data: AnalysisData,
    residual: np.ndarray,
    fit_weights: np.ndarray,
    peaks: Sequence[int],
    segment_results: Sequence[Any],
    config: FitConfig,
) -> list[dict[str, Any]]:
    """Describe failed sweeps and rank, without blaming, their peak neighborhoods."""
    diagnostics: list[dict[str, Any]] = []
    residual = np.asarray(residual, dtype=float)
    weights = np.asarray(fit_weights, dtype=float)
    neighborhood_s = max(
        float(config.peak_center_window_v) / float(data.scan_rate_v_s),
        float(np.median(np.diff(data.time))),
    )

    for sweep_number, ((start, end), segment_result) in enumerate(
        zip(data.segment_bounds, segment_results), start=1
    ):
        if segment_result is None or bool(getattr(segment_result, "success", False)):
            continue
        selected = [
            (number, index)
            for number, index in enumerate(peaks, start=1)
            if start <= index <= end
        ]
        message = " ".join(str(getattr(segment_result, "message", "")).split())
        reason = message or "optimizer stopped before convergence"
        lowered = reason.lower()
        if "function evaluations" in lowered or "max_nfev" in lowered:
            reason = "evaluation limit reached"
        elif len(reason) > 90:
            reason = reason[:87].rstrip() + "…"
        if not selected:
            diagnostics.append({"sweep": sweep_number, "reason": reason, "regions": []})
            continue

        centers = []
        for number, selected_index in selected:
            parameter_name = f"peak{number}_center"
            parameter = getattr(segment_result, "params", {}).get(parameter_name)
            center = (
                float(parameter.value)
                if parameter is not None
                else float(data.time[selected_index])
            )
            centers.append((number, center))

        local_indices = np.arange(start, end + 1)
        local_time = np.asarray(data.time[start : end + 1], dtype=float)
        distances = np.abs(local_time[:, None] - np.asarray([item[1] for item in centers]))
        nearest = np.argmin(distances, axis=1)
        included = (
            np.isfinite(residual[local_indices])
            & np.isfinite(weights[local_indices])
            & (weights[local_indices] > 0)
        )
        entries = []
        for local_peak, (number, center) in enumerate(centers):
            mask = included & (nearest == local_peak) & (distances[:, local_peak] <= neighborhood_s)
            indices = local_indices[mask]
            if not len(indices):
                continue
            weighted_residual = residual[indices] * weights[indices]
            error = float(np.sum(weighted_residual**2))
            rmse = float(np.sqrt(np.mean(weighted_residual**2)))
            center_index = int(np.argmin(np.abs(data.time - center)))
            entries.append((error, number, float(data.potential[center_index]), rmse))

        if not entries:
            diagnostics.append({"sweep": sweep_number, "reason": reason, "regions": []})
            continue
        entries.sort(reverse=True)
        neighborhood_error = max(sum(item[0] for item in entries), np.finfo(float).tiny)
        ranked: list[dict[str, float | int]] = []
        for error, number, potential, rmse in entries[:3]:
            share = 100.0 * error / neighborhood_error
            ranked.append({
                "peak": number,
                "potential_v": potential,
                "weighted_rmse": rmse,
                "share_pct": share,
            })
        diagnostics.append({"sweep": sweep_number, "reason": reason, "regions": ranked})
    return diagnostics


def quality_warnings(metrics: dict[str, Any], peaks: Sequence[PeakResult]) -> list[str]:
    warnings: list[str] = []
    if not metrics.get("success", False):
        failure_regions = list(metrics.get("optimizer_failure_regions", []))
        if failure_regions:
            lines = ["Optimizer incomplete"]
            for diagnostic in failure_regions:
                if not isinstance(diagnostic, Mapping):
                    lines.append(str(diagnostic))
                    continue
                lines.append(
                    f"Sweep {diagnostic.get('sweep', '?')} — {diagnostic.get('reason', 'stopped early')}"
                )
                regions = list(diagnostic.get("regions", []))
                if regions:
                    lines.append("Difficult peak regions:")
                    for region in regions:
                        lines.append(
                            f"  P{region['peak']} · {region['share_pct']:.0f}% residual · "
                            f"RMSE {region['weighted_rmse']:.3g} {metrics.get('scale_prefix', '')}A s^-1/2 · "
                            f"{region['potential_v']:.4g} V"
                        )
            lines.append("Residual ranking locates difficulty; it does not assign cause.")
            warnings.append("\n".join(lines))
        else:
            warnings.append("optimizer did not report success")
    lag = float(metrics.get("sd_residual_lag1", np.nan))
    if np.isfinite(lag) and abs(lag) > 0.95:
        warnings.append("semi-derivative residual is strongly serially correlated")
    low = [f"P{peak.number}" for peak in peaks if peak.exponent < 0.8]
    if low:
        warnings.append("low Pearson-IV exponent: " + ", ".join(low))
    if not metrics.get("background_fit_success", True):
        warnings.append("background optimizer did not report success")
    bound_names = list(metrics.get("background_parameters_at_bounds", []))
    if bound_names:
        warnings.append("CPE exponent at a fit bound: " + ", ".join(bound_names))
    condition = float(metrics.get("background_jacobian_condition_max", np.nan))
    if np.isfinite(condition) and condition > 1e8:
        warnings.append(
            "CPE knot fit is poorly conditioned; add separated knots or compare another background"
        )
    return warnings


def _plot_segments(ax: Any, data: AnalysisData, y: np.ndarray, **kwargs: Any) -> None:
    label = kwargs.pop("label", None)
    for number, (start, end) in enumerate(data.segment_bounds):
        ax.plot(data.potential[start : end + 1], y[start : end + 1], label=label if number == 0 else None, **kwargs)


def peak_summary_frame(outcome: FitOutcome) -> pd.DataFrame:
    current_unit = f"{outcome.data.report_scale_prefix}A"
    charge_factor, _, charge_report_prefix = peak_charge_display_scale(outcome)
    charge_unit = f"{charge_report_prefix}C"
    return pd.DataFrame(
        [
            {
                "Peak": peak.number,
                "t (s)": peak.time_s,
                "Eₚ (V)": peak.potential_v,
                f"iₚ ({current_unit})": peak.current_scaled,
                f"Qₚ ({charge_unit})": peak.charge_scaled * charge_factor,
                "m": peak.exponent,
                "ν": peak.skew,
            }
            for peak in outcome.peak_results
        ]
    )


def peak_charge_display_scale(outcome: FitOutcome) -> tuple[float, str, str]:
    """Return scaled-charge multiplier and readable Unicode/ASCII prefixes."""
    charges_c = np.asarray(
        [peak.charge_scaled for peak in outcome.peak_results], dtype=float
    ) / float(outcome.data.scaler)
    charge_scaler, prefix, report_prefix = current_scale(
        charges_c if len(charges_c) else [0.0]
    )
    return float(charge_scaler / outcome.data.scaler), prefix, report_prefix


def formatted_peak_table_frame(
    outcome: FitOutcome,
    *,
    ascii_units: bool = False,
) -> pd.DataFrame:
    """Build the presentation table using the scan's recorded resolution."""
    display_time, time_unit, time_scale = display_time_axis(outcome.data.time)
    time_decimals = experimental_decimal_places(
        display_time, smallest_step=False
    )
    potential_decimals = experimental_decimal_places(
        outcome.data.potential, smallest_step=True
    )
    current_prefix = (
        outcome.data.report_scale_prefix
        if ascii_units
        else outcome.data.scale_prefix
    )
    charge_factor, charge_prefix, charge_report_prefix = peak_charge_display_scale(outcome)
    if ascii_units:
        charge_prefix = charge_report_prefix
    rows = []
    for peak in outcome.peak_results:
        rows.append(
            {
                "Peak": str(int(peak.number)),
                f"t ({time_unit})": f"{peak.time_s * time_scale:.{time_decimals}f}",
                "Eₚ (V)": f"{peak.potential_v:.{potential_decimals}f}",
                f"iₚ ({current_prefix}A)": format_significant(peak.current_scaled),
                f"Qₚ ({charge_prefix}C)": format_significant(
                    peak.charge_scaled * charge_factor
                ),
                "m": format_significant(peak.exponent),
                "ν": format_significant(peak.skew),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "Peak",
            f"t ({time_unit})",
            "Eₚ (V)",
            f"iₚ ({current_prefix}A)",
            f"Qₚ ({charge_prefix}C)",
            "m",
            "ν",
        ],
    )


def write_pdf(outcome: FitOutcome, path: str | Path) -> Path:
    path = Path(path)
    data = outcome.data
    result = outcome.lmfit_result
    residual = data.transform - np.asarray(result.best_fit)
    palette = ["#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#0072B2"]
    measured_for_plot = np.asarray(data.current, dtype=float).copy()
    display_time, time_unit, _ = display_time_axis(data.time)
    with PdfPages(path) as pdf:
        fig, axes = plt.subplots(2, 1, figsize=(8.27, 11.69), gridspec_kw={"height_ratios": [1.25, 0.85]})
        _plot_segments(axes[0], data, measured_for_plot, color="#0072B2", linewidth=1.15, label="measured CV")
        _plot_segments(axes[0], data, outcome.reconstructed_current, color="#222222", linestyle="--", linewidth=1.15, label="full reconstruction")
        _plot_segments(axes[0], data, outcome.segment_baseline_current, color="#009E73", linestyle=":", linewidth=1.0, label="total baseline")
        for peak in outcome.peak_results:
            color = palette[(peak.number - 1) % len(palette)]
            axes[0].vlines(
                peak.potential_v,
                peak.baseline_current_scaled,
                peak.baseline_current_scaled + peak.current_scaled,
                color=color,
                linestyle="--",
                linewidth=1.15,
                alpha=0.95,
            )
            axes[0].scatter(data.potential[peak.current_index], data.current[peak.current_index], color=color, zorder=5)
            axes[0].annotate(f"P{peak.number}", (data.potential[peak.current_index], data.current[peak.current_index]), xytext=(4, 7), textcoords="offset points")
        axes[0].set(title="Measured and reconstructed cyclic voltammogram", xlabel="Potential (V)", ylabel=f"i ({data.scale_prefix}A)")
        axes[0].legend(fontsize=8, frameon=False)
        axes[1].axis("off")
        table_values = formatted_peak_table_frame(outcome)
        table = axes[1].table(
            cellText=table_values.values,
            colLabels=table_values.columns,
            cellLoc="center",
            loc="center",
            colWidths=[1.0 / len(table_values.columns)] * len(table_values.columns),
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.4)
        for (row, _), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(fontweight="bold")
        axes[1].set_title("Revised peak-current summary", fontsize=13)
        fig.tight_layout(pad=2.0)
        pdf.savefig(fig, dpi=250)
        plt.close(fig)

        fig, axes = plt.subplots(3, 1, figsize=(8.27, 11.69), gridspec_kw={"height_ratios": [1.15, 0.85, 0.65]})
        axes[0].plot(display_time, data.transform, color="#0072B2", linewidth=0.9, label="semi-derivative")
        axes[0].plot(display_time, result.best_fit, color="#D55E00", linewidth=1.1, linestyle="--", label="total fit")
        axes[0].plot(
            display_time,
            outcome.spline_background,
            color="#009E73",
            linewidth=0.8,
            alpha=0.65,
            zorder=1,
            label=f"{outcome.config.background_scope} {outcome.config.background_kind} background",
        )
        if outcome.config.background_kind == "cubic":
            axes[0].plot(display_time, outcome.polynomial_background, color="#777777", linewidth=0.9, linestyle=":", label="polynomial reference")
        anchor_heights = [outcome.knot_values[index] for index in outcome.knot_indices]
        axes[0].scatter(display_time[outcome.knot_indices], anchor_heights, color="#000000", marker="x", label="background knots", zorder=5)
        for number, (_, component) in enumerate((item for item in outcome.components.items() if item[0].startswith("peak")), start=1):
            axes[0].plot(display_time, component, color=palette[(number - 1) % len(palette)], linestyle=":", linewidth=0.9, label=f"P{number} component")
        for index in data.switching_indices:
            axes[0].axvline(display_time[index], color="#BBBBBB", linewidth=0.7)
        axes[0].set(title="Background and Pearson-IV components", xlabel=f"Time ({time_unit})", ylabel=f"e ({data.scale_prefix}A s$^{{-1/2}}$)")
        axes[0].legend(fontsize=6.8, ncol=2)
        axes[1].axhline(0, color="#888888", linewidth=0.6)
        axes[1].plot(display_time, outcome.normalized_signal, color="#0072B2", linewidth=1.0)
        axes[1].set(title="Normalized peak signal above background", xlabel=f"Time ({time_unit})", ylabel="Normalized signal")
        axes[2].axhline(0, color="#888888", linewidth=0.6)
        residual_for_plot = np.asarray(residual, dtype=float).copy()
        residual_for_plot[data.excluded_mask] = np.nan
        axes[2].plot(display_time, residual_for_plot, color="#D55E00", linewidth=0.85)
        for start, end in data.excluded_intervals:
            for axis in axes:
                axis.axvspan(display_time[start], display_time[end], color="#777777", alpha=0.10)
        axes[2].set(
            title="Semi-derivative residual",
            xlabel=f"Time ({time_unit})",
            ylabel=f"Residual ({data.scale_prefix}A s$^{{-1/2}}$)",
        )
        fig.tight_layout(pad=2.0)
        pdf.savefig(fig, dpi=250)
        plt.close(fig)

        report_lines: list[str] = []
        for line in fit_report_without_r_squared(result).splitlines():
            report_lines.extend(textwrap.wrap(line, 100) or [""])
        first_report_page = report_lines[:34]
        remaining_report_pages = [
            report_lines[index : index + 62]
            for index in range(34, len(report_lines), 62)
        ]

        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis("off")
        warning_text = "; ".join(outcome.metrics["warnings"]) or "None"
        chronology_window = data.scan.metadata.get("chronology_window")
        metadata = [
            f"Makima Fitter version: {PACKAGE_VERSION}",
            f"Input: {data.scan.source_path}",
            f"Input format: {data.scan.source_format}",
            (
                f"Chronology window: {chronology_window} (scan coordinates)"
                if chronology_window
                else f"Complete scan: {data.scan.scan_number} of {data.scan.scan_count}"
            ),
            f"Automatic switching indices: {', '.join(map(str, data.switching_indices)) or 'none'}",
            f"Excluded transient intervals: {len(data.excluded_intervals)}",
            f"Excluded transient samples: {int(np.count_nonzero(data.excluded_mask))}",
            f"Background scope: {outcome.config.background_scope}",
            f"Chronological sweep segments: {len(data.segment_bounds)}",
            f"Background construction: {outcome.config.background_kind}",
            "Switch treatment: automatically detected boundaries constrain sweep-local PIV fits; the fractional inverse retains full chronological history",
            f"Knot placement: {'locked to data' if outcome.config.knots_locked_to_data else 'free height'}",
            f"Background treatment: {background_treatment_description(outcome.config.background_kind, outcome.config.background_scope)}",
            f"Background knots: {len(outcome.knot_indices)}",
            f"Signal processing: {data.filter_diagnostics.get('summary', 'Filtering off')}",
            *[
                f"Background parameter {name}: {value:.12g}"
                for name, value in outcome.background_parameters.items()
            ],
            f"Pearson-IV peaks: {len(outcome.peak_results)}",
            f"SD RMSE: {outcome.metrics['sd_rmse']:.7g} {data.scale_prefix}A s^-1/2",
            f"Current reconstruction RMSE: {outcome.metrics['current_rmse']:.7g} {data.scale_prefix}A",
            f"Initial inverse-boundary correction: {outcome.current_offset_scaled:.7g} {data.scale_prefix}A",
            f"Reduced chi-square: {outcome.metrics['reduced_chi_square']:.7g}",
            f"Residual lag-1 correlation: {outcome.metrics['sd_residual_lag1']:.7g}",
            f"QC warnings: {warning_text}",
        ]
        metadata_lines: list[str] = []
        for line in metadata:
            metadata_lines.extend(textwrap.wrap(line, 92) or [""])
        report_top = max(0.37, 0.89 - 0.023 * len(metadata_lines))
        ax.text(0.02, 0.98, "Makima Fitter analysis audit", va="top", fontsize=15, fontweight="bold")
        ax.text(0.02, 0.93, "\n".join(metadata_lines), va="top", fontsize=9.0, linespacing=1.35)
        ax.text(0.02, report_top, "lmfit report (page 1)\n\n" + "\n".join(first_report_page), va="top", fontsize=7.2, family="monospace")
        pdf.savefig(fig, dpi=250)
        plt.close(fig)
        for page_number, lines in enumerate(remaining_report_pages, start=2):
            fig, ax = plt.subplots(figsize=(8.27, 11.69))
            ax.axis("off")
            fig.text(0.08, 0.95, f"lmfit report (page {page_number})", va="top", fontsize=14, fontweight="bold")
            fig.text(0.08, 0.91, "\n".join(lines), va="top", fontsize=7.2, family="monospace")
            pdf.savefig(fig, dpi=250)
            plt.close(fig)
    return path


def export_results(
    outcome: FitOutcome,
    output_dir: str | Path,
    name: str,
    detailed_csv: bool = True,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = name.strip() or "makima_fit"
    paths = {
        "pdf": write_pdf(outcome, output_dir / f"{safe_name}.pdf"),
        "peak_summary": output_dir / f"{safe_name}_peak_summary.csv",
        "knots": output_dir / f"{safe_name}_background_knots.csv",
        "diagnostics": output_dir / f"{safe_name}_fit_diagnostics.csv",
    }
    peak_summary_frame(outcome).to_csv(paths["peak_summary"], index=False)
    pd.DataFrame(
        [
            {
                "Knot no.": number,
                "Sweep segment": _segment_number_for_index(outcome.data, index),
                "Data index": index,
                "Time (s)": outcome.data.time[index],
                "Potential (V)": outcome.data.potential[index],
                f"Measured semi-derivative ({outcome.data.report_scale_prefix}A s^-1/2)": outcome.data.transform[index],
                f"Background-knot semi-derivative ({outcome.data.report_scale_prefix}A s^-1/2)": outcome.knot_values[index],
                f"Constructed background value ({outcome.data.report_scale_prefix}A s^-1/2)": outcome.spline_background[index],
            }
            for number, index in enumerate(outcome.knot_indices, start=1)
        ]
    ).to_csv(paths["knots"], index=False)
    background_treatment = background_treatment_description(
        outcome.config.background_kind, outcome.config.background_scope
    )
    configured_stages = list(outcome.config.filter_stages)
    if not configured_stages and outcome.config.filter_kind != "none":
        configured_stages = [FilterStage(
            kind=outcome.config.filter_kind,
            domain=outcome.config.filter_domain,
            scope=outcome.config.filter_scope,
            range=outcome.config.filter_range,
            frequency_hz=outcome.config.filter_frequency_hz,
            bandwidth_hz=outcome.config.filter_bandwidth_hz,
            attenuation_db=outcome.config.filter_attenuation_db,
            window_mv=outcome.config.filter_window_mv,
            polynomial_order=outcome.config.filter_polynomial_order,
            hampel_sigma=outcome.config.filter_hampel_sigma,
            harmonics=outcome.config.filter_harmonics,
            feather_mv=outcome.config.filter_feather_mv,
            envelope_prominence_pct=outcome.config.filter_envelope_prominence_pct,
        )]
    stage_rows = []
    for number, stage in enumerate(configured_stages, start=1):
        stage_rows.extend([
            (f"Filter stage {number}", FILTER_LABELS.get(stage.kind, stage.kind)),
            (f"Filter stage {number} domain", stage.domain),
            (f"Filter stage {number} scope", stage.scope),
            (f"Filter stage {number} range indices", "none" if stage.range is None else f"{stage.range[0]}-{stage.range[1]}"),
            (f"Filter stage {number} frequency (Hz)", stage.frequency_hz),
            (f"Filter stage {number} bandwidth (Hz)", stage.bandwidth_hz),
            (f"Filter stage {number} attenuation (dB)", stage.attenuation_db),
            (f"Filter stage {number} window (mV)", stage.window_mv),
            (f"Filter stage {number} polynomial order", stage.polynomial_order),
            (f"Filter stage {number} Hampel threshold (sigma)", stage.hampel_sigma),
            (f"Filter stage {number} harmonics", stage.harmonics),
            (f"Filter stage {number} envelope prominence (%)", stage.envelope_prominence_pct),
            (f"Filter stage {number} regional feather (mV)", stage.feather_mv),
        ])
    chronology_window = outcome.data.scan.metadata.get("chronology_window")
    diagnostic_rows = [
        ("Package version", PACKAGE_VERSION),
        ("Background model", background_treatment),
        ("Background scope", outcome.config.background_scope),
        ("Switch treatment", "automatic switching points constrain independent Pearson-IV fits; background is continuous only when continuous scope is selected; inverse-half-derivative uses the complete chronological history"),
        ("Knot placement", "locked to measured semi-derivative" if outcome.config.knots_locked_to_data else "free manual heights"),
        ("Signal-processing stages", len(configured_stages)),
        *stage_rows,
        ("Upper-envelope maxima indices", "; ".join(map(str, outcome.data.filter_diagnostics.get("maxima_indices", []))) or "none"),
        ("Applied filter response", outcome.data.filter_diagnostics.get("summary", "Filtering off")),
        ("Startup treatment", "none; represented by the selected background"),
        ("Electrolysis background", "none; represented by the selected background"),
        ("Piecewise CPE background", "knot-fitted" if outcome.config.background_kind == "cpe" else "not selected"),
        ("Diffusional baseline analysis", "not included"),
        ("Hogan EC analysis", "not included"),
        ("Source file", outcome.data.scan.source_path),
        ("Source input format", outcome.data.scan.source_format),
        (
            "Chronology window" if chronology_window else "Complete scan",
            chronology_window or outcome.data.scan.scan_number,
        ),
        ("Automatic switching points", len(outcome.data.switching_indices)),
        ("Automatic switching indices", "; ".join(map(str, outcome.data.switching_indices))),
        ("Excluded transient intervals", len(outcome.data.excluded_intervals)),
        ("Excluded transient samples", int(np.count_nonzero(outcome.data.excluded_mask))),
        ("Initial inverse-boundary correction", outcome.current_offset_scaled),
        ("Background knots", len(outcome.knot_indices)),
        *[(f"Background parameter {name}", value) for name, value in outcome.background_parameters.items()],
        *[(f"Background diagnostic {name}", value) for name, value in outcome.background_diagnostics.items()],
        ("Selected peaks", len(outcome.peak_results)),
        (f"Semi-derivative RMSE ({outcome.data.scale_prefix}A s^-1/2)", outcome.metrics["sd_rmse"]),
        (f"Current reconstruction RMSE ({outcome.data.scale_prefix}A)", outcome.metrics["current_rmse"]),
        *[
            (key.replace("_", " ").title(), value)
            for key, value in outcome.metrics.items()
            if key not in {"warnings", "sd_rmse", "current_rmse"}
        ],
        ("QC warnings", "; ".join(outcome.metrics["warnings"])),
    ]
    for segment_number, (start, end) in enumerate(outcome.data.segment_bounds, start=1):
        segment_knots = [
            index for index in outcome.knot_indices if start <= index <= end
        ]
        diagnostic_rows.extend(
            [
                (f"Segment {segment_number} index range", f"{start}-{end}"),
                (f"Segment {segment_number} knot count", len(segment_knots)),
                (f"Segment {segment_number} knot indices", "; ".join(map(str, segment_knots))),
                (f"Segment {segment_number} knot time range (s)", f"{outcome.data.time[min(segment_knots)]:.12g}-{outcome.data.time[max(segment_knots)]:.12g}" if segment_knots else "none"),
                (f"Segment {segment_number} background treatment", background_treatment),
            ]
        )
    for interval_number, (start, end) in enumerate(outcome.data.excluded_intervals, start=1):
        diagnostic_rows.extend(
            [
                (f"Transient interval {interval_number} indices", f"{start}-{end}"),
                (f"Transient interval {interval_number} time range (s)", f"{outcome.data.time[start]:.12g}-{outcome.data.time[end]:.12g}"),
                (f"Transient interval {interval_number} treatment", "measured current and semi-derivative retained unchanged; zero fit weight; omitted from residual metrics"),
            ]
        )
    diagnostic_rows.extend((key.replace("_", " ").title(), value) for key, value in outcome.data.scan.metadata.items())
    pd.DataFrame(diagnostic_rows, columns=["Metric", "Value"]).to_csv(paths["diagnostics"], index=False)
    if detailed_csv:
        detailed_path = output_dir / f"{safe_name}_fit_trace.csv"
        raw_current_scaled = outcome.data.current_a * outcome.data.scaler
        raw_transform = fractional_differintegral(outcome.data.time, raw_current_scaled, 0.5)
        trace_columns = {
                "Time (s)": outcome.data.time,
                "Potential (V)": outcome.data.potential,
                "Original measured current (A)": outcome.data.current_a,
                f"Original measured current ({outcome.data.report_scale_prefix}A)": raw_current_scaled,
                f"Filtered/working current ({outcome.data.report_scale_prefix}A)": outcome.data.current,
                "Excluded transient sample": outcome.data.excluded_mask,
                "Fit weight": outcome.data.fit_weights,
                "Upper-envelope maximum": np.isin(
                    np.arange(len(outcome.data.time)),
                    outcome.data.filter_diagnostics.get("maxima_indices", []),
                ),
                f"Original semi-derivative ({outcome.data.report_scale_prefix}A s^-1/2)": raw_transform,
                f"Filtered/working semi-derivative ({outcome.data.report_scale_prefix}A s^-1/2)": outcome.data.transform,
                f"Background ({outcome.data.report_scale_prefix}A s^-1/2)": outcome.spline_background,
                f"Total semi-derivative fit ({outcome.data.report_scale_prefix}A s^-1/2)": outcome.lmfit_result.best_fit,
                f"Semi-derivative residual ({outcome.data.report_scale_prefix}A s^-1/2)": outcome.data.transform - outcome.lmfit_result.best_fit,
                f"Reconstructed current ({outcome.data.report_scale_prefix}A)": outcome.reconstructed_current,
                f"Background-only current ({outcome.data.report_scale_prefix}A)": outcome.background_current,
                f"Total peak-reporting baseline ({outcome.data.report_scale_prefix}A)": outcome.segment_baseline_current,
                f"Inverse-transform boundary correction ({outcome.data.report_scale_prefix}A)": outcome.inverse_boundary_correction,
                "Automatic switching point": np.isin(np.arange(len(outcome.data.time)), outcome.data.switching_indices),
                "Selected background knot": np.isin(np.arange(len(outcome.data.time)), outcome.knot_indices),
                "Selected peak": np.isin(np.arange(len(outcome.data.time)), outcome.peak_indices),
                "Background source": background_source_labels(
                    outcome.data,
                    outcome.knot_indices,
                    outcome.config.background_kind,
                    outcome.config.background_scope,
                ),
                "Background scope": np.full(len(outcome.data.time), outcome.config.background_scope),
            }
        if outcome.config.background_kind == "cubic":
            trace_columns["Polynomial reference"] = outcome.polynomial_background
        frame = pd.DataFrame(trace_columns)
        for name_key, values in outcome.components.items():
            frame[f"Component: {name_key.removesuffix('_')}"] = values
        frame.to_csv(detailed_path, index=False)
        paths["trace"] = detailed_path
    return paths


__all__ = [
    "AnalysisData",
    "FILTER_LABELS",
    "apply_signal_filter",
    "apply_transient_exclusions",
    "build_exclusion_mask",
    "background_treatment_description",
    "construct_background",
    "current_scale",
    "FitConfig",
    "FilterStage",
    "FitOutcome",
    "PACKAGE_VERSION",
    "detect_switching_points",
    "display_time_axis",
    "experimental_decimal_places",
    "export_results",
    "fit_spline_peaks",
    "fractional_differintegral",
    "format_significant",
    "formatted_peak_table_frame",
    "minimum_background_anchors",
    "normalize_exclusion_intervals",
    "peak_charge_display_scale",
    "peak_summary_frame",
    "polynomial_reference",
    "prepare_analysis",
    "segment_bounds",
    "suggested_knot_indices",
    "validate_background_selection",
    "validate_knots",
    "write_pdf",
]
