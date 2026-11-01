"""Single-window interactive interface for Makima Fitter."""

from __future__ import annotations

import ctypes
from dataclasses import asdict, replace as dataclass_replace
import os
from pathlib import Path
import sys
import tempfile
import traceback

import numpy as np
from scipy.signal import find_peaks

from PyQt5.QtCore import QByteArray, QEvent, QTimer, Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QApplication,
    QAbstractSpinBox,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from data_io import (
    SUPPORTED_FORMATS,
    detect_input_format,
    load_scan,
    load_scan_window,
    scan_count,
)
from session_io import SESSION_EXTENSION, load_session_archive, save_session_archive
from spline_core import (
    PACKAGE_VERSION,
    FILTER_LABELS,
    FitConfig,
    FilterStage,
    apply_signal_filter,
    apply_transient_exclusions,
    construct_background,
    display_time_axis,
    export_results,
    fit_spline_peaks,
    format_significant,
    formatted_peak_table_frame,
    minimum_background_anchors,
    normalize_exclusion_intervals,
    peak_charge_display_scale,
    polynomial_reference,
    prepare_analysis,
    preview_interpolated_background,
    suggested_knot_indices,
)


LIVE_TABLE_ROWS = 10
FOUR_PLOT_LAYOUT_RECT = (0.0, 0.0, 0.982, 1.0)
WINDOWS_APP_ID = "MakimaFitter.Desktop.2.0.0"
MAX_DISPLAY_POINTS = 20000
BACKGROUND_LINE_STYLE = {
    "color": "#009E73",
    "linewidth": 0.8,
    "alpha": 0.65,
    "zorder": 1,
}


def application_icon_path() -> Path:
    application_path = (
        Path(sys._MEIPASS)
        if getattr(sys, "frozen", False)
        else Path(__file__).resolve().parent
    )
    return application_path / "icon.ico"


def configure_windows_app_identity() -> None:
    """Give Windows a stable taskbar identity instead of the Python host icon."""
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            WINDOWS_APP_ID
        )
    except (AttributeError, OSError):
        pass


def apply_windows_taskbar_icon(window: QWidget) -> bool:
    """Assign the SD icon to the native window class used by the taskbar."""
    if sys.platform != "win32":
        return False
    icon_path = application_icon_path()
    if not icon_path.is_file():
        return False
    try:
        user32 = ctypes.windll.user32
        load_image = user32.LoadImageW
        load_image.argtypes = [
            ctypes.c_void_p,
            ctypes.c_wchar_p,
            ctypes.c_uint,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint,
        ]
        load_image.restype = ctypes.c_void_p
        send_message = user32.SendMessageW
        send_message.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_size_t,
            ctypes.c_void_p,
        ]
        send_message.restype = ctypes.c_ssize_t
        set_class_icon = getattr(user32, "SetClassLongPtrW", user32.SetClassLongW)
        set_class_icon.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        set_class_icon.restype = ctypes.c_void_p

        image_icon = 1
        load_from_file = 0x0010
        wm_seticon = 0x0080
        icon_small, icon_big = 0, 1
        gclp_hicon, gclp_hiconsm = -14, -34
        big_size = (
            max(16, int(user32.GetSystemMetrics(11))),
            max(16, int(user32.GetSystemMetrics(12))),
        )
        small_size = (
            max(16, int(user32.GetSystemMetrics(49))),
            max(16, int(user32.GetSystemMetrics(50))),
        )
        big_handle = load_image(
            None, str(icon_path), image_icon, big_size[0], big_size[1], load_from_file
        )
        small_handle = load_image(
            None, str(icon_path), image_icon, small_size[0], small_size[1], load_from_file
        )
        if not big_handle or not small_handle:
            return False
        hwnd = ctypes.c_void_p(int(window.winId()))
        send_message(hwnd, wm_seticon, icon_big, big_handle)
        send_message(hwnd, wm_seticon, icon_small, small_handle)
        set_class_icon(hwnd, gclp_hicon, big_handle)
        set_class_icon(hwnd, gclp_hiconsm, small_handle)
        window._windows_icon_handles = (big_handle, small_handle)
        return True
    except (AttributeError, OSError, TypeError, ValueError):
        return False


class SplineNavigationToolbar(NavigationToolbar):
    """Route the toolbar Home button through the same all-panel reset as H."""

    def __init__(self, canvas, parent=None):
        self._application_home_callback = None
        super().__init__(canvas, parent)

    def set_application_home_callback(self, callback) -> None:
        self._application_home_callback = callback

    def home(self, *args) -> None:
        if self._application_home_callback is not None:
            self._application_home_callback()
        else:
            super().home(*args)

    def matplotlib_home(self) -> None:
        super().home()

    def set_message(self, message) -> None:
        """Use the application's Move terminology in toolbar status text."""
        text = str(message).replace("pan/zoom", "move").replace("Pan/Zoom", "Move")
        super().set_message(text)


def wrapped_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    label.setMinimumWidth(0)
    label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Minimum)
    return label


def compact_layout(layout, *, margins=(8, 10, 8, 6), spacing=2) -> None:
    layout.setContentsMargins(*margins)
    layout.setSpacing(spacing)


def _odd_window_from_mv_for_display(
    potential: np.ndarray,
    requested_mv: float,
    minimum: int,
) -> int:
    """Mirror the core window conversion for immediate UI guidance."""
    steps = np.abs(np.diff(np.asarray(potential, dtype=float)))
    steps = steps[np.isfinite(steps) & (steps > np.finfo(float).eps)]
    if not len(steps):
        return 0
    points = max(minimum, int(round((requested_mv / 1000.0) / float(np.median(steps)))))
    if points % 2 == 0:
        points += 1
    maximum = len(potential) if len(potential) % 2 else len(potential) - 1
    return min(points, maximum)


def _peak_candidates_outside_exclusions(
    residual: np.ndarray,
    excluded_mask: np.ndarray,
    *,
    distance: int,
    prominence: float,
) -> list[tuple[int, float]]:
    """Find signed residual peaks only inside contiguous included regions."""
    residual = np.asarray(residual, dtype=float)
    excluded_mask = np.asarray(excluded_mask, dtype=bool)
    included = np.flatnonzero(~excluded_mask & np.isfinite(residual))
    if not len(included):
        return []
    runs = np.split(included, np.flatnonzero(np.diff(included) > 1) + 1)
    candidates: list[tuple[int, float]] = []
    for run in runs:
        if len(run) < 3:
            continue
        local = residual[run]
        for sign in (1.0, -1.0):
            peaks, properties = find_peaks(
                sign * local,
                distance=distance,
                prominence=prominence,
            )
            candidates.extend(
                (int(run[int(index)]), float(value))
                for index, value in zip(peaks, properties.get("prominences", []))
            )
    return candidates


class MainWindow(QMainWindow):
    def __init__(self, initial_input: str | None = None):
        super().__init__()
        self.setWindowTitle(f"Makima Fitter {PACKAGE_VERSION}")
        self.setWindowIcon(QIcon(str(application_icon_path())))
        application = QApplication.instance()
        if application is not None:
            application.setWindowIcon(self.windowIcon())
        self.resize(1480, 1000)
        self.setMinimumSize(1000, 650)

        self.input_file: str | None = None
        self._source_display_path: str | None = None
        self.session_file: str | None = None
        self._session_temp_dir: tempfile.TemporaryDirectory | None = None
        self.output_dir: str | None = None
        self.analysis = None
        self.outcome = None
        self.knot_indices: list[int] = []
        self.knot_values: dict[int, float] = {}
        self.peak_indices: list[int] = []
        self.transient_intervals: list[tuple[int, int]] = []
        self.pending_transient_index: int | None = None
        self.filter_range: tuple[int, int] | None = None
        self.pending_filter_index: int | None = None
        self._filter_range_armed = False
        self.filter_stages: list[FilterStage] = []
        self._active_filter_stage = -1
        self._loading_filter_stage = False
        self._filtered_analysis_cache = None
        self._filter_cache_key = None
        self._filter_error: str | None = None
        self.history: list[dict] = []
        self.future: list[dict] = []
        self._scan_states: dict[int | str, dict] = {}
        self._active_scan_number: int | str | None = None
        self._pending_scan_history = False
        self._loading_scan = False
        self._preserve_plot_limits = False
        self._home_limits = None
        self._quality_panel_mode = "table"
        self._table_selected_cells: list[tuple[int, int]] = []
        self._table_calculation: dict[int, str] = {}
        self._table_calculation_symbols: dict[int, str] = {}
        self._table_active_column: int | None = None
        self._live_peak_table = None
        self._top_right_mode = "normalized"
        self._bottom_left_mode = "potential"
        self._focused_axis_index: int | None = None
        self._was_maximized_before_plot_focus = False
        self._controls_visible_before_plot_focus = True
        self._controls_width_before_hide = 410
        self._was_maximized_before_window_fullscreen = False
        self._navigation_drag_active = False
        self._measurement_indices: list[int] = []
        self._measurement_axis_index: int | None = None
        self._fit_timer = QTimer(self)
        self._fit_timer.setSingleShot(True)
        self._fit_timer.setInterval(220)
        self._fit_timer.timeout.connect(self.update_live_fit)
        self._scan_reload_timer = QTimer(self)
        self._scan_reload_timer.setSingleShot(True)
        self._scan_reload_timer.setInterval(160)
        self._scan_reload_timer.timeout.connect(self.load_selected_scan)

        self.main_splitter = QSplitter(Qt.Horizontal)
        self.controls_panel = self._build_controls()
        self.canvas_panel = self._build_canvas()
        self.main_splitter.addWidget(self.controls_panel)
        self.main_splitter.addWidget(self.canvas_panel)
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setSizes([410, 1010])
        self.setCentralWidget(self.main_splitter)
        self.setFocusPolicy(Qt.StrongFocus)
        self.statusBar().showMessage("Select an input file to begin")
        QApplication.instance().installEventFilter(self)

        if initial_input:
            if Path(initial_input).suffix.lower() == SESSION_EXTENSION:
                self.load_session_path(initial_input)
            else:
                self.load_input_path(initial_input)
        QTimer.singleShot(0, self._prime_keyboard_focus)
        QTimer.singleShot(150, self._prime_keyboard_focus)

    def _build_controls(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        panel = QWidget()
        layout = QVBoxLayout(panel)
        compact_layout(layout, margins=(6, 6, 6, 6), spacing=5)
        panel.setStyleSheet(
            "QGroupBox { margin-top: 8px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 7px; padding: 0 2px; }"
            "QPushButton { min-height: 22px; max-height: 26px; padding: 1px 5px; }"
            "QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit { min-height: 22px; max-height: 24px; }"
        )

        input_group = QGroupBox("1. Input")
        input_layout = QVBoxLayout(input_group)
        compact_layout(input_layout)
        self.file_button = QPushButton("Select CV")
        self.file_button.clicked.connect(self.select_file)
        self.file_label = wrapped_label("No file selected")
        self.file_label.setMaximumWidth(385)
        input_layout.addWidget(self.file_button)
        self.save_session_button = QPushButton("Save session")
        self.save_session_button.clicked.connect(self.save_session)
        self.save_session_button.setEnabled(False)
        self.save_session_button.setToolTip(
            "Save the source CV, selections, filters, fit settings, and current views"
        )
        input_layout.addWidget(self.save_session_button)
        input_layout.addWidget(self.file_label)
        self.format_dropdown = QComboBox()
        self.format_dropdown.addItems(SUPPORTED_FORMATS)
        self.format_dropdown.currentTextChanged.connect(self._format_changed)
        input_layout.addWidget(self.format_dropdown)
        scan_row = QHBoxLayout()
        scan_row.addWidget(QLabel("Complete scan:"))
        self.scan_number = QSpinBox()
        self.scan_number.setRange(1, 1)
        self.scan_number.valueChanged.connect(self._schedule_scan_reload)
        scan_row.addWidget(self.scan_number)
        self.scan_count_label = QLabel("of 1")
        scan_row.addWidget(self.scan_count_label)
        scan_row.addStretch(1)
        input_layout.addLayout(scan_row)
        self.custom_scan_window = QCheckBox("Custom chronology window")
        self.custom_scan_window.setToolTip(
            "Select a continuous portion of the recorded chronology; for example, "
            "3.75 to 4.75 is one full cycle shifted three quarters of a scan"
        )
        self.custom_scan_window.toggled.connect(self._custom_scan_window_toggled)
        input_layout.addWidget(self.custom_scan_window)
        self.scan_window_row = QWidget()
        chronology_row = QHBoxLayout(self.scan_window_row)
        chronology_row.setContentsMargins(0, 0, 0, 0)
        chronology_row.addWidget(QLabel("From scan:"))
        self.scan_window_start = QDoubleSpinBox()
        self.scan_window_start.setRange(1.0, 1.99)
        self.scan_window_start.setDecimals(2)
        self.scan_window_start.setSingleStep(0.25)
        self.scan_window_start.setValue(1.0)
        chronology_row.addWidget(self.scan_window_start)
        chronology_row.addWidget(QLabel("to:"))
        self.scan_window_end = QDoubleSpinBox()
        self.scan_window_end.setRange(1.01, 2.0)
        self.scan_window_end.setDecimals(2)
        self.scan_window_end.setSingleStep(0.25)
        self.scan_window_end.setValue(2.0)
        chronology_row.addWidget(self.scan_window_end)
        chronology_row.addStretch(1)
        self.scan_window_start.valueChanged.connect(self._scan_window_limit_changed)
        self.scan_window_end.valueChanged.connect(self._scan_window_limit_changed)
        self.scan_window_row.hide()
        input_layout.addWidget(self.scan_window_row)
        rate_row = QHBoxLayout()
        rate_row.addWidget(QLabel("Scan rate:"))
        self.scan_rate = QDoubleSpinBox()
        self.scan_rate.setRange(0.0, 1000.0)
        self.scan_rate.setDecimals(6)
        self.scan_rate.setValue(0.0)
        self.scan_rate.setSuffix(" V/s")
        self.scan_rate.setSpecialValueText("Auto")
        self.scan_rate.editingFinished.connect(self.load_selected_scan)
        rate_row.addWidget(self.scan_rate)
        input_layout.addLayout(rate_row)
        self.input_group = input_group
        layout.addWidget(input_group)
        layout.addStretch(1)

        selection_group = QGroupBox("2. Selections")
        selection_layout = QVBoxLayout(selection_group)
        compact_layout(selection_layout)
        self.selection_group = QButtonGroup(self)
        self.knot_radio = QRadioButton("Background knots")
        self.peak_radio = QRadioButton("Pearson-IV peaks")
        self.transient_radio = QRadioButton("Transient exclusions")
        self.knot_radio.setChecked(True)
        self.selection_group.addButton(self.knot_radio)
        self.selection_group.addButton(self.peak_radio)
        self.selection_group.addButton(self.transient_radio)
        selection_layout.addWidget(self.knot_radio)
        selection_layout.addWidget(self.peak_radio)
        selection_layout.addWidget(self.transient_radio)
        background_row = QHBoxLayout()
        background_row.addWidget(QLabel("Background:"))
        self.background_kind = QComboBox()
        self.background_kind.addItem("Makima", "makima")
        self.background_kind.addItem("PCHIP", "pchip")
        self.background_kind.addItem("Linear segments", "linear")
        self.background_kind.addItem("Cubic spline", "cubic")
        self.background_kind.addItem("Piecewise CPE fit", "cpe")
        self.background_kind.addItem("Constant fit", "constant")
        self.background_kind.setToolTip(
            "Makima/PCHIP/linear/cubic interpolate the knots. "
            "CPE and constant models fit their parameters only to the selected knots."
        )
        self.background_kind.currentIndexChanged.connect(self._background_changed)
        background_row.addWidget(self.background_kind)
        selection_layout.addLayout(background_row)
        scope_row = QHBoxLayout()
        scope_row.addWidget(QLabel("Scope:"))
        self.background_scope = QComboBox()
        self.background_scope.addItem("Independent sweeps", "independent")
        self.background_scope.addItem("Continuous scan", "continuous")
        self.background_scope.currentIndexChanged.connect(self._background_changed)
        scope_row.addWidget(self.background_scope)
        selection_layout.addLayout(scope_row)
        self.lock_knots = QCheckBox("Lock knots to data")
        self.lock_knots.setChecked(True)
        self.lock_knots.toggled.connect(self._knot_lock_changed)
        selection_layout.addWidget(self.lock_knots)
        self.suggest_peaks_button = QPushButton("Find peaks")
        self.suggest_peaks_button.clicked.connect(self.suggest_peaks)
        self.suggest_peaks_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        selection_layout.addWidget(self.suggest_peaks_button)
        self.clear_button_row = QHBoxLayout()
        self.reset_knots_button = QPushButton("Clear knots")
        self.reset_knots_button.clicked.connect(self.clear_knots)
        self.clear_peaks_button = QPushButton("Clear peaks")
        self.clear_peaks_button.clicked.connect(self.clear_peaks)
        self.reset_knots_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.clear_peaks_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.clear_button_row.addWidget(self.reset_knots_button, 1)
        self.clear_button_row.addWidget(self.clear_peaks_button, 1)
        selection_layout.addLayout(self.clear_button_row)
        self.selection_action_row = QHBoxLayout()
        self.clear_transients_button = QPushButton("Clear exclusions")
        self.clear_transients_button.clicked.connect(self.clear_transients)
        self.undo_button = QPushButton("Undo")
        self.undo_button.clicked.connect(self.undo_selection)
        self.clear_transients_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.undo_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.selection_action_row.addWidget(self.clear_transients_button, 1)
        self.selection_action_row.addWidget(self.undo_button, 1)
        selection_layout.addLayout(self.selection_action_row)
        self.selections_panel = selection_group
        layout.addWidget(selection_group)
        layout.addStretch(1)

        filter_group = QGroupBox("3. Signal processing")
        filter_layout = QVBoxLayout(filter_group)
        compact_layout(filter_layout)
        filter_stage_row = QHBoxLayout()
        filter_stage_row.addWidget(QLabel("Stages:"))
        self.filter_stage_selector = QComboBox()
        self.filter_stage_selector.addItem("No filters")
        self.filter_stage_selector.currentIndexChanged.connect(self._select_filter_stage)
        filter_stage_row.addWidget(self.filter_stage_selector, 1)
        filter_layout.addLayout(filter_stage_row)
        filter_stage_actions = QHBoxLayout()
        self.add_filter_button = QPushButton("Add")
        self.remove_filter_button = QPushButton("Remove")
        self.move_filter_up_button = QPushButton("↑")
        self.move_filter_down_button = QPushButton("↓")
        self.add_filter_button.setToolTip("Add a filter stage after the selected stage")
        self.remove_filter_button.setToolTip("Remove the selected filter stage")
        self.move_filter_up_button.setToolTip("Apply the selected stage earlier")
        self.move_filter_down_button.setToolTip("Apply the selected stage later")
        self.add_filter_button.clicked.connect(self.add_filter_stage)
        self.remove_filter_button.clicked.connect(self.remove_filter_stage)
        self.move_filter_up_button.clicked.connect(lambda: self.move_filter_stage(-1))
        self.move_filter_down_button.clicked.connect(lambda: self.move_filter_stage(1))
        for button in (
            self.add_filter_button,
            self.remove_filter_button,
            self.move_filter_up_button,
            self.move_filter_down_button,
        ):
            filter_stage_actions.addWidget(button, 1)
        filter_layout.addLayout(filter_stage_actions)
        filter_kind_row = QHBoxLayout()
        filter_kind_row.addWidget(QLabel("Method:"))
        self.filter_kind = QComboBox()
        self.filter_kind.addItem("None", "none")
        self.filter_kind.addItem("Savitzky-Golay", "savgol")
        self.filter_kind.addItem("Modified sinc", "modified_sinc")
        self.filter_kind.addItem("Butterworth low-pass (zero phase)", "lowpass")
        self.filter_kind.addItem("IIR notch (zero phase)", "notch")
        self.filter_kind.addItem("FFT band-stop", "fft_bandstop")
        self.filter_kind.addItem("Hampel filter", "hampel")
        self.filter_kind.addItem("Current-maxima Makima envelope", "upper_envelope")
        filter_kind_row.addWidget(self.filter_kind, 1)
        filter_layout.addLayout(filter_kind_row)

        filter_domain_row = QHBoxLayout()
        filter_domain_row.addWidget(QLabel("Apply to:"))
        self.filter_domain = QComboBox()
        self.filter_domain.addItem("Base current", "current")
        self.filter_domain.addItem("Semi-derivative", "semiderivative")
        filter_domain_row.addWidget(self.filter_domain, 1)
        filter_layout.addLayout(filter_domain_row)

        filter_scope_row = QHBoxLayout()
        filter_scope_row.addWidget(QLabel("Range:"))
        self.filter_scope = QComboBox()
        self.filter_scope.addItem("Global per sweep", "global")
        self.filter_scope.addItem("Selected range", "regional")
        filter_scope_row.addWidget(self.filter_scope, 1)
        filter_layout.addLayout(filter_scope_row)
        self.filter_range_button = QPushButton("Set voltage range")
        self.filter_range_button.clicked.connect(self.begin_filter_range)
        filter_layout.addWidget(self.filter_range_button)
        self.filter_range_label = wrapped_label("No range selected")
        filter_layout.addWidget(self.filter_range_label)

        self.filter_advanced_button = QPushButton("Show method settings")
        self.filter_advanced_button.setCheckable(True)
        self.filter_advanced_button.toggled.connect(self._update_filter_controls)
        filter_layout.addWidget(self.filter_advanced_button)
        self.filter_advanced_panel = QWidget()
        advanced_layout = QVBoxLayout(self.filter_advanced_panel)
        advanced_layout.setContentsMargins(0, 0, 0, 0)
        advanced_layout.setSpacing(2)

        self.filter_frequency_row = QWidget()
        frequency_row = QHBoxLayout(self.filter_frequency_row)
        frequency_row.setContentsMargins(0, 0, 0, 0)
        self.filter_frequency_label = QLabel("Frequency:")
        frequency_row.addWidget(self.filter_frequency_label)
        self.filter_frequency = QDoubleSpinBox()
        self.filter_frequency.setRange(0.000001, 1000000.0)
        self.filter_frequency.setDecimals(4)
        self.filter_frequency.setValue(5.0)
        self.filter_frequency.setSuffix(" Hz")
        frequency_row.addWidget(self.filter_frequency, 1)
        advanced_layout.addWidget(self.filter_frequency_row)

        self.filter_bandwidth_row = QWidget()
        bandwidth_row = QHBoxLayout(self.filter_bandwidth_row)
        bandwidth_row.setContentsMargins(0, 0, 0, 0)
        self.filter_bandwidth_label = QLabel("Bandwidth:")
        bandwidth_row.addWidget(self.filter_bandwidth_label)
        self.filter_bandwidth = QDoubleSpinBox()
        self.filter_bandwidth.setRange(0.000001, 1000000.0)
        self.filter_bandwidth.setDecimals(4)
        self.filter_bandwidth.setValue(1.0)
        self.filter_bandwidth.setSuffix(" Hz")
        bandwidth_row.addWidget(self.filter_bandwidth, 1)
        advanced_layout.addWidget(self.filter_bandwidth_row)

        self.filter_attenuation_row = QWidget()
        attenuation_row = QHBoxLayout(self.filter_attenuation_row)
        attenuation_row.setContentsMargins(0, 0, 0, 0)
        attenuation_row.addWidget(QLabel("Attenuation:"))
        self.filter_attenuation = QDoubleSpinBox()
        self.filter_attenuation.setRange(1.0, 120.0)
        self.filter_attenuation.setDecimals(1)
        self.filter_attenuation.setValue(10.0)
        self.filter_attenuation.setSuffix(" dB")
        attenuation_row.addWidget(self.filter_attenuation, 1)
        advanced_layout.addWidget(self.filter_attenuation_row)

        self.filter_window_row = QWidget()
        window_row = QHBoxLayout(self.filter_window_row)
        window_row.setContentsMargins(0, 0, 0, 0)
        window_row.addWidget(QLabel("Window width:"))
        self.filter_window = QDoubleSpinBox()
        self.filter_window.setRange(0.001, 1000.0)
        self.filter_window.setDecimals(2)
        self.filter_window.setValue(10.0)
        self.filter_window.setSuffix(" mV")
        window_row.addWidget(self.filter_window, 1)
        advanced_layout.addWidget(self.filter_window_row)

        self.filter_shape_row = QWidget()
        shape_row = QHBoxLayout(self.filter_shape_row)
        shape_row.setContentsMargins(0, 0, 0, 0)
        self.filter_shape_label = QLabel("Degree / order:")
        shape_row.addWidget(self.filter_shape_label)
        self.filter_polynomial = QSpinBox()
        self.filter_polynomial.setRange(1, 10)
        self.filter_polynomial.setValue(4)
        shape_row.addWidget(self.filter_polynomial)
        shape_row.addStretch(1)
        advanced_layout.addWidget(self.filter_shape_row)

        self.filter_harmonics_row = QWidget()
        harmonics_row = QHBoxLayout(self.filter_harmonics_row)
        harmonics_row.setContentsMargins(0, 0, 0, 0)
        harmonics_row.addWidget(QLabel("Harmonics:"))
        self.filter_harmonics = QSpinBox()
        self.filter_harmonics.setRange(1, 8)
        self.filter_harmonics.setValue(1)
        harmonics_row.addWidget(self.filter_harmonics, 1)
        advanced_layout.addWidget(self.filter_harmonics_row)

        self.filter_robust_row = QWidget()
        robust_row = QHBoxLayout(self.filter_robust_row)
        robust_row.setContentsMargins(0, 0, 0, 0)
        robust_row.addWidget(QLabel("Hampel threshold:"))
        self.filter_hampel_sigma = QDoubleSpinBox()
        self.filter_hampel_sigma.setRange(1.0, 12.0)
        self.filter_hampel_sigma.setDecimals(1)
        self.filter_hampel_sigma.setValue(3.5)
        self.filter_hampel_sigma.setSuffix(" σ")
        robust_row.addWidget(self.filter_hampel_sigma, 1)
        advanced_layout.addWidget(self.filter_robust_row)

        self.filter_prominence_row = QWidget()
        prominence_row = QHBoxLayout(self.filter_prominence_row)
        prominence_row.setContentsMargins(0, 0, 0, 0)
        prominence_row.addWidget(QLabel("Minimum prominence:"))
        self.filter_envelope_prominence = QDoubleSpinBox()
        self.filter_envelope_prominence.setRange(0.0, 100.0)
        self.filter_envelope_prominence.setDecimals(1)
        self.filter_envelope_prominence.setValue(10.0)
        self.filter_envelope_prominence.setSuffix(" %")
        self.filter_envelope_prominence.setToolTip(
            "Minimum local-maximum prominence as a percentage of the detrended oscillation span"
        )
        prominence_row.addWidget(self.filter_envelope_prominence, 1)
        advanced_layout.addWidget(self.filter_prominence_row)

        self.filter_feather_row = QWidget()
        feather_row = QHBoxLayout(self.filter_feather_row)
        feather_row.setContentsMargins(0, 0, 0, 0)
        feather_row.addWidget(QLabel("Regional feather:"))
        self.filter_feather = QDoubleSpinBox()
        self.filter_feather.setRange(0.0, 500.0)
        self.filter_feather.setDecimals(2)
        self.filter_feather.setValue(10.0)
        self.filter_feather.setSuffix(" mV")
        feather_row.addWidget(self.filter_feather, 1)
        advanced_layout.addWidget(self.filter_feather_row)
        filter_layout.addWidget(self.filter_advanced_panel)

        self.filter_density_label = wrapped_label(
            "Sampling: load a CV to view point density"
        )
        self.filter_density_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        filter_layout.addWidget(self.filter_density_label)

        self.filter_response_label = wrapped_label("Filtering off")
        self.filter_response_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        filter_layout.addWidget(self.filter_response_label)
        self.filter_group = filter_group
        layout.addWidget(filter_group)
        layout.addStretch(1)

        for control in (
            self.filter_kind,
            self.filter_domain,
            self.filter_frequency,
            self.filter_bandwidth,
            self.filter_attenuation,
            self.filter_window,
            self.filter_polynomial,
            self.filter_harmonics,
            self.filter_hampel_sigma,
            self.filter_envelope_prominence,
            self.filter_feather,
        ):
            if isinstance(control, QComboBox):
                control.currentIndexChanged.connect(self._filter_changed)
            else:
                control.valueChanged.connect(self._filter_changed)
        self.filter_scope.currentIndexChanged.connect(self._filter_scope_changed)
        self._refresh_filter_stage_selector()
        self._update_filter_controls()

        fit_group = QGroupBox("4. Fit")
        fit_layout = QVBoxLayout(fit_group)
        compact_layout(fit_layout)
        exponent_row = QHBoxLayout()
        exponent_row.addWidget(QLabel("Pearson-IV m minimum:"))
        self.exponent_min = QDoubleSpinBox()
        self.exponent_min.setRange(0.5, 100.0)
        self.exponent_min.setDecimals(3)
        self.exponent_min.setSingleStep(0.1)
        self.exponent_min.setValue(0.5)
        self.exponent_min.valueChanged.connect(self.schedule_fit)
        exponent_row.addWidget(self.exponent_min)
        fit_layout.addLayout(exponent_row)
        center_row = QHBoxLayout()
        center_row.addWidget(QLabel("Peak center window:"))
        self.center_window = QDoubleSpinBox()
        self.center_window.setRange(0.01, 1.0)
        self.center_window.setDecimals(3)
        self.center_window.setValue(0.25)
        self.center_window.setSuffix(" V")
        self.center_window.valueChanged.connect(self.schedule_fit)
        center_row.addWidget(self.center_window)
        fit_layout.addLayout(center_row)
        self.fit_quality_label = wrapped_label("No fit yet")
        self.fit_quality_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        fit_layout.addWidget(self.fit_quality_label)
        self.qc_details_button = QPushButton("QC details (0)")
        self._qc_warning_count = 0
        self.qc_details_button.setCheckable(True)
        self.qc_details_button.setEnabled(False)
        self.qc_details_button.toggled.connect(self._toggle_qc_details)
        fit_layout.addWidget(self.qc_details_button)
        self.qc_details_label = wrapped_label("")
        self.qc_details_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.qc_details_label.hide()
        fit_layout.addWidget(self.qc_details_label)
        copy_row = QHBoxLayout()
        self.copy_window_button = QPushButton("Copy window")
        self.copy_window_button.clicked.connect(self.copy_whole_window)
        self.copy_table_button = QPushButton("Copy table")
        self.copy_table_button.clicked.connect(self.copy_peak_table)
        self.copy_table_button.setEnabled(False)
        copy_row.addWidget(self.copy_window_button, 1)
        copy_row.addWidget(self.copy_table_button, 1)
        self.copy_button_row = copy_row
        fit_layout.addLayout(copy_row)
        self.fit_group = fit_group
        layout.addWidget(fit_group)
        layout.addStretch(1)

        output_group = QGroupBox("5. Export")
        output_layout = QVBoxLayout(output_group)
        compact_layout(output_layout)
        self.output_button = QPushButton("Choose output folder")
        self.output_button.clicked.connect(self.select_output_directory)
        self.output_label = wrapped_label("No output folder selected")
        self.output_label.setMaximumWidth(385)
        output_layout.addWidget(self.output_button)
        output_layout.addWidget(self.output_label)
        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Output name:"))
        self.output_name = QLineEdit("makima_fit")
        name_row.addWidget(self.output_name)
        output_layout.addLayout(name_row)
        self.detailed_csv = QCheckBox("Include detailed fit-trace CSV")
        self.detailed_csv.setChecked(True)
        output_layout.addWidget(self.detailed_csv)
        self.export_button = QPushButton("Export PDF and CSV")
        self.export_button.clicked.connect(self.finalize_export)
        self.export_button.setEnabled(False)
        output_layout.addWidget(self.export_button)
        self.output_group = output_group
        layout.addWidget(output_group)
        layout.addStretch(1)

        self.hotkey_button = QPushButton("Keyboard shortcuts")
        self.hotkey_button.clicked.connect(self.show_hotkeys)
        layout.addWidget(self.hotkey_button)
        layout.addStretch(1)
        scroll.setWidget(panel)
        scroll.setMinimumWidth(375)
        scroll.setMaximumWidth(490)
        return scroll

    def _build_canvas(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(6, 6, 6, 6)
        self.figure = Figure(figsize=(11, 8), constrained_layout=True)
        self.figure.get_layout_engine().set(rect=FOUR_PLOT_LAYOUT_RECT)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setFocusPolicy(Qt.StrongFocus)
        self.toolbar = SplineNavigationToolbar(self.canvas, widget)
        self.toolbar.set_application_home_callback(self.home_all)
        self.toolbar.setToolButtonStyle(Qt.ToolButtonIconOnly)
        move_action = self.toolbar._actions.get("pan")
        if move_action is not None:
            move_action.setText("Move")
            move_action.setToolTip("Move plots")
            move_action.setStatusTip("Move plots")
        self.toolbar.addSeparator()
        self.measure_button = QPushButton("Δt ↔ f")
        self.measure_button.setCheckable(True)
        self.measure_button.setToolTip(
            "Select two plotted data points; shows their time separation and reciprocal frequency (T)"
        )
        self.measure_button.toggled.connect(self.toggle_interval_measurement)
        self.toolbar.addWidget(self.measure_button)
        self.measure_result_label = QLabel("  Δt: —   f: —")
        self.measure_result_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.measure_result_label.setToolTip(
            "Frequency is 1/Δt, useful when the two points are consecutive oscillation peaks"
        )
        self.toolbar.addWidget(self.measure_result_label)
        self.axes = self.figure.subplots(2, 2)
        self.canvas.mpl_connect("button_press_event", self.canvas_clicked)
        self.canvas.mpl_connect("button_press_event", self._navigation_press)
        self.canvas.mpl_connect("button_release_event", self._navigation_release)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, 1)
        self.plot_hint = QLabel(
            "Ctrl-click a plot to focus on it; Ctrl-click again or press Esc to return."
        )
        self.plot_hint.setWordWrap(False)
        layout.addWidget(self.plot_hint)
        self._draw_empty()
        return widget

    def _draw_empty(self) -> None:
        self._preserve_plot_limits = False
        self._home_limits = None
        for axis in self.axes.flat:
            axis.clear()
            axis.text(0.5, 0.5, "Load a CV scan", ha="center", va="center", transform=axis.transAxes, color="#777777")
        self.canvas.draw_idle()

    def _prime_keyboard_focus(self) -> None:
        """Give the application a reliable key target before the first click."""
        if not self.isVisible() or QApplication.activeModalWidget() is not None:
            return
        QApplication.setActiveWindow(self)
        self.activateWindow()
        focus = QApplication.focusWidget()
        if focus is None or focus is self or not self.isAncestorOf(focus):
            self.canvas.setFocus(Qt.ShortcutFocusReason)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        QTimer.singleShot(0, self._prime_keyboard_focus)
        QTimer.singleShot(150, self._prime_keyboard_focus)

    def _navigation_press(self, event) -> None:
        if self.toolbar.mode and event.inaxes in self._active_navigation_axes():
            self._navigation_drag_active = True

    def _navigation_release(self, _event) -> None:
        self._navigation_drag_active = False

    def eventFilter(self, watched, event):
        if watched is self and event.type() == QEvent.WindowActivate:
            QTimer.singleShot(0, self._prime_keyboard_focus)
        if event.type() == QEvent.KeyPress and self._handle_hotkey(event):
            return True
        return super().eventFilter(watched, event)

    def closeEvent(self, event) -> None:
        application = QApplication.instance()
        if application is not None:
            application.removeEventFilter(self)
        self._release_session_temp()
        super().closeEvent(event)

    def _handle_hotkey(self, event) -> bool:
        key = event.key()
        modifiers = event.modifiers()
        control = bool(modifiers & Qt.ControlModifier)
        shift = bool(modifiers & Qt.ShiftModifier)
        focus = QApplication.focusWidget()

        if control:
            if key == Qt.Key_Q:
                self.close()
                return True
            if key == Qt.Key_O:
                self.select_file()
                return True
            if key == Qt.Key_S:
                self.save_session(force_dialog=shift)
                return True
            if key == Qt.Key_R:
                self.reset_selections()
                return True
            if key == Qt.Key_M:
                self.toggle_controls_panel()
                return True
            if key == Qt.Key_Backspace:
                if isinstance(focus, (QAbstractSpinBox, QLineEdit, QComboBox)):
                    return False
                self.clear_table_calculation(all_columns=True)
                return True
            if key in (Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down):
                if self.scan_number.hasFocus() or focus is self.scan_number.lineEdit():
                    step = 5 if key in (Qt.Key_Right, Qt.Key_Up) else -5
                    self.scan_number.stepBy(step)
                    return True
                if isinstance(focus, (QAbstractSpinBox, QLineEdit, QComboBox)):
                    return False
                self.move_all(key, fast=True)
                return True
            if key == Qt.Key_C:
                self.copy_whole_window()
                return True
            if key == Qt.Key_T:
                self.copy_peak_table()
                return True
            if key == Qt.Key_Z and shift:
                self.redo_selection()
                return True
            if key == Qt.Key_Z:
                self.undo_selection()
                return True
            if key == Qt.Key_Y:
                self.redo_selection()
                return True
            if key in (Qt.Key_Equal, Qt.Key_Plus):
                self.zoom_all(0.80)
                return True
            if key in (Qt.Key_Minus, Qt.Key_Underscore):
                self.zoom_all(1.25)
                return True
            return False

        if modifiers & (Qt.AltModifier | Qt.MetaModifier):
            return False

        if key in (Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down):
            if self.scan_number.hasFocus() or focus is self.scan_number.lineEdit():
                step = 1 if key in (Qt.Key_Right, Qt.Key_Up) else -1
                self.scan_number.stepBy(step)
                return True
            if isinstance(focus, (QAbstractSpinBox, QLineEdit, QComboBox)):
                return False
            self.move_all(key, fast=shift)
            return True

        if isinstance(focus, (QAbstractSpinBox, QLineEdit)):
            return False
        if event.isAutoRepeat():
            return False

        calculator_actions = {
            Qt.Key_Plus: "add",
            Qt.Key_Minus: "subtract",
            Qt.Key_Asterisk: "multiply",
            Qt.Key_Slash: "divide",
        }
        if key in calculator_actions:
            self.calculate_table_selection(calculator_actions[key])
            return True
        if key == Qt.Key_Backspace:
            self.clear_table_calculation(all_columns=False)
            return True

        actions = {
            Qt.Key_H: self.home_all,
            Qt.Key_M: lambda: self.toggle_navigation("move"),
            Qt.Key_Z: lambda: self.toggle_navigation("zoom"),
            Qt.Key_W: self.toggle_window_full_screen,
            Qt.Key_F: self.suggest_peaks,
            Qt.Key_K: lambda: self.set_selection_mode("spline"),
            Qt.Key_B: self.cycle_background_kind,
            Qt.Key_S: self.cycle_background_scope,
            Qt.Key_V: self.toggle_filter_scope,
            Qt.Key_L: self.toggle_knot_lock,
            Qt.Key_P: lambda: self.set_selection_mode("peak"),
            Qt.Key_X: lambda: self.set_selection_mode("transient"),
            Qt.Key_C: self.clear_active_selection,
            Qt.Key_R: self.reset_settings,
            Qt.Key_Q: self.toggle_qc_details_hotkey,
            Qt.Key_E: self.finalize_export,
            Qt.Key_I: self.select_file,
            Qt.Key_N: self.focus_scan_number,
            Qt.Key_T: self.measure_button.toggle,
            Qt.Key_Escape: self.exit_plot_focus_or_navigation,
        }
        action = actions.get(key)
        if action is None:
            return False
        action()
        return True

    def _active_navigation_axes(self) -> list:
        axes = list(self.axes.flat)
        if self._focused_axis_index is not None:
            return [axes[self._focused_axis_index]]
        return [axis for axis in axes if axis.get_visible()]

    def _sync_navigation_axes(self) -> None:
        active = set(self._active_navigation_axes())
        for axis in self.axes.flat:
            axis.set_navigate(axis in active)

    def toggle_navigation(self, mode: str) -> None:
        if self.measure_button.isChecked():
            self.measure_button.setChecked(False)
        self._sync_navigation_axes()
        if mode == "move":
            self.toolbar.pan()
        elif mode == "zoom":
            self.toolbar.zoom()
        self.canvas.setFocus(Qt.ShortcutFocusReason)
        state = str(self.toolbar.mode).lower() or "selection"
        if "pan" in state:
            state = "move"
        self.statusBar().showMessage(f"Plot mode: {state}")

    def toggle_interval_measurement(self, checked: bool) -> None:
        if checked:
            self.cancel_navigation()
            self._measurement_indices = []
            self._measurement_axis_index = None
            self.measure_result_label.setText("  Select point 1 of 2")
            self.statusBar().showMessage(
                "Interval ruler active: click two plotted data points; right-click to clear"
            )
        else:
            self._measurement_indices = []
            self._measurement_axis_index = None
            self.measure_result_label.setText("  Δt: —   f: —")
        self.canvas.setFocus(Qt.ShortcutFocusReason)
        if self.analysis is not None:
            self.draw_live_view()

    def _measurement_series(self, axis, data):
        display_time, _, _ = display_time_axis(data.time)
        ax_sd, ax_norm, ax_cv, ax_resid = self.axes.flat
        if axis is ax_sd:
            return display_time, np.asarray(data.transform, dtype=float)
        if axis is ax_norm:
            if self._top_right_mode == "potential":
                return np.asarray(data.potential, dtype=float), np.asarray(data.transform, dtype=float)
            values = (
                np.asarray(self.outcome.normalized_signal, dtype=float)
                if self.outcome is not None
                else np.asarray(data.transform, dtype=float)
            )
            return display_time, values
        if axis is ax_cv:
            x_values = (
                np.asarray(data.potential, dtype=float)
                if self._bottom_left_mode == "potential"
                else display_time
            )
            return x_values, np.asarray(data.current, dtype=float)
        if axis is ax_resid and self._quality_panel_mode != "table":
            values = np.zeros(len(data.time), dtype=float)
            if self.outcome is not None:
                values = np.asarray(data.transform - self.outcome.lmfit_result.best_fit, dtype=float)
            return display_time, values
        return None

    def _nearest_measurement_index(self, event) -> int | None:
        if self.analysis is None or event.inaxes not in list(self.axes.flat):
            return None
        data = self.working_analysis()
        series = self._measurement_series(event.inaxes, data)
        if series is None:
            return None
        x_values, y_values = series
        finite = np.isfinite(x_values) & np.isfinite(y_values)
        indices = np.flatnonzero(finite)
        if not len(indices):
            return None
        coordinates = event.inaxes.transData.transform(
            np.column_stack((x_values[indices], y_values[indices]))
        )
        distances = (coordinates[:, 0] - event.x) ** 2 + (coordinates[:, 1] - event.y) ** 2
        return int(indices[int(np.argmin(distances))])

    def _measurement_text(self) -> str:
        if self.analysis is None or len(self._measurement_indices) != 2:
            return ""
        first, second = self._measurement_indices
        delta_t = abs(float(self.analysis.time[second] - self.analysis.time[first]))
        delta_e_mv = abs(float(self.analysis.potential[second] - self.analysis.potential[first])) * 1000.0
        frequency = np.inf if delta_t <= np.finfo(float).eps else 1.0 / delta_t
        frequency_text = "∞" if not np.isfinite(frequency) else f"{frequency:.6g} Hz"
        return (
            f"Δt = {delta_t:.6g} s   f = {frequency_text}   "
            f"ΔE = {delta_e_mv:.6g} mV   Δindex = {abs(second - first)}"
        )

    def _handle_interval_measurement_click(self, event) -> None:
        if event.button == 3:
            self._measurement_indices = []
            self._measurement_axis_index = None
            self.measure_result_label.setText("  Select point 1 of 2")
            self.draw_live_view()
            return
        if event.button != 1:
            return
        index = self._nearest_measurement_index(event)
        if index is None:
            self.statusBar().showMessage("Choose a plot containing a data trace")
            return
        if len(self._measurement_indices) == 2:
            self._measurement_indices = []
        self._measurement_indices.append(index)
        self._measurement_axis_index = list(self.axes.flat).index(event.inaxes)
        if len(self._measurement_indices) == 1:
            self.measure_result_label.setText("  Select point 2 of 2")
            self.statusBar().showMessage(
                f"First point selected at t = {self.analysis.time[index]:.6g} s"
            )
        else:
            result = self._measurement_text()
            self.measure_result_label.setText(f"  {result}")
            self.statusBar().showMessage(
                result + "; frequency is the reciprocal point separation"
            )
        self.draw_live_view()

    def cancel_navigation(self) -> None:
        mode = str(self.toolbar.mode).lower()
        if "pan" in mode:
            self.toolbar.pan()
        elif "zoom" in mode:
            self.toolbar.zoom()
        self.canvas.setFocus(Qt.ShortcutFocusReason)
        self.statusBar().showMessage("Plot navigation off; selection clicks enabled")

    def exit_plot_focus_or_navigation(self) -> None:
        if self._focused_axis_index is not None:
            self.toggle_plot_focus(self.axes.flat[self._focused_axis_index])
        else:
            self.cancel_navigation()

    def home_all(self) -> None:
        self.cancel_navigation()
        if self._home_limits is None:
            self.toolbar.matplotlib_home()
        else:
            for axis, (xlim, ylim) in zip(self.axes.flat, self._home_limits):
                axis.set_xlim(xlim)
                axis.set_ylim(ylim)
        self._preserve_plot_limits = True
        self.canvas.draw_idle()
        self.statusBar().showMessage("All plots restored to home view")

    def zoom_all(self, factor: float) -> None:
        if self.analysis is None:
            return
        self.cancel_navigation()
        for axis in self._active_navigation_axes():
            x_min, x_max = axis.get_xlim()
            y_min, y_max = axis.get_ylim()
            x_center = (x_min + x_max) * 0.5
            y_center = (y_min + y_max) * 0.5
            x_half = (x_max - x_min) * factor * 0.5
            y_half = (y_max - y_min) * factor * 0.5
            axis.set_xlim(x_center - x_half, x_center + x_half)
            axis.set_ylim(y_center - y_half, y_center + y_half)
        self._preserve_plot_limits = True
        self.canvas.draw_idle()
        direction = "in" if factor < 1 else "out"
        self.statusBar().showMessage(f"Zoomed {direction} on all plots")

    def _view_is_zoomed(self) -> bool:
        if self._home_limits is None:
            return False
        active = set(self._active_navigation_axes())
        return any(
            axis in active
            and (
                not np.allclose(axis.get_xlim(), xlim)
                or not np.allclose(axis.get_ylim(), ylim)
            )
            for axis, (xlim, ylim) in zip(self.axes.flat, self._home_limits)
        )

    def move_all(self, key: int, fast: bool = False) -> None:
        fraction = 0.25 if fast else 0.10
        for axis in self._active_navigation_axes():
            x_min, x_max = axis.get_xlim()
            y_min, y_max = axis.get_ylim()
            if key in (Qt.Key_Left, Qt.Key_Right):
                offset = (x_max - x_min) * fraction
                if key == Qt.Key_Left:
                    offset *= -1
                axis.set_xlim(x_min + offset, x_max + offset)
            else:
                offset = (y_max - y_min) * fraction
                if key == Qt.Key_Down:
                    offset *= -1
                axis.set_ylim(y_min + offset, y_max + offset)
        self._preserve_plot_limits = True
        self.canvas.draw_idle()
        self.statusBar().showMessage("Plots moved")

    def set_selection_mode(self, mode: str) -> None:
        self.cancel_navigation()
        button = {
            "spline": self.knot_radio,
            "peak": self.peak_radio,
            "transient": self.transient_radio,
        }[mode]
        button.setChecked(True)
        self.canvas.setFocus(Qt.ShortcutFocusReason)
        labels = {
            "spline": "Background-knot selection",
            "peak": "Pearson-IV peak selection",
            "transient": "Transient exclusion: select interval start and end",
        }
        self.statusBar().showMessage(labels[mode])

    def cycle_background_kind(self) -> None:
        self.cancel_navigation()
        self.background_kind.setCurrentIndex(
            (self.background_kind.currentIndex() + 1) % self.background_kind.count()
        )
        self.background_kind.setFocus(Qt.ShortcutFocusReason)
        self.statusBar().showMessage(
            f"Background: {self.background_kind.currentText()}; use the arrow keys to change it"
        )

    def cycle_background_scope(self) -> None:
        self.cancel_navigation()
        self.background_scope.setCurrentIndex(
            (self.background_scope.currentIndex() + 1) % self.background_scope.count()
        )
        self.background_scope.setFocus(Qt.ShortcutFocusReason)
        self.statusBar().showMessage(
            f"Scope: {self.background_scope.currentText()}; use the arrow keys to change it"
        )

    def _filter_editor_controls(self) -> tuple:
        return (
            self.filter_kind,
            self.filter_domain,
            self.filter_scope,
            self.filter_frequency,
            self.filter_bandwidth,
            self.filter_attenuation,
            self.filter_window,
            self.filter_polynomial,
            self.filter_harmonics,
            self.filter_hampel_sigma,
            self.filter_envelope_prominence,
            self.filter_feather,
        )

    def _stage_from_controls(self) -> FilterStage:
        return FilterStage(
            kind=str(self.filter_kind.currentData()),
            domain=str(self.filter_domain.currentData()),
            scope=str(self.filter_scope.currentData()),
            range=self.filter_range,
            frequency_hz=self.filter_frequency.value(),
            bandwidth_hz=self.filter_bandwidth.value(),
            attenuation_db=self.filter_attenuation.value(),
            window_mv=self.filter_window.value(),
            polynomial_order=self.filter_polynomial.value(),
            hampel_sigma=self.filter_hampel_sigma.value(),
            harmonics=self.filter_harmonics.value(),
            feather_mv=self.filter_feather.value(),
            envelope_prominence_pct=self.filter_envelope_prominence.value(),
        )

    def _set_filter_editor_values(self, stage: FilterStage | None) -> None:
        self._loading_filter_stage = True
        for control in self._filter_editor_controls():
            control.blockSignals(True)
        if stage is None:
            self.filter_kind.setCurrentIndex(self.filter_kind.findData("none"))
            self.filter_domain.setCurrentIndex(self.filter_domain.findData("current"))
            self.filter_scope.setCurrentIndex(self.filter_scope.findData("global"))
            self.filter_range = None
        else:
            self.filter_kind.setCurrentIndex(self.filter_kind.findData(stage.kind))
            if stage.kind == "modified_sinc":
                self.filter_polynomial.setRange(2, 10)
                self.filter_polynomial.setSingleStep(2)
            else:
                self.filter_polynomial.setRange(1, 7)
                self.filter_polynomial.setSingleStep(1)
            self.filter_domain.setCurrentIndex(self.filter_domain.findData(stage.domain))
            self.filter_scope.setCurrentIndex(self.filter_scope.findData(stage.scope))
            self.filter_frequency.setValue(stage.frequency_hz)
            self.filter_bandwidth.setValue(stage.bandwidth_hz)
            self.filter_attenuation.setValue(stage.attenuation_db)
            self.filter_window.setValue(stage.window_mv)
            self.filter_polynomial.setValue(stage.polynomial_order)
            self.filter_harmonics.setValue(stage.harmonics)
            self.filter_hampel_sigma.setValue(stage.hampel_sigma)
            self.filter_envelope_prominence.setValue(stage.envelope_prominence_pct)
            self.filter_feather.setValue(stage.feather_mv)
            self.filter_range = stage.range
        for control in self._filter_editor_controls():
            control.blockSignals(False)
        if stage is not None and 0 <= self._active_filter_stage < len(self.filter_stages):
            self.filter_stages[self._active_filter_stage] = self._stage_from_controls()
            stage = self.filter_stages[self._active_filter_stage]
        self.pending_filter_index = None
        self._filter_range_armed = bool(
            stage is not None and stage.scope == "regional" and stage.range is None
        )
        self._loading_filter_stage = False
        self._update_filter_range_label()
        self._update_filter_controls()
        self._update_sampling_density_label()

    def _refresh_filter_stage_selector(self) -> None:
        self.filter_stage_selector.blockSignals(True)
        self.filter_stage_selector.clear()
        if not self.filter_stages:
            self.filter_stage_selector.addItem("No filters")
            self._active_filter_stage = -1
            selected = 0
        else:
            for number, stage in enumerate(self.filter_stages, start=1):
                scope = "global" if stage.scope == "global" else "selected"
                self.filter_stage_selector.addItem(
                    f"{number}. {FILTER_LABELS.get(stage.kind, stage.kind)} · {scope}"
                )
            self._active_filter_stage = min(
                max(self._active_filter_stage, 0), len(self.filter_stages) - 1
            )
            selected = self._active_filter_stage
        self.filter_stage_selector.setCurrentIndex(selected)
        self.filter_stage_selector.blockSignals(False)
        self.remove_filter_button.setEnabled(bool(self.filter_stages))
        self.move_filter_up_button.setEnabled(self._active_filter_stage > 0)
        self.move_filter_down_button.setEnabled(
            0 <= self._active_filter_stage < len(self.filter_stages) - 1
        )

    def _select_filter_stage(self, index: int) -> None:
        if not self.filter_stages:
            self._active_filter_stage = -1
            self._set_filter_editor_values(None)
            return
        self._active_filter_stage = min(max(int(index), 0), len(self.filter_stages) - 1)
        self._set_filter_editor_values(self.filter_stages[self._active_filter_stage])
        self._refresh_filter_stage_selector()

    def add_filter_stage(self) -> None:
        insert_at = self._active_filter_stage + 1 if self.filter_stages else 0
        stage = FilterStage()
        self.filter_stages.insert(insert_at, stage)
        self._active_filter_stage = insert_at
        self._refresh_filter_stage_selector()
        self._set_filter_editor_values(stage)
        self._after_filter_chain_changed()
        self.statusBar().showMessage(f"Filter stage {insert_at + 1} added")

    def remove_filter_stage(self) -> None:
        if not (0 <= self._active_filter_stage < len(self.filter_stages)):
            return
        removed = self._active_filter_stage
        self.filter_stages.pop(removed)
        self._active_filter_stage = min(removed, len(self.filter_stages) - 1)
        self._refresh_filter_stage_selector()
        stage = self.filter_stages[self._active_filter_stage] if self.filter_stages else None
        self._set_filter_editor_values(stage)
        self._after_filter_chain_changed()
        self.statusBar().showMessage("Filter stage removed")

    def move_filter_stage(self, offset: int) -> None:
        source = self._active_filter_stage
        target = source + int(offset)
        if not (0 <= source < len(self.filter_stages) and 0 <= target < len(self.filter_stages)):
            return
        self.filter_stages[source], self.filter_stages[target] = (
            self.filter_stages[target], self.filter_stages[source]
        )
        self._active_filter_stage = target
        self._refresh_filter_stage_selector()
        self._set_filter_editor_values(self.filter_stages[target])
        self._after_filter_chain_changed()
        self.statusBar().showMessage(f"Filter stage moved to position {target + 1}")

    def _update_filter_controls(self) -> None:
        kind = str(self.filter_kind.currentData())
        enabled = bool(self.filter_stages and self._active_filter_stage >= 0 and kind != "none")
        spectral = kind in {"lowpass", "notch", "fft_bandstop"}
        envelope = kind == "upper_envelope"
        advanced = bool(enabled and self.filter_advanced_button.isChecked())
        self.filter_domain.setEnabled(enabled and not envelope)
        self.filter_scope.setEnabled(enabled and not envelope)
        self.filter_range_button.setEnabled(enabled and self.filter_scope.currentData() == "regional")
        self.filter_range_button.setVisible(enabled and self.filter_scope.currentData() == "regional")
        self.filter_range_label.setVisible(enabled and self.filter_scope.currentData() == "regional")
        self.filter_advanced_button.setVisible(enabled)
        self.filter_advanced_button.setText(
            "Hide method settings" if self.filter_advanced_button.isChecked() else "Show method settings"
        )
        self.filter_advanced_panel.setVisible(advanced)
        self.filter_frequency.setEnabled(spectral or envelope)
        self.filter_bandwidth.setEnabled(spectral)
        self.filter_attenuation.setEnabled(spectral)
        self.filter_window.setEnabled(kind in {"modified_sinc", "savgol", "hampel"})
        self.filter_polynomial.setEnabled(kind in {"modified_sinc", "savgol"})
        self.filter_harmonics.setEnabled(kind in {"notch", "fft_bandstop"})
        self.filter_hampel_sigma.setEnabled(kind == "hampel")
        self.filter_envelope_prominence.setEnabled(envelope)
        self.filter_feather.setEnabled(enabled and self.filter_scope.currentData() == "regional")
        self.filter_frequency_row.setVisible(advanced and (spectral or envelope))
        self.filter_bandwidth_row.setVisible(advanced and spectral)
        self.filter_attenuation_row.setVisible(advanced and spectral)
        self.filter_window_row.setVisible(advanced and kind in {"modified_sinc", "savgol", "hampel"})
        self.filter_shape_row.setVisible(advanced and kind in {"modified_sinc", "savgol"})
        self.filter_harmonics_row.setVisible(advanced and kind in {"notch", "fft_bandstop"})
        self.filter_robust_row.setVisible(advanced and kind == "hampel")
        self.filter_prominence_row.setVisible(advanced and envelope)
        self.filter_feather_row.setVisible(
            advanced and enabled and self.filter_scope.currentData() == "regional"
        )
        self.filter_kind.setToolTip("")
        if kind == "lowpass":
            self.filter_frequency_label.setText("Passband edge:")
            self.filter_bandwidth_label.setText("Transition width:")
        elif kind in {"notch", "fft_bandstop"}:
            self.filter_frequency_label.setText("Center frequency:")
            self.filter_bandwidth_label.setText("Stop bandwidth:")
        elif envelope:
            self.filter_frequency_label.setText("Expected oscillation frequency:")
            self.filter_bandwidth_label.setText("Bandwidth:")
            self.filter_kind.setToolTip(
                "Detects oscillation maxima in base current, builds a Makima envelope through "
                "those knots, then semi-differentiates the reconstructed current for fitting"
            )
        else:
            self.filter_frequency_label.setText("Frequency:")
            self.filter_bandwidth_label.setText("Bandwidth:")
        self.filter_polynomial.blockSignals(True)
        if kind == "modified_sinc":
            self.filter_shape_label.setText("Filter degree:")
            self.filter_polynomial.setRange(2, 10)
            self.filter_polynomial.setSingleStep(2)
            value = self.filter_polynomial.value()
            if value % 2:
                self.filter_polynomial.setValue(min(10, value + 1))
            self.filter_polynomial.setToolTip(
                "Modified-sinc filters support even degrees 2–10; 4 is a balanced starting point"
            )
        else:
            self.filter_shape_label.setText("Polynomial order:")
            self.filter_polynomial.setRange(1, 7)
            self.filter_polynomial.setSingleStep(1)
            self.filter_polynomial.setToolTip(
                "Polynomial order for the Savitzky-Golay fit"
                if kind == "savgol"
                else "Used by Modified sinc and Savitzky-Golay filters"
            )
        self.filter_polynomial.blockSignals(False)
        QTimer.singleShot(0, self._resize_controls_to_contents)

    def _update_sampling_density_label(self) -> None:
        if self.analysis is None:
            self.filter_density_label.setText(
                "Sampling: load a CV to view point density"
            )
            return
        densities: list[float] = []
        point_counts: list[int] = []
        kind = str(self.filter_kind.currentData())
        degree = int(self.filter_polynomial.value())
        minimum = 3
        if kind == "savgol":
            minimum = degree + 2
        elif kind == "modified_sinc":
            minimum = 2 * (degree // 2 + 2) + 1
        for start, end in self.analysis.segment_bounds:
            potential = self.analysis.potential[start : end + 1]
            steps_mv = np.abs(np.diff(potential)) * 1000.0
            steps_mv = steps_mv[np.isfinite(steps_mv) & (steps_mv > np.finfo(float).eps)]
            if not len(steps_mv):
                continue
            densities.append(1.0 / float(np.median(steps_mv)))
            point_counts.append(
                _odd_window_from_mv_for_display(
                    potential,
                    self.filter_window.value(),
                    minimum,
                )
            )
        if not densities:
            self.filter_density_label.setText("Sampling density unavailable for this scan")
            return
        if np.ptp(densities) <= max(1e-9, 0.01 * float(np.mean(densities))):
            density_text = f"{float(np.mean(densities)):.4g} points/mV"
        else:
            density_text = f"{min(densities):.4g}–{max(densities):.4g} points/mV across sweeps"
        if min(point_counts) == max(point_counts):
            points_text = f"{point_counts[0]} points"
        else:
            points_text = f"{min(point_counts)}–{max(point_counts)} points"
        mv_per_point = 1.0 / float(np.mean(densities))
        text = f"Sampling: {density_text} · {mv_per_point:.4g} mV/point"
        if kind in {"modified_sinc", "savgol", "hampel"}:
            text += (
                f"\nWindow: {self.filter_window.value():.4g} mV · {points_text}"
            )
        elif kind == "upper_envelope":
            dt = float(np.median(np.diff(self.analysis.time)))
            period_points = 1.0 / max(dt * self.filter_frequency.value(), np.finfo(float).eps)
            text += f"\nPeriod: ~{period_points:.3g} acquisition points"
        self.filter_density_label.setText(text)

    def _invalidate_filter_cache(self) -> None:
        self._filtered_analysis_cache = None
        self._filter_cache_key = None
        self._filter_error = None

    def _after_filter_chain_changed(self) -> None:
        self._update_filter_controls()
        self._update_sampling_density_label()
        self._invalidate_filter_cache()
        filtered = self._filtered_analysis()
        if (
            self.analysis is not None
            and self.lock_knots.isChecked()
            and self.knot_indices
            and self._filter_error is None
        ):
            self.knot_values = {
                index: float(filtered.transform[index]) for index in self.knot_indices
            }
        self.outcome = None
        self.export_button.setEnabled(False)
        self.copy_table_button.setEnabled(False)
        if self.analysis is not None:
            self.schedule_fit()

    def _filter_changed(self, *_args) -> None:
        if self._loading_filter_stage:
            return
        if self.filter_kind.currentData() == "upper_envelope":
            self.filter_domain.blockSignals(True)
            self.filter_scope.blockSignals(True)
            self.filter_domain.setCurrentIndex(0)
            self.filter_scope.setCurrentIndex(1)
            self.filter_domain.blockSignals(False)
            self.filter_scope.blockSignals(False)
            self._filter_range_armed = self.filter_range is None
            self._update_filter_range_label()
        kind = str(self.filter_kind.currentData())
        if self._active_filter_stage < 0 and kind != "none":
            self.filter_stages.append(self._stage_from_controls())
            self._active_filter_stage = 0
        elif 0 <= self._active_filter_stage < len(self.filter_stages):
            if kind == "none":
                self.remove_filter_stage()
                return
            self.filter_stages[self._active_filter_stage] = self._stage_from_controls()
        self._refresh_filter_stage_selector()
        self._after_filter_chain_changed()

    def _filter_scope_changed(self, *_args) -> None:
        regional = self.filter_scope.currentData() == "regional"
        self._filter_range_armed = bool(regional and self.filter_range is None)
        if not regional:
            self.pending_filter_index = None
        self._update_filter_range_label()
        self._filter_changed()
        if regional and self.filter_range is None:
            self.statusBar().showMessage(
                "Regional filter: click two limits on the semi-derivative or current plot"
            )

    def toggle_filter_scope(self) -> None:
        if self.filter_kind.currentData() == "upper_envelope":
            self.begin_filter_range()
            return
        if not self.filter_stages:
            self.add_filter_stage()
        next_index = 1 if self.filter_scope.currentData() == "global" else 0
        self.filter_scope.setCurrentIndex(next_index)
        self.canvas.setFocus(Qt.ShortcutFocusReason)
        self.statusBar().showMessage(
            "Filter range: selected voltage range; click two limits on the upper-left plot"
            if self.filter_scope.currentData() == "regional"
            else "Filter range: global, processed independently on each sweep"
        )

    def begin_filter_range(self) -> None:
        if not self.filter_stages:
            self.add_filter_stage()
        if self.filter_scope.currentData() != "regional":
            self.filter_scope.setCurrentIndex(1)
        self.filter_range = None
        if 0 <= self._active_filter_stage < len(self.filter_stages):
            self.filter_stages[self._active_filter_stage] = dataclass_replace(
                self.filter_stages[self._active_filter_stage],
                scope="regional",
                range=None,
            )
            self._refresh_filter_stage_selector()
        self.pending_filter_index = None
        self._filter_range_armed = True
        self._update_filter_range_label()
        self._invalidate_filter_cache()
        self.outcome = None
        self.canvas.setFocus(Qt.ShortcutFocusReason)
        self.draw_live_view()
        self.statusBar().showMessage(
            "Click the first and second filter limits on the semi-derivative or current plot"
        )

    def _update_filter_range_label(self) -> None:
        if self.analysis is None or self.filter_range is None:
            text = "No range selected"
        else:
            start, end = sorted(self.filter_range)
            text = (
                f"Range: {self.analysis.potential[start]:.5g}–"
                f"{self.analysis.potential[end]:.5g} V · samples {start}–{end}"
            )
        if self.pending_filter_index is not None and self.analysis is not None:
            text = (
                f"First limit: {self.analysis.potential[self.pending_filter_index]:.5g} V · "
                "select the second limit"
            )
        self.filter_range_label.setText(text)

    def _configure_filter_limits(self) -> None:
        if self.analysis is None or len(self.analysis.time) < 2:
            return
        nyquist = 0.5 / float(np.median(np.diff(self.analysis.time)))
        maximum = max(nyquist * 0.999, 0.000002)
        adjusted_stages = []
        for stage in self.filter_stages:
            frequency = stage.frequency_hz
            bandwidth = stage.bandwidth_hz
            if frequency >= maximum:
                frequency = maximum * 0.35
            if frequency + bandwidth >= maximum:
                bandwidth = max(maximum * 0.1, 0.000001)
            adjusted_stages.append(
                dataclass_replace(stage, frequency_hz=frequency, bandwidth_hz=bandwidth)
            )
        self.filter_stages = adjusted_stages
        for control in (self.filter_frequency, self.filter_bandwidth):
            control.blockSignals(True)
            control.setMaximum(maximum)
            control.blockSignals(False)
        if self.filter_frequency.value() >= maximum:
            self.filter_frequency.blockSignals(True)
            self.filter_frequency.setValue(maximum * 0.35)
            self.filter_frequency.blockSignals(False)
        if self.filter_frequency.value() + self.filter_bandwidth.value() >= maximum:
            self.filter_bandwidth.blockSignals(True)
            self.filter_bandwidth.setValue(max(maximum * 0.1, 0.000001))
            self.filter_bandwidth.blockSignals(False)
        self._invalidate_filter_cache()

    def _filter_key(self) -> tuple:
        config = self._config()
        return (id(self.analysis), config.filter_stages)

    def _filtered_analysis(self):
        if self.analysis is None:
            return None
        key = self._filter_key()
        if self._filtered_analysis_cache is not None and key == self._filter_cache_key:
            return self._filtered_analysis_cache
        try:
            filtered = apply_signal_filter(self.analysis, self._config())
            self._filter_error = None
            diagnostics = filtered.filter_diagnostics
            summary = diagnostics.get("summary", "Filtering off")
            stages = list(diagnostics.get("stages", []))
            if not stages and diagnostics.get("kind") in FILTER_LABELS:
                stages = [diagnostics]
            if not stages:
                self.filter_response_label.setText("Filtering off")
                self.filter_response_label.setToolTip("")
            else:
                lines = ["Filters:"]
                for number, details in enumerate(stages, start=1):
                    method = FILTER_LABELS.get(details.get("kind"), "Filter")
                    domain = (
                        "base current"
                        if details.get("domain") == "current"
                        else "semi-derivative"
                    )
                    scope = (
                        "each sweep"
                        if details.get("scope") == "global"
                        else "selected range"
                    )
                    response = str(details.get("summary", "")).partition("; ")[2]
                    lines.append(f"{number}. {method} — {domain}, {scope}")
                    if response:
                        lines.append(f"   {response.replace('; ', ' · ')}")
                self.filter_response_label.setText("\n".join(lines))
                self.filter_response_label.setToolTip(summary)
        except Exception as error:
            filtered = self.analysis
            self._filter_error = str(error)
            self.filter_response_label.setText(f"Filter needs attention:\n{error}")
        self._filtered_analysis_cache = filtered
        self._filter_cache_key = key
        return filtered

    def toggle_knot_lock(self) -> None:
        self.cancel_navigation()
        self.lock_knots.toggle()
        self.lock_knots.setFocus(Qt.ShortcutFocusReason)
        state = "locked to measured data" if self.lock_knots.isChecked() else "unlocked"
        self.statusBar().showMessage(f"Background knots: {state}")

    def _minimum_knots(self) -> int:
        return minimum_background_anchors(str(self.background_kind.currentData()))

    def _background_changed(self) -> None:
        kind = str(self.background_kind.currentData())
        self.knot_radio.setText(
            "CPE fit knots"
            if kind == "cpe"
            else "Constant fit knots"
            if kind == "constant"
            else "Background knots"
        )
        if kind == "cpe":
            self.background_kind.setToolTip(
                "CPE fits Q and alpha to at least two knots on every sweep. "
                "Independent scope fits chronologically; continuous scope fits all sweeps jointly."
            )
        elif kind == "constant":
            self.background_kind.setToolTip(
                "Fits the mean manual-knot level independently per sweep or once across the continuous scan."
            )
        else:
            self.background_kind.setToolTip(
                "Interpolates every manual knot. Makima is the recommended local smooth default."
            )
        self.outcome = None
        self.export_button.setEnabled(False)
        self.copy_table_button.setEnabled(False)
        self.schedule_fit()

    def _knot_lock_changed(self, checked: bool) -> None:
        if checked and self.analysis is not None and self.knot_indices:
            self._remember()
            self.knot_values = {
                index: float(self.working_analysis().transform[index])
                for index in self.knot_indices
            }
        self.outcome = None
        self.export_button.setEnabled(False)
        self.copy_table_button.setEnabled(False)
        self.schedule_fit()

    def _control_click(self, event) -> bool:
        key = str(getattr(event, "key", "") or "").lower()
        gui_event = getattr(event, "guiEvent", None)
        modifiers = gui_event.modifiers() if gui_event is not None and hasattr(gui_event, "modifiers") else Qt.NoModifier
        return "control" in key or "ctrl" in key or bool(modifiers & Qt.ControlModifier)

    def toggle_plot_focus(self, axis) -> None:
        self.cancel_navigation()
        axes = list(self.axes.flat)
        index = axes.index(axis)
        if self._focused_axis_index == index:
            self._focused_axis_index = None
            self.controls_panel.setVisible(self._controls_visible_before_plot_focus)
            self.toolbar.show()
            self.plot_hint.show()
            self.statusBar().show()
            if self._was_maximized_before_plot_focus:
                self.showMaximized()
            else:
                self.showNormal()
            self._restore_four_plot_grid()
            self.statusBar().showMessage("Four-plot view restored")
            return

        if self._focused_axis_index is None:
            self._was_maximized_before_plot_focus = self.isMaximized()
            self._controls_visible_before_plot_focus = self.controls_panel.isVisible()
        self._focused_axis_index = index
        self.figure.set_layout_engine(None)
        for number, item in enumerate(axes):
            item.set_visible(number == index)
            item.set_navigate(number == index)
        axis.set_position([0.12, 0.13, 0.84, 0.79])
        self.controls_panel.hide()
        self.toolbar.hide()
        self.plot_hint.hide()
        self.statusBar().hide()
        self.showFullScreen()
        QTimer.singleShot(0, lambda: self.canvas.setFocus(Qt.ShortcutFocusReason))
        self._apply_focused_plot_typography(axis)
        QTimer.singleShot(
            0,
            lambda: self._apply_focused_plot_typography(axis)
            if self._focused_axis_index == index
            else None,
        )
        self.canvas.draw_idle()

    def toggle_window_full_screen(self) -> None:
        """Toggle the complete application window between full-screen and its prior state."""
        if self._focused_axis_index is not None:
            self.exit_plot_focus_or_navigation()
            return
        if self.isFullScreen():
            if self._was_maximized_before_window_fullscreen:
                self.showMaximized()
            else:
                self.showNormal()
            state = "off"
        else:
            self._was_maximized_before_window_fullscreen = self.isMaximized()
            self.showFullScreen()
            state = "on"
        QTimer.singleShot(0, lambda: self.canvas.setFocus(Qt.ShortcutFocusReason))
        self.statusBar().showMessage(f"Full screen {state}")

    def toggle_controls_panel(self) -> None:
        """Hide or restore the left control menu without changing the plots."""
        if self._focused_axis_index is not None:
            self.statusBar().showMessage("Restore the four-plot view before showing the menu")
            return
        visible = self.controls_panel.isVisible()
        if visible:
            sizes = self.main_splitter.sizes()
            if sizes and sizes[0] > 0:
                self._controls_width_before_hide = int(sizes[0])
            self.controls_panel.hide()
            state = "hidden"
        else:
            self.controls_panel.show()
            total = max(1, sum(self.main_splitter.sizes()))
            width = min(max(375, self._controls_width_before_hide), max(375, total - 300))
            self.main_splitter.setSizes([width, max(1, total - width)])
            state = "shown"
        self.canvas.setFocus(Qt.ShortcutFocusReason)
        self.statusBar().showMessage(f"Control menu {state}")

    def toggle_qc_details_hotkey(self) -> None:
        if not self.qc_details_button.isEnabled():
            self.statusBar().showMessage("No QC warnings for the current fit")
            return
        self.qc_details_button.toggle()
        self.canvas.setFocus(Qt.ShortcutFocusReason)

    def _apply_focused_plot_typography(self, axis) -> None:
        """Scale plot text for a single panel occupying the full display."""
        scale = float(
            np.clip(
                min(self.canvas.width() / 1000.0, self.canvas.height() / 650.0),
                1.25,
                1.75,
            )
        )
        axis.title.set_fontsize(12.0 * scale)
        axis.xaxis.label.set_fontsize(10.0 * scale)
        axis.yaxis.label.set_fontsize(10.0 * scale)
        axis.tick_params(axis="both", labelsize=8.5 * scale)
        legend = axis.get_legend()
        if legend is not None:
            for text_item in legend.get_texts():
                text_item.set_fontsize(7.0 * scale)
        for text_item in axis.texts:
            text_item.set_fontsize(max(float(text_item.get_fontsize()), 8.0 * scale))
        for table in axis.tables:
            table.set_fontsize(9.0 * scale)
        self.canvas.draw_idle()

    def _restore_four_plot_grid(self) -> None:
        """Recreate the subplot grid after a plot used full-window geometry."""
        self._navigation_drag_active = False
        self.figure.clear()
        self.figure.set_layout_engine("constrained")
        self.figure.get_layout_engine().set(rect=FOUR_PLOT_LAYOUT_RECT)
        self.axes = self.figure.subplots(2, 2)
        for axis in self.axes.flat:
            axis.set_navigate(True)
        self.toolbar.update()
        self._preserve_plot_limits = False
        self._home_limits = None
        self.draw_live_view()
        self.toolbar.push_current()
        self.canvas.setFocus(Qt.ShortcutFocusReason)

    def toggle_quality_panel(self) -> None:
        self._quality_panel_mode = (
            "table" if self._quality_panel_mode == "residual" else "residual"
        )
        self.draw_live_view()
        label = "peak table" if self._quality_panel_mode == "table" else "residuals"
        self.statusBar().showMessage(f"Lower-right: {label}")

    def _clear_table_calculator(self) -> None:
        self._table_selected_cells = []
        self._table_calculation = {}
        self._table_calculation_symbols = {}
        self._table_active_column = None

    def clear_table_calculation(self, *, all_columns: bool) -> None:
        if self._quality_panel_mode != "table":
            return
        column = self._table_active_column
        if all_columns:
            if not self._table_selected_cells and not self._table_calculation:
                return
            self._clear_table_calculator()
            message = "Cleared the complete calculator row"
        elif column in {1, 2, 3, 4}:
            self._table_selected_cells = [
                coordinates
                for coordinates in self._table_selected_cells
                if coordinates[1] != column
            ]
            self._table_calculation.pop(column, None)
            self._table_calculation_symbols.pop(column, None)
            message = "Cleared the active calculator column"
        else:
            self.statusBar().showMessage("Select a calculator column to clear")
            return
        self.draw_live_view()
        self.statusBar().showMessage(message)

    def _table_cell_from_event(self, event) -> tuple[int, int] | None:
        event_x = getattr(event, "x", None)
        event_y = getattr(event, "y", None)
        if self._live_peak_table is None or event_x is None or event_y is None:
            return None
        renderer = self.canvas.get_renderer()
        for coordinates, cell in self._live_peak_table.get_celld().items():
            if cell.get_window_extent(renderer).contains(float(event_x), float(event_y)):
                return coordinates
        return None

    def _handle_table_cell_click(self, event) -> bool:
        coordinates = self._table_cell_from_event(event)
        if coordinates is None:
            return False
        row, column = coordinates
        peak_count = min(9, len(self.outcome.peak_results)) if self.outcome is not None else 0
        if event.button == 3:
            self._clear_table_calculator()
            self.draw_live_view()
            self.statusBar().showMessage("Peak-table calculator selection cleared")
            return True
        if event.button != 1 or column not in {1, 2, 3, 4}:
            return False
        self._table_active_column = column
        if row == LIVE_TABLE_ROWS:
            self.draw_live_view()
            self.statusBar().showMessage(
                "Calculator column active; use +, −, ×, ÷, or Backspace"
            )
            return True
        if not 1 <= row <= peak_count:
            return False
        if coordinates in self._table_selected_cells:
            self._table_selected_cells.remove(coordinates)
        else:
            self._table_selected_cells.append(coordinates)
        self._table_calculation.pop(column, None)
        self._table_calculation_symbols.pop(column, None)
        self.draw_live_view()
        selected_in_column = sum(
            selected_column == column
            for _, selected_column in self._table_selected_cells
        )
        self.statusBar().showMessage(
            f"Selected {selected_in_column} value(s) in this column; use +, −, ×, or ÷"
        )
        return True

    def calculate_table_selection(self, operation: str) -> None:
        if self._quality_panel_mode != "table" or self.outcome is None:
            self.statusBar().showMessage("Open the peak table and select values first")
            return
        column = self._table_active_column
        valid = [
            (row, selected_column)
            for row, selected_column in self._table_selected_cells
            if selected_column == column
            and selected_column in {1, 2, 3, 4}
            and 1 <= row <= min(9, len(self.outcome.peak_results))
        ]
        if not valid:
            self.statusBar().showMessage(
                "Select time, potential, current, or charge cells in one column first"
            )
            return
        values = []
        for row, _ in valid:
            peak = self.outcome.peak_results[row - 1]
            if column == 1:
                _, _, time_scale = display_time_axis(self.outcome.data.time)
                values.append(peak.time_s * time_scale)
            elif column == 2:
                values.append(peak.potential_v)
            elif column == 3:
                values.append(peak.current_scaled)
            else:
                charge_factor, _, _ = peak_charge_display_scale(self.outcome)
                values.append(peak.charge_scaled * charge_factor)
        result = float(values[0])
        if operation == "add":
            result = float(np.sum(values))
        elif operation == "subtract":
            for subtrahend in values[1:]:
                result -= float(subtrahend)
        elif operation == "multiply":
            result = float(np.prod(values))
        elif operation == "divide":
            for divisor in values[1:]:
                if abs(float(divisor)) <= np.finfo(float).eps:
                    self.statusBar().showMessage("Cannot divide by a selected zero value")
                    return
                result /= float(divisor)
        else:
            raise ValueError(f"Unknown peak-table calculation: {operation}")
        symbols = {"add": "+", "subtract": "−", "multiply": "×", "divide": "÷"}
        names = {"add": "Addition", "subtract": "Subtraction", "multiply": "Multiplication", "divide": "Division"}
        self._table_calculation[column] = format_significant(result)
        self._table_calculation_symbols[column] = symbols[operation]
        self.draw_live_view()
        self.statusBar().showMessage(
            f"{names[operation]} shown in the active calculator cell"
        )

    def toggle_top_right_panel(self) -> None:
        self._top_right_mode = (
            "potential" if self._top_right_mode == "normalized" else "normalized"
        )
        self.draw_live_view()
        label = "semi-derivative vs potential" if self._top_right_mode == "potential" else "normalized peaks"
        self.statusBar().showMessage(f"Upper-right: {label}")

    def toggle_bottom_left_panel(self) -> None:
        self._bottom_left_mode = (
            "time" if self._bottom_left_mode == "potential" else "potential"
        )
        self.draw_live_view()
        label = "current reconstruction vs time" if self._bottom_left_mode == "time" else "CV reconstruction"
        self.statusBar().showMessage(f"Lower-left: {label}")

    def working_analysis(self):
        if self.analysis is None:
            return None
        return apply_transient_exclusions(self._filtered_analysis(), self.transient_intervals)

    def copy_peak_table(self) -> None:
        if self.outcome is None:
            self.statusBar().showMessage("A valid live fit is needed before copying the peak table")
            return
        frame = formatted_peak_table_frame(self.outcome, ascii_units=True)
        QApplication.clipboard().setText(frame.to_csv(sep="\t", index=False, lineterminator="\n"))
        self.statusBar().showMessage(
            f"Copied {len(frame)} peak row(s) as tab-separated spreadsheet data"
        )

    def copy_whole_window(self) -> None:
        """Copy the complete current application view as a clipboard image."""
        self.canvas.draw()
        QApplication.processEvents()
        screen = self.screen() or QApplication.primaryScreen()
        pixmap = screen.grabWindow(int(self.winId())) if screen is not None else self.grab()
        if pixmap.isNull():
            pixmap = self.grab()
        if pixmap.isNull():
            self.statusBar().showMessage("Could not capture the Makima Fitter window")
            return
        QApplication.clipboard().setPixmap(pixmap)
        self.statusBar().showMessage(
            f"Copied the whole Makima Fitter window ({pixmap.width()} × {pixmap.height()} px)"
        )

    def clear_active_selection(self) -> None:
        if self.selection_group.checkedButton() is self.knot_radio:
            self.clear_knots()
        elif self.selection_group.checkedButton() is self.peak_radio:
            self.clear_peaks()
        else:
            self.clear_transients()

    def reset_settings(self) -> None:
        """Restore analysis defaults and auto rate while preserving selected features."""
        if self.analysis is None:
            return
        reload_for_auto_rate = self.scan_rate.value() != 0.0
        retained_stages = [
            FilterStage(
                kind=stage.kind,
                scope=stage.scope,
                range=stage.range,
            )
            for stage in self.filter_stages
        ]
        for control in (
            self.background_kind,
            self.background_scope,
            self.lock_knots,
            self.exponent_min,
            self.center_window,
            self.filter_advanced_button,
            self.scan_rate,
        ):
            control.blockSignals(True)
        self.background_kind.setCurrentIndex(self.background_kind.findData("makima"))
        self.background_scope.setCurrentIndex(self.background_scope.findData("independent"))
        self.lock_knots.setChecked(True)
        self.exponent_min.setValue(0.5)
        self.center_window.setValue(0.25)
        self.filter_advanced_button.setChecked(False)
        self.scan_rate.setValue(0.0)
        for control in (
            self.background_kind,
            self.background_scope,
            self.lock_knots,
            self.exponent_min,
            self.center_window,
            self.filter_advanced_button,
            self.scan_rate,
        ):
            control.blockSignals(False)
        self.knot_radio.setChecked(True)
        self.knot_radio.setText("Background knots")
        self.filter_stages = retained_stages
        self._active_filter_stage = min(
            max(self._active_filter_stage, 0), len(self.filter_stages) - 1
        )
        self._configure_filter_limits()
        self._refresh_filter_stage_selector()
        active_stage = (
            self.filter_stages[self._active_filter_stage]
            if self.filter_stages
            else None
        )
        self._set_filter_editor_values(active_stage)
        self.filter_advanced_panel.hide()
        self._invalidate_filter_cache()
        if self.knot_indices:
            working = self.working_analysis()
            self.knot_values = {
                index: float(working.transform[index])
                for index in self.knot_indices
            }
        self.outcome = None
        self.export_button.setEnabled(False)
        self.copy_table_button.setEnabled(False)
        self.qc_details_button.setChecked(False)
        if reload_for_auto_rate:
            self._store_active_scan_state()
            self.load_selected_scan()
        else:
            self.schedule_fit()
        self.statusBar().showMessage(
            "Settings restored and scan rate set to Auto; selected features retained"
        )

    def reset_selections(self) -> None:
        if self.analysis is None:
            return
        if (
            self.knot_indices
            or self.peak_indices
            or self.transient_intervals
            or self.pending_transient_index is not None
            or self.filter_stages
        ):
            self._remember()
        self.knot_indices = []
        self.knot_values = {}
        self.peak_indices = []
        self.transient_intervals = []
        self.pending_transient_index = None
        self.filter_range = None
        self.pending_filter_index = None
        self._filter_range_armed = False
        self.filter_stages = []
        self._active_filter_stage = -1
        self.outcome = None
        self.export_button.setEnabled(False)
        self.copy_table_button.setEnabled(False)
        self._quality_panel_mode = "table"
        self._clear_table_calculator()
        self._top_right_mode = "normalized"
        self._bottom_left_mode = "potential"
        self.knot_radio.setChecked(True)
        self.background_kind.blockSignals(True)
        self.background_scope.blockSignals(True)
        self.lock_knots.blockSignals(True)
        self.background_kind.setCurrentIndex(0)
        self.background_scope.setCurrentIndex(0)
        self.lock_knots.setChecked(True)
        self.background_kind.blockSignals(False)
        self.background_scope.blockSignals(False)
        self.lock_knots.blockSignals(False)
        self.exponent_min.blockSignals(True)
        self.center_window.blockSignals(True)
        self.exponent_min.setValue(0.5)
        self.center_window.setValue(0.25)
        self.exponent_min.blockSignals(False)
        self.center_window.blockSignals(False)
        filter_controls = self._filter_editor_controls()
        for control in filter_controls:
            control.blockSignals(True)
        self.filter_kind.setCurrentIndex(0)
        self.filter_domain.setCurrentIndex(0)
        self.filter_scope.setCurrentIndex(0)
        self.filter_frequency.setValue(min(5.0, self.filter_frequency.maximum()))
        self.filter_bandwidth.setValue(min(1.0, self.filter_bandwidth.maximum()))
        self.filter_attenuation.setValue(10.0)
        self.filter_window.setValue(10.0)
        self.filter_polynomial.setValue(4)
        self.filter_harmonics.setValue(1)
        self.filter_hampel_sigma.setValue(3.5)
        self.filter_envelope_prominence.setValue(10.0)
        self.filter_feather.setValue(10.0)
        for control in filter_controls:
            control.blockSignals(False)
        self.filter_advanced_button.blockSignals(True)
        self.filter_advanced_button.setChecked(False)
        self.filter_advanced_button.blockSignals(False)
        self._refresh_filter_stage_selector()
        self._update_filter_controls()
        self._update_sampling_density_label()
        self._update_filter_range_label()
        self._invalidate_filter_cache()
        self.home_all()
        self.schedule_fit()
        self.statusBar().showMessage("Selections, fit settings, and plot view reset")

    def focus_scan_number(self) -> None:
        if not self.scan_number.isEnabled():
            self.statusBar().showMessage("The current file contains one complete scan")
            return
        self.scan_number.setFocus(Qt.ShortcutFocusReason)
        self.scan_number.selectAll()
        self.statusBar().showMessage("Scan number active; use the arrow keys to change scans")

    def select_file(self) -> None:
        start = str(Path(self._source_display_path).parent) if self._source_display_path else ""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select CV or Makima Fitter session",
            start,
            "CV data and sessions (*.mpt *.txt *.csv *.makima);;"
            "Makima Fitter sessions (*.makima);;Supported CV files (*.mpt *.txt *.csv);;"
            "All files (*.*)",
        )
        if path:
            if Path(path).suffix.lower() == SESSION_EXTENSION:
                self.load_session_path(path)
            else:
                self._release_session_temp()
                self.session_file = None
                self.load_input_path(path)

    def _release_session_temp(self) -> None:
        temporary = self._session_temp_dir
        self._session_temp_dir = None
        if temporary is not None:
            try:
                temporary.cleanup()
            except OSError:
                pass

    @staticmethod
    def _serialize_selection_snapshot(snapshot: dict) -> dict:
        return {
            "knots": [int(value) for value in snapshot.get("knots", [])],
            "knot_values": {
                str(int(index)): float(value)
                for index, value in snapshot.get("knot_values", {}).items()
            },
            "peaks": [int(value) for value in snapshot.get("peaks", [])],
            "transients": [
                [int(start), int(stop)]
                for start, stop in snapshot.get("transients", [])
            ],
            "pending": snapshot.get("pending"),
            "filter_range": list(snapshot["filter_range"])
            if snapshot.get("filter_range") is not None
            else None,
            "pending_filter": snapshot.get("pending_filter"),
            "filter_stages": [asdict(stage) for stage in snapshot.get("filter_stages", [])],
            "active_filter_stage": int(snapshot.get("active_filter_stage", -1)),
        }

    @staticmethod
    def _deserialize_selection_snapshot(payload: dict) -> dict:
        if not isinstance(payload, dict):
            raise ValueError("A saved scan state is invalid")
        allowed_stage_fields = set(FilterStage.__dataclass_fields__)
        stages = []
        for item in payload.get("filter_stages", []):
            if not isinstance(item, dict):
                raise ValueError("A saved filter stage is invalid")
            values = {key: value for key, value in item.items() if key in allowed_stage_fields}
            if values.get("range") is not None:
                values["range"] = tuple(int(value) for value in values["range"])
            stages.append(FilterStage(**values))

        def optional_index(value):
            return None if value is None else int(value)

        saved_range = payload.get("filter_range")
        return {
            "knots": [int(value) for value in payload.get("knots", [])],
            "knot_values": {
                int(index): float(value)
                for index, value in payload.get("knot_values", {}).items()
            },
            "peaks": [int(value) for value in payload.get("peaks", [])],
            "transients": [
                (int(start), int(stop))
                for start, stop in payload.get("transients", [])
            ],
            "pending": optional_index(payload.get("pending")),
            "filter_range": tuple(int(value) for value in saved_range)
            if saved_range is not None
            else None,
            "pending_filter": optional_index(payload.get("pending_filter")),
            "filter_stages": stages,
            "active_filter_stage": int(payload.get("active_filter_stage", -1)),
        }

    @staticmethod
    def _set_combo_to_data(combo: QComboBox, value) -> None:
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    def _session_metadata(self) -> dict:
        self._store_active_scan_state()
        selection_mode = (
            "peak"
            if self.peak_radio.isChecked()
            else "transient"
            if self.transient_radio.isChecked()
            else "knot"
        )
        axes = [
            {
                "xlim": [float(value) for value in axis.get_xlim()],
                "ylim": [float(value) for value in axis.get_ylim()],
            }
            for axis in self.axes.flat
        ]
        return {
            "app_version": PACKAGE_VERSION,
            "source_original_path": self._source_display_path or self.input_file,
            "source_original_name": Path(
                self._source_display_path or self.input_file or "source-data"
            ).name,
            "settings": {
                "source_format": self.format_dropdown.currentText(),
                "active_scan": int(self.scan_number.value()),
                "custom_scan_window": self.custom_scan_window.isChecked(),
                "scan_window_start": float(self.scan_window_start.value()),
                "scan_window_end": float(self.scan_window_end.value()),
                "scan_rate_v_s": float(self.scan_rate.value()),
                "background_kind": self.background_kind.currentData(),
                "background_scope": self.background_scope.currentData(),
                "lock_knots": self.lock_knots.isChecked(),
                "pearson4_exponent_min": float(self.exponent_min.value()),
                "peak_center_window_v": float(self.center_window.value()),
                "selection_mode": selection_mode,
                "output_dir": self.output_dir,
                "output_name": self.output_name.text(),
                "detailed_csv": self.detailed_csv.isChecked(),
                "quality_panel_mode": self._quality_panel_mode,
                "top_right_mode": self._top_right_mode,
                "bottom_left_mode": self._bottom_left_mode,
                "filter_advanced": self.filter_advanced_button.isChecked(),
                "splitter_sizes": [int(value) for value in self.main_splitter.sizes()],
                "window_geometry": bytes(self.saveGeometry().toBase64()).decode("ascii"),
                "axes": axes,
            },
            "scan_states": {
                str(scan_number): self._serialize_selection_snapshot(snapshot)
                for scan_number, snapshot in self._scan_states.items()
            },
        }

    def save_session(self, checked: bool = False, *, force_dialog: bool = False) -> None:
        del checked
        if self.input_file is None or self.analysis is None:
            self.statusBar().showMessage("Load a CV before saving a session")
            return
        target = self.session_file if self.session_file and not force_dialog else None
        if target is None:
            source = Path(self._source_display_path or self.input_file)
            suggested = source.with_suffix(SESSION_EXTENSION)
            target, _ = QFileDialog.getSaveFileName(
                self,
                "Save editable Makima Fitter session",
                str(suggested),
                "Makima Fitter sessions (*.makima)",
            )
        if not target:
            return
        self.save_session_path(target)

    def save_session_path(self, path: str | Path) -> Path | None:
        if self.input_file is None or self.analysis is None:
            return None
        try:
            saved = save_session_archive(path, self.input_file, self._session_metadata())
            self.session_file = str(saved.resolve())
            self.statusBar().showMessage(f"Saved editable session: {saved}")
            return saved
        except Exception as error:
            QMessageBox.critical(self, "Session save failed", str(error))
            traceback.print_exc()
            self.statusBar().showMessage("Session save failed")
            return None

    def _restore_session_settings(self, settings: dict) -> None:
        controls = [
            self.background_kind,
            self.background_scope,
            self.lock_knots,
            self.exponent_min,
            self.center_window,
            self.detailed_csv,
            self.filter_advanced_button,
            self.custom_scan_window,
            self.scan_window_start,
            self.scan_window_end,
        ]
        for control in controls:
            control.blockSignals(True)
        self._set_combo_to_data(self.background_kind, settings.get("background_kind"))
        self._set_combo_to_data(self.background_scope, settings.get("background_scope"))
        self.lock_knots.setChecked(bool(settings.get("lock_knots", True)))
        self.exponent_min.setValue(float(settings.get("pearson4_exponent_min", 0.5)))
        self.center_window.setValue(float(settings.get("peak_center_window_v", 0.25)))
        self.detailed_csv.setChecked(bool(settings.get("detailed_csv", True)))
        self.filter_advanced_button.setChecked(bool(settings.get("filter_advanced", False)))
        custom_window = bool(settings.get("custom_scan_window", False))
        self.scan_window_start.setValue(float(settings.get("scan_window_start", 1.0)))
        self.scan_window_end.setValue(float(settings.get("scan_window_end", 2.0)))
        self.custom_scan_window.setChecked(custom_window)
        for control in controls:
            control.blockSignals(False)
        self.filter_advanced_panel.setVisible(self.filter_advanced_button.isChecked())
        self.scan_window_row.setVisible(custom_window)
        self.scan_number.setEnabled(
            not custom_window and self.scan_number.maximum() > 1
        )
        self.output_dir = settings.get("output_dir") or self.output_dir
        self.output_label.setText(self.output_dir or "No output folder selected")
        if settings.get("output_name"):
            self.output_name.setText(str(settings["output_name"]))
        selection_mode = settings.get("selection_mode", "knot")
        {
            "knot": self.knot_radio,
            "peak": self.peak_radio,
            "transient": self.transient_radio,
        }.get(selection_mode, self.knot_radio).setChecked(True)
        self._quality_panel_mode = str(settings.get("quality_panel_mode", "table"))
        self._top_right_mode = str(settings.get("top_right_mode", "normalized"))
        self._bottom_left_mode = str(settings.get("bottom_left_mode", "potential"))
        sizes = settings.get("splitter_sizes")
        if isinstance(sizes, list) and len(sizes) == 2:
            self.main_splitter.setSizes([int(value) for value in sizes])

    def load_session_path(self, path: str | Path) -> bool:
        try:
            metadata, source_bytes, source_suffix = load_session_archive(path)
            settings = metadata.get("settings", {})
            if not isinstance(settings, dict):
                raise ValueError("The session settings are invalid")
            saved_states = {}
            for scan_key, snapshot in metadata.get("scan_states", {}).items():
                text_key = str(scan_key)
                key = int(text_key) if text_key.isdigit() else text_key
                saved_states[key] = self._deserialize_selection_snapshot(snapshot)
            new_temporary = tempfile.TemporaryDirectory(prefix="MakimaFitter-session-")
            source_name = str(metadata.get("source_original_name", "source-data"))
            suffix = source_suffix or Path(source_name).suffix
            embedded_path = Path(new_temporary.name) / f"source-data{suffix}"
            embedded_path.write_bytes(source_bytes)
        except Exception as error:
            QMessageBox.critical(self, "Could not open session", str(error))
            traceback.print_exc()
            self.statusBar().showMessage("Could not open session")
            return False

        self._release_session_temp()
        self._session_temp_dir = new_temporary
        try:
            self.scan_rate.blockSignals(True)
            self.scan_rate.setValue(float(settings.get("scan_rate_v_s", 0.0)))
            self.scan_rate.blockSignals(False)
            self.load_input_path(
                str(embedded_path),
                display_path=str(metadata.get("source_original_path") or source_name),
                source_format=settings.get("source_format"),
                scan_number=int(settings.get("active_scan", 1)),
            )
            if self.analysis is None:
                raise ValueError("The embedded CV data could not be loaded")
            self._restore_session_settings(settings)
            self._scan_states = saved_states
            self._active_scan_number = None
            self.load_selected_scan()
            self._restore_session_settings(settings)
            self.draw_live_view()
            self.history.clear()
            self.future.clear()
            self.session_file = str(Path(path).resolve())
            geometry = settings.get("window_geometry")
            if geometry:
                self.restoreGeometry(QByteArray.fromBase64(str(geometry).encode("ascii")))
            axes = settings.get("axes")
            if isinstance(axes, list) and len(axes) == 4:
                for axis, limits in zip(self.axes.flat, axes):
                    axis.set_xlim(*[float(value) for value in limits["xlim"]])
                    axis.set_ylim(*[float(value) for value in limits["ylim"]])
                self._preserve_plot_limits = True
                self.canvas.draw_idle()
            self.statusBar().showMessage(f"Opened editable session: {Path(path).name}")
            return True
        except Exception as error:
            self._release_session_temp()
            QMessageBox.critical(self, "Could not open session", str(error))
            traceback.print_exc()
            self.statusBar().showMessage("Could not open session")
            return False

    def load_input_path(
        self,
        path: str,
        *,
        display_path: str | None = None,
        source_format: str | None = None,
        scan_number: int | None = None,
    ) -> None:
        resolved = Path(path).resolve()
        self._scan_reload_timer.stop()
        self._scan_states.clear()
        self._active_scan_number = None
        self._pending_scan_history = False
        self.history.clear()
        self.future.clear()
        self.input_file = str(resolved)
        self._source_display_path = str(display_path or resolved)
        self.file_label.setText(self._source_display_path)
        self.custom_scan_window.blockSignals(True)
        self.custom_scan_window.setChecked(False)
        self.custom_scan_window.blockSignals(False)
        self.scan_window_row.hide()
        detected = source_format or detect_input_format(resolved)
        if detected:
            self.format_dropdown.blockSignals(True)
            self.format_dropdown.setCurrentText(detected)
            self.format_dropdown.blockSignals(False)
        display_source = Path(self._source_display_path)
        default_output = display_source.parent / "Makima Fitter output"
        self.output_dir = str(default_output)
        self.output_label.setText(str(default_output))
        self.output_name.setText(f"{display_source.stem}_makima")
        self.refresh_scan_count()
        if scan_number is not None:
            self.scan_number.blockSignals(True)
            self.scan_number.setValue(int(scan_number))
            self.scan_number.blockSignals(False)
        self.load_selected_scan()
        self.save_session_button.setEnabled(self.analysis is not None)

    def _format_changed(self) -> None:
        if self.input_file:
            self._scan_reload_timer.stop()
            self._scan_states.clear()
            self._active_scan_number = None
            self._pending_scan_history = False
            self.history.clear()
            self.future.clear()
            self.refresh_scan_count()
            self.load_selected_scan()

    def _requested_scan_key(self) -> int | str:
        if self.custom_scan_window.isChecked():
            return (
                f"window:{self.scan_window_start.value():.8g}:"
                f"{self.scan_window_end.value():.8g}"
            )
        return int(self.scan_number.value())

    def _custom_scan_window_toggled(self, checked: bool) -> None:
        self.scan_window_row.setVisible(bool(checked))
        self.scan_number.setEnabled(not checked and self.scan_number.maximum() > 1)
        if checked:
            start = float(self.scan_number.value())
            maximum = float(self.scan_window_end.maximum())
            self.scan_window_start.blockSignals(True)
            self.scan_window_end.blockSignals(True)
            self.scan_window_start.setValue(min(start, maximum - 0.01))
            self.scan_window_end.setValue(min(start + 1.0, maximum))
            self.scan_window_start.blockSignals(False)
            self.scan_window_end.blockSignals(False)
        self._schedule_scan_reload()

    def _scan_window_limit_changed(self, _value: float) -> None:
        minimum_gap = 0.01
        start = self.scan_window_start.value()
        end = self.scan_window_end.value()
        sender = self.sender()
        if start >= end:
            if sender is self.scan_window_start:
                self.scan_window_end.blockSignals(True)
                self.scan_window_end.setValue(
                    min(self.scan_window_end.maximum(), start + minimum_gap)
                )
                self.scan_window_end.blockSignals(False)
            else:
                self.scan_window_start.blockSignals(True)
                self.scan_window_start.setValue(
                    max(self.scan_window_start.minimum(), end - minimum_gap)
                )
                self.scan_window_start.blockSignals(False)
        self._schedule_scan_reload()

    def refresh_scan_count(self) -> None:
        if not self.input_file:
            return
        try:
            count = scan_count(self.input_file, self.format_dropdown.currentText())
        except Exception as error:
            self.scan_number.setRange(1, 1)
            self.scan_count_label.setText("count unavailable")
            self.scan_count_label.setToolTip(str(error))
            return
        self.scan_number.blockSignals(True)
        previous = self.scan_number.value()
        self.scan_number.setRange(1, max(1, count))
        self.scan_number.setValue(min(previous, count))
        self.scan_number.blockSignals(False)
        self.scan_count_label.setText(f"of {count}")
        enabled = count > 1
        self.scan_number.setEnabled(enabled and not self.custom_scan_window.isChecked())
        maximum = float(max(2, count + 1))
        self.scan_window_start.blockSignals(True)
        self.scan_window_end.blockSignals(True)
        self.scan_window_start.setRange(1.0, maximum - 0.01)
        self.scan_window_end.setRange(1.01, maximum)
        self.scan_window_start.blockSignals(False)
        self.scan_window_end.blockSignals(False)
        self.custom_scan_window.setEnabled(enabled)
        if not enabled and self.custom_scan_window.isChecked():
            self.custom_scan_window.setChecked(False)

    def _schedule_scan_reload(self) -> None:
        if not self.input_file or self._loading_scan:
            return
        target_scan = self._requested_scan_key()
        if self._active_scan_number is not None and target_scan == self._active_scan_number:
            self._scan_reload_timer.stop()
            if self._pending_scan_history and self.history:
                self.history.pop()
            self._pending_scan_history = False
            return
        if self._active_scan_number is not None and not self._pending_scan_history:
            self._store_active_scan_state()
            self._remember()
            self._pending_scan_history = True
        self._scan_reload_timer.start()

    def load_selected_scan(self) -> None:
        if not self.input_file or self._loading_scan:
            return
        self._loading_scan = True
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self._store_active_scan_state()
            rate = self.scan_rate.value() or None
            requested_key = self._requested_scan_key()
            if self.custom_scan_window.isChecked():
                scan = load_scan_window(
                    self.input_file,
                    self.format_dropdown.currentText(),
                    self.scan_window_start.value(),
                    self.scan_window_end.value(),
                    rate,
                )
            else:
                scan = load_scan(
                    self.input_file,
                    self.format_dropdown.currentText(),
                    self.scan_number.value(),
                    rate,
                )
            if self._source_display_path:
                scan = dataclass_replace(scan, source_path=Path(self._source_display_path))
            self.analysis = prepare_analysis(scan)
            self._invalidate_filter_cache()
            self._configure_filter_limits()
            self._update_sampling_density_label()
            self._measurement_indices = []
            self._measurement_axis_index = None
            self.measure_button.blockSignals(True)
            self.measure_button.setChecked(False)
            self.measure_button.blockSignals(False)
            self.measure_result_label.setText("  Δt: —   f: —")
            self._preserve_plot_limits = False
            self._home_limits = None
            saved_state = self._scan_states.get(requested_key)
            if saved_state is None:
                self.knot_indices = suggested_knot_indices(self.analysis)
                self.knot_values = {}
                self.peak_indices = []
                self.transient_intervals = []
                self.pending_transient_index = None
                self.filter_stages = [
                    dataclass_replace(stage, range=None)
                    if stage.scope == "regional"
                    else stage
                    for stage in self.filter_stages
                ]
                self._refresh_filter_stage_selector()
                stage = self.filter_stages[self._active_filter_stage] if self.filter_stages else None
                self._set_filter_editor_values(stage)
                self.filter_range = stage.range if stage is not None else None
                self.pending_filter_index = None
                self._filter_range_armed = bool(self.filter_scope.currentData() == "regional")
            else:
                self._apply_selection_snapshot(saved_state)
                if self.lock_knots.isChecked():
                    working = self.working_analysis()
                    self.knot_values = {
                        index: float(working.transform[index])
                        for index in self.knot_indices
                    }
            self._update_filter_range_label()
            self._active_scan_number = requested_key
            self._store_active_scan_state()
            self._quality_panel_mode = "table"
            self._clear_table_calculator()
            self._top_right_mode = "normalized"
            self._bottom_left_mode = "potential"
            self.outcome = None
            switch_text = ", ".join(
                f"{self.analysis.potential[index]:.4g} V" for index in self.analysis.switching_indices
            ) or "none inside the selected scan"
            selection_text = (
                f"chronology {self.scan_window_start.value():.2f} to "
                f"{self.scan_window_end.value():.2f}"
                if self.custom_scan_window.isChecked()
                else f"scan {scan.scan_number} of {scan.scan_count}"
            )
            self.statusBar().showMessage(
                f"Loaded {selection_text}; automatic switching potential(s): {switch_text}"
            )
            self.fit_quality_label.setText(
                "No fit yet"
            )
            self._set_qc_warnings([], collapse=True)
            self.export_button.setEnabled(False)
            self.copy_table_button.setEnabled(False)
            self.draw_live_view()
            if self.knot_indices or self.peak_indices or self.transient_intervals:
                self.schedule_fit()
        except Exception as error:
            self.analysis = None
            self.outcome = None
            self.export_button.setEnabled(False)
            self.copy_table_button.setEnabled(False)
            QMessageBox.critical(self, "Could not load scan", str(error))
            traceback.print_exc()
            self._draw_empty()
        finally:
            QApplication.restoreOverrideCursor()
            self._loading_scan = False
            self._pending_scan_history = False

    def select_output_directory(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select output folder", self.output_dir or "")
        if path:
            self.output_dir = path
            self.output_label.setText(path)

    def _remember(self) -> None:
        self.history.append(self._snapshot())
        self.future.clear()
        if len(self.history) > 40:
            self.history.pop(0)

    def _snapshot(self) -> dict:
        snapshot = self._selection_snapshot()
        snapshot["scan_number"] = self._active_scan_number
        return snapshot

    def _selection_snapshot(self) -> dict:
        return {
            "knots": self.knot_indices.copy(),
            "knot_values": self.knot_values.copy(),
            "peaks": self.peak_indices.copy(),
            "transients": self.transient_intervals.copy(),
            "pending": self.pending_transient_index,
            "filter_range": self.filter_range,
            "pending_filter": self.pending_filter_index,
            "filter_stages": self.filter_stages.copy(),
            "active_filter_stage": self._active_filter_stage,
        }

    def _store_active_scan_state(self) -> None:
        if self._active_scan_number is not None and self.analysis is not None:
            self._scan_states[self._active_scan_number] = self._selection_snapshot()

    def _apply_selection_snapshot(self, snapshot: dict) -> None:
        self.knot_indices = snapshot["knots"].copy()
        self.knot_values = snapshot["knot_values"].copy()
        self.peak_indices = snapshot["peaks"].copy()
        self.transient_intervals = snapshot["transients"].copy()
        self.pending_transient_index = snapshot["pending"]
        self.filter_stages = snapshot.get("filter_stages", self.filter_stages).copy()
        self._active_filter_stage = int(snapshot.get("active_filter_stage", self._active_filter_stage))
        self._refresh_filter_stage_selector()
        stage = self.filter_stages[self._active_filter_stage] if self.filter_stages else None
        self._set_filter_editor_values(stage)
        self.filter_range = snapshot.get("filter_range", self.filter_range)
        self.pending_filter_index = snapshot.get("pending_filter")
        self._filter_range_armed = bool(
            self.filter_scope.currentData() == "regional" and self.filter_range is None
        )
        self._invalidate_filter_cache()
        self._update_filter_range_label()

    def _restore_snapshot(self, snapshot: dict) -> None:
        self._scan_reload_timer.stop()
        self._pending_scan_history = False
        target_scan = snapshot.get("scan_number")
        if target_scan is not None and target_scan != self._active_scan_number:
            self._store_active_scan_state()
            self._scan_states[target_scan] = {
                key: value.copy() if hasattr(value, "copy") else value
                for key, value in snapshot.items()
                if key != "scan_number"
            }
            is_window = isinstance(target_scan, str) and target_scan.startswith("window:")
            self.custom_scan_window.blockSignals(True)
            self.custom_scan_window.setChecked(is_window)
            self.custom_scan_window.blockSignals(False)
            self.scan_window_row.setVisible(is_window)
            if is_window:
                _, start, end = target_scan.split(":", 2)
                self.scan_window_start.blockSignals(True)
                self.scan_window_end.blockSignals(True)
                self.scan_window_start.setValue(float(start))
                self.scan_window_end.setValue(float(end))
                self.scan_window_start.blockSignals(False)
                self.scan_window_end.blockSignals(False)
            else:
                self.scan_number.blockSignals(True)
                self.scan_number.setValue(int(target_scan))
                self.scan_number.blockSignals(False)
            self.load_selected_scan()
        else:
            self._apply_selection_snapshot(snapshot)
            self._store_active_scan_state()
        self.outcome = None
        self.export_button.setEnabled(False)
        self.copy_table_button.setEnabled(False)

    def canvas_clicked(self, event) -> None:
        if (
            self.measure_button.isChecked()
            and not self.toolbar.mode
            and event.inaxes in list(self.axes.flat)
        ):
            self._handle_interval_measurement_click(event)
            return
        if (
            self.analysis is not None
            and self.filter_scope.currentData() == "regional"
            and self._filter_range_armed
            and not self.toolbar.mode
            and event.inaxes in {self.axes[0, 0], self.axes[1, 0]}
        ):
            if event.inaxes is self.axes[0, 0] and event.xdata is not None:
                _, _, time_scale = display_time_axis(self.analysis.time)
                clicked_time_s = float(event.xdata) / time_scale
                index = int(np.argmin(np.abs(self.analysis.time - clicked_time_s)))
            else:
                index = self._nearest_measurement_index(event)
            if index is not None:
                self._handle_filter_range_click(index, event.button)
            return
        if (
            event.inaxes in list(self.axes.flat)
            and event.button == 1
            and not self.toolbar.mode
            and self._control_click(event)
        ):
            self.toggle_plot_focus(event.inaxes)
            return
        if (
            event.inaxes is self.axes[1, 1]
            and event.button in {1, 3}
            and not self.toolbar.mode
        ):
            if self._quality_panel_mode == "table" and self._handle_table_cell_click(event):
                return
            if event.button != 1:
                return
            self.toggle_quality_panel()
            return
        if event.inaxes is self.axes[0, 1] and event.button == 1 and not self.toolbar.mode:
            self.toggle_top_right_panel()
            return
        if event.inaxes is self.axes[1, 0] and event.button == 1 and not self.toolbar.mode:
            self.toggle_bottom_left_panel()
            return
        if (
            self.analysis is None
            or event.inaxes is not self.axes[0, 0]
            or event.xdata is None
            or self.toolbar.mode
        ):
            return
        _, _, time_scale = display_time_axis(self.analysis.time)
        clicked_time_s = float(event.xdata) / time_scale
        index = int(np.argmin(np.abs(self.analysis.time - clicked_time_s)))
        if self.selection_group.checkedButton() is self.transient_radio:
            self._handle_transient_click(index, event.button)
            return
        target = (
            self.knot_indices
            if self.selection_group.checkedButton() is self.knot_radio
            else self.peak_indices
        )
        working = self.working_analysis()
        if event.button == 1 and working.excluded_mask[index]:
            self.statusBar().showMessage("Selections cannot be placed inside an excluded interval")
            return
        self._remember()
        if event.button == 1:
            if target is self.knot_indices:
                if not self.lock_knots.isChecked() and event.ydata is None:
                    self.history.pop()
                    return
                value = (
                    float(working.transform[index])
                    if self.lock_knots.isChecked()
                    else float(event.ydata)
                )
                if not np.isfinite(value):
                    self.history.pop()
                    return
                self.knot_values[index] = value
            if index not in target:
                target.append(index)
                target.sort()
        elif event.button == 3 and target:
            nearest = min(target, key=lambda candidate: abs(candidate - index))
            target.remove(nearest)
            if target is self.knot_indices:
                self.knot_values.pop(nearest, None)
        else:
            self.history.pop()
            return
        self.outcome = None
        self.export_button.setEnabled(False)
        self.copy_table_button.setEnabled(False)
        self.schedule_fit()

    def _handle_filter_range_click(self, index: int, button: int) -> None:
        if button == 3:
            self.filter_range = None
            self.pending_filter_index = None
            self._filter_range_armed = True
            self._update_filter_range_label()
            self._filter_changed()
            self.statusBar().showMessage("Regional filter range cleared; click a new first limit")
            return
        if button != 1:
            return
        if self.pending_filter_index is None:
            self.pending_filter_index = int(index)
            self._filter_range_armed = True
            self._update_filter_range_label()
            self.draw_live_view()
            self.statusBar().showMessage("First filter limit selected; click the second limit")
            return
        first, second = sorted((self.pending_filter_index, int(index)))
        self.filter_range = (first, second)
        self.pending_filter_index = None
        self._filter_range_armed = False
        self._update_filter_range_label()
        self._filter_changed()
        self.statusBar().showMessage(
            f"Regional filter range set: {self.analysis.potential[first]:.5g} to "
            f"{self.analysis.potential[second]:.5g} V"
        )

    def _handle_transient_click(self, index: int, button: int) -> None:
        if button == 1:
            self._remember()
            if self.pending_transient_index is None:
                self.pending_transient_index = index
                self.statusBar().showMessage("Transient start selected; click the interval end")
                self.draw_live_view()
                return
            self.transient_intervals.append((self.pending_transient_index, index))
            self.transient_intervals = normalize_exclusion_intervals(
                len(self.analysis.time), self.transient_intervals
            )
            self.pending_transient_index = None
            working = self.working_analysis()
            self.knot_indices = [index for index in self.knot_indices if not working.excluded_mask[index]]
            self.knot_values = {
                index: value for index, value in self.knot_values.items()
                if index in self.knot_indices
            }
            self.peak_indices = [index for index in self.peak_indices if not working.excluded_mask[index]]
            self.outcome = None
            self.export_button.setEnabled(False)
            self.copy_table_button.setEnabled(False)
            self.schedule_fit()
            return
        if button == 3:
            if self.pending_transient_index is not None:
                self._remember()
                self.pending_transient_index = None
                self.draw_live_view()
                return
            if self.transient_intervals:
                self._remember()
                nearest = int(
                    np.argmin(
                        [abs((start + end) * 0.5 - index) for start, end in self.transient_intervals]
                    )
                )
                del self.transient_intervals[nearest]
                self.outcome = None
                self.export_button.setEnabled(False)
                self.copy_table_button.setEnabled(False)
                self.schedule_fit()

    def undo_selection(self) -> None:
        if not self.history:
            return
        self.future.append(self._snapshot())
        restored = self.history.pop()
        self._restore_snapshot(restored)
        self.schedule_fit()
        if restored.get("scan_number") is not None:
            self.statusBar().showMessage(f"Undo: restored scan {restored['scan_number']} selections")

    def redo_selection(self) -> None:
        if not self.future:
            return
        self.history.append(self._snapshot())
        restored = self.future.pop()
        self._restore_snapshot(restored)
        self.schedule_fit()
        if restored.get("scan_number") is not None:
            self.statusBar().showMessage(f"Redo: restored scan {restored['scan_number']} selections")

    def clear_knots(self) -> None:
        if self.analysis is None or not self.knot_indices:
            return
        self._remember()
        self.knot_indices = []
        self.knot_values = {}
        self.outcome = None
        self.export_button.setEnabled(False)
        self.copy_table_button.setEnabled(False)
        self.schedule_fit()

    def clear_peaks(self) -> None:
        if not self.peak_indices:
            return
        self._remember()
        self.peak_indices = []
        self.outcome = None
        self.export_button.setEnabled(False)
        self.copy_table_button.setEnabled(False)
        self.schedule_fit()

    def clear_transients(self) -> None:
        if not self.transient_intervals and self.pending_transient_index is None:
            return
        self._remember()
        self.transient_intervals = []
        self.pending_transient_index = None
        self.outcome = None
        self.export_button.setEnabled(False)
        self.copy_table_button.setEnabled(False)
        self.schedule_fit()

    def suggest_peaks(self) -> None:
        if self.analysis is None:
            return
        try:
            working = self.working_analysis()
            reference, _, _, _ = construct_background(
                working,
                self.knot_indices,
                self.knot_values,
                str(self.background_kind.currentData()),
                str(self.background_scope.currentData()),
            )
            residual = working.transform - reference
            distance = max(5, int(round(0.06 / working.scan_rate_v_s / np.median(np.diff(working.time)))))
            included_residual = residual[~working.excluded_mask]
            prominence = max(float(np.std(included_residual)) * 0.35, np.finfo(float).eps)
            candidates = _peak_candidates_outside_exclusions(
                residual,
                working.excluded_mask,
                distance=distance,
                prominence=prominence,
            )
            candidates.sort(key=lambda item: float(item[1]), reverse=True)
            suggested = sorted(int(item[0]) for item in candidates[:6])
            if not suggested:
                raise ValueError("No prominent peak candidates were found")
            self._remember()
            self.peak_indices = suggested
            self.schedule_fit()
        except Exception as error:
            QMessageBox.warning(self, "Peak suggestion unavailable", str(error))

    def schedule_fit(self) -> None:
        if self.analysis is None:
            return
        self._fit_timer.start()
        self.draw_live_view()

    def _config(self) -> FitConfig:
        return FitConfig(
            pearson4_exponent_min=self.exponent_min.value(),
            peak_center_window_v=self.center_window.value(),
            background_kind=str(self.background_kind.currentData()),
            background_scope=str(self.background_scope.currentData()),
            knots_locked_to_data=self.lock_knots.isChecked(),
            filter_kind=str(self.filter_kind.currentData()),
            filter_domain=str(self.filter_domain.currentData()),
            filter_scope=str(self.filter_scope.currentData()),
            filter_range=self.filter_range,
            filter_frequency_hz=self.filter_frequency.value(),
            filter_bandwidth_hz=self.filter_bandwidth.value(),
            filter_attenuation_db=self.filter_attenuation.value(),
            filter_window_mv=self.filter_window.value(),
            filter_polynomial_order=self.filter_polynomial.value(),
            filter_hampel_sigma=self.filter_hampel_sigma.value(),
            filter_harmonics=self.filter_harmonics.value(),
            filter_feather_mv=self.filter_feather.value(),
            filter_envelope_prominence_pct=self.filter_envelope_prominence.value(),
            filter_stages=tuple(self.filter_stages),
        )

    def _toggle_qc_details(self, checked: bool) -> None:
        action = "Hide QC details" if checked else "QC details"
        self.qc_details_button.setText(f"{action} ({self._qc_warning_count})")
        self.qc_details_label.setVisible(
            bool(checked and self.qc_details_button.isEnabled())
        )
        QTimer.singleShot(0, self._resize_controls_to_contents)

    def _resize_controls_to_contents(self) -> None:
        if hasattr(self, "controls_panel"):
            content = self.controls_panel.widget()
            if content is not None:
                content.adjustSize()
                target_height = max(
                    content.sizeHint().height(),
                    self.controls_panel.viewport().height(),
                )
                content.setFixedHeight(target_height)
                content.resize(
                    max(content.width(), self.controls_panel.viewport().width()),
                    target_height,
                )

    def _set_qc_warnings(self, warnings, *, collapse: bool = False) -> None:
        warnings = list(warnings)
        if collapse:
            self.qc_details_button.setChecked(False)
        count = len(warnings)
        self._qc_warning_count = count
        suffix = "warning" if count == 1 else "warnings"
        action = "Hide QC details" if self.qc_details_button.isChecked() else "QC details"
        self.qc_details_button.setText(f"{action} ({count})")
        self.qc_details_button.setEnabled(bool(warnings))
        self.qc_details_label.setText("\n\n".join(f"• {warning}" for warning in warnings))
        self.qc_details_label.setVisible(
            bool(warnings and self.qc_details_button.isChecked())
        )
        QTimer.singleShot(0, self._resize_controls_to_contents)
        return count, suffix

    def update_live_fit(self) -> None:
        if self.analysis is None:
            return
        if self._navigation_drag_active:
            self._fit_timer.start()
            return
        if not self.peak_indices:
            self.outcome = None
            self._clear_table_calculator()
            self.export_button.setEnabled(False)
            self.copy_table_button.setEnabled(False)
            self.fit_quality_label.setText("Select at least one peak to start fitting.")
            self._set_qc_warnings([], collapse=True)
            self.draw_live_view()
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.statusBar().showMessage("Updating background and Pearson-IV fit…")
        QApplication.processEvents()
        try:
            working = self.working_analysis()
            if self._filter_error is not None:
                raise ValueError(self._filter_error)
            self.outcome = fit_spline_peaks(
                working,
                self.knot_indices,
                self.peak_indices,
                self._config(),
                preview=True,
                knot_values=self.knot_values,
            )
            self.knot_indices = self.outcome.knot_indices.copy()
            self.knot_values = self.outcome.knot_values.copy()
            self.peak_indices = self.outcome.peak_indices.copy()
            self._table_selected_cells = [
                (row, column)
                for row, column in self._table_selected_cells
                if column in {1, 2, 3, 4}
                and 1 <= row <= min(9, len(self.outcome.peak_results))
            ]
            self._table_calculation = {}
            self._table_calculation_symbols = {}
            warning_count, warning_word = self._set_qc_warnings(
                self.outcome.metrics["warnings"]
            )
            sd_unit = f"{self.analysis.scale_prefix}A s^-1/2"
            self.fit_quality_label.setText(
                f"SD RMSE: {self.outcome.metrics['sd_rmse']:.4g} {sd_unit}\n"
                f"Current RMSE: {self.outcome.metrics['current_rmse']:.4g} {self.analysis.scale_prefix}A\n"
                f"QC: {warning_count} {warning_word}"
            )
            self.export_button.setEnabled(True)
            self.copy_table_button.setEnabled(True)
            self.statusBar().showMessage("Live fit updated")
        except Exception as error:
            self.outcome = None
            self._table_calculation = {}
            self._table_calculation_symbols = {}
            self.export_button.setEnabled(False)
            self.copy_table_button.setEnabled(False)
            self.fit_quality_label.setText(f"Fit needs attention: {error}")
            self._set_qc_warnings([], collapse=True)
            self.statusBar().showMessage(str(error))
        finally:
            QApplication.restoreOverrideCursor()
            self.draw_live_view()

    def _plot_segments(self, axis, y, label=None, **kwargs) -> None:
        for number, (start, end) in enumerate(self.analysis.segment_bounds):
            self._plot_series(
                axis,
                self.analysis.potential[start : end + 1],
                y[start : end + 1],
                label=label if number == 0 else None,
                **kwargs,
            )

    @staticmethod
    def _plot_series(axis, x, y, *args, **kwargs):
        """Render dense lines cheaply without changing analysis resolution."""
        x_values = np.asarray(x)
        y_values = np.asarray(y)
        count = len(x_values)
        if count > MAX_DISPLAY_POINTS:
            indices = np.linspace(0, count - 1, MAX_DISPLAY_POINTS, dtype=int)
            indices = np.unique(indices)
            x_values = x_values[indices]
            y_values = y_values[indices]
        return axis.plot(x_values, y_values, *args, **kwargs)

    def _draw_filter_region(self, axis, x_values) -> None:
        colors = ("#56B4E9", "#E69F00", "#CC79A7", "#009E73")
        for number, stage in enumerate(self.filter_stages, start=1):
            if stage.scope != "regional" or stage.range is None:
                continue
            start, end = sorted(stage.range)
            axis.axvspan(
                x_values[start],
                x_values[end],
                color=colors[(number - 1) % len(colors)],
                alpha=0.09,
                label=f"Stage {number} range",
                zorder=0,
            )
        if self.filter_scope.currentData() == "regional" and self.pending_filter_index is not None:
            axis.axvline(
                x_values[self.pending_filter_index],
                color="#56B4E9",
                linestyle="--",
                linewidth=1.0,
            )

    def _draw_upper_envelope_maxima(self, axis, x_values, y_values, data, *, label: bool) -> None:
        indices = np.asarray(
            data.filter_diagnostics.get("maxima_indices", []),
            dtype=int,
        )
        indices = indices[(indices >= 0) & (indices < len(x_values))]
        if not len(indices):
            return
        axis.scatter(
            np.asarray(x_values)[indices],
            np.asarray(y_values)[indices],
            marker="v",
            s=28,
            facecolors="#F0E442",
            edgecolors="#6B5D00",
            linewidths=0.7,
            label="Makima-envelope maxima" if label else None,
            zorder=10,
        )

    def _draw_interval_measurement(self, data) -> None:
        if not self._measurement_indices:
            return
        axes = list(self.axes.flat)
        axis_index = self._measurement_axis_index
        if axis_index is None or not 0 <= axis_index < len(axes):
            return
        axis = axes[axis_index]
        series = self._measurement_series(axis, data)
        if series is None:
            return
        x_values, y_values = series
        indices = [
            index for index in self._measurement_indices
            if 0 <= index < len(x_values) and np.isfinite(x_values[index]) and np.isfinite(y_values[index])
        ]
        if not indices:
            return
        axis.plot(
            x_values[indices],
            y_values[indices],
            color="#D55E00",
            marker="o",
            markerfacecolor="white",
            markeredgewidth=1.5,
            linewidth=1.1,
            zorder=12,
        )
        for number, index in enumerate(indices, start=1):
            axis.annotate(
                str(number),
                (x_values[index], y_values[index]),
                xytext=(5, 6),
                textcoords="offset points",
                color="#A33B16",
                fontsize=8,
                fontweight="bold",
                zorder=13,
            )
        result = self._measurement_text()
        if result:
            axis.text(
                0.98,
                0.97,
                result,
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                color="#A33B16",
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "#D55E00"},
                zorder=14,
            )

    def _draw_live_peak_table(self, axis, outcome) -> None:
        data = outcome.data if outcome is not None else self.working_analysis()
        if outcome is not None:
            formatted = formatted_peak_table_frame(outcome)
            columns = list(formatted.columns)
        else:
            _, time_unit, _ = display_time_axis(data.time)
            columns = [
                "Peak",
                f"t ({time_unit})",
                "Eₚ (V)",
                f"iₚ ({data.scale_prefix}A)",
                f"Qₚ ({data.scale_prefix}C)",
                "m",
                "ν",
            ]
        rows = [["" for _ in columns] for _ in range(LIVE_TABLE_ROWS)]
        if outcome is not None:
            for row, values in zip(rows[:9], formatted.values.tolist()[:9]):
                row[:] = values
        rows[-1][0] = "Calc"
        for column, value in self._table_calculation.items():
            if 0 <= column < len(columns):
                symbol = self._table_calculation_symbols.get(column, "")
                rows[-1][column] = f"{value} ({symbol})" if symbol else value
        table = axis.table(
            cellText=rows,
            colLabels=columns,
            cellLoc="center",
            loc="center",
            bbox=[0.0, 0.0, 1.0, 1.0],
            colWidths=[1.0 / len(columns)] * len(columns),
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9.0)
        for (row, column), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(fontweight="bold")
                cell.set_facecolor("#F0F0F0")
            elif row == LIVE_TABLE_ROWS:
                cell.set_facecolor("#E8EEF5")
                if column == 0:
                    cell.set_text_props(fontweight="bold")
                elif column == self._table_active_column:
                    cell.set_edgecolor("#3274A1")
                    cell.set_linewidth(2.0)
            elif (row, column) in self._table_selected_cells:
                cell.set_facecolor("#FFF2A8")
                cell.set_edgecolor("#B38F00")
                cell.set_linewidth(1.4)
        self._live_peak_table = table
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)

    def draw_live_view(self) -> None:
        self._live_peak_table = None
        if self.analysis is None:
            self._draw_empty()
            return
        data = self.working_analysis()
        display_time, time_unit, _ = display_time_axis(data.time)
        raw_current = np.asarray(self.analysis.current_a, dtype=float) * self.analysis.scaler
        raw_transform = np.asarray(self.analysis.transform, dtype=float)
        filter_active = bool(self.filter_stages) and self._filter_error is None
        ax_sd, ax_norm, ax_cv, ax_resid = self.axes.flat
        palette = ["#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#0072B2"]
        saved_limits = None
        if self._preserve_plot_limits and self._view_is_zoomed():
            saved_limits = [
                (axis.get_xlim(), axis.get_ylim()) for axis in self.axes.flat
            ]
        for axis in self.axes.flat:
            axis.clear()
        if filter_active:
            self._plot_series(ax_sd, display_time, raw_transform, color="#999999", linewidth=0.65, alpha=0.75, label="Original semi-derivative")
        self._plot_series(
            ax_sd,
            display_time,
            data.transform,
            color="#0072B2",
            linewidth=0.85,
            label="Filtered semi-derivative" if filter_active else "Semi-derivative",
        )
        self._draw_filter_region(ax_sd, display_time)
        if self.background_kind.currentData() == "cubic":
            try:
                poly = polynomial_reference(
                    data,
                    self.knot_indices,
                    require_complete=False,
                    knot_values=self.knot_values,
                    minimum_per_segment=self._minimum_knots(),
                    background_scope=str(self.background_scope.currentData()),
                )
                if np.any(np.isfinite(poly)):
                    self._plot_series(ax_sd, display_time, poly, color="#777777", linestyle=":", linewidth=1.0, label="Polynomial")
            except Exception:
                pass
        if self.outcome is None and self.knot_indices:
            preview = preview_interpolated_background(
                data,
                self.knot_indices,
                self.knot_values,
                str(self.background_kind.currentData()),
                str(self.background_scope.currentData()),
            )
            if np.any(np.isfinite(preview)):
                self._plot_series(
                    ax_sd,
                    display_time,
                    preview,
                    label="Background preview",
                    **BACKGROUND_LINE_STYLE,
                )
        knot_heights = [
            self.knot_values.get(index, float(data.transform[index]))
            for index in self.knot_indices
        ]
        ax_sd.scatter(display_time[self.knot_indices], knot_heights, marker="x", color="#000000", s=36, label="Knots", zorder=6)
        if self.peak_indices:
            ax_sd.scatter(display_time[self.peak_indices], data.transform[self.peak_indices], marker="o", facecolors="none", edgecolors="#CC79A7", s=50, label="Peaks", zorder=6)
        for number, index in enumerate(data.switching_indices, start=1):
            ax_sd.axvline(display_time[index], color="#BBBBBB", linewidth=0.75)
            ax_sd.annotate(f"S{number}", (display_time[index], data.transform[index]), xytext=(3, 5), textcoords="offset points", fontsize=7, color="#666666")
        for start, end in data.excluded_intervals:
            ax_sd.axvspan(
                display_time[start],
                display_time[end],
                color="#777777",
                alpha=0.12,
            )
        if self.pending_transient_index is not None:
            ax_sd.axvline(display_time[self.pending_transient_index], color="#D55E00", linestyle=":", linewidth=1.1)

        if self.outcome is not None:
            outcome = self.outcome
            kind_label = {
                "makima": "Makima",
                "pchip": "PCHIP",
                "linear": "Linear",
                "cubic": "Cubic",
                "cpe": "CPE fit",
                "constant": "Constant fit",
            }.get(outcome.config.background_kind, outcome.config.background_kind.title())
            background_label = f"{kind_label} background"
            self._plot_series(
                ax_sd,
                display_time,
                outcome.spline_background,
                label=background_label,
                **BACKGROUND_LINE_STYLE,
            )
            self._plot_series(ax_sd, display_time, outcome.lmfit_result.best_fit, color="#D55E00", linestyle="--", linewidth=1.1, label="Total fit")
            peak_components = [(name, values) for name, values in outcome.components.items() if name.startswith("peak")]
            scale = max(float(np.max(np.abs(outcome.lmfit_result.best_fit - outcome.spline_background))), np.finfo(float).eps)

            if self._top_right_mode == "normalized":
                ax_norm.axhline(0, color="#999999", linewidth=0.65)
                self._plot_series(ax_norm, display_time, outcome.normalized_signal, color="#222222", linewidth=1.0, label="Peak signal")
                for number, (_, values) in enumerate(peak_components, start=1):
                    component = np.asarray(values, dtype=float).copy()
                    component[data.excluded_mask] = np.nan
                    self._plot_series(ax_norm, display_time, component / scale, linestyle=":", linewidth=0.95, color=palette[(number - 1) % len(palette)], label=f"P{number}")
                for start, end in data.excluded_intervals:
                    ax_norm.axvspan(display_time[start], display_time[end], color="#777777", alpha=0.10)
            else:
                measured_sd = np.asarray(data.transform, dtype=float).copy()
                self._plot_segments(ax_norm, measured_sd, color="#0072B2", linewidth=0.9, label="Semi-derivative")
                self._plot_segments(ax_norm, outcome.lmfit_result.best_fit, color="#D55E00", linestyle="--", linewidth=1.1, label="Total fit")
                self._plot_segments(
                    ax_norm,
                    outcome.spline_background,
                    label="Background",
                    **BACKGROUND_LINE_STYLE,
                )
                for number, (_, values) in enumerate(peak_components, start=1):
                    self._plot_segments(ax_norm, values, color=palette[(number - 1) % len(palette)], linestyle=":", linewidth=0.9, label=f"P{number}")

            measured_current = np.asarray(data.current, dtype=float).copy()
            if self._bottom_left_mode == "potential":
                if filter_active:
                    self._plot_segments(ax_cv, raw_current, color="#999999", linewidth=0.7, alpha=0.75, label="Original")
                self._plot_segments(ax_cv, measured_current, color="#0072B2", linewidth=1.0, label="Filtered" if filter_active else "Measured")
                self._plot_segments(ax_cv, outcome.reconstructed_current, color="#222222", linestyle="--", linewidth=1.1, label="Reconstruction")
                self._plot_segments(ax_cv, outcome.segment_baseline_current, color="#009E73", linestyle=":", linewidth=1.0, label="Total baseline")
                for peak in outcome.peak_results:
                    color = palette[(peak.number - 1) % len(palette)]
                    ax_cv.vlines(
                        peak.potential_v,
                        peak.baseline_current_scaled,
                        peak.baseline_current_scaled + peak.current_scaled,
                        color=color,
                        linestyle="--",
                        linewidth=1.1,
                    )
                    ax_cv.annotate(
                        f"P{peak.number}",
                        (peak.potential_v, peak.baseline_current_scaled + peak.current_scaled),
                        xytext=(3, 5),
                        textcoords="offset points",
                        fontsize=7,
                        color=color,
                    )
            else:
                if filter_active:
                    self._plot_series(ax_cv, display_time, raw_current, color="#999999", linewidth=0.7, alpha=0.75, label="Original")
                self._plot_series(ax_cv, display_time, measured_current, color="#0072B2", linewidth=1.0, label="Filtered" if filter_active else "Measured")
                self._draw_filter_region(ax_cv, display_time)
                self._plot_series(ax_cv, display_time, outcome.reconstructed_current, color="#222222", linestyle="--", linewidth=1.1, label="Reconstruction")
                self._plot_series(
                    ax_cv,
                    display_time,
                    outcome.segment_baseline_current,
                    label="Total baseline",
                    **BACKGROUND_LINE_STYLE,
                )
                for number, (name, values) in enumerate(
                    ((name, values) for name, values in outcome.current_components.items() if name.startswith("peak")),
                    start=1,
                ):
                    self._plot_series(ax_cv, display_time, values, color=palette[(number - 1) % len(palette)], linestyle=":", linewidth=0.95, label=f"P{number}")
                for start, end in data.excluded_intervals:
                    ax_cv.axvspan(display_time[start], display_time[end], color="#777777", alpha=0.10)

            if self._quality_panel_mode == "table":
                self._draw_live_peak_table(ax_resid, outcome)
            else:
                residual = data.transform - outcome.lmfit_result.best_fit
                residual = np.asarray(residual, dtype=float)
                residual[data.excluded_mask] = np.nan
                ax_resid.axhline(0, color="#999999", linewidth=0.65)
                self._plot_series(ax_resid, display_time, residual, color="#D55E00", linewidth=0.85)
                ax_resid.text(
                    0.02,
                    0.97,
                    f"SD RMSE = {outcome.metrics['sd_rmse']:.4g} {data.scale_prefix}A s^-1/2\n"
                    f"Current RMSE = {outcome.metrics['current_rmse']:.4g} {data.scale_prefix}A\n"
                    f"lag-1 = {outcome.metrics['sd_residual_lag1']:.4f}",
                    transform=ax_resid.transAxes,
                    va="top",
                    fontsize=8,
                    bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "#CCCCCC"},
                )
                for start, end in data.excluded_intervals:
                    ax_resid.axvspan(display_time[start], display_time[end], color="#777777", alpha=0.10)
        else:
            ax_norm.text(0.5, 0.5, "Fit a peak to populate this view", ha="center", va="center", transform=ax_norm.transAxes, color="#777777")
            measured_current = np.asarray(data.current, dtype=float).copy()
            if self._bottom_left_mode == "potential":
                if filter_active:
                    self._plot_segments(ax_cv, raw_current, color="#999999", linewidth=0.7, alpha=0.75, label="Original")
                self._plot_segments(ax_cv, measured_current, color="#0072B2", linewidth=1.0, label="Filtered" if filter_active else "Measured")
            else:
                if filter_active:
                    self._plot_series(ax_cv, display_time, raw_current, color="#999999", linewidth=0.7, alpha=0.75, label="Original")
                self._plot_series(ax_cv, display_time, measured_current, color="#0072B2", linewidth=1.0, label="Filtered" if filter_active else "Measured")
                self._draw_filter_region(ax_cv, display_time)
            if self._quality_panel_mode == "table":
                self._draw_live_peak_table(ax_resid, None)
            else:
                ax_resid.text(0.5, 0.5, "Fit a peak to populate residuals", ha="center", va="center", transform=ax_resid.transAxes, color="#777777")

        self._draw_upper_envelope_maxima(
            ax_sd,
            display_time,
            data.transform,
            data,
            label=True,
        )
        cv_x_values = data.potential if self._bottom_left_mode == "potential" else display_time
        self._draw_upper_envelope_maxima(
            ax_cv,
            cv_x_values,
            data.current,
            data,
            label=True,
        )
        ax_sd.set(title="Semi-derivative fit", xlabel=f"Time ({time_unit})", ylabel=f"Semi-derivative ({data.scale_prefix}A s$^{{-1/2}}$)")
        if self._top_right_mode == "normalized":
            ax_norm.set(title="Normalized peaks (click: potential view)", xlabel=f"Time ({time_unit})", ylabel="Normalized signal")
        else:
            ax_norm.set(title="Semi-derivative fit vs potential (click: normalized view)", xlabel="Potential (V)", ylabel=f"Semi-derivative ({data.scale_prefix}A s$^{{-1/2}}$)")
        if self._bottom_left_mode == "potential":
            ax_cv.set(title="CV reconstruction (click: time view)", xlabel="Potential (V)", ylabel=f"Current ({data.scale_prefix}A)")
        else:
            ax_cv.set(title="Current reconstruction vs time (click: CV view)", xlabel=f"Time ({time_unit})", ylabel=f"Current ({data.scale_prefix}A)")
        if self._quality_panel_mode == "table":
            ax_resid.set(title="Peak table (click: residuals)", xlabel="", ylabel="")
        else:
            ax_resid.set(
                title="Semi-derivative residuals (click: table)",
                xlabel=f"Time ({time_unit})",
                ylabel=f"Residual ({data.scale_prefix}A s$^{{-1/2}}$)",
            )
        self._draw_interval_measurement(data)
        for axis in self.axes.flat:
            handles, labels = axis.get_legend_handles_labels()
            if handles:
                axis.legend(fontsize=7, frameon=False, ncol=2)
            if axis is ax_resid and self._quality_panel_mode == "table":
                axis.grid(False)
            else:
                axis.grid(alpha=0.14)
        if saved_limits is not None:
            for axis, (xlim, ylim) in zip(self.axes.flat, saved_limits):
                axis.set_xlim(xlim)
                axis.set_ylim(ylim)
        else:
            for axis in self.axes.flat:
                if axis is ax_resid and self._quality_panel_mode == "table":
                    continue
                axis.relim()
                axis.autoscale_view()
                axis.margins(x=0.04, y=0.08)
            self._home_limits = [
                (axis.get_xlim(), axis.get_ylim()) for axis in self.axes.flat
            ]
        if self._focused_axis_index is not None:
            for number, axis in enumerate(self.axes.flat):
                axis.set_visible(number == self._focused_axis_index)
            self.axes.flat[self._focused_axis_index].set_position(
                [0.12, 0.13, 0.84, 0.79]
            )
            self._apply_focused_plot_typography(
                self.axes.flat[self._focused_axis_index]
            )
        self._preserve_plot_limits = True
        self.canvas.draw_idle()

    def finalize_export(self) -> None:
        if self.analysis is None or not self.peak_indices:
            return
        if not self.output_dir:
            QMessageBox.warning(self, "Missing output folder", "Choose an output folder first.")
            return
        name = self.output_name.text().strip()
        if not name:
            name = "makima_fit"
        if any(character in name for character in '<>:"/\\|?*'):
            QMessageBox.warning(self, "Invalid output name", "The output name contains a character Windows cannot use.")
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.statusBar().showMessage("Finalizing the fit and writing the report…")
        QApplication.processEvents()
        try:
            working = self.working_analysis()
            if self._filter_error is not None:
                raise ValueError(self._filter_error)
            final_outcome = fit_spline_peaks(
                working,
                self.knot_indices,
                self.peak_indices,
                self._config(),
                preview=False,
                knot_values=self.knot_values,
            )
            paths = export_results(final_outcome, self.output_dir, name, self.detailed_csv.isChecked())
            self.outcome = final_outcome
            self.copy_table_button.setEnabled(True)
            self.draw_live_view()
            path_lines = "\n".join(str(path) for path in paths.values())
            QMessageBox.information(self, "Export complete", f"Makima Fitter wrote:\n\n{path_lines}")
            self.statusBar().showMessage(f"Export complete: {paths['pdf']}")
        except Exception as error:
            QMessageBox.critical(self, "Export failed", str(error))
            traceback.print_exc()
            self.statusBar().showMessage("Export failed")
        finally:
            QApplication.restoreOverrideCursor()

    def show_hotkeys(self) -> None:
        groups = [
            (
                "Plot navigation",
                [
                    ("H", "Restore the home view"),
                    ("M", "Toggle the move tool"),
                    ("Z", "Toggle the zoom-box tool"),
                    ("W", "Toggle the full-screen window"),
                    ("Ctrl+M", "Hide/show the left menu"),
                    ("Ctrl+= / Ctrl+-", "Zoom all visible plots in / out"),
                    ("Arrow keys", "Move plots"),
                    ("Shift/Ctrl+Arrow", "Move plots farther"),
                    ("Esc", "Leave navigation or full-screen focus"),
                    ("Ctrl+click", "Focus a plot full-screen; repeat to restore"),
                    ("T / Δt ↔ f button", "Toggle the two-point interval ruler"),
                ],
            ),
            (
                "Selections and models",
                [
                    ("K / P / X", "Knot, peak, or transient mode"),
                    ("B", "Cycle background models"),
                    ("S", "Toggle background scope"),
                    ("V", "Toggle global/selected-range filtering"),
                    ("L", "Lock/unlock knots to measured data"),
                    ("F", "Find peak candidates"),
                    ("C", "Clear the active selection type"),
                    ("R", "Reset settings; keep selected features"),
                    ("Ctrl+R", "Reset selections, settings, and view"),
                    ("Q", "Show/hide QC warning details"),
                    ("Ctrl+Z", "Undo selections or scan changes"),
                    ("Ctrl+Shift+Z / Ctrl+Y", "Redo"),
                ],
            ),
            (
                "Input, output, and panels",
                [
                    ("I / Ctrl+O", "Select a CV or .makima session"),
                    ("Ctrl+S", "Save the current session"),
                    ("Ctrl+Shift+S", "Save the session with a new name"),
                    ("N", "Focus the complete-scan selector"),
                    ("+ / − / × / ÷", "Calculate the active table column"),
                    ("Backspace", "Clear the active calculation"),
                    ("Ctrl+Backspace", "Clear the calculator row"),
                    ("E", "Export the PDF and CSV files"),
                    ("Ctrl+T", "Copy the peak table"),
                    ("Ctrl+C", "Copy the application window"),
                    ("Ctrl+Q", "Quit Makima Fitter"),
                    ("Click upper-right", "Switch normalized/potential view"),
                    ("Click lower-left", "Switch CV/time view"),
                    ("Click lower-right", "Switch residual/table view"),
                ],
            ),
        ]
        dialog = QDialog(self)
        dialog.setObjectName("keyboardShortcutsDialog")
        dialog.setWindowTitle("Makima Fitter keyboard shortcuts")
        layout = QVBoxLayout(dialog)
        columns = QHBoxLayout()
        for title, rows in groups:
            group = QGroupBox(title)
            group.setObjectName("shortcutColumn")
            group_layout = QVBoxLayout(group)
            table = "<table cellspacing='5'>" + "".join(
                f"<tr><td valign='top'><b>{keys}</b></td>"
                f"<td valign='top'>{description}</td></tr>"
                for keys, description in rows
            ) + "</table>"
            label = QLabel(table)
            label.setWordWrap(True)
            label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            label.setMinimumWidth(250)
            label.setMaximumWidth(350)
            group_layout.addWidget(label)
            group_layout.addStretch(1)
            columns.addWidget(group, 1)
        layout.addLayout(columns)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        buttons.accepted.connect(dialog.accept)
        layout.addWidget(buttons)
        screen = self.screen() or QApplication.primaryScreen()
        available = screen.availableGeometry() if screen is not None else None
        if available is not None:
            dialog.setMaximumSize(int(available.width() * 0.94), int(available.height() * 0.90))
            dialog.resize(min(1050, int(available.width() * 0.90)), min(520, int(available.height() * 0.80)))
        self._hotkey_dialog = dialog
        dialog.exec_()


def main() -> int:
    configure_windows_app_identity()
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    app.setApplicationName("Makima Fitter")
    app.setApplicationDisplayName("Makima Fitter")
    app.setOrganizationName("Makima Fitter")
    app.setWindowIcon(QIcon(str(application_icon_path())))
    initial_input = sys.argv[1] if len(sys.argv) > 1 else None
    window = MainWindow(initial_input=initial_input)
    window.show()
    apply_windows_taskbar_icon(window)
    window.raise_()
    window.activateWindow()
    QApplication.setActiveWindow(window)
    QTimer.singleShot(0, window._prime_keyboard_focus)
    QTimer.singleShot(150, window._prime_keyboard_focus)
    QTimer.singleShot(250, lambda: apply_windows_taskbar_icon(window))
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
