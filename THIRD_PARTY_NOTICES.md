# Third-party notices

## Modified-sinc smoothing algorithm

The modified-sinc kernel construction and weighted linear edge-extrapolation
logic in `source/spline_core.py` is a Python port of the reference algorithm
published with:

M. Schmid, D. Rath, and U. Diebold, “Why and How Savitzky-Golay Filters Should
Be Replaced,” *ACS Measurement Science Au* 2 (2022), 185-196.
https://doi.org/10.1021/acsmeasuresciau.1c00054

Reference implementation copyright Michael Schmid, IAP/TU Wien, 2021. The
authors distribute the accompanying implementation under GNU General Public
License version 3 and request citation of the paper for scientific use. Makima
Fitter is also distributed under GNU General Public License version 3. The
Python port changes language, array layout, and integration with the
sweep-aware/regional filtering pipeline; the published correction constants
and algorithmic equations are retained.
