"""Metrics / evaluation helpers.

This package is intentionally small at first.

Rationale (research workflow):
- For denoise comparison, exporting videos is convenient for qualitative review
  but inefficient for parameter tuning.
- We usually want numeric summaries like:
  - total events before/after
  - kept_ratio / removed_ratio
  - per-polarity counts

CLI commands `myevs stats`, `myevs compare-stats`, and `myevs sweep` are the
first step. More metrics (SNR/TP/FP with labels, ROI-based metrics, plots) can
be added here later.
"""

from __future__ import annotations

from .esr import event_structural_ratio_mean_from_xy
from .aocc import aocc_from_xyt
