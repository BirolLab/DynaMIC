#!/usr/bin/env python3

import os
import subprocess
import sys
import tempfile
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd

from dynamic import __version__ as TOOL_VERSION
from dynamic.utils import (LogoArgumentParser, console_write, emit_message,
                           get_run_logo_lines, style_text)

TESTDATA = resources.files(__package__) / "testdata"
RESULTS_PATH = Path("modelling") / "tables" / "dynamic_results.csv"
KEYS = ["treatment", "organism", "dose_layout"]
CHECKS = {"dose_response_label": "Class calls", "fit_status": "Fit status",
          "discrete_mic": "Discrete MIC", "mic_qc": "MIC QC flags",
          "midpoint_qc": "Midpoint QC flags", "slope_qc": "Slope QC flags",
          "replicate_outliers": "Replicate outliers", "sc_imputed": "Control imputation",
          "fit_slope": "Fitted slope", "fit_midpoint": "Fitted midpoint",
          "continuous_mic": "Continuous MIC"}
NUMERIC = {"fit_slope", "fit_midpoint", "continuous_mic"}
TOLERANCE = 1e-6
MAX_WORKERS = 4


def detail(label, value):
    return f"{style_text(f'{label}:', '1')} {value}"


def run_pipeline(input_path, run_dir):
    package_root = str(Path(__file__).resolve().parents[1])
    environment = {**os.environ, "DYNAMIC_SUPPRESS_STAGE_LOGO": "1",
                   "PYTHONPATH": os.pathsep.join(
                       filter(None, [package_root, os.environ.get("PYTHONPATH", "")]))}
    return subprocess.run([sys.executable, "-m", "dynamic.dynamic", "--input", str(input_path),
                           "--output-dir", str(run_dir), "--plot", "no", "--report", "no",
                           "--max-workers", str(MAX_WORKERS)],
                          env=environment, capture_output=True, text=True)


def summarize(results):
    aggregations = {column: (column, "first") for column in CHECKS if column in results.columns}
    aggregations["replicate_outliers"] = ("replicate_curve_outlier", "sum")
    aggregations["sc_imputed"] = ("SC_imputed", "any")
    return results.groupby(KEYS, as_index=False).agg(**aggregations)


def compare(merged, column):
    observed, expected = merged[column], merged[f"{column}_expected"]
    if column in NUMERIC:
        return np.isclose(pd.to_numeric(observed, errors="coerce"),
                          pd.to_numeric(expected, errors="coerce"), rtol=TOLERANCE, equal_nan=True)
    return (observed.fillna("").astype(str) == expected.fillna("").astype(str)).to_numpy()


def main():
    LogoArgumentParser(description="DynaMIC installation test").parse_args()
    for line in get_run_logo_lines():
        console_write(line)
    emit_message("info", "DynaMIC Test", lines=[detail("Version", TOOL_VERSION),
                                                detail("Dataset", "test_input.csv")])

    with tempfile.TemporaryDirectory(prefix="dynamic_test_") as scratch:
        run_dir = Path(scratch) / "run"
        with resources.as_file(TESTDATA / "test_input.csv") as input_path:
            completed = run_pipeline(input_path, run_dir)
        if completed.returncode != 0:
            emit_message("error", "DynaMIC Test Failed",
                         lines=detail("Reason", "the pipeline did not complete"),
                         items=completed.stderr.strip().splitlines()[-8:] or ["no error output"])
            return 1
        observed = summarize(pd.read_csv(run_dir / RESULTS_PATH, low_memory=False))

    merged = observed.merge(pd.read_csv(TESTDATA / "expected_results.csv"), on=KEYS, how="outer",
                            suffixes=("", "_expected"), indicator=True)
    matches = {column: compare(merged, column) for column in CHECKS}
    lines = [detail(label, f"{int(matches[column].sum())}/{len(merged)}")
             for column, label in CHECKS.items()]
    failures = [f"{label} differs for {treatment}" for column, label in CHECKS.items()
                for treatment in merged.loc[~matches[column], "treatment"].head(3)]
    missing = int(merged["_merge"].ne("both").sum())
    if missing:
        failures.insert(0, f"{missing} treatment(s) missing from the results")

    if failures:
        emit_message("error", "DynaMIC Test Failed", lines=lines, items=failures)
        return 1
    emit_message("success", "DynaMIC Test Passed", lines=lines)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
