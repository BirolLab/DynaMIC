#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
import subprocess
import sys
import traceback

from dynamic.utils import (ASCII_logo, LogoArgumentParser, RunLogger, TEXT_COLOURS,
                           auto_or_positive_int, create_cwd_dated_run_directory, emit_message, format_detail,
                           format_dose_layout_full, open_probability, parse_csv_list,
                           positive_float, positive_int, probability, prepare_output_directories,
                           style_text, write_run_parameters)


from dynamic import __version__ as TOOL_VERSION
DEFAULT_RUN_DIR_PREFIX = "dynamic_results"
DEFAULT_Q_THRESHOLD = 0.05
AUTO_NULL_CURVES = "auto"
DEFAULT_MIN_SLOPE = 0.01
DEFAULT_MAX_SLOPE = 30.0
DEFAULT_REPLICATE_OUTLIER_ALPHA = 0.05
DEFAULT_MIC_INHIBITION_THRESHOLD = 0.95
DEFAULT_FIT_MODEL = "2pl"
DEFAULT_N_BOOTSTRAPS = 500
DEFAULT_MAX_WORKERS = 30
CONTROL_TREATMENTS = {"Growth Control", "Sterility Control"}


class WorkflowValidationError(RuntimeError):
    def __init__(self, message, reported=False):
        super().__init__(message)
        self.reported = reported


def parse_args():
    parser = LogoArgumentParser(description="Full DynaMIC Workflow", formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Raw input CSV.")
    parser.add_argument("--output-dir", default=None, help="Parent directory for the complete DynaMIC run.")
    parser.add_argument("--organism", action="append", default=None, help="Organism to analyze. Repeat to select multiple. Default: all organisms.")
    parser.add_argument("--plot", choices=["yes", "no"], default="yes", help="Save production plots. Default: yes.")
    parser.add_argument("--report", choices=["yes", "no"], default="yes", help="Write the interactive Report.html at the end of the run. Default: yes.")

    parser.add_argument("--missing-controls", choices=["impute", "exclude"], default="impute", help="Normalization handling for missing assay-level controls. Default: impute.")
    parser.add_argument("--within-assay-outlier-controls", choices=["keep", "exclude"], default="exclude", help="Normalization handling for within-assay control outliers. Default: exclude.")
    parser.add_argument("--across-assay-outlier-controls", dest="outlier_controls", choices=["report", "impute", "exclude"], default="report", help="Handle across-assay Growth Control/Sterility Control outliers: report only, impute from organism medians, or exclude assay. Default: report.")

    parser.add_argument("--calibration-dir", default=None, help="Use an existing calibration directory and skip calibration.")
    parser.add_argument("--n-null-curves", type=auto_or_positive_int, default=AUTO_NULL_CURVES, help="Calibration null curves per null: 'auto' sizes each set from --q-threshold (floor 50,000) or a positive integer applied uniformly. Default: auto.")
    parser.add_argument("--min-slope", type=positive_float, default=DEFAULT_MIN_SLOPE, help="Minimum template and fitted slope. Default: 0.01.")
    parser.add_argument("--max-slope", type=positive_float, default=DEFAULT_MAX_SLOPE, help="Maximum template and fitted slope. Default: 30.")

    parser.add_argument("--q-threshold", type=probability, default=DEFAULT_Q_THRESHOLD, help="Inhibition q threshold. Default: 0.05.")
    parser.add_argument("--replicate-outlier-alpha", type=probability, default=DEFAULT_REPLICATE_OUTLIER_ALPHA, help="Iterative extreme-value replicate outlier alpha. Default: 0.05.")
    parser.add_argument("--replicate-outliers", choices=["exclude", "include", "skip"], default="exclude", help="Exclude or include replicate outliers in treatment-level curves, or skip outlier detection. Default: exclude.")

    parser.add_argument("--fit-label", default="dose-variant", help="Inhibition class to fit. Default: dose-variant.")
    parser.add_argument("--fit-model", choices=["2pl", "3pl", "4pl"], default=DEFAULT_FIT_MODEL, help="Inhibition model for curve fitting. Default: 2pl.")
    parser.add_argument("--mic-threshold", type=open_probability, default=DEFAULT_MIC_INHIBITION_THRESHOLD, help="Inhibition threshold for fitted MIC. Default: 0.95.")
    parser.add_argument("--bootstrap", choices=["on", "off"], default="on", help="Turn bootstrap uncertainty on or off. Default: on.")
    parser.add_argument("--n-bootstraps", type=positive_int, default=DEFAULT_N_BOOTSTRAPS, help="Maximum bootstrap draws. Default: 500.")
    parser.add_argument("--max-workers", "--threads", dest="max_workers", type=positive_int, default=DEFAULT_MAX_WORKERS, help=f"Parallel workers for fitting. Default: {DEFAULT_MAX_WORKERS}.")
    args = parser.parse_args()
    if args.min_slope >= args.max_slope:
        parser.error("--min-slope must be less than --max-slope")
    return args


def run_command(command, stage_name, env):
    emit_message("stage", stage_name)
    process = subprocess.run(command, env=env)
    return_code = process.returncode
    if return_code != 0:
        raise RuntimeError(f"{stage_name} failed with exit code {return_code}.")


def stage_command(stage_name, *arguments):
    return [sys.executable, "-m", f"{__package__}.{stage_name}", *map(str, arguments)]


def stage_environment(env):
    package_root = str(Path(__file__).resolve().parents[1])
    existing = env.get("PYTHONPATH", "")
    if package_root not in existing.split(os.pathsep):
        env["PYTHONPATH"] = os.pathsep.join([package_root, existing]) if existing else package_root
    return env


def maybe_add_organisms(command, organisms):
    for organism in organisms or []:
        command.extend(["--organism", organism])


def format_compact_list(values, max_items=6):
    values = [str(value) for value in values]
    if len(values) <= max_items:
        return ", ".join(values)
    return f"{', '.join(values[:max_items])}, +{len(values) - max_items} more"


def validate_reused_calibration(calibration_dir, normalized_data_path, q_threshold, selected_organisms=None):
    import math
    import pandas as pd
    manifest_path = Path(calibration_dir) / "calibration_manifest.csv"
    if not manifest_path.exists():
        raise WorkflowValidationError(f"Calibration manifest was not found: {manifest_path}")

    manifest = pd.read_csv(manifest_path)
    missing_manifest_columns = sorted({"organism", "inhibition_columns"} - set(manifest.columns))
    if missing_manifest_columns:
        raise WorkflowValidationError(
            "Calibration manifest is missing required column(s): "
            + ", ".join(missing_manifest_columns)
        )

    baked_null_sizes = {}
    null_curve_summary_path = Path(calibration_dir) / "tables" / "null_curve_summary.csv"
    if null_curve_summary_path.exists():
        null_curve_summary = pd.read_csv(null_curve_summary_path)
        if {"organism", "dose_layout", "n_null_curves"} <= set(null_curve_summary.columns):
            baked_null_sizes = null_curve_summary.groupby(["organism", "dose_layout"])["n_null_curves"].min().to_dict()

    normalized_headers = pd.read_csv(normalized_data_path, nrows=0).columns
    missing_normalized_columns = sorted({"organism", "treatment"} - set(normalized_headers))
    if missing_normalized_columns:
        raise WorkflowValidationError(
            "Normalized data are missing required column(s): "
            + ", ".join(missing_normalized_columns)
        )

    normalized_data = pd.read_csv(normalized_data_path, usecols=["organism", "treatment"])
    normalized_organisms = sorted(
        normalized_data.loc[
            ~normalized_data["treatment"].isin(CONTROL_TREATMENTS),
            "organism",
        ].dropna().unique()
    )
    calibration_organisms = sorted(manifest["organism"].dropna().unique())
    requested_organisms = list(dict.fromkeys(selected_organisms or []))
    analysis_organisms = requested_organisms or normalized_organisms

    issues = []
    under_resolved = []
    if requested_organisms:
        missing_from_calibration = [organism for organism in analysis_organisms if organism not in calibration_organisms]
        missing_from_normalized = [organism for organism in analysis_organisms if organism not in normalized_organisms]
        if missing_from_calibration:
            issues.append(f"Missing from calibration: {format_compact_list(missing_from_calibration)}")
        if missing_from_normalized:
            issues.append(f"Missing from normalized data: {format_compact_list(missing_from_normalized)}")
    else:
        uncalibrated_organisms = [organism for organism in normalized_organisms if organism not in calibration_organisms]
        unused_calibrations = [organism for organism in calibration_organisms if organism not in normalized_organisms]
        if uncalibrated_organisms:
            issues.append(f"Uncalibrated normalized organism(s): {format_compact_list(uncalibrated_organisms)}")
        if unused_calibrations:
            issues.append(f"Calibration organism(s) absent from normalized data: {format_compact_list(unused_calibrations)}")

    manifest_for_check = manifest[manifest["organism"].isin(analysis_organisms)]
    normalized_header_set = set(normalized_headers)
    dose_layout_records = []
    for row in manifest_for_check.itertuples(index=False):
        inhibition_columns = parse_csv_list(row.inhibition_columns)
        dose_layout = getattr(row, "dose_layout", format_dose_layout_full(inhibition_columns))
        missing_dose_columns = [column for column in inhibition_columns if column not in normalized_header_set]
        if missing_dose_columns:
            issues.append(f"{row.organism} ({dose_layout}): missing dose column(s) {format_compact_list(missing_dose_columns)}")
            continue
        dose_layout_records.append((row.organism, dose_layout, inhibition_columns))

    if dose_layout_records:
        dose_columns = sorted({column for _, _, columns in dose_layout_records for column in columns})
        curve_data = pd.read_csv(normalized_data_path, usecols=["organism", "treatment", *dose_columns])
        for organism, dose_layout, inhibition_columns in dose_layout_records:
            organism_rows = curve_data["organism"].eq(organism) & ~curve_data["treatment"].isin(CONTROL_TREATMENTS)
            extra_dose_columns = [column for column in dose_columns if column not in set(inhibition_columns)]
            exact_set = curve_data[inhibition_columns].notna().all(axis=1)
            if extra_dose_columns:
                exact_set &= curve_data[extra_dose_columns].isna().all(axis=1)
            selected_rows = organism_rows & exact_set
            if selected_rows.sum() == 0:
                issues.append(f"{organism} ({dose_layout}): no exact-set curves for doses {format_dose_layout_full(inhibition_columns)}")
                continue
            baked_null_size = baked_null_sizes.get((organism, dose_layout))
            if baked_null_size and q_threshold > 0:
                complete_treatments = curve_data.loc[selected_rows, ["organism", "treatment"]].drop_duplicates().shape[0]
                required_null_size = math.ceil(complete_treatments / q_threshold)
                if required_null_size > int(baked_null_size):
                    under_resolved.append(f"{organism} ({dose_layout}): {complete_treatments:,} treatments need {required_null_size:,} null curves but calibration has {int(baked_null_size):,}")

    if issues:
        emit_message(
            "error",
            "Existing calibration does not match normalized data",
            lines=[
                format_detail("Calibration", style_text(str(calibration_dir), TEXT_COLOURS["accent"])),
                "Rerun without --calibration-dir, or provide a calibration generated for this dataset.",
            ],
            items=issues,
        )
        raise WorkflowValidationError(
            "Existing calibration is not compatible with the normalized data.",
            reported=True,
        )

    if under_resolved:
        emit_message(
            "warn",
            "Reused calibration null resolution is below the current dataset",
            lines=[
                format_detail("Calibration", style_text(str(calibration_dir), TEXT_COLOURS["accent"])),
                "Null curves were sized for the calibration set; the current set is larger, so the empirical null no longer spans the target q threshold. Recalibrate for this dataset or set --n-null-curves to restore full resolution.",
            ],
            items=under_resolved,
        )

    emit_message(
        "info",
        "Existing calibration validated",
        lines=format_detail("Organisms", f"{len(analysis_organisms):,}"),
    )


def run_pipeline(args, run_dir):
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp")
    env.setdefault("XDG_CACHE_HOME", "/tmp")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env["DYNAMIC_SUPPRESS_STAGE_LOGO"] = "1"
    env = stage_environment(env)

    normalization_dir = run_dir / "normalization"
    calibration_dir = Path(args.calibration_dir) if args.calibration_dir else run_dir / "calibration"
    classification_dir = run_dir / "classification"
    modelling_dir = run_dir / "modelling"
    normalization_tables_dir = normalization_dir / "tables"
    normalized_data_path = normalization_tables_dir / f"{Path(args.input).stem}_with_if.csv"
    control_data_path = normalization_tables_dir / "control_pool.csv"

    emit_message("info", "DynaMIC Full Workflow", lines=[
        *ASCII_logo,
        format_detail("Version", TOOL_VERSION),
        format_detail("Input", style_text(str(args.input), TEXT_COLOURS["accent"])),
        format_detail("Output", style_text(str(run_dir), TEXT_COLOURS["accent"])),
        format_detail("Organisms", ", ".join(args.organism) if args.organism else "all"),
        format_detail("Calibration", "existing" if args.calibration_dir else "new"),
        format_detail("Null curves", "auto" if args.n_null_curves == AUTO_NULL_CURVES else f"{args.n_null_curves:,}"),
        format_detail("Fit model", args.fit_model),
        format_detail("Bootstrap", args.bootstrap),
        format_detail("Fitting workers", args.max_workers),
    ])

    normalization_command = stage_command(
        "normalization",
        "--input", args.input,
        "--output-dir", normalization_dir,
        "--missing-controls", args.missing_controls,
        "--within-assay-outlier-controls", args.within_assay_outlier_controls,
        "--across-assay-outlier-controls", args.outlier_controls,
    )
    maybe_add_organisms(normalization_command, args.organism)
    if args.plot == "no":
        normalization_command.extend(["--plot", "N"])
    run_command(normalization_command, "Step 1/4: Normalization", env)

    if not normalized_data_path.exists():
        raise RuntimeError(f"Normalization output was not found at expected path: {normalized_data_path}")
    if not control_data_path.exists():
        raise RuntimeError(f"Control pool was not found at expected path: {control_data_path}")

    if args.calibration_dir is None:
        calibration_command = stage_command(
            "calibration",
            "--normalized-data", normalized_data_path,
            "--control-data", control_data_path,
            "--output-dir", calibration_dir,
            "--n-null-curves", args.n_null_curves,
            "--q-threshold", args.q_threshold,
            "--min-slope", args.min_slope,
            "--max-slope", args.max_slope,
            "--plot", args.plot,
        )
        maybe_add_organisms(calibration_command, args.organism)
        run_command(calibration_command, "Step 2/4: Calibration", env)
    else:
        emit_message("stage", "Step 2/4: Calibration skipped", lines=format_detail("Using", style_text(str(calibration_dir), TEXT_COLOURS["accent"])))
        validate_reused_calibration(calibration_dir, normalized_data_path, args.q_threshold, args.organism)

    classification_command = stage_command(
        "classification",
        "--normalized-data", normalized_data_path,
        "--calibration-dir", calibration_dir,
        "--output-dir", classification_dir,
        "--q-threshold", args.q_threshold,
        "--replicate-outlier-alpha", args.replicate_outlier_alpha,
        "--replicate-outliers", args.replicate_outliers,
        "--max-workers", args.max_workers,
        "--plot", args.plot,
    )
    maybe_add_organisms(classification_command, args.organism)
    run_command(classification_command, "Step 3/4: Classification", env)

    modelling_command = stage_command(
        "modelling",
        "--normalized-data", normalized_data_path,
        "--classification-dir", classification_dir,
        "--calibration-dir", calibration_dir,
        "--output-dir", modelling_dir,
        "--fit-label", args.fit_label,
        "--fit-model", args.fit_model,
        "--min-slope", args.min_slope,
        "--max-slope", args.max_slope,
        "--mic-threshold", args.mic_threshold,
        "--bootstrap", args.bootstrap,
        "--n-bootstraps", args.n_bootstraps,
        "--max-workers", args.max_workers,
    )
    maybe_add_organisms(modelling_command, args.organism)
    run_command(modelling_command, "Step 4/4: Modelling", env)

    if args.report == "yes":
        report_command = stage_command("report", "--run-dir", run_dir)
        run_command(report_command, "Report", env)

    emit_message("success", "DynaMIC workflow complete", lines=[
        format_detail("Run directory", style_text(str(run_dir), TEXT_COLOURS["accent"])),
        format_detail("Normalization", "normalization/"),
        format_detail("Calibration", str(calibration_dir) if args.calibration_dir else "calibration/"),
        format_detail("Classification", "classification/"),
        format_detail("Modelling", "modelling/"),
        *([format_detail("Report", "Report.html")] if args.report == "yes" else []),
    ])


def main():
    args = parse_args()
    run_dir = create_cwd_dated_run_directory(args.output_dir, DEFAULT_RUN_DIR_PREFIX)
    directories = prepare_output_directories(run_dir, ["logs"])
    with RunLogger(directories["logs"] / "run_log.txt"):
        write_run_parameters(args, directories["logs"], "dynamic.py", TOOL_VERSION, {"output_dir": str(run_dir)})
        try:
            run_pipeline(args, run_dir)
        except WorkflowValidationError as error:
            if not error.reported:
                emit_message("error", "DynaMIC workflow stopped", lines=format_detail("Reason", str(error)))
            raise SystemExit(1)
        except Exception as error:
            emit_message("error", "DynaMIC workflow failed", lines=format_detail("Reason", str(error)))
            traceback.print_exc()
            raise


if __name__ == "__main__":
    main()
