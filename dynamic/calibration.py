#!/usr/bin/env python3

import argparse
import traceback
import math
import os
import re
from concurrent.futures import ProcessPoolExecutor
from typing import NamedTuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from dynamic.utils import (BackgroundRenderer, LogoArgumentParser, RunLogger, TEXT_COLOURS,
                           auto_or_positive_int, create_dated_run_directory, create_progress_bar, emit_message, format_detail,
                           format_dose_layout_full, format_stage_label, get_dose_labels,
                           get_run_logo_lines, nonnegative_int, parse_csv_list,
                           positive_float, positive_int, preload_fork_worker_modules,
                           prepare_output_directories, probability, require_columns, save_plot_figure,
                           set_progress_stage, slugify_filename_token, style_text, write_run_parameters)


from dynamic import __version__ as TOOL_VERSION
DEFAULT_RUN_DIR_PREFIX = "calibration_results"
DEFAULT_NULL_CURVES = 50_000
AUTO_NULL_CURVES = "auto"
DEFAULT_Q_THRESHOLD = 0.05
DEFAULT_NULL_CURVE_SEED = 1
DEFAULT_BLOCK_SIZE = 10_000
DEFAULT_N_SLOPES = 9
DEFAULT_N_MIDPOINTS = 25
DEFAULT_MIN_SLOPE = 0.01
DEFAULT_MAX_SLOPE = 30.0
DEFAULT_MIN_TEMPLATE_NORM = 1e-4
INHIBITION_COLUMN_RE = re.compile(r"I\d+(?:\.\d+)?")
GROWTH_CONTROL = "Growth Control"
STERILITY_CONTROL = "Sterility Control"
FULL_INHIBITION_NULL = "full inhibition null"
NO_INHIBITION_NULL = "no inhibition null"
NULL_SOURCE_CONTROLS = {FULL_INHIBITION_NULL: STERILITY_CONTROL, NO_INHIBITION_NULL: GROWTH_CONTROL}
CONTROL_INHIBITION_SOURCES = {
    "growth_control_inhibition": GROWTH_CONTROL,
    "sterility_control_inhibition": STERILITY_CONTROL,
}
NULL_COLOURS = {FULL_INHIBITION_NULL: "#DDA94B", NO_INHIBITION_NULL: "#1E4174"}
SCORE_COLUMN = "two_pl_signal_score"
MAX_POOL_WORKERS = 8


def load_normalized_inhibition_data(normalized_data_path, metadata_columns=("organism", "treatment")):
    headers = pd.read_csv(normalized_data_path, nrows=0)
    require_columns(headers.columns, ["organism", "treatment"], "Normalized data", normalized_data_path)
    metadata = set(metadata_columns)
    used_columns = [column for column in headers.columns
                    if column in metadata or INHIBITION_COLUMN_RE.fullmatch(column)]
    return pd.read_csv(normalized_data_path, usecols=used_columns)


def parse_args():
    parser = LogoArgumentParser(description="DynaMIC empirical null calibration", formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--normalized-data", required=True, help="CSV produced by normalization.py.")
    parser.add_argument("--control-data", required=True, help="control_pool.csv produced by normalization.py.")
    parser.add_argument("--output-dir", default=None, help="Directory for calibration outputs.")
    parser.add_argument("--organism", action="append", default=None, help="Organism to calibrate. Repeat to select multiple. Default: all organisms.")
    parser.add_argument("--n-null-curves", type=auto_or_positive_int, default=AUTO_NULL_CURVES, help="Calibration null curves per null: 'auto' sizes each set from --q-threshold (floor 50,000) or a positive integer applied uniformly. Default: auto.")
    parser.add_argument("--q-threshold", type=probability, default=DEFAULT_Q_THRESHOLD, help="Target FDR used to size 'auto' null curves. Default: 0.05.")
    parser.add_argument("--min-slope", type=positive_float, default=DEFAULT_MIN_SLOPE, help="Minimum template slope. Default: 0.01.")
    parser.add_argument("--max-slope", type=positive_float, default=DEFAULT_MAX_SLOPE, help="Maximum template slope. Default: 30.")
    parser.add_argument("--n-slopes", type=positive_int, default=DEFAULT_N_SLOPES, help="Number of template slope values. Default: 9.")
    parser.add_argument("--n-midpoints", type=positive_int, default=DEFAULT_N_MIDPOINTS, help="Number of template midpoints. Default: 25.")
    parser.add_argument("--null-seed", type=nonnegative_int, default=DEFAULT_NULL_CURVE_SEED, help="Calibration null random seed. Default: 1.")
    parser.add_argument("--plot", choices=["yes", "no"], default="yes", help="Save null-score plots. Default: yes.")
    args = parser.parse_args()
    if args.min_slope >= args.max_slope:
        parser.error("--min-slope must be less than --max-slope")
    return args


def resolve_null_curves(n_null_curves, n_experimental_treatments, q_threshold):
    if n_null_curves != AUTO_NULL_CURVES:
        return n_null_curves
    if q_threshold <= 0:
        return DEFAULT_NULL_CURVES
    return max(DEFAULT_NULL_CURVES, math.ceil(n_experimental_treatments / q_threshold))


def benjamini_hochberg(p_values):
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = ranked * len(p_values) / np.arange(1, len(p_values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.clip(adjusted, 0, 1)
    return output


def get_inhibition_columns(data_path, preloaded_data=None):
    columns = preloaded_data.columns if preloaded_data is not None else pd.read_csv(data_path, nrows=0).columns
    require_columns(columns, ["organism", "treatment"], "Normalized data", data_path)
    return sorted([column for column in columns if INHIBITION_COLUMN_RE.fullmatch(column)], key=lambda column: float(column[1:]), reverse=True)


def get_analysis_organisms(normalized_data_path, selected_organisms=None, preloaded_data=None):
    if preloaded_data is not None:
        require_columns(preloaded_data.columns, ["organism", "treatment"], "Normalized data", normalized_data_path)
        data = preloaded_data
    else:
        headers = pd.read_csv(normalized_data_path, nrows=0)
        require_columns(headers.columns, ["organism", "treatment"], "Normalized data", normalized_data_path)
        data = pd.read_csv(normalized_data_path, usecols=["organism", "treatment"])
    organisms = sorted(data.loc[~data["treatment"].isin(NULL_SOURCE_CONTROLS.values()), "organism"].dropna().unique())
    if selected_organisms:
        selected = set(selected_organisms)
        organisms = [organism for organism in organisms if organism in selected]
    if not organisms:
        raise ValueError("No organisms were available for calibration.")
    return organisms


def detect_experimental_dose_sets(normalized_data_path, organism, preloaded_data=None, inhibition_columns=None):
    if inhibition_columns is None:
        inhibition_columns = get_inhibition_columns(normalized_data_path, preloaded_data=preloaded_data)
    inhibition_columns = np.asarray(inhibition_columns, dtype=object)
    if preloaded_data is not None:
        data = preloaded_data
    else:
        data = pd.read_csv(normalized_data_path, usecols=["organism", "treatment", *inhibition_columns])
    experimental_rows = data["organism"].eq(organism) & ~data["treatment"].isin(NULL_SOURCE_CONTROLS.values())
    experimental = data.loc[experimental_rows, ["organism", "treatment"]].reset_index(drop=True)
    dose_present = data.loc[experimental_rows, list(inhibition_columns)].notna().to_numpy(bool)
    if dose_present.size == 0:
        raise ValueError(f"No experimental rows found for organism: {organism}")

    dose_patterns, pattern_index, counts = np.unique(dose_present, axis=0, return_inverse=True, return_counts=True)
    treatment_counts = (experimental.assign(dose_pattern=pattern_index.reshape(-1))
                        .drop_duplicates(["dose_pattern", "organism", "treatment"]).groupby("dose_pattern").size())
    order = np.argsort(counts)[::-1]
    dose_layout_records = []
    for rank, index in enumerate([index for index in order if dose_patterns[index].any()], start=1):
        columns = list(inhibition_columns[dose_patterns[index]])
        dose_layout_records.append({
            "dose_layout": format_dose_layout(columns),
            "dose_layout_full": format_dose_layout_full(columns),
            "dose_layout_token": make_dose_layout_token(columns),
            "dose_layout_rank": rank,
            "n_experimental_curves": int(counts[index]),
            "n_experimental_treatments": int(treatment_counts.get(index, 0)),
            "inhibition_columns": columns,
            "concentration_values": np.asarray([column[1:] for column in columns], dtype=float),
        })
    if not dose_layout_records:
        raise ValueError(f"No populated inhibition columns found for organism: {organism}")
    return dose_layout_records


def format_dose_layout(inhibition_columns):
    doses = get_dose_labels(inhibition_columns)
    return doses[0] if len(doses) == 1 else f"{doses[0]}-{doses[-1]}"


def make_dose_layout_token(inhibition_columns):
    return "doses_" + slugify_filename_token(format_dose_layout(inhibition_columns), default="dose")


def ensure_dose_layout_columns(manifest):
    if "inhibition_columns" not in manifest.columns:
        return manifest
    parsed = manifest["inhibition_columns"].map(parse_csv_list)
    if "dose_layout" not in manifest.columns:
        manifest["dose_layout"] = parsed.map(format_dose_layout)
    if "dose_layout_full" not in manifest.columns:
        manifest["dose_layout_full"] = parsed.map(format_dose_layout_full)
    if "dose_layout_token" not in manifest.columns:
        manifest["dose_layout_token"] = parsed.map(make_dose_layout_token)
    return manifest


def emit_dose_layout_selection(organism, dose_layout_records):
    if len(dose_layout_records) <= 1:
        return

    emit_message(
        "warn",
        f"{organism}: multiple dose layouts detected",
        lines=format_detail("Calibration sets", f"{len(dose_layout_records):,} evaluated separately"),
        items=[
            f"set {record['dose_layout_rank']} | curves = {record['n_experimental_curves']:,} | doses = {record['dose_layout']}"
            for record in dose_layout_records
        ],
    )


def get_control_inliers(control_data):
    require_columns(control_data.columns, ["organism", "treatment", "I"], "Control data", "control data")
    outlier_columns = [column for column in ["within_assay_mZ_score", "across_assay_mZ_score"] if column in control_data.columns]
    control_outlier_rows = control_data[outlier_columns].fillna(False).astype(bool).any(axis=1) if outlier_columns else pd.Series(False, index=control_data.index)
    return control_data.loc[control_data["treatment"].isin(NULL_SOURCE_CONTROLS.values()) & ~control_outlier_rows].copy()


def get_control_inhibition_measurements(control_inliers, organism, source_control):
    matching_control_rows = control_inliers["organism"].eq(organism) & control_inliers["treatment"].eq(source_control)
    values = pd.to_numeric(control_inliers.loc[matching_control_rows, "I"], errors="coerce").dropna().to_numpy(float)
    if values.size == 0:
        raise ValueError(
            f"No usable {source_control} inhibition measurements were found for {organism} after control QC. "
            "Calibration requires both Growth Control and Sterility Control measurements. "
            "For classification of a dataset without controls, provide --calibration-dir from a previous calibration run."
        )
    return values


def centre_vector(values):
    values = np.asarray(values, dtype=float)
    return values - values.mean()


def centre_matrix(curves):
    curves = np.asarray(curves, dtype=np.float32)
    return curves - curves.mean(axis=1)[:, None]


def two_parameter_logistic(concentrations, slope, midpoint):
    concentrations = np.asarray(concentrations, dtype=float)
    return 1.0 / (1.0 + (midpoint / concentrations) ** slope)


class TemplateLibrary(NamedTuple):
    matrix: np.ndarray
    transpose: np.ndarray
    candidate_slopes: np.ndarray
    candidate_midpoints: np.ndarray
    slopes: np.ndarray
    midpoints: np.ndarray


def build_2pl_template_matrix(concentration_values, n_slopes, n_midpoints, min_slope, max_slope, min_template_norm=DEFAULT_MIN_TEMPLATE_NORM):
    concentration_values = np.asarray(concentration_values, dtype=float)
    candidate_slopes = np.geomspace(min_slope, max_slope, n_slopes)
    candidate_midpoints = np.geomspace(concentration_values.min() / 2, concentration_values.max() * 2, n_midpoints)
    template_vectors, template_slopes, template_midpoints = [], [], []

    for slope in candidate_slopes:
        for midpoint in candidate_midpoints:
            centred_template = centre_vector(two_parameter_logistic(concentration_values, slope, midpoint))
            template_norm = np.linalg.norm(centred_template)
            if np.isfinite(template_norm) and template_norm >= min_template_norm:
                template_vectors.append(centred_template / template_norm)
                template_slopes.append(slope)
                template_midpoints.append(midpoint)

    template_matrix = np.ascontiguousarray(np.vstack(template_vectors), dtype=np.float32)
    return TemplateLibrary(template_matrix, np.ascontiguousarray(template_matrix.T),
                           candidate_slopes, candidate_midpoints,
                           np.asarray(template_slopes, dtype=np.float32),
                           np.asarray(template_midpoints, dtype=np.float32))


def calculate_2pl_signal_scores(curves, template_transpose, block_size=DEFAULT_BLOCK_SIZE, return_direction=True):
    curves = np.asarray(curves, dtype=np.float32)
    signal_scores = np.empty(curves.shape[0], dtype=np.float32)
    best_projections = np.empty(curves.shape[0], dtype=np.float32) if return_direction else None

    for start in range(0, curves.shape[0], block_size):
        stop = min(start + block_size, curves.shape[0])
        projections = centre_matrix(curves[start:stop]) @ template_transpose
        if return_direction:
            best_index = np.argmax(np.abs(projections), axis=1)
            best_projection = projections[np.arange(projections.shape[0]), best_index]
            signal_scores[start:stop] = np.abs(best_projection)
            best_projections[start:stop] = best_projection
        else:
            np.abs(projections, out=projections)
            signal_scores[start:stop] = projections.max(axis=1)

    if not return_direction:
        return pd.DataFrame({SCORE_COLUMN: signal_scores})
    return pd.DataFrame({SCORE_COLUMN: signal_scores, "best_projection": best_projections})


def empirical_right_tail_p_values(observed_statistics, null_statistics):
    observed_statistics = np.asarray(observed_statistics, dtype=float)
    null_statistics = np.sort(np.asarray(null_statistics, dtype=float))
    counts_ge = null_statistics.size - np.searchsorted(null_statistics, observed_statistics, side="left")
    return (counts_ge + 1.0) / (null_statistics.size + 1.0)


def simulate_null_curves(inhibition_values, n_curves, n_doses, random_generator):
    return random_generator.choice(np.asarray(inhibition_values, dtype=np.float32), size=(n_curves, n_doses), replace=True)


def compute_dose_layout_calibration(payload):
    null_rng = np.random.default_rng(payload["null_seed"])
    control_inhibition_values = payload["control_inhibition_values"]
    template_transpose = payload["template_transpose"]
    concentration_values = payload["concentration_values"]
    n_null_curves = payload["n_null_curves"]
    score_block_size = payload["score_block_size"]
    organism = payload["organism"]
    dose_layout_fields = payload["dose_layout_fields"]

    null_score_tables = {}
    curve_summary_records = []
    for null, source_control in NULL_SOURCE_CONTROLS.items():
        simulated_curves = simulate_null_curves(control_inhibition_values[source_control], n_null_curves, len(concentration_values), null_rng)
        curve_summary_records.append({
            "organism": organism,
            **dose_layout_fields,
            "null": null,
            "n_null_curves": int(n_null_curves),
            "simulated_curve_mean": float(simulated_curves.mean()),
            "simulated_curve_variance": float(simulated_curves.var(ddof=1)),
        })
        null_score_tables[null] = calculate_2pl_signal_scores(simulated_curves, template_transpose, block_size=score_block_size, return_direction=False)

    np.savez_compressed(payload["null_scores_path"], **{null.replace(" ", "_"): null_score_tables[null][SCORE_COLUMN].to_numpy(np.float32) for null in NULL_SOURCE_CONTROLS})
    np.savez_compressed(payload["control_inhibition_path"], **{key: control_inhibition_values[source].astype(np.float32) for key, source in CONTROL_INHIBITION_SOURCES.items()})
    np.savez_compressed(
        payload["templates_path"],
        concentration_values=concentration_values.astype(float),
        inhibition_columns=np.asarray(payload["inhibition_columns"], dtype=object),
        template_transpose=template_transpose,
        template_slopes=payload["template_slopes"],
        template_midpoints=payload["template_midpoints"],
        candidate_slopes=payload["candidate_slopes"],
        candidate_midpoints=payload["candidate_midpoints"],
    )

    organism_summary = pd.concat([table.assign(null=null) for null, table in null_score_tables.items()], ignore_index=True)
    threshold_summary = organism_summary.groupby("null").agg(
        n_null=(SCORE_COLUMN, "size"),
        signal_zero_fraction=(SCORE_COLUMN, lambda values: np.mean(np.asarray(values) == 0)),
        signal_95=(SCORE_COLUMN, lambda values: np.percentile(values, 95)),
        signal_99=(SCORE_COLUMN, lambda values: np.percentile(values, 99)),
        signal_999=(SCORE_COLUMN, lambda values: np.percentile(values, 99.9)),
    ).reset_index()
    for column, value in reversed([("organism", organism), *dose_layout_fields.items()]):
        threshold_summary.insert(0, column, value)

    return {
        "null_score_tables": null_score_tables,
        "curve_summary_records": curve_summary_records,
        "threshold_records": threshold_summary.to_dict("records"),
    }


def plot_null_score_distribution(dose_layout_plot_records, organism, output_path, log_bin_width=0.005):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(dose_layout_plot_records), 1, figsize=(6.4, 2.9 * len(dose_layout_plot_records) + 1.2), dpi=300, sharex=True)
    axes = np.ravel(np.atleast_1d(axes))
    for axis, record in zip(axes, dose_layout_plot_records):
        null_score_tables = record["null_score_tables"]
        score_values = {
            null: null_score_tables[null][SCORE_COLUMN].to_numpy(float)
            for null in NULL_SOURCE_CONTROLS
        }
        positive_values = [values[values > 0] for values in score_values.values()]
        all_positive = np.concatenate([values for values in positive_values if values.size > 0])
        if all_positive.size == 0:
            axis.text(0.5, 0.5, "No positive scores", ha="center", va="center", transform=axis.transAxes)
        else:
            log_min = np.floor(np.log10(all_positive.min()) / log_bin_width) * log_bin_width
            log_max = np.ceil(np.log10(all_positive.max()) / log_bin_width) * log_bin_width
            log_bins = np.arange(log_min, log_max + log_bin_width, log_bin_width)
            for null in NULL_SOURCE_CONTROLS:
                values = score_values[null]
                axis.hist(np.log10(values[values > 0]), bins=log_bins, density=True, alpha=0.42, color=NULL_COLOURS[null], label=f"{null} | exact zero={np.mean(values == 0):.3%}")
            tick_powers = np.arange(math.floor(log_min), math.ceil(log_max) + 1)
            axis.set_xticks(tick_powers)
            axis.set_xticklabels([f"{power:g}" for power in tick_powers])
            axis.legend(frameon=False, fontsize=7)
        axis.set_title(f"Set {record['dose_layout_rank']}: {record['dose_layout']} | N={record['n_experimental_curves']:,}", fontweight="bold", fontsize=9)
        axis.set_ylabel("Density", fontweight="bold")
        axis.grid(True, axis="y", alpha=0.25)
        axis.grid(False, axis="x")
    axes[-1].set_xlabel("Log10(Signal Score)", fontweight="bold")
    fig.suptitle(f"Empirical Control Nulls | {organism}", fontweight="bold")
    fig.tight_layout()
    save_plot_figure(fig, output_path)
    plt.close(fig)


def run_calibration(args, output_dir, directories):
    plots_dir = directories["plots"]
    calibration_files_dir = directories["calibration_files"]
    tables_dir = directories["tables"]
    emit_message("info", "Calibration run", lines=[
        *get_run_logo_lines(),
        format_detail("Version", TOOL_VERSION),
        format_detail("Normalized data", style_text(str(args.normalized_data), TEXT_COLOURS["accent"])),
        format_detail("Control data", style_text(str(args.control_data), TEXT_COLOURS["accent"])),
        format_detail("Output", style_text(str(output_dir), TEXT_COLOURS["accent"])),
        format_detail("Null curves", "auto" if args.n_null_curves == AUTO_NULL_CURVES else f"{args.n_null_curves:,}"),
    ])

    normalized_data = load_normalized_inhibition_data(args.normalized_data)
    organisms = get_analysis_organisms(args.normalized_data, args.organism, preloaded_data=normalized_data)
    inhibition_columns_all = get_inhibition_columns(args.normalized_data, preloaded_data=normalized_data)
    dose_layouts_by_organism = {
        organism: detect_experimental_dose_sets(args.normalized_data, organism, preloaded_data=normalized_data,
                                              inhibition_columns=inhibition_columns_all)
        for organism in organisms
    }
    pool_workers = min(MAX_POOL_WORKERS, os.cpu_count() or 1)
    worker_pool = None
    if pool_workers > 1:
        preload_fork_worker_modules("scipy.stats", *(["matplotlib.pyplot"] if args.plot == "yes" else []))
        worker_pool = ProcessPoolExecutor(max_workers=pool_workers)
    renderer = BackgroundRenderer(worker_pool)
    total_stages = 1 + sum(2 + 4 * len(dose_layouts) for dose_layouts in dose_layouts_by_organism.values())
    progress = create_progress_bar(total_stages, format_stage_label(1, total_stages, "Load controls"))
    stage_number = 1

    def complete_stage(next_stage=None):
        nonlocal stage_number
        progress.update(1)
        stage_number += 1
        if next_stage is not None and stage_number <= total_stages:
            set_progress_stage(progress, stage_number, total_stages, next_stage)

    control_inliers = get_control_inliers(pd.read_csv(args.control_data))
    complete_stage("Calibration organisms")
    manifest_records = []
    threshold_records = []
    curve_summary_records = []

    try:
        organism_jobs = []
        for organism in organisms:
            organism_token = slugify_filename_token(organism, default="organism")
            dose_layout_records = dose_layouts_by_organism[organism]
            set_progress_stage(progress, stage_number, total_stages, f"{organism}: collect controls")
            emit_message("info", f"Calibrating {organism}")
            emit_dose_layout_selection(organism, dose_layout_records)
            control_inhibition_values = {
                source: get_control_inhibition_measurements(control_inliers, organism, source)
                for source in CONTROL_INHIBITION_SOURCES.values()
            }
            complete_stage(f"{organism}: calibration sets")

            set_jobs = []
            for dose_layout_record in dose_layout_records:
                dose_layout_fields = {key: dose_layout_record[key] for key in ["dose_layout", "dose_layout_full", "dose_layout_token", "dose_layout_rank", "n_experimental_curves"]}
                concentration_values = dose_layout_record["concentration_values"]
                record_token = f"{organism_token}_{dose_layout_record['dose_layout_token']}"
                n_null_curves = resolve_null_curves(args.n_null_curves, dose_layout_record["n_experimental_treatments"], args.q_threshold)
                score_block_size = min(DEFAULT_BLOCK_SIZE, n_null_curves)
                set_progress_stage(progress, stage_number, total_stages, f"{organism}: set {dose_layout_record['dose_layout_rank']} templates")
                templates = build_2pl_template_matrix(
                    concentration_values,
                    n_slopes=args.n_slopes,
                    n_midpoints=args.n_midpoints,
                    min_slope=args.min_slope,
                    max_slope=args.max_slope,
                )
                emit_message("info", "Template library", lines=[
                    format_detail("Dose layout", dose_layout_record["dose_layout"]),
                    format_detail("Null curves", f"{n_null_curves:,}"),
                    format_detail("Total templates meeting criteria", f"{templates.matrix.shape[0]:,}"),
                ])
                complete_stage(f"{organism}: set {dose_layout_record['dose_layout_rank']} nulls")

                payload = {
                    "organism": organism,
                    "dose_layout_fields": dose_layout_fields,
                    "concentration_values": concentration_values,
                    "inhibition_columns": dose_layout_record["inhibition_columns"],
                    "control_inhibition_values": control_inhibition_values,
                    "template_transpose": templates.transpose,
                    "template_slopes": templates.slopes,
                    "template_midpoints": templates.midpoints,
                    "candidate_slopes": templates.candidate_slopes,
                    "candidate_midpoints": templates.candidate_midpoints,
                    "n_null_curves": n_null_curves,
                    "score_block_size": score_block_size,
                    "null_seed": args.null_seed + dose_layout_record["dose_layout_rank"] - 1,
                    "null_scores_path": calibration_files_dir / f"{record_token}_null_scores.npz",
                    "templates_path": calibration_files_dir / f"{record_token}_templates.npz",
                    "control_inhibition_path": calibration_files_dir / f"{record_token}_control_inhibition.npz",
                }
                job = worker_pool.submit(compute_dose_layout_calibration, payload) if worker_pool is not None else compute_dose_layout_calibration(payload)
                set_jobs.append((dose_layout_record, dose_layout_fields, payload, job))
            organism_jobs.append((organism, organism_token, set_jobs))

        for organism, organism_token, set_jobs in organism_jobs:
            dose_layout_plot_records = []
            for dose_layout_record, dose_layout_fields, payload, job in set_jobs:
                set_progress_stage(progress, stage_number, total_stages, f"{organism}: set {dose_layout_record['dose_layout_rank']} save files")
                result = job.result() if worker_pool is not None else job
                complete_stage(f"{organism}: set {dose_layout_record['dose_layout_rank']} summaries")

                curve_summary_records.extend(result["curve_summary_records"])
                threshold_records.extend(result["threshold_records"])
                manifest_records.append({
                    "organism": organism,
                    "organism_token": organism_token,
                    **dose_layout_fields,
                    "inhibition_columns": ",".join(payload["inhibition_columns"]),
                    "concentration_values": ",".join(map(str, payload["concentration_values"])),
                    "n_doses": len(payload["concentration_values"]),
                    "templates_path": str(payload["templates_path"].relative_to(output_dir)),
                    "null_scores_path": str(payload["null_scores_path"].relative_to(output_dir)),
                    "control_inhibition_path": str(payload["control_inhibition_path"].relative_to(output_dir)),
                })
                dose_layout_plot_records.append({**dose_layout_fields, "null_score_tables": result["null_score_tables"]})
                complete_stage(f"{organism}: calibration sets")

            set_progress_stage(progress, stage_number, total_stages, f"{organism}: save plots")
            if args.plot == "yes":
                renderer.submit(plot_null_score_distribution, dose_layout_plot_records, organism, plots_dir / f"{organism_token}_null_scores.png")
            complete_stage("Calibration organisms")
        renderer.wait()
    finally:
        progress.close()
        if worker_pool is not None:
            worker_pool.shutdown()

    pd.DataFrame(manifest_records).to_csv(output_dir / "calibration_manifest.csv", index=False)
    pd.DataFrame(threshold_records).to_csv(tables_dir / "null_threshold_summary.csv", index=False)
    pd.DataFrame(curve_summary_records).to_csv(tables_dir / "null_curve_summary.csv", index=False)
    emit_message("success", "Calibration complete", lines=[
        format_detail("Output", style_text(str(output_dir), TEXT_COLOURS["accent"])),
        format_detail("Manifest", "calibration_manifest.csv"),
    ])


def main():
    args = parse_args()
    output_dir = create_dated_run_directory(args.output_dir, args.normalized_data, DEFAULT_RUN_DIR_PREFIX)
    directories = prepare_output_directories(output_dir, ["calibration_files", "logs", "plots", "tables"])
    with RunLogger(directories["logs"] / "run_log.txt"):
        write_run_parameters(args, directories["logs"], "calibration.py", TOOL_VERSION, {"output_dir": str(output_dir)})
        try:
            run_calibration(args, output_dir, directories)
        except Exception as error:
            emit_message("error", "Calibration failed", lines=format_detail("Reason", str(error)))
            traceback.print_exc()
            raise


if __name__ == "__main__":
    main()
