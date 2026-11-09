#!/usr/bin/env python3

import argparse
import math
import os
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from dynamic.calibration import (NO_INHIBITION_NULL, FULL_INHIBITION_NULL, NULL_SOURCE_CONTROLS,
                                 SCORE_COLUMN, ensure_dose_layout_columns,
                                 load_normalized_inhibition_data, benjamini_hochberg,
                                 calculate_2pl_signal_scores, empirical_right_tail_p_values,
                                 get_inhibition_columns)
from dynamic.utils import (BackgroundRenderer, LogoArgumentParser, RunLogger, TEXT_COLOURS,
                           create_dated_run_directory, create_progress_bar, emit_message, format_detail,
                           format_stage_label, get_run_logo_lines, parse_csv_list, positive_int,
                           preload_fork_worker_modules, probability, prepare_output_directories,
                           require_columns, save_plot_figure, set_progress_stage, style_text,
                           write_run_parameters)


from dynamic import __version__ as TOOL_VERSION
DEFAULT_RUN_DIR_PREFIX = "classification_results"
DEFAULT_Q_THRESHOLD = 0.05
DEFAULT_REPLICATE_OUTLIER_ALPHA = 0.05
DEFAULT_REPLICATE_MAX_OUTLIER_FRACTION = 1 / 3
DEFAULT_REPLICATE_DISTANCE_EPSILON = 1e-8
DEFAULT_MINIMUM_REPLICATES_FOR_OUTLIER_TEST = 3
DEFAULT_MAX_WORKERS = 30
OUTLIER_TASK_CHUNKS_PER_WORKER = 4
PARALLEL_OUTLIER_TREATMENT_THRESHOLD = 2000
LABEL_VARIANT = "dose-variant"
LABEL_INVERSE_VARIANT = "inverse dose-variant"
LABEL_INVARIANT = "dose-invariant"
SIGNAL_LABEL_ORDER = [LABEL_VARIANT, LABEL_INVERSE_VARIANT, LABEL_INVARIANT]
REPLICATE_OUTLIER_COLUMNS = ["treatment_index", "replicate_index", "n_replicates", "curve_discordance", "replicate_curve_outlier"]

MISSING_INHIBITION_COLUMNS_MESSAGE = (
    "The normalized data are missing inhibition column(s) required by the selected calibration: "
    "{columns}. Use a calibration generated with the same dose layout, "
    "or rerun calibration for this normalized dataset."
)
EMPTY_CURVE_ROWS_MESSAGE = "No exact-match curves found for {organism} using the calibration concentration columns."


def parse_args():
    parser = LogoArgumentParser(description="DynaMIC dose-response classification", formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--normalized-data", required=True, help="CSV produced by normalization.py.")
    parser.add_argument("--calibration-dir", required=True, help="Directory produced by calibration.py.")
    parser.add_argument("--output-dir", default=None, help="Directory for classification outputs.")
    parser.add_argument("--organism", action="append", default=None, help="Organism to classify. Repeat to select multiple. Default: all calibration organisms.")
    parser.add_argument("--q-threshold", type=probability, default=DEFAULT_Q_THRESHOLD, help="Inhibition q threshold. Default: 0.05.")
    parser.add_argument("--replicate-outlier-alpha", type=probability, default=DEFAULT_REPLICATE_OUTLIER_ALPHA, help="Iterative ESD replicate outlier alpha. Default: 0.05.")
    parser.add_argument("--replicate-outliers", choices=["exclude", "include", "skip"], default="exclude", help="Exclude or include replicate outliers in treatment-level curves, or skip outlier detection. Default: exclude.")
    parser.add_argument("--replicate-max-outlier-fraction", type=probability, default=DEFAULT_REPLICATE_MAX_OUTLIER_FRACTION, help="Maximum fraction of replicate curves tested as outliers. Default: 1/3.")
    parser.add_argument("--minimum-replicates-for-outlier-test", type=positive_int, default=DEFAULT_MINIMUM_REPLICATES_FOR_OUTLIER_TEST, help="Minimum replicates for iterative ESD testing. Default: 3.")
    parser.add_argument("--max-workers", "--threads", dest="max_workers", type=positive_int, default=DEFAULT_MAX_WORKERS, help=f"Parallel workers for replicate outlier detection. Default: {DEFAULT_MAX_WORKERS}.")
    parser.add_argument("--plot", choices=["yes", "no"], default="yes", help="Save curve heatmaps. Default: yes.")
    return parser.parse_args()


def load_manifest(calibration_dir, selected_organisms=None):
    manifest_path = Path(calibration_dir) / "calibration_manifest.csv"
    if not manifest_path.exists():
        raise ValueError(f"Calibration manifest was not found: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    require_columns(manifest.columns, ["organism", "organism_token", "inhibition_columns", "templates_path", "null_scores_path"], "Calibration manifest", manifest_path)
    manifest = ensure_dose_layout_columns(manifest)
    if "dose_layout_rank" not in manifest.columns:
        manifest["dose_layout_rank"] = manifest.groupby("organism", sort=False).cumcount() + 1
    if "n_experimental_curves" not in manifest.columns:
        manifest["n_experimental_curves"] = pd.NA
    if selected_organisms:
        manifest = manifest[manifest["organism"].isin(selected_organisms)].copy()
    if manifest.empty:
        raise ValueError("No calibration records matched the requested organism selection.")
    return manifest


def load_null_scores(calibration_dir, null_scores_path):
    data = np.load(Path(calibration_dir) / null_scores_path)
    return {null: data[null.replace(" ", "_")] for null in NULL_SOURCE_CONTROLS}


def load_templates(calibration_dir, templates_path):
    data = np.load(Path(calibration_dir) / templates_path, allow_pickle=True)
    return data["template_transpose"], data["concentration_values"]


def load_exact_layout_rows(normalized_data_path, organism, inhibition_columns, metadata_columns,
                           missing_columns_message, empty_message, preloaded_data=None):
    headers = preloaded_data if preloaded_data is not None else pd.read_csv(normalized_data_path, nrows=0)
    require_columns(headers.columns, ["organism", "treatment"], "Normalized data", normalized_data_path)
    all_inhibition_columns = get_inhibition_columns(normalized_data_path, preloaded_data=preloaded_data)
    missing_inhibition_columns = sorted(set(inhibition_columns) - set(headers.columns))
    if missing_inhibition_columns:
        raise ValueError(missing_columns_message.format(columns=", ".join(missing_inhibition_columns)))
    inhibition_column_set = set(inhibition_columns)
    extra_inhibition_columns = [column for column in all_inhibition_columns if column not in inhibition_column_set]
    use_columns = [column for column in metadata_columns if column in headers.columns]
    columns_to_use = list(dict.fromkeys(use_columns + all_inhibition_columns))
    data = preloaded_data.loc[:, preloaded_data.columns.isin(columns_to_use)] if preloaded_data is not None else pd.read_csv(normalized_data_path, usecols=columns_to_use)
    exact_match = data[inhibition_columns].notna().all(axis=1)
    if extra_inhibition_columns:
        exact_match &= data[extra_inhibition_columns].isna().all(axis=1)
    curve_rows = data.loc[
        data["organism"].eq(organism)
        & ~data["treatment"].isin(NULL_SOURCE_CONTROLS.values())
        & exact_match
    ].reset_index(drop=True)
    if curve_rows.empty:
        raise ValueError(empty_message.format(organism=organism))
    return curve_rows


def load_experimental_rows(normalized_data_path, organism, inhibition_columns, preloaded_data=None):
    return load_exact_layout_rows(
        normalized_data_path, organism, inhibition_columns,
        metadata_columns=["organism", "treatment", "data_id", "assay_id", "sequence"],
        missing_columns_message=MISSING_INHIBITION_COLUMNS_MESSAGE,
        empty_message=EMPTY_CURVE_ROWS_MESSAGE,
        preloaded_data=preloaded_data,
    )


def group_treatment_replicates(curve_rows, inhibition_columns):
    replicate_keys = ["organism", "treatment"]
    treatment_groups = curve_rows.groupby(replicate_keys, sort=False, dropna=False)
    treatment_metadata = treatment_groups.size().reset_index(name="n_replicates")
    if "sequence" in curve_rows.columns:
        treatment_metadata["sequence"] = treatment_groups["sequence"].first().to_numpy()
    treatment_metadata.insert(0, "treatment_index", np.arange(len(treatment_metadata)))

    inhibition_values = curve_rows[inhibition_columns].to_numpy(dtype=np.float32)
    treatment_indices = list(treatment_groups.indices.values())
    treatment_replicates_curves = [inhibition_values[index] for index in treatment_indices]

    metadata_columns = [column for column in ["organism", "treatment", "sequence", "data_id", "assay_id"] if column in curve_rows.columns]
    treatment_sizes = np.array([len(index) for index in treatment_indices])
    replicate_metadata = curve_rows.iloc[np.concatenate(treatment_indices)][metadata_columns].reset_index(drop=True)
    replicate_metadata.insert(0, "n_replicates", np.repeat(treatment_sizes, treatment_sizes))
    replicate_metadata.insert(0, "replicate_index", np.concatenate([np.arange(size) for size in treatment_sizes]))
    replicate_metadata.insert(0, "treatment_index", np.repeat(np.arange(len(treatment_sizes)), treatment_sizes))
    return treatment_metadata, treatment_replicates_curves, replicate_metadata


def calculate_replicate_curve_discordance(replicate_curves):
    replicate_curves = np.asarray(replicate_curves, dtype=np.float32)
    n_replicates = replicate_curves.shape[0]
    leave_one_out_index = (np.arange(n_replicates)[:, None] + np.arange(1, n_replicates)[None, :]) % n_replicates
    median_of_others = np.median(replicate_curves[leave_one_out_index], axis=1)
    return np.abs(replicate_curves - median_of_others).mean(axis=1).astype(float)


def calculate_esd_p_values(r_statistics, remaining_counts):
    from scipy import stats
    r_statistics = np.asarray(r_statistics, dtype=float)
    remaining_counts = np.asarray(remaining_counts, dtype=float)
    p_values = np.zeros(r_statistics.shape[0], dtype=float)
    denominators = (remaining_counts - 1) ** 2 - remaining_counts * r_statistics ** 2
    valid = denominators > 0
    if valid.any():
        n_replicates = remaining_counts[valid]
        t_statistics = r_statistics[valid] * np.sqrt(n_replicates * (n_replicates - 2) / denominators[valid])
        p_values[valid] = np.minimum(1.0, n_replicates * (1.0 - stats.t.cdf(t_statistics, n_replicates - 2)))
    return p_values


def iterative_esd_candidates(values, max_outliers):
    remaining_values = np.asarray(values, dtype=float).copy()
    remaining_replicate_indices = np.arange(len(values))
    candidate_indices, candidate_ratios, candidate_counts = [], [], []

    for _ in range(max_outliers):
        n_replicates = remaining_values.size
        if n_replicates < 3:
            break
        sd = float(remaining_values.std(ddof=1))
        highest_position = int(np.argmax(remaining_values))
        r_statistic = (float(remaining_values[highest_position]) - float(remaining_values.mean())) / sd if np.isfinite(sd) and sd > DEFAULT_REPLICATE_DISTANCE_EPSILON else 0.0
        candidate_indices.append(int(remaining_replicate_indices[highest_position]))
        candidate_ratios.append(r_statistic)
        candidate_counts.append(n_replicates)
        remaining_values = np.delete(remaining_values, highest_position)
        remaining_replicate_indices = np.delete(remaining_replicate_indices, highest_position)

    return candidate_indices, candidate_ratios, candidate_counts


def detect_replicate_outlier_chunk(chunk):
    prepared_treatments = []
    all_ratios, all_counts = [], []
    for treatment_index, replicate_curves, alpha, max_outlier_fraction in chunk:
        n_replicates = len(replicate_curves)
        discordance = calculate_replicate_curve_discordance(replicate_curves)
        max_tested_outliers = int(np.floor(n_replicates * max_outlier_fraction + 1e-12))
        candidate_indices, candidate_ratios, candidate_counts = iterative_esd_candidates(discordance, max_tested_outliers)
        prepared_treatments.append((treatment_index, n_replicates, discordance, candidate_indices, len(all_ratios), alpha))
        all_ratios.extend(candidate_ratios)
        all_counts.extend(candidate_counts)

    p_values = calculate_esd_p_values(all_ratios, all_counts) if all_ratios else np.empty(0)
    records = []
    for treatment_index, n_replicates, discordance, candidate_indices, offset, alpha in prepared_treatments:
        treatment_p_values = p_values[offset:offset + len(candidate_indices)]
        n_outliers = max([rank for rank, p_value in enumerate(treatment_p_values, start=1) if p_value < alpha], default=0)
        replicate_is_outlier = np.zeros(n_replicates, dtype=bool)
        replicate_is_outlier[candidate_indices[:n_outliers]] = True
        records.extend(
            {
                "treatment_index": treatment_index,
                "replicate_index": replicate_index,
                "n_replicates": n_replicates,
                "curve_discordance": float(discordance[replicate_index]),
                "replicate_curve_outlier": bool(replicate_is_outlier[replicate_index]),
            }
            for replicate_index in range(n_replicates)
        )
    return records


def detect_replicate_outliers(treatment_replicates_curves, replicate_metadata, alpha, max_outlier_fraction, minimum_replicates, max_workers, executor=None):
    outlier_tasks = [
        (treatment_index, replicate_curves, alpha, max_outlier_fraction)
        for treatment_index, replicate_curves in enumerate(treatment_replicates_curves)
        if len(replicate_curves) >= minimum_replicates
    ]
    if not outlier_tasks:
        replicate_records = []
    elif executor is None or max_workers <= 1 or len(outlier_tasks) < PARALLEL_OUTLIER_TREATMENT_THRESHOLD:
        replicate_records = detect_replicate_outlier_chunk(outlier_tasks)
    else:
        chunksize = max(1, math.ceil(len(outlier_tasks) / (max_workers * OUTLIER_TASK_CHUNKS_PER_WORKER)))
        chunks = [outlier_tasks[start:start + chunksize] for start in range(0, len(outlier_tasks), chunksize)]
        replicate_records = []
        for chunk_records in executor.map(detect_replicate_outlier_chunk, chunks):
            replicate_records.extend(chunk_records)

    replicate_outliers = pd.DataFrame(replicate_records, columns=REPLICATE_OUTLIER_COLUMNS)
    if not replicate_outliers.empty:
        replicate_outliers = replicate_outliers.sort_values(["treatment_index", "replicate_index"]).reset_index(drop=True)
        replicate_outliers = replicate_outliers.merge(replicate_metadata, on=["treatment_index", "replicate_index", "n_replicates"], how="left", sort=False)
    return replicate_outliers


def build_outlier_replicates_by_treatment(replicate_outliers):
    if replicate_outliers.empty or "replicate_curve_outlier" not in replicate_outliers.columns:
        return {}
    return {
        int(treatment_index): set(group["replicate_index"].astype(int))
        for treatment_index, group in replicate_outliers.loc[replicate_outliers["replicate_curve_outlier"]].groupby("treatment_index")
    }


def build_analysis_treatment_curves(treatment_replicates_curves, replicate_outliers, exclude_outliers=True):
    outliers_by_treatment = build_outlier_replicates_by_treatment(replicate_outliers)
    analysis_replicate_curves = []
    for treatment_index, replicate_curves in enumerate(treatment_replicates_curves):
        replicate_curves = np.asarray(replicate_curves, dtype=np.float32)
        outliers = outliers_by_treatment.get(treatment_index, set())
        keep = np.array([replicate_index not in outliers for replicate_index in range(replicate_curves.shape[0])], dtype=bool) if exclude_outliers else np.ones(replicate_curves.shape[0], dtype=bool)
        if keep.sum() == 0:
            keep[:] = True
        analysis_replicate_curves.append(replicate_curves[keep])
    treatment_sizes = np.array([curves.shape[0] for curves in analysis_replicate_curves])
    segment_starts = np.concatenate([[0], np.cumsum(treatment_sizes)[:-1]])
    analysis_mean_curves = np.add.reduceat(np.concatenate(analysis_replicate_curves), segment_starts) / treatment_sizes[:, None].astype(np.float32)
    return analysis_replicate_curves, np.asarray(analysis_mean_curves, dtype=np.float32)


def classify_curves(analysis_metadata, inhibition_curves, template_transpose, null_scores, q_threshold):
    scores = calculate_2pl_signal_scores(inhibition_curves, template_transpose=template_transpose, return_direction=True)
    full_inhibition_null_p = empirical_right_tail_p_values(scores[SCORE_COLUMN], null_scores[FULL_INHIBITION_NULL])
    no_inhibition_null_p = empirical_right_tail_p_values(scores[SCORE_COLUMN], null_scores[NO_INHIBITION_NULL])
    result = analysis_metadata.reset_index(drop=True).copy()
    result[SCORE_COLUMN] = scores[SCORE_COLUMN].to_numpy(float)
    result[f"{SCORE_COLUMN}_p"] = np.maximum(full_inhibition_null_p, no_inhibition_null_p)
    result[f"{SCORE_COLUMN}_q"] = benjamini_hochberg(result[f"{SCORE_COLUMN}_p"])
    selected = result[f"{SCORE_COLUMN}_q"] <= q_threshold
    expected_direction = scores["best_projection"].to_numpy(float) >= 0
    result["dose_response_label"] = np.select(
        [selected & expected_direction, selected & ~expected_direction],
        [LABEL_VARIANT, LABEL_INVERSE_VARIANT],
        default=LABEL_INVARIANT,
    )
    leading_columns = [column for column in ["organism", "treatment", "dose_layout", "dose_layout_full", "sequence"] if column in result.columns]
    output_columns = leading_columns + [
        "replicate_outlier_exclusion",
        SCORE_COLUMN,
        f"{SCORE_COLUMN}_p",
        f"{SCORE_COLUMN}_q",
        "dose_response_label",
    ]
    return result[output_columns]


def prepare_replicate_outlier_output(replicate_outliers):
    table = replicate_outliers.drop(columns=["treatment_index"], errors="ignore").copy()
    leading_columns = [column for column in ["organism", "treatment", "dose_layout", "dose_layout_full", "sequence", "replicate_index", "n_replicates"] if column in table.columns]
    return table[leading_columns + [column for column in table.columns if column not in leading_columns]]


def plot_curve_heatmap(result, inhibition_curves, concentration_values, organism, output_path, dose_layout=None):
    import matplotlib.pyplot as plt
    plot_order = np.argsort(concentration_values)
    x_plot = concentration_values[plot_order]
    label_order = [label for label in SIGNAL_LABEL_ORDER if label in result["dose_response_label"].unique()]
    if not label_order:
        return
    heatmap_table = result.copy()
    heatmap_table["mean_inhibition"] = inhibition_curves.mean(axis=1)
    heatmap_table["inhibition_range"] = np.ptp(inhibition_curves, axis=1)
    fig = plt.figure(figsize=(5.8 * len(label_order) + 0.5, 5.4), dpi=300, constrained_layout=True)
    grid = fig.add_gridspec(1, len(label_order) + 1, width_ratios=[1] * len(label_order) + [0.04], wspace=0.08)
    axes = [fig.add_subplot(grid[0, index]) for index in range(len(label_order))]
    colourbar_axis = fig.add_subplot(grid[0, -1])
    image = None
    for axis, label in zip(axes, label_order):
        label_rows = heatmap_table.loc[heatmap_table["dose_response_label"].eq(label)].sort_values(["mean_inhibition", "inhibition_range"], ascending=[False, False])
        curves = inhibition_curves[label_rows.index.to_numpy()][:, plot_order]
        image = axis.imshow(curves, aspect="auto", interpolation="nearest", vmin=0, vmax=1, cmap=plt.get_cmap("gist_earth"))
        axis.set_title(f"{label}\nN={len(label_rows):,}", fontweight="bold")
        axis.set_xlabel("Concentration", fontweight="bold")
        axis.set_xticks(np.arange(len(x_plot)))
        axis.set_xticklabels([f"{x:g}" for x in x_plot], rotation=45, ha="right")
        axis.set_ylabel("Curves sorted by mean inhibition", fontweight="bold")
        axis.tick_params(axis="y", left=False, labelleft=False)
        axis.grid(False)
    colourbar = fig.colorbar(image, cax=colourbar_axis)
    colourbar.set_label("Inhibition Fraction", fontweight="bold")
    fig.suptitle(f"Curve Heatmap by Dose-Response Class | {organism}" + (f" | {dose_layout}" if dose_layout else ""), fontweight="bold")
    save_plot_figure(fig, output_path)
    plt.close(fig)


def run_classification(args, output_dir, directories):
    plots_dir = directories["plots"]
    tables_dir = directories["tables"]
    manifest = load_manifest(args.calibration_dir, args.organism)
    emit_message("info", "Classification run", lines=[
        *get_run_logo_lines(),
        format_detail("Version", TOOL_VERSION),
        format_detail("Normalized data", style_text(str(args.normalized_data), TEXT_COLOURS["accent"])),
        format_detail("Calibration", style_text(str(args.calibration_dir), TEXT_COLOURS["accent"])),
        format_detail("Output", style_text(str(output_dir), TEXT_COLOURS["accent"])),
        format_detail("Replicate outliers", args.replicate_outliers),
        format_detail("q threshold", args.q_threshold),
    ])

    total_stages = max(1, 5 * len(manifest))
    progress = create_progress_bar(total_stages, format_stage_label(1, total_stages, "Classify organisms"))
    stage_number = 1

    def complete_stage(next_stage=None):
        nonlocal stage_number
        progress.update(1)
        stage_number += 1
        if next_stage is not None and stage_number <= total_stages:
            set_progress_stage(progress, stage_number, total_stages, next_stage)

    output_records = []
    normalized_data = load_normalized_inhibition_data(
        args.normalized_data, metadata_columns=("organism", "treatment", "data_id", "assay_id", "sequence"))
    worker_pool = None
    if args.max_workers > 1 and (args.plot == "yes" or args.replicate_outliers != "skip"):
        preload_fork_worker_modules(*(["scipy.stats"] if args.replicate_outliers != "skip" else []),
                                    *(["matplotlib.pyplot"] if args.plot == "yes" else []))
        worker_pool = ProcessPoolExecutor(max_workers=min(args.max_workers, os.cpu_count() or 1))
    renderer = BackgroundRenderer(worker_pool)
    try:
        for row in manifest.itertuples(index=False):
            organism = row.organism
            organism_token = row.organism_token
            dose_layout = row.dose_layout
            dose_layout_full = row.dose_layout_full
            dose_layout_token = row.dose_layout_token
            record_token = f"{organism_token}_{dose_layout_token}"
            set_progress_stage(progress, stage_number, total_stages, f"{organism}: load calibration")
            emit_message("info", f"Classifying {organism}", lines=format_detail("Dose layout", dose_layout))
            inhibition_columns = parse_csv_list(row.inhibition_columns)
            template_transpose, concentration_values = load_templates(args.calibration_dir, row.templates_path)
            null_scores = load_null_scores(args.calibration_dir, row.null_scores_path)
            complete_stage(f"{organism}: group replicates")

            curve_rows = load_experimental_rows(args.normalized_data, organism, inhibition_columns, preloaded_data=normalized_data)
            treatment_metadata, treatment_replicates_curves, replicate_metadata = group_treatment_replicates(curve_rows, inhibition_columns)
            treatment_metadata["dose_layout"] = dose_layout
            treatment_metadata["dose_layout_full"] = dose_layout_full
            replicate_metadata["dose_layout"] = dose_layout
            replicate_metadata["dose_layout_full"] = dose_layout_full
            complete_stage(f"{organism}: detect outlier curves")

            if args.replicate_outliers == "skip":
                replicate_outliers = pd.DataFrame(columns=REPLICATE_OUTLIER_COLUMNS)
            else:
                replicate_outliers = detect_replicate_outliers(
                    treatment_replicates_curves,
                    replicate_metadata,
                    alpha=args.replicate_outlier_alpha,
                    max_outlier_fraction=args.replicate_max_outlier_fraction,
                    minimum_replicates=args.minimum_replicates_for_outlier_test,
                    max_workers=args.max_workers,
                    executor=worker_pool,
                )
            exclude_outliers = args.replicate_outliers == "exclude"
            _, inhibition_curves = build_analysis_treatment_curves(treatment_replicates_curves, replicate_outliers, exclude_outliers)
            complete_stage(f"{organism}: score curves")

            analysis_metadata = treatment_metadata.copy()
            analysis_metadata["replicate_outlier_exclusion"] = exclude_outliers
            result = classify_curves(analysis_metadata, inhibition_curves, template_transpose, null_scores, args.q_threshold)
            complete_stage(f"{organism}: save results")

            result_path = tables_dir / f"{record_token}_classification_results.csv"
            replicate_outliers_path = tables_dir / f"{record_token}_replicate_outliers.csv"
            result.to_csv(result_path, index=False)
            prepare_replicate_outlier_output(replicate_outliers).to_csv(replicate_outliers_path, index=False)
            if args.plot == "yes":
                renderer.submit(plot_curve_heatmap, result, inhibition_curves, concentration_values, organism, plots_dir / f"{record_token}_heatmap.png", dose_layout=dose_layout)

            output_records.append({
                "organism": organism,
                "organism_token": organism_token,
                "dose_layout": dose_layout,
                "dose_layout_full": dose_layout_full,
                "dose_layout_token": dose_layout_token,
                "dose_layout_rank": row.dose_layout_rank,
                "n_experimental_curves": row.n_experimental_curves,
                "inhibition_columns": ",".join(inhibition_columns),
                "concentration_values": ",".join(map(str, concentration_values)),
                "classification_results_path": str(result_path.relative_to(output_dir)),
                "replicate_outliers_path": str(replicate_outliers_path.relative_to(output_dir)),
            })
            emit_message("info", "Classification summary", items=[
                f"{label}: {int(result['dose_response_label'].eq(label).sum()):,}"
                for label in SIGNAL_LABEL_ORDER
            ])
            complete_stage("Classify organisms")
        renderer.wait()
    finally:
        progress.close()
        if worker_pool is not None:
            worker_pool.shutdown()

    pd.DataFrame(output_records).to_csv(output_dir / "classification_manifest.csv", index=False)
    emit_message("success", "Classification complete", lines=[
        format_detail("Output", style_text(str(output_dir), TEXT_COLOURS["accent"])),
        format_detail("Manifest", "classification_manifest.csv"),
    ])


def main():
    args = parse_args()
    output_dir = create_dated_run_directory(args.output_dir, args.normalized_data, DEFAULT_RUN_DIR_PREFIX)
    directories = prepare_output_directories(output_dir, ["logs", "plots", "tables"])
    with RunLogger(directories["logs"] / "run_log.txt"):
        write_run_parameters(args, directories["logs"], "classification.py", TOOL_VERSION, {"output_dir": str(output_dir)})
        try:
            run_classification(args, output_dir, directories)
        except Exception as error:
            emit_message("error", "Classification failed", lines=format_detail("Reason", str(error)))
            traceback.print_exc()
            raise


if __name__ == "__main__":
    main()
