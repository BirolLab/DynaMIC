#!/usr/bin/env python3

import argparse
import math
import os
from pathlib import Path
import re
import traceback
from concurrent.futures import ProcessPoolExecutor
from typing import NamedTuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import numpy as np
import pandas as pd

from dynamic.utils import (BackgroundRenderer, LogoArgumentParser, PLOT_DPI, RunLogger, TEXT_COLOURS,
                           create_cwd_dated_run_directory, create_progress_bar, emit_message,
                           ensure_unique_filename, format_detail, format_stage_label, get_run_logo_lines,
                           prepare_output_directories, save_plot_figure,
                           set_progress_stage, set_progress_total, slugify_filename_token, style_text,
                           write_run_parameters)


DEFAULT_RUN_DIR_PREFIX = "normalization_results"
DEFAULT_OUTPUT_DIR_NAME = "output"
DEFAULT_PLOT_OUTPUT_NAME = "across_assay_outlier_control_check.png"
DEFAULT_WITHIN_ASSAY_PLOT_PREFIX = "within_assay_outlier_control_check"
DEFAULT_RAW_INHIBITION_PLOT_NAME = "raw_inhibition_fraction_across_concentrations.png"
DEFAULT_CONTROL_POOL_OUTLIER_CLASSES_PLOT_NAME = "control_pool_outlier_classes.png"
DEFAULT_CONTROL_POOL_INLIER_MEASUREMENT_PLOT_NAME = "control_pool_inlier_measurement.png"
CONTROL_POOL_OUTPUT_NAME = "control_pool.csv"
CONTROL_TREATMENT_NAMES = ["Growth Control", "Sterility Control"]
CONTROL_QC_COLOURS = {"Sterility Control": "#1b9e77", "Growth Control": "#d95f02"}
ASSAY_KEY_COLUMNS = ["organism", "assay_id"]
MEASUREMENT_COLUMN_RE = re.compile(r"A\d+(?:\.\d+)?")
MODIFIED_Z_SCORE_SCALE = 0.6745
MODIFIED_Z_SCORE_THRESHOLD = 3.5
from dynamic import __version__ as TOOL_VERSION
PLOT_DEFAULT_TOKEN = "__default_plot__"
PLOT_DISABLE_TOKENS = {"0", "n", "no", "false", "off"}
PLOT_TITLE_Y = 0.97
PLOT_LEGEND_Y = 0.94
PLOT_ORGANISM_TOP_OFFSET = 1.05
MAX_RENDER_WORKERS = 8

CENTRE_COLUMN = "median"
CENTRE_LABEL = "Median"
CONTROL_SUMMARY_MODE = "medians"


def parse_args():
    parser = LogoArgumentParser(
        description="High-throughput Grouped Signal Normalization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", dest="input_csv", default=None, help="Input CSV path. Required.")
    parser.add_argument("--output", dest="output_csv", default=None, help="Output CSV filename to place inside the generated run directory.")
    parser.add_argument("--output-dir", default=None, help="Run directory for normalization outputs. Default: output/normalization_results_<date>.")
    parser.add_argument("--organism", action="append", default=None, help="Organism to normalize. Repeat to select multiple. Default: all organisms.")
    parser.add_argument("--missing-controls", choices=["impute", "exclude"], default="impute", help="Impute or exclude assay-level control stats. Default: impute")
    parser.add_argument("--within-assay-outlier-controls", choices=["keep", "exclude"], default="exclude",
                        help="Keep or exclude control measurements flagged as within-assay outliers before summary statistics are computed. Default: exclude")
    parser.add_argument("--across-assay-outlier-controls", dest="outlier_controls",
                        choices=["report", "impute", "exclude"], default="report",
                        help="Handle across-assay GC/SC outliers: report only, impute from organism medians, or exclude assay. Default: report.")
    parser.add_argument("--plot", nargs="?", default=PLOT_DEFAULT_TOKEN, const=PLOT_DEFAULT_TOKEN, metavar="N|PATH", help=("Use '--plot N' to disable plotting, or '--plot <path>' to choose the saved plot path. "f"Default: {DEFAULT_PLOT_OUTPUT_NAME}."))
    args = parser.parse_args()
    if args.input_csv is None:
        parser.error("The following argument is required: --input")
    return args


def standardize_filename(filename, default_suffix):
    name = Path(filename).name
    if Path(name).suffix == "":
        name = f"{name}{default_suffix}"
    return name


def resolve_output_paths(input_csv, output_csv, tables_dir):
    if output_csv is None:
        output_csv_name = f"{Path(input_csv).stem}_with_if.csv"
    else:
        output_csv_name = standardize_filename(output_csv, ".csv")
    return ensure_unique_filename(Path(tables_dir) / output_csv_name)


def parse_plot_option(plot_option, plots_dir):
    if isinstance(plot_option, str) and plot_option.lower() in PLOT_DISABLE_TOKENS:
        return False, None
    if plot_option == PLOT_DEFAULT_TOKEN:
        return True, ensure_unique_filename(Path(plots_dir) / DEFAULT_PLOT_OUTPUT_NAME)
    plot_path = Path(plot_option).expanduser()
    if plot_path.suffix == "":
        plot_path = plot_path.with_suffix(".png")
    if not plot_path.is_absolute() and plot_path.parent == Path("."):
        plot_path = Path(plots_dir) / plot_path.name
    return True, ensure_unique_filename(plot_path)


def describe_plot_option(plot_option):
    if isinstance(plot_option, str) and plot_option.lower() in PLOT_DISABLE_TOKENS:
        return "disabled"
    if plot_option == PLOT_DEFAULT_TOKEN:
        return f"enabled ({DEFAULT_PLOT_OUTPUT_NAME})"
    plot_path = Path(plot_option).expanduser()
    if plot_path.suffix == "":
        plot_path = plot_path.with_suffix(".png")
    return f"enabled ({plot_path})"


def emit_run_configuration(args, output_csv_path):
    emit_message(
        "info",
        "Run Configuration",
        lines=[*get_run_logo_lines(), format_detail("Version", TOOL_VERSION),
               format_detail("Input", style_text(str(args.input_csv), TEXT_COLOURS["accent"])),
               format_detail("Output", style_text(str(output_csv_path), TEXT_COLOURS["accent"])),
               format_detail("Organisms", ", ".join(args.organism) if args.organism else "all"),
               format_detail("Assay centre", CENTRE_COLUMN),
               format_detail("Missing controls", args.missing_controls),
               format_detail("Within-assay outliers", args.within_assay_outlier_controls),
               format_detail("Across-assay outliers", args.outlier_controls),
               format_detail("Plot", describe_plot_option(args.plot))])


def validate_input_columns(data):
    required_columns = {"assay_id", "data_id", "treatment_id", "treatment", "organism"}
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError("The input CSV is missing required column(s): " + ", ".join(sorted(missing_columns)))
    validate_assay_keys(data)


def assay_keys(data, unique=True):
    if data.empty:
        return []
    assay_key_data = data[ASSAY_KEY_COLUMNS].drop_duplicates() if unique else data[ASSAY_KEY_COLUMNS]
    return list(map(tuple, assay_key_data.to_numpy()))


def assay_key_series(data):
    return pd.Series(assay_keys(data, unique=False), index=data.index)


def assay_label(assay_key):
    organism, assay_id = assay_key
    return f"{organism} | assay_id {assay_id}"


def sorted_assay_keys(assay_key_iterable):
    return sorted(assay_key_iterable, key=lambda key: (str(key[0]), str(key[1])))


def validate_assay_keys(data):
    missing_key_rows = data[ASSAY_KEY_COLUMNS].isna().any(axis=1)
    blank_key_rows = data[ASSAY_KEY_COLUMNS].astype(str).apply(lambda column: column.str.strip().eq("")).any(axis=1)
    if (missing_key_rows | blank_key_rows).any():
        raise ValueError("Invalid assay keys: organism and assay_id are required for every row.")


def filter_selected_organisms(data, selected_organisms):
    if not selected_organisms:
        return data

    selected_organisms = list(dict.fromkeys(selected_organisms))
    available_organisms = set(data["organism"].dropna().unique())
    missing_organisms = [organism for organism in selected_organisms if organism not in available_organisms]
    if missing_organisms:
        available = ", ".join(map(str, sorted(available_organisms)))
        raise ValueError(
            "Requested organism(s) were not found in the input data: "
            f"{', '.join(missing_organisms)}. Available organisms: {available}"
        )

    filtered_data = data.loc[data["organism"].isin(selected_organisms)].copy()
    emit_message("info", "Organism filter", lines=[
        format_detail("Selected", ", ".join(selected_organisms)),
        format_detail("Rows retained", style_text(f"{len(filtered_data):,}", "1")),
    ])
    return filtered_data


def filter_valid_organisms(data):
    all_organisms = set(data["organism"])
    organisms_missing_controls = set()
    for control in CONTROL_TREATMENT_NAMES:
        organisms_missing_controls |= all_organisms - set(data.loc[data["treatment"].eq(control), "organism"])
    if organisms_missing_controls:
        emit_message("warn", "Normalization skipped for the following organisms:", items=sorted(organisms_missing_controls))
    else:
        emit_message("info", "All organisms have both Growth Control and Sterility Control.")
    organisms_with_controls = all_organisms - organisms_missing_controls
    return data[data["organism"].isin(organisms_with_controls)]


def find_assays_missing_controls(data_valid_organisms, controls):
    assays_with_control = {control: set(assay_keys(data_valid_organisms.loc[data_valid_organisms["treatment"].eq(control)])) for control in controls}
    assays_missing_controls = {assay_key: absent_controls for assay_key in sorted_assay_keys(assay_keys(data_valid_organisms))
                               if (absent_controls := [control for control in controls if assay_key not in assays_with_control[control]])}
    if assays_missing_controls:
        emit_message("warn", "Following assays are missing assay-level controls:",
                     items=[f"{assay_label(assay_key)}: missing {' & '.join(absent_controls)}" for assay_key, absent_controls in assays_missing_controls.items()])
    else:
        emit_message("info", "All assays have both assay-level controls.")
    return assays_missing_controls


def get_measurement_columns(data, controls=None):
    measurement_columns = [column for column in data.columns if MEASUREMENT_COLUMN_RE.fullmatch(column)]
    if not measurement_columns:
        return []
    rows = ~data["treatment"].isin(controls) if controls is not None else pd.Series(True, index=data.index)
    has_values = data.loc[rows, measurement_columns].apply(pd.to_numeric, errors="coerce").notna().any()
    return [column for column in measurement_columns if has_values[column]]


def summarize_control_readings_by_assay(control_readings_by_assay):
    return (control_readings_by_assay.groupby(["assay_id", "organism", "treatment"])["measurement"]
            .agg(median="median", n_measurements="count")
            .reset_index()
            .sort_values(["organism", "assay_id", "treatment"])
            .reset_index(drop=True))


def summarize_control_readings_by_organism(control_readings_by_organism):
    return (control_readings_by_organism.groupby(["organism", "treatment"])["measurement"]
            .agg(median="median", n_measurements="count")
            .reset_index())


def combine_missing_control_maps(*missing_control_maps):
    combined_missing_controls = {}
    for missing_control_map in missing_control_maps:
        for assay_key, absent_controls in missing_control_map.items():
            combined_missing_controls.setdefault(tuple(assay_key), set()).update(absent_controls)
    return {assay_key: sorted(combined_missing_controls[assay_key]) for assay_key in sorted_assay_keys(combined_missing_controls)}


def calculate_control_stats_by_assay(data_valid_organisms, controls, measurement_columns):
    control_readings_by_assay = (data_valid_organisms.loc[data_valid_organisms["treatment"].isin(controls),
                                                      ["assay_id", "organism", "treatment"] + measurement_columns].melt(
                                                          id_vars=["assay_id", "organism", "treatment"],
                                                          value_vars=measurement_columns, var_name="column",
                                                          value_name="measurement"))
    control_stats_by_assay = summarize_control_readings_by_assay(control_readings_by_assay)
    return control_readings_by_assay, control_stats_by_assay


def add_modified_z_score_outlier_flag(data, group_columns, value_column, flag_column):
    values_by_group = data.groupby(group_columns)[value_column]
    median = values_by_group.transform("median")
    absolute_deviation = (data[value_column] - median).abs()
    mad = absolute_deviation.groupby([data[column] for column in group_columns]).transform("median")
    valid_mad = mad.gt(0)
    modified_z_score = (MODIFIED_Z_SCORE_SCALE * (data[value_column] - median) / mad).where(valid_mad)
    data[flag_column] = modified_z_score.abs().gt(MODIFIED_Z_SCORE_THRESHOLD).fillna(False)
    return data


def add_within_assay_outlier_flags(control_readings, measurement_column):
    control_readings = add_modified_z_score_outlier_flag(
        control_readings,
        ["assay_id", "organism", "treatment"],
        measurement_column,
        "within_assay_mZ_score",
    )
    control_readings["is_extreme_outlier"] = control_readings["within_assay_mZ_score"]
    return control_readings


def find_assays_missing_controls_after_within_assay(control_stats_by_assay_comparison):
    assays_missing_controls = (control_stats_by_assay_comparison[
        control_stats_by_assay_comparison["raw_n_measurements"].gt(0)
        &
        control_stats_by_assay_comparison["cleaned_n_measurements"].fillna(0).eq(0)
    ].groupby(ASSAY_KEY_COLUMNS)["treatment"].apply(lambda x: sorted(set(x))).to_dict())
    if assays_missing_controls:
        emit_message("warn", "Within-assay cleaning removed all control measurements",
                     lines="These controls will be treated as missing if within-assay outlier exclusion is enabled.",
                     items=[f"{assay_label(assay_key)}: removed {' & '.join(assays_missing_controls[assay_key])}"
                            for assay_key in sorted_assay_keys(assays_missing_controls)])
    return assays_missing_controls


class WithinAssayOutlierCheck(NamedTuple):
    readings_checked: pd.DataFrame
    readings_cleaned: pd.DataFrame
    stats_raw: pd.DataFrame
    stats_cleaned: pd.DataFrame
    stats_comparison: pd.DataFrame


def perform_within_assay_outlier_check(control_readings_by_assay, controls, control_stats_by_assay_raw):
    control_readings_by_assay_checked = control_readings_by_assay[control_readings_by_assay["treatment"].isin(controls)].copy()
    control_readings_by_assay_checked = add_within_assay_outlier_flags(
        control_readings_by_assay_checked, "measurement"
    )
    control_readings_by_assay_cleaned = (
        control_readings_by_assay_checked[~control_readings_by_assay_checked["is_extreme_outlier"]].copy()
    )
    control_stats_by_assay_cleaned = summarize_control_readings_by_assay(control_readings_by_assay_cleaned)
    control_stats_by_assay_comparison = (
        control_stats_by_assay_raw.rename(
            columns={"median": "raw_median", "n_measurements": "raw_n_measurements"}
        ).merge(
            control_stats_by_assay_cleaned.rename(
                columns={"median": "cleaned_median", "n_measurements": "cleaned_n_measurements"}
            ),
            on=["assay_id", "organism", "treatment"],
            how="left",
        )
    )
    control_stats_by_assay_comparison["outlier_count"] = (
        control_stats_by_assay_comparison["raw_n_measurements"]
        - control_stats_by_assay_comparison["cleaned_n_measurements"].fillna(0)
    ).astype(int)
    control_stats_changed = (
        control_stats_by_assay_comparison[control_stats_by_assay_comparison["outlier_count"].gt(0)]
        .sort_values(["organism", "assay_id", "treatment"])
        .reset_index(drop=True)
    )
    if control_stats_changed.empty:
        emit_message("info", "No extreme within-assay control outliers detected.")
    else:
        emit_message("warn", "Extreme within-assay control outliers detected",
                     lines="Affected assay/control groups are listed below.",
                     items=[
                         f"Assay ID {row.assay_id} | {row.organism} | {row.treatment} | outliers = {row.outlier_count}"
                         for row in control_stats_changed.itertuples(index=False)
                     ])
    return WithinAssayOutlierCheck(
        control_readings_by_assay_checked,
        control_readings_by_assay_cleaned,
        control_stats_by_assay_raw,
        control_stats_by_assay_cleaned,
        control_stats_by_assay_comparison,
    )


def perform_across_assay_outlier_check(control_stats_by_assay, controls):
    control_stats_by_assay_checked = control_stats_by_assay[control_stats_by_assay["treatment"].isin(controls)].copy()
    centre_label_lower = CENTRE_LABEL.lower()

    control_stats_by_assay_checked = add_modified_z_score_outlier_flag(
        control_stats_by_assay_checked,
        ["organism", "treatment"],
        CENTRE_COLUMN,
        "across_assay_mZ_score",
    )
    control_stats_by_assay_checked["is_extreme_outlier"] = control_stats_by_assay_checked["across_assay_mZ_score"]
    control_outliers = (control_stats_by_assay_checked[control_stats_by_assay_checked["is_extreme_outlier"]]
                       .sort_values(["organism", "treatment", "assay_id"])
                       .reset_index(drop=True))

    if not control_outliers.empty:
        emit_message("warn", "Extreme across-assay control outliers detected",
                     lines=f"Inspect the raw data for these assays. Detection used modified z-scores on assay-level {centre_label_lower}.",
                     items=[f"Assay ID {row.assay_id} | {row.organism} | {row.treatment} | {centre_label_lower} = {getattr(row, CENTRE_COLUMN):.4f}"
                            for row in control_outliers.itertuples(index=False)])
    else:
        emit_message("info", "No extreme across-assay control outliers detected.",
                     lines=f"Detection used modified z-scores on assay-level {centre_label_lower}.")
    return control_stats_by_assay_checked, control_outliers


def calculate_control_stats_by_organism_for_imputation(control_readings_by_assay, controls, control_outliers):
    outlier_assays_by_control = {control: set(assay_keys(control_outliers.loc[control_outliers["treatment"].eq(control)]))
                                 for control in controls}
    if control_outliers.empty:
        emit_message("info", "Organism-level control stats for imputation used all control assays")
    else:
        emit_message("warn", "Organism-level control stats for imputation excluded flagged control outliers",
                     items=[f"{control}: excluded {', '.join(assay_label(assay_key) for assay_key in sorted_assay_keys(outlier_assays_by_control[control]))}"
                            for control in controls
                            if outlier_assays_by_control[control]])
    control_assay_keys = assay_key_series(control_readings_by_assay)
    control_readings_by_organism = control_readings_by_assay.loc[
        ~(
            control_readings_by_assay["treatment"].eq("Sterility Control")
            & control_assay_keys.isin(outlier_assays_by_control["Sterility Control"])
        )
        & ~(
            control_readings_by_assay["treatment"].eq("Growth Control")
            & control_assay_keys.isin(outlier_assays_by_control["Growth Control"])
        ),
        ["organism", "treatment", "measurement"],
    ]
    return summarize_control_readings_by_organism(control_readings_by_organism)


def create_organism_grid(n_organisms, max_columns, organism_width, organism_height, height_padding=0.0, **subplot_kwargs):
    import matplotlib.pyplot as plt
    num_columns = min(max_columns, n_organisms) or 1
    num_rows = math.ceil(n_organisms / num_columns)
    fig, subplot_axes = plt.subplots(num_rows, num_columns,
                                     figsize=(organism_width * num_columns, organism_height * num_rows + height_padding),
                                     **subplot_kwargs)
    return fig, np.ravel(np.atleast_1d(subplot_axes))


def collect_legend_entries(axes):
    legend_items = {}
    for axis in axes:
        current_handles, current_labels = axis.get_legend_handles_labels()
        for handle, label in zip(current_handles, current_labels):
            if label and label not in legend_items:
                legend_items[label] = handle
    return legend_items


def ordered_legend(legend_items, legend_order):
    ordered_labels = [label for label in legend_order if label in legend_items]
    return [legend_items[label] for label in ordered_labels], ordered_labels


def plot_across_assay_control_outliers(control_stats_by_assay_checked, control_stats_by_organism,
                                      control_stats_by_organism_all, controls,
                                      label_size=12, show_plot=True, plot_output=None):
    import matplotlib.pyplot as plt
    centre_label_all = f"{CENTRE_LABEL} (all assays)"
    centre_label_imputed = f"{CENTRE_LABEL} (used for imputation)"
    organism_control_centres = control_stats_by_organism.set_index(["organism", "treatment"])[CENTRE_COLUMN]
    organism_control_centres_all = control_stats_by_organism_all.set_index(["organism", "treatment"])[CENTRE_COLUMN]
    all_centre_values = pd.concat([control_stats_by_assay_checked[CENTRE_COLUMN], control_stats_by_organism[CENTRE_COLUMN],
                                   control_stats_by_organism_all[CENTRE_COLUMN]]).dropna()
    y_padding = 0.1 * (all_centre_values.max() - all_centre_values.min()) or 0.1
    unique_organisms = sorted(control_stats_by_assay_checked["organism"].unique())
    control_stats_lookup = {
        (organism_name, control_name): control_stats_group.sort_values("assay_id").reset_index(drop=True)
        for (organism_name, control_name), control_stats_group in control_stats_by_assay_checked.groupby(["organism", "treatment"], sort=False)
    }
    fig, subplot_axes = create_organism_grid(len(unique_organisms), 3, 6, 3.35, 0.4, sharey=True)
    control_colours = CONTROL_QC_COLOURS
    legend_labels_seen = set()
    for axis, organism_name in zip(subplot_axes, unique_organisms):
        for control_name in controls:
            control_stats_for_organism = control_stats_lookup.get((organism_name, control_name))
            if control_stats_for_organism is None or control_stats_for_organism.empty:
                continue
            outlier_mask = control_stats_for_organism["is_extreme_outlier"]
            point_colour = control_colours[control_name]
            axis.scatter(control_stats_for_organism.loc[~outlier_mask, "assay_id"],
                         control_stats_for_organism.loc[~outlier_mask, CENTRE_COLUMN],
                         s=50, alpha=0.8, color=point_colour, label=control_name if control_name not in legend_labels_seen else None)
            legend_labels_seen.add(control_name)
            if outlier_mask.any():
                axis.scatter(control_stats_for_organism.loc[outlier_mask, "assay_id"],
                             control_stats_for_organism.loc[outlier_mask, CENTRE_COLUMN],
                             s=110, facecolors="none", edgecolors="crimson", linewidths=1.8,
                             label="Flagged Outlier" if "Flagged Outlier" not in legend_labels_seen else None)
                legend_labels_seen.add("Flagged Outlier")
            organism_control_centre_all = organism_control_centres_all.get((organism_name, control_name), np.nan)
            organism_control_centre_used = organism_control_centres.get((organism_name, control_name), np.nan)

            if (pd.notna(organism_control_centre_all) and pd.notna(organism_control_centre_used)
                    and not np.isclose(organism_control_centre_all, organism_control_centre_used)):
                axis.axhline(organism_control_centre_all, linestyle="--", linewidth=1.2, color="grey",
                             alpha=0.95, label=centre_label_all if centre_label_all not in legend_labels_seen else None)
                legend_labels_seen.add(centre_label_all)
            if pd.notna(organism_control_centre_used):
                axis.axhline(
                    organism_control_centre_used,
                    linestyle="--",
                    linewidth=1.2,
                    color=point_colour,
                    label=centre_label_imputed if centre_label_imputed not in legend_labels_seen else None)
                legend_labels_seen.add(centre_label_imputed)
            for assay_id, centre_value in control_stats_for_organism.loc[outlier_mask, ["assay_id", CENTRE_COLUMN]].itertuples(index=False):
                axis.annotate(int(assay_id), (assay_id, centre_value), xytext=(-7, 10), textcoords="offset points", fontsize=8, color="crimson")
        axis.set(ylim=(all_centre_values.min() - y_padding, all_centre_values.max() + y_padding))
        axis.set_title(organism_name, fontweight="bold")
        axis.set_xlabel("Assay ID", fontweight="bold", fontsize=label_size)
        axis.set_ylabel(f"Control {CENTRE_LABEL}", fontweight="bold", fontsize=label_size)
        axis.tick_params(axis="x", rotation=0)
    for axis in subplot_axes[len(unique_organisms):]:
        axis.remove()

    legend_items = collect_legend_entries(subplot_axes[:len(unique_organisms)])
    legend_order = ["Growth Control", "Sterility Control", "Flagged Outlier", centre_label_all, centre_label_imputed]
    ordered_legend_handles, ordered_legend_labels = ordered_legend(legend_items, legend_order)
    figure_width, figure_height = fig.get_size_inches()
    top_margin = 1 - PLOT_ORGANISM_TOP_OFFSET / figure_height
    bottom_margin = 0.28 / figure_height
    left_margin = 0.72 / figure_width
    right_margin = 1 - 0.08 / figure_width
    fig.legend(ordered_legend_handles, ordered_legend_labels, loc="upper center",
               bbox_to_anchor=(0.5, PLOT_LEGEND_Y), ncol=max(1, len(ordered_legend_labels)),
               frameon=False, fontsize=12)
    fig.suptitle("Across Assay Control Outliers", fontsize=15, fontweight="bold", y=PLOT_TITLE_Y)
    fig.subplots_adjust(top=top_margin, bottom=bottom_margin, left=left_margin, right=right_margin,
                        wspace=0.22, hspace=0.30)
    if plot_output is not None:
        plot_path = save_plot_figure(fig, plot_output)
        emit_message("info", "Across-assay control plot saved",
                     lines=[format_detail("Statistic", CENTRE_LABEL),
                            format_detail("Path", style_text(str(plot_path), TEXT_COLOURS["accent"]))])
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def build_within_assay_plot_output_path(plot_output, organism_name):
    plot_path = Path(plot_output)
    organism_token = slugify_filename_token(organism_name, default="organism")
    return ensure_unique_filename(plot_path.with_name(f"{DEFAULT_WITHIN_ASSAY_PLOT_PREFIX}_{organism_token}{plot_path.suffix}"))


def build_raw_inhibition_plot_output_path(plot_output):
    plot_path = Path(plot_output)
    plot_name = Path(DEFAULT_RAW_INHIBITION_PLOT_NAME)
    return ensure_unique_filename(plot_path.with_name(f"{plot_name.stem}{plot_path.suffix}"))


def render_within_assay_organism_figure(organism_name, assay_ids, organism_measurements, organism_summaries,
                                     all_measurement_values, controls, within_assay_outlier_controls,
                                     label_size, show_plot, organism_plot_output):
    import matplotlib.pyplot as plt
    control_colours = CONTROL_QC_COLOURS
    control_short_labels = {"Growth Control": "GC", "Sterility Control": "SC"}
    raw_centre_label = f"{CENTRE_LABEL} (raw)"
    cleaned_centre_label = f"{CENTRE_LABEL} (within-assay cleaned)"
    y_padding = 0.1 * (all_measurement_values.max() - all_measurement_values.min()) or 0.1
    fig, subplot_axes = create_organism_grid(len(assay_ids), 4, 4.8, 3.05, 0.3, sharey=True)
    legend_labels_seen = set()

    for axis, assay_id in zip(subplot_axes, assay_ids):
        for control_index, control_name in enumerate(controls):
            control_measurements = organism_measurements.get((assay_id, control_name))
            if control_measurements is None or control_measurements.empty:
                continue
            jitter = np.zeros(len(control_measurements)) if len(control_measurements) == 1 else np.linspace(-0.16, 0.16, len(control_measurements))
            x_values = control_index + jitter
            outlier_mask = control_measurements["is_extreme_outlier"].to_numpy()
            axis.scatter(x_values[~outlier_mask],
                         control_measurements.loc[~control_measurements["is_extreme_outlier"], "measurement"],
                         s=50, alpha=0.75, color=control_colours[control_name],
                         label=control_name if control_name not in legend_labels_seen else None)
            legend_labels_seen.add(control_name)
            if outlier_mask.any():
                axis.scatter(x_values[outlier_mask],
                             control_measurements.loc[control_measurements["is_extreme_outlier"], "measurement"],
                             s=50, alpha=0.75, color="crimson",
                             label="Flagged Outlier" if "Flagged Outlier" not in legend_labels_seen else None)
                legend_labels_seen.add("Flagged Outlier")

            if (assay_id, control_name) not in organism_summaries:
                continue
            raw_centre, cleaned_centre = organism_summaries[(assay_id, control_name)]
            axis.hlines(raw_centre, control_index - 0.25, control_index + 0.25, color="grey",
                        linestyle="--", linewidth=1.2,
                        label=raw_centre_label if raw_centre_label not in legend_labels_seen else None)
            legend_labels_seen.add(raw_centre_label)
            if pd.notna(cleaned_centre):
                axis.hlines(cleaned_centre, control_index - 0.25, control_index + 0.25,
                            color=control_colours[control_name], linestyle="-", linewidth=1.6,
                            label=cleaned_centre_label if cleaned_centre_label not in legend_labels_seen else None)
                legend_labels_seen.add(cleaned_centre_label)
        axis.set(title=f"Assay ID {int(assay_id)}",
                 ylim=(all_measurement_values.min() - y_padding, all_measurement_values.max() + y_padding))
        axis.set_xticks(range(len(controls)))
        axis.set_xticklabels([control_short_labels[control_name] for control_name in controls], fontsize=label_size - 1)
        axis.set_xlabel("Control", fontweight="bold", fontsize=label_size)
        axis.set_ylabel("Measurement", fontweight="bold", fontsize=label_size)

    for axis in subplot_axes[len(assay_ids):]:
        axis.remove()

    legend_items = collect_legend_entries(subplot_axes[:len(assay_ids)])
    legend_order = ["Growth Control", "Sterility Control", "Flagged Outlier", raw_centre_label, cleaned_centre_label]
    ordered_legend_handles, ordered_legend_labels = ordered_legend(legend_items, legend_order)
    figure_width, figure_height = fig.get_size_inches()
    legend_y = 1 - 0.34 / figure_height
    title_y = 1 + 0.020 / figure_height
    top_margin = 1 - 0.80 / figure_height
    bottom_margin = 0.50 / figure_height
    left_margin = 0.55 / figure_width
    right_margin = 1 - 0.05 / figure_width
    fig.legend(ordered_legend_handles, ordered_legend_labels, loc="center",
               bbox_to_anchor=(0.5, legend_y), ncol=max(1, len(ordered_legend_labels)),
               frameon=False, fontsize=12)
    fig.suptitle(f"{organism_name} | Within-assay outliers {within_assay_outlier_controls}",
                 fontsize=14, fontweight="bold", y=title_y)
    fig.subplots_adjust(top=top_margin, bottom=bottom_margin, left=left_margin, right=right_margin,
                        wspace=0.34, hspace=0.42)

    if organism_plot_output is not None:
        save_plot_figure(fig, organism_plot_output)
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def submit_within_assay_control_plots(control_readings_by_assay_checked, control_stats_by_assay_comparison,
                                     controls, within_assay_outlier_controls="keep",
                                     label_size=12, show_plot=False, plot_output=None,
                                     renderer=None):
    saved_plot_paths = []
    if plot_output is None and not show_plot:
        return saved_plot_paths
    if renderer is None or show_plot:
        renderer = BackgroundRenderer()
    raw_centre_column = f"raw_{CENTRE_COLUMN}"
    cleaned_centre_column = f"cleaned_{CENTRE_COLUMN}"
    unique_organisms = sorted(control_readings_by_assay_checked["organism"].unique())
    measurement_lookup = {}
    for (organism_name, assay_id, control_name), control_measurements in control_readings_by_assay_checked.groupby(
            ["organism", "assay_id", "treatment"], sort=False):
        measurement_lookup.setdefault(organism_name, {})[(assay_id, control_name)] = (
            control_measurements.sort_values(["column", "measurement"]).reset_index(drop=True))
    summary_lookup = {}
    for row in control_stats_by_assay_comparison.itertuples(index=False):
        summary_lookup.setdefault(row.organism, {})[(row.assay_id, row.treatment)] = (
            getattr(row, raw_centre_column), getattr(row, cleaned_centre_column))
    assay_ids_by_organism = (
        control_readings_by_assay_checked.groupby("organism")["assay_id"].apply(lambda x: sorted(x.unique())).to_dict()
    )
    organism_measurement_values = control_readings_by_assay_checked.groupby("organism")["measurement"].apply(list).to_dict()
    raw_centres_by_organism = control_stats_by_assay_comparison.groupby("organism")[raw_centre_column].apply(list).to_dict()
    cleaned_centres_by_organism = control_stats_by_assay_comparison.groupby("organism")[cleaned_centre_column].apply(list).to_dict()

    for organism_name in unique_organisms:
        assay_ids = assay_ids_by_organism.get(organism_name, [])
        if not assay_ids:
            continue
        all_measurement_values = pd.concat([
            pd.Series(organism_measurement_values.get(organism_name, []), dtype=float),
            pd.Series(raw_centres_by_organism.get(organism_name, []), dtype=float),
            pd.Series(cleaned_centres_by_organism.get(organism_name, []), dtype=float),
        ]).dropna()
        if all_measurement_values.empty:
            all_measurement_values = pd.Series([0.0])
        organism_plot_output = None
        if plot_output is not None:
            organism_plot_output = build_within_assay_plot_output_path(plot_output, organism_name)
            saved_plot_paths.append(organism_plot_output)
        renderer.submit(
            render_within_assay_organism_figure,
            organism_name, assay_ids,
            measurement_lookup.get(organism_name, {}),
            summary_lookup.get(organism_name, {}),
            all_measurement_values, controls, within_assay_outlier_controls,
            label_size, show_plot, organism_plot_output,
        )
    return saved_plot_paths


def report_within_assay_control_plots(saved_plot_paths):
    if saved_plot_paths:
        emit_message("info", "Within-assay control plots saved",
                     lines=["One plot was generated per organism.",
                            format_detail("Statistic", CENTRE_LABEL)],
                     items=[style_text(str(plot_path), TEXT_COLOURS["accent"]) for plot_path in saved_plot_paths])


def render_raw_inhibition_figure(plot_data, concentration_order_template, label_size, show_plot, plot_path):
    import matplotlib.pyplot as plt
    raw_inhibition_group_order = ["<0", "[0,1]", ">1"]
    rng = np.random.default_rng(42)

    unique_organisms = sorted(plot_data["organism"].dropna().unique())
    fig, subplot_axes = create_organism_grid(len(unique_organisms), 3, 5.04, 4.2, sharey=True)
    fig.set_dpi(PLOT_DPI)
    fig.patch.set_facecolor("white")
    palette = {"<0": "#d62728", "[0,1]": "#2ca02c", ">1": "#ff7f0e"}
    legend_labels_seen = set()

    for axis, organism_name in zip(subplot_axes, unique_organisms):
        organism_data = plot_data[plot_data["organism"].eq(organism_name)].copy()
        organism_concentrations = set(organism_data["concentration"])
        concentration_order = [concentration for concentration in concentration_order_template if concentration in organism_concentrations]
        x_map = {concentration: index for index, concentration in enumerate(concentration_order)}
        organism_data["x"] = organism_data["concentration"].map(x_map).astype(float)
        organism_data["x_jitter"] = organism_data["x"] + rng.uniform(-0.14, 0.14, size=len(organism_data))
        for raw_inhibition_group in raw_inhibition_group_order:
            group_data = organism_data[organism_data["raw_inhibition_group"].eq(raw_inhibition_group)]
            if group_data.empty:
                continue
            axis.scatter(group_data["x_jitter"], group_data["raw_inhibition"],
                         s=10, alpha=0.55, color=palette[raw_inhibition_group],
                         edgecolors="white", linewidths=0.25,
                         label=raw_inhibition_group if raw_inhibition_group not in legend_labels_seen else None)
            legend_labels_seen.add(raw_inhibition_group)
        axis.set_title(organism_name, size=12, fontweight="bold")
        axis.set_facecolor("#fbfbfb")
        axis.set_xticks(range(len(concentration_order)))
        axis.set_xticklabels(concentration_order, rotation=0, ha="center")
        axis.tick_params(axis="y", labelsize=10)
        axis.tick_params(axis="x", labelsize=10)
        axis.set_xlabel("Concentration", fontweight="bold", fontsize=label_size)
        axis.set_ylabel("Raw Inhibition Fraction", fontweight="bold", fontsize=label_size)
        axis.grid(True, axis="y", linestyle="--", alpha=0.9, color="#d9d9d9", linewidth=0.7)
        axis.grid(False, axis="x")
        axis.axhline(0, color="#666666", lw=0.9, alpha=0.9)
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.1)
            spine.set_color("#333333")

    for axis in subplot_axes[len(unique_organisms):]:
        axis.remove()

    legend_items = collect_legend_entries(subplot_axes[:len(unique_organisms)])
    ordered_legend_handles, ordered_legend_labels = ordered_legend(legend_items, raw_inhibition_group_order)
    figure_width, figure_height = fig.get_size_inches()
    top_margin = 1 - PLOT_ORGANISM_TOP_OFFSET / figure_height
    bottom_margin = 0.42 / figure_height
    left_margin = 0.62 / figure_width
    right_margin = 1 - 0.05 / figure_width
    fig.legend(ordered_legend_handles, ordered_legend_labels, loc="upper center",
               bbox_to_anchor=(0.5, PLOT_LEGEND_Y),
               ncol=max(1, len(ordered_legend_labels)), frameon=False,
               fontsize=12, markerscale=2.4)
    fig.suptitle("Raw Inhibition Fraction Across Concentrations",
                 fontsize=15, fontweight="bold", y=PLOT_TITLE_Y)
    fig.subplots_adjust(top=top_margin, bottom=bottom_margin, left=left_margin, right=right_margin,
                        wspace=0.18, hspace=0.28)

    if plot_path is not None:
        save_plot_figure(fig, plot_path)
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def plot_raw_inhibition_fractions(data_with_if, raw_inhibition_columns, controls,
                                label_size=12, show_plot=False, plot_output=None, renderer=None):
    if plot_output is None and not show_plot:
        return
    if renderer is None or show_plot:
        renderer = BackgroundRenderer()
    raw_inhibition_columns = [column for column in raw_inhibition_columns if column in data_with_if.columns]
    if not raw_inhibition_columns:
        emit_message("warn", "Raw inhibition plot skipped",
                     lines="No RI columns were available in the normalized data.")
        return

    identifier_column = "data_id"
    filter_column = "treatment"
    plot_data = data_with_if.loc[
        ~data_with_if[filter_column].isin(controls),
        ["organism", identifier_column] + raw_inhibition_columns,
    ].copy()
    plot_data = plot_data.melt(id_vars=["organism", identifier_column],
                               value_vars=raw_inhibition_columns,
                               var_name="concentration", value_name="raw_inhibition")
    plot_data["concentration"] = plot_data["concentration"].str.replace("RI", "", regex=False)
    plot_data["raw_inhibition"] = pd.to_numeric(plot_data["raw_inhibition"], errors="coerce")
    plot_data = plot_data.dropna(subset=["raw_inhibition", "organism"]).copy()
    if plot_data.empty:
        emit_message("warn", "Raw inhibition plot skipped",
                     lines="No non-control RI values were available after filtering.")
        return

    raw_inhibition_group_order = ["<0", "[0,1]", ">1"]
    plot_data["raw_inhibition_group"] = np.select(
        [
            plot_data["raw_inhibition"] < 0,
            plot_data["raw_inhibition"].between(0, 1, inclusive="both"),
            plot_data["raw_inhibition"] > 1,
        ],
        raw_inhibition_group_order,
        default="",
    )
    plot_data = plot_data[plot_data["raw_inhibition_group"].ne("")].copy()
    concentration_order_template = [column[2:] for column in raw_inhibition_columns]

    plot_path = build_raw_inhibition_plot_output_path(plot_output) if plot_output is not None else None
    renderer.submit(render_raw_inhibition_figure, plot_data, concentration_order_template, label_size, show_plot, plot_path)
    if plot_path is not None:
        emit_message("info", "Raw inhibition plot saved",
                     lines=format_detail("Path", style_text(str(plot_path), TEXT_COLOURS["accent"])))


def render_control_pool_outlier_classes_figure(plot_data, plot_path, label_size=12):
    import matplotlib.pyplot as plt
    controls = ["Sterility Control", "Growth Control"]
    outlier_classes = ["Inlier", "Within Assay Outlier", "Across Assay Outlier"]
    outlier_colours = {"Inlier": "#333333", "Within Assay Outlier": "#f59e0b", "Across Assay Outlier": "crimson"}
    x_positions = {
        ("Sterility Control", "Inlier"): 0,
        ("Sterility Control", "Within Assay Outlier"): 1,
        ("Sterility Control", "Across Assay Outlier"): 2,
        ("Growth Control", "Inlier"): 4,
        ("Growth Control", "Within Assay Outlier"): 5,
        ("Growth Control", "Across Assay Outlier"): 6,
    }
    unique_organisms = sorted(plot_data["organism"].dropna().unique())
    fig, subplot_axes = create_organism_grid(len(unique_organisms), 3, 6, 3.6, sharey=True)
    rng = np.random.default_rng(42)

    for axis, organism_name in zip(subplot_axes, unique_organisms):
        organism_data = plot_data[plot_data["organism"].eq(organism_name)]
        for control_name in controls:
            for outlier_class in outlier_classes:
                measurement = organism_data.loc[
                    organism_data["treatment"].eq(control_name)
                    & organism_data["outlier_class"].eq(outlier_class),
                    "measurement",
                ].dropna()
                if measurement.empty:
                    continue
                x_values = x_positions[(control_name, outlier_class)] + rng.uniform(-0.18, 0.18, len(measurement))
                axis.scatter(x_values, measurement, s=10, alpha=0.50,
                             color=outlier_colours[outlier_class], label=outlier_class)
        axis.axvline(3, color="#999999", lw=1.0, ls="--")
        axis.set_xlim(-0.75, 6.75)
        axis.text(0.5, 1.11, organism_name, transform=axis.transAxes,
                  ha="center", va="bottom", fontsize=11, fontweight="bold",
                  clip_on=False)
        axis.set_xticks([0, 1, 2, 4, 5, 6])
        axis.set_xticklabels(["Inlier", "Within", "Across", "Inlier", "Within", "Across"], rotation=35, ha="right")
        axis.text(1, 1.02, "Sterility Control", transform=axis.get_xaxis_transform(),
                  ha="center", va="bottom", fontsize=9, fontweight="bold", clip_on=False)
        axis.text(5, 1.02, "Growth Control", transform=axis.get_xaxis_transform(),
                  ha="center", va="bottom", fontsize=9, fontweight="bold", clip_on=False)
        axis.set_ylabel("Measurement", fontweight="bold", fontsize=label_size)
        axis.grid(True, axis="y", alpha=0.25)
        axis.grid(False, axis="x")
        for spine in axis.spines.values():
            spine.set_linewidth(1.1)
            spine.set_color("#333333")

    for axis in subplot_axes[len(unique_organisms):]:
        axis.remove()

    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=outlier_colours[outlier_class],
                   label=outlier_class, markersize=7)
        for outlier_class in outlier_classes
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, PLOT_LEGEND_Y),
               ncol=3, frameon=False)
    fig.suptitle("Control Pool Outlier Classes", fontsize=15, fontweight="bold", y=PLOT_TITLE_Y)
    figure_width, figure_height = fig.get_size_inches()
    fig.subplots_adjust(top=1 - 1.25 / figure_height,
                        bottom=0.64 / figure_height,
                        left=0.72 / figure_width,
                        right=1 - 0.08 / figure_width,
                        wspace=0.22, hspace=0.55)
    save_plot_figure(fig, plot_path)
    plt.close(fig)


def plot_control_pool_outlier_classes(control_pool, output_dir, label_size=12, renderer=None):
    if control_pool.empty:
        return None
    if renderer is None:
        renderer = BackgroundRenderer()
    controls = ["Sterility Control", "Growth Control"]
    plot_data = control_pool[control_pool["treatment"].isin(controls)].copy()
    plot_data["measurement"] = pd.to_numeric(plot_data["measurement"], errors="coerce")
    plot_data = plot_data.dropna(subset=["measurement", "organism"]).copy()
    if plot_data.empty:
        return None
    within_mask = plot_data["within_assay_mZ_score"].fillna(False).astype(bool)
    across_mask = plot_data["across_assay_mZ_score"].fillna(False).astype(bool)
    plot_data["outlier_class"] = np.select(
        [across_mask, within_mask],
        ["Across Assay Outlier", "Within Assay Outlier"],
        default="Inlier",
    )
    plot_path = Path(output_dir) / DEFAULT_CONTROL_POOL_OUTLIER_CLASSES_PLOT_NAME
    renderer.submit(render_control_pool_outlier_classes_figure, plot_data, plot_path)
    emit_message("info", "Control pool outlier classes plot saved",
                 lines=format_detail("Path", style_text(str(plot_path), TEXT_COLOURS["accent"])))
    return plot_path


def render_control_pool_inlier_measurement_figure(plot_data, plot_path, label_size=12):
    import matplotlib.pyplot as plt
    controls = ["Sterility Control", "Growth Control"]
    control_colours = {"Sterility Control": "#DDA94B", "Growth Control": "#1E4174"}
    x_positions = {"Sterility Control": 0, "Growth Control": 1}
    unique_organisms = sorted(plot_data["organism"].dropna().unique())
    fig, subplot_axes = create_organism_grid(len(unique_organisms), 3, 5.2, 3.6, sharey=True)
    rng = np.random.default_rng(42)

    for axis, organism_name in zip(subplot_axes, unique_organisms):
        organism_data = plot_data[plot_data["organism"].eq(organism_name)]
        for control_name in controls:
            measurement = organism_data.loc[organism_data["treatment"].eq(control_name), "measurement"].dropna()
            if measurement.empty:
                continue
            median_measurement = measurement.median()
            x_values = x_positions[control_name] + rng.uniform(-0.12, 0.12, len(measurement))
            axis.scatter(x_values, measurement, s=14, alpha=0.55,
                         color=control_colours[control_name], edgecolor="none",
                         label=control_name)
            axis.hlines(median_measurement, x_positions[control_name] - 0.22,
                        x_positions[control_name] + 0.22,
                        color="black", linewidth=1.6)
            axis.text(x_positions[control_name] + 0.25, median_measurement,
                      f"N={len(measurement)}", ha="left", va="center",
                      fontsize=10, color="black")
        axis.set_title(organism_name, fontsize=11, fontweight="bold")
        axis.set_xlim(-0.5, 1.65)
        axis.set_xticks([0, 1])
        axis.set_xticklabels(["Sterility\nControl", "Growth\nControl"])
        axis.set_ylabel("Measurement", fontweight="bold", fontsize=label_size)
        axis.grid(True, axis="y", alpha=0.25)
        axis.grid(False, axis="x")
        for spine in axis.spines.values():
            spine.set_linewidth(1.1)
            spine.set_color("#333333")

    for axis in subplot_axes[len(unique_organisms):]:
        axis.remove()

    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=control_colours[control_name],
                   label=control_name, markersize=7)
        for control_name in controls
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, PLOT_LEGEND_Y),
               ncol=2, frameon=False)
    fig.suptitle("Inlier Control Measurement by Organism", fontsize=15, fontweight="bold", y=PLOT_TITLE_Y)
    figure_width, figure_height = fig.get_size_inches()
    fig.subplots_adjust(top=1 - PLOT_ORGANISM_TOP_OFFSET / figure_height,
                        bottom=0.58 / figure_height,
                        left=0.62 / figure_width,
                        right=1 - 0.05 / figure_width,
                        wspace=0.22, hspace=0.30)
    save_plot_figure(fig, plot_path)
    plt.close(fig)


def plot_control_pool_inlier_measurement(control_pool, output_dir, label_size=12, renderer=None):
    if control_pool.empty:
        return None
    if renderer is None:
        renderer = BackgroundRenderer()
    controls = ["Sterility Control", "Growth Control"]
    within_mask = control_pool["within_assay_mZ_score"].fillna(False).astype(bool)
    across_mask = control_pool["across_assay_mZ_score"].fillna(False).astype(bool)
    plot_data = control_pool.loc[
        control_pool["treatment"].isin(controls)
        & (~within_mask)
        & (~across_mask)
    ].copy()
    plot_data["measurement"] = pd.to_numeric(plot_data["measurement"], errors="coerce")
    plot_data = plot_data.dropna(subset=["measurement", "organism"]).copy()
    if plot_data.empty:
        return None
    plot_path = Path(output_dir) / DEFAULT_CONTROL_POOL_INLIER_MEASUREMENT_PLOT_NAME
    renderer.submit(render_control_pool_inlier_measurement_figure, plot_data, plot_path)
    emit_message("info", "Control pool inlier measurement plot saved",
                 lines=format_detail("Path", style_text(str(plot_path), TEXT_COLOURS["accent"])))
    return plot_path


def impute_missing_control_stats_by_assay(assays_missing_controls, control_stats_by_assay, control_stats_by_organism):
    imputed_control_stats_by_assay = (
        pd.DataFrame([(*assay_key, absent_control)
                      for assay_key, absent_controls in assays_missing_controls.items()
                      for absent_control in absent_controls],
                     columns=ASSAY_KEY_COLUMNS + ["treatment"]).merge(
                         control_stats_by_organism,
                         on=["organism", "treatment"], how="left"))
    return (
        pd.concat([control_stats_by_assay.assign(imputed=False),
                   imputed_control_stats_by_assay.assign(imputed=True)], ignore_index=True)
        .sort_values(["organism", "assay_id", "treatment"])
        .reset_index(drop=True))


def impute_outlier_control_stats_by_assay(control_stats_by_assay_imputed, control_outliers, control_stats_by_organism):
    if control_outliers.empty:
        return control_stats_by_assay_imputed

    flagged_controls = control_outliers[["assay_id", "organism", "treatment"]].drop_duplicates()
    imputed_outlier_control_stats_by_assay = flagged_controls.merge(
        control_stats_by_organism,
        on=["organism", "treatment"],
        how="left",
    ).assign(imputed=True)
    control_stats_without_outliers = (
        control_stats_by_assay_imputed.merge(
            flagged_controls[ASSAY_KEY_COLUMNS + ["treatment"]].assign(replace_control=True),
            on=ASSAY_KEY_COLUMNS + ["treatment"],
            how="left",
        )
    )
    control_stats_without_outliers = (
        control_stats_without_outliers[control_stats_without_outliers["replace_control"].isna()]
        .drop(columns="replace_control")
    )
    return (
        pd.concat([control_stats_without_outliers, imputed_outlier_control_stats_by_assay], ignore_index=True)
        .sort_values(["organism", "assay_id", "treatment"])
        .reset_index(drop=True)
    )


def validate_inhibition_inputs(data_with_if):
    invalid_control_rows = (
        ~np.isfinite(pd.to_numeric(data_with_if["GC_median"], errors="coerce"))
        | ~np.isfinite(pd.to_numeric(data_with_if["SC_median"], errors="coerce"))
        | ~np.isfinite(pd.to_numeric(data_with_if["dynamic_range"], errors="coerce"))
        | data_with_if["dynamic_range"].le(0)
    )
    if invalid_control_rows.any():
        assay_key = assay_keys(data_with_if.loc[invalid_control_rows])[0]
        raise ValueError(f"Invalid assay controls: finite GC/SC and positive dynamic_range required ({assay_label(assay_key)}).")


def add_inhibition_fractions(data_valid_organisms, control_stats_by_assay_imputed, measurement_columns):
    control_stats_by_assay_wide = control_stats_by_assay_imputed.pivot(index=ASSAY_KEY_COLUMNS, columns="treatment", values="median").reset_index()
    control_stats_by_assay_wide = control_stats_by_assay_wide.rename(
        columns={"Growth Control": "GC_median", "Sterility Control": "SC_median"}
    )
    control_imputation_flags = (control_stats_by_assay_imputed.pivot(index=ASSAY_KEY_COLUMNS, columns="treatment", values="imputed").reset_index().rename(
        columns={"Growth Control": "GC_imputed", "Sterility Control": "SC_imputed"}))
    for column in ["GC_imputed", "SC_imputed"]:
        if column not in control_imputation_flags.columns:
            control_imputation_flags[column] = False
    control_imputation_flags[["GC_imputed", "SC_imputed"]] = control_imputation_flags[["GC_imputed", "SC_imputed"]].fillna(False).astype(bool)
    data_with_if = (data_valid_organisms.copy()
                      .merge(control_stats_by_assay_wide, on=ASSAY_KEY_COLUMNS, how="left")
                      .merge(control_imputation_flags[ASSAY_KEY_COLUMNS + ["GC_imputed", "SC_imputed"]], on=ASSAY_KEY_COLUMNS, how="left"))
    data_with_if["GC_imputed"] = data_with_if["GC_imputed"].fillna(False).astype(bool)
    data_with_if["SC_imputed"] = data_with_if["SC_imputed"].fillna(False).astype(bool)
    growth_control = data_with_if["GC_median"]
    sterility_control = data_with_if["SC_median"]
    dynamic_range = growth_control - sterility_control
    data_with_if["dynamic_range"] = dynamic_range
    denom = dynamic_range.where(dynamic_range > 0, np.nan)

    measurements = data_with_if[measurement_columns].apply(pd.to_numeric, errors="coerce")
    raw_inhibitions = measurements.rsub(growth_control, axis=0).div(denom, axis=0)
    raw_inhibition_columns = [f"RI{column[1:]}" for column in measurement_columns]
    inhibition_columns = [f"I{column[1:]}" for column in measurement_columns]
    raw_inhibitions.columns = raw_inhibition_columns
    inhibitions = raw_inhibitions.clip(0, 1)
    inhibitions.columns = inhibition_columns
    data_with_if = pd.concat([data_with_if, raw_inhibitions, inhibitions], axis=1)

    validate_inhibition_inputs(data_with_if)
    provenance_columns = ["GC_imputed", "SC_imputed", "dynamic_range"]
    excluded_from_leading = set(measurement_columns + raw_inhibition_columns + inhibition_columns
                                + ["GC_median", "SC_median"] + provenance_columns)
    non_measurement_columns = [column for column in data_with_if.columns if column not in excluded_from_leading]
    ordered_columns = (non_measurement_columns + provenance_columns
                       + measurement_columns + raw_inhibition_columns
                       + inhibition_columns)
    return data_with_if[ordered_columns], inhibition_columns, raw_inhibition_columns


def save_control_pool(data_with_if, measurement_columns, control_stats_by_assay_checked,
                     tables_dir, plots_dir, save_plots=False, renderer=None):
    id_columns = ["assay_id", "data_id", "treatment_id", "treatment", "organism"]
    control_data = data_with_if[data_with_if["treatment"].isin(CONTROL_TREATMENT_NAMES)]
    control_pool = pd.concat(
        [
            control_data.loc[control_data[column].notna(), id_columns + [column, f"RI{column[1:]}", f"I{column[1:]}"]]
            .assign(concentration=column[1:])
            .rename(columns={column: "measurement", f"RI{column[1:]}": "RI", f"I{column[1:]}": "I"})
            for column in measurement_columns
            if control_data[column].notna().any()
        ],
        ignore_index=True,
    )
    measurement = pd.to_numeric(control_pool["measurement"], errors="coerce")
    control_pool = control_pool[measurement.notna()].copy()
    control_pool["measurement"] = measurement.loc[control_pool.index]
    control_pool = add_within_assay_outlier_flags(control_pool, "measurement")
    across_flags = control_stats_by_assay_checked[["assay_id", "organism", "treatment", "is_extreme_outlier"]].rename(
        columns={"is_extreme_outlier": "across_assay_mZ_score"}
    )
    control_pool = control_pool.merge(across_flags, on=["assay_id", "organism", "treatment"], how="left")
    control_pool["across_assay_mZ_score"] = control_pool["across_assay_mZ_score"].fillna(False).astype(bool)
    control_pool = control_pool[
        id_columns
        + ["concentration", "measurement", "RI", "I", "within_assay_mZ_score", "across_assay_mZ_score"]
    ]
    output_path = Path(tables_dir) / CONTROL_POOL_OUTPUT_NAME
    control_pool.to_csv(output_path, index=False)
    if save_plots:
        plot_control_pool_outlier_classes(control_pool, plots_dir, renderer=renderer)
        plot_control_pool_inlier_measurement(control_pool, plots_dir, renderer=renderer)
    return output_path


def report_missing_control_handling(assays_missing_controls, missing_control_strategy):
    if missing_control_strategy == "impute":
        if assays_missing_controls:
            emit_message("info", "Missing-control handling: impute",
                         lines=(f"Assays missing assay-level controls will use organism-level control {CONTROL_SUMMARY_MODE} for the missing GC and/or SC"))
        else:
            emit_message("info", "Missing-control handling: impute", lines="No assays required imputation.")
        return set()
    excluded_assays = set(assays_missing_controls)
    if excluded_assays:
        emit_message("warn", "Missing-control handling: exclude",
                     lines="The following assays were excluded from the analysis.",
                     items=[f"{assay_label(assay_key)}: missing {' & '.join(assays_missing_controls[assay_key])}"
                            for assay_key in sorted_assay_keys(excluded_assays)])
    else:
        emit_message("info", "Missing-control handling: exclude",
                     lines="No assays were excluded because no assay-level controls were missing.")
    return excluded_assays


def report_outlier_handling(control_outliers, outlier_strategy):
    if outlier_strategy == "report":
        if control_outliers.empty:
            emit_message("info", "Across-assay outlier handling: report", lines="No flagged GC/SC outlier assays were found.")
        else:
            emit_message("info", "Across-assay outlier handling: report",
                         lines="Flagged GC/SC outlier controls were reported; assay-level controls were retained.")
        return set()
    if outlier_strategy == "impute":
        if control_outliers.empty:
            emit_message("info", "Across-assay outlier handling: impute", lines="No flagged GC/SC outlier assays were found.")
        else:
            emit_message(
                "info",
                "Across-assay outlier handling: impute",
                lines=f"Flagged GC/SC outlier controls will use organism-level control {CONTROL_SUMMARY_MODE} in the final analysis.",
            )
        return set()
    if control_outliers.empty:
        emit_message("info", "Across-assay outlier handling: exclude", lines="No flagged GC/SC outlier assays were found, so none were excluded.")
        return set()
    outlier_controls_by_assay = (control_outliers.groupby(ASSAY_KEY_COLUMNS)["treatment"].apply(lambda x: sorted(set(x))).to_dict())
    excluded_assays = set(outlier_controls_by_assay)
    emit_message("warn", "Across-assay outlier handling: exclude", lines="The following assays were excluded from the analysis.",
                 items=[f"{assay_label(assay_key)}: flagged {' & '.join(outlier_controls_by_assay[assay_key])}"
                        for assay_key in sorted_assay_keys(excluded_assays)])
    return excluded_assays


def filter_analysis_inputs(data_valid_organisms, control_stats_by_assay, control_stats_by_assay_imputed,
                           missing_control_strategy, assays_missing_controls, outlier_strategy, control_outliers):
    missing_control_exclusions = report_missing_control_handling(assays_missing_controls, missing_control_strategy)
    outlier_exclusions = report_outlier_handling(control_outliers, outlier_strategy)
    excluded_assays = missing_control_exclusions | outlier_exclusions
    data_for_analysis = data_valid_organisms[~assay_key_series(data_valid_organisms).isin(excluded_assays)].copy()
    if missing_control_strategy == "impute" or outlier_strategy == "impute":
        control_stats_for_analysis = control_stats_by_assay_imputed[
            ~assay_key_series(control_stats_by_assay_imputed).isin(excluded_assays)].copy()
    else:
        control_stats_for_analysis = control_stats_by_assay[
            ~assay_key_series(control_stats_by_assay).isin(excluded_assays)].assign(imputed=False)
    emit_message("info", "Assay selection summary", lines=[format_detail("Excluded assays", style_text(f"{len(excluded_assays):,}", "1")),
                                                           format_detail("Remaining assays", style_text(f"{len(assay_keys(data_for_analysis)):,}", "1"))])
    return data_for_analysis, control_stats_for_analysis


def select_within_assay_summary_inputs(within_assay_outlier_controls, within_assay_qc,
                                       assays_missing_controls_raw):
    if within_assay_outlier_controls == "exclude":
        emit_message("info", "Within-assay outlier handling: exclude",
                     lines="Flagged control measurements were excluded from assay- and organism-level control summaries.")
        assays_missing_controls = combine_missing_control_maps(
            assays_missing_controls_raw,
            find_assays_missing_controls_after_within_assay(within_assay_qc.stats_comparison),
        )
        return within_assay_qc.readings_cleaned, within_assay_qc.stats_cleaned, assays_missing_controls

    emit_message("info", "Within-assay outlier handling: keep",
                 lines="Flagged control measurements were retained in assay- and organism-level control summaries.")
    return within_assay_qc.readings_checked, within_assay_qc.stats_raw, assays_missing_controls_raw


class ControlQCTables(NamedTuple):
    readings_checked: pd.DataFrame
    readings_for_summary: pd.DataFrame
    stats_by_assay: pd.DataFrame
    stats_comparison: pd.DataFrame
    assays_missing_controls: dict


def compute_control_qc_tables(control_readings_by_assay, controls, control_stats_by_assay_raw,
                             assays_missing_controls_raw, within_assay_outlier_controls):
    within_assay_qc = perform_within_assay_outlier_check(
        control_readings_by_assay,
        controls,
        control_stats_by_assay_raw=control_stats_by_assay_raw,
    )
    control_readings_by_assay_for_summary, control_stats_by_assay, assays_missing_controls = select_within_assay_summary_inputs(
        within_assay_outlier_controls,
        within_assay_qc,
        assays_missing_controls_raw,
    )
    return ControlQCTables(within_assay_qc.readings_checked, control_readings_by_assay_for_summary,
                          control_stats_by_assay, within_assay_qc.stats_comparison,
                          assays_missing_controls)


def generate_normalization_plots(should_plot, progress, stages, total_stages, plot_step_count,
                           control_stats_by_assay_checked, control_stats_by_organism,
                           control_stats_by_organism_all, control_readings_by_assay_checked,
                           control_stats_by_assay_comparison, controls,
                           within_assay_outlier_controls, plot_output_path, renderer=None):
    if should_plot:
        generated_plot_count = 0

        def advance_plot_progress():
            nonlocal generated_plot_count
            generated_plot_count += 1
            set_progress_stage(progress, 11, total_stages, f"{stages[10]} ({generated_plot_count}/{plot_step_count})")
            progress.update(1)

        set_progress_stage(progress, 11, total_stages, f"{stages[10]} (0/{plot_step_count})")
        within_assay_plot_paths = submit_within_assay_control_plots(
            control_readings_by_assay_checked,
            control_stats_by_assay_comparison,
            controls,
            within_assay_outlier_controls=within_assay_outlier_controls,
            show_plot=False,
            plot_output=plot_output_path,
            renderer=renderer)
        plot_across_assay_control_outliers(control_stats_by_assay_checked, control_stats_by_organism,
                                          control_stats_by_organism_all, controls,
                                          show_plot=True,
                                          plot_output=plot_output_path)
        advance_plot_progress()
        for _ in within_assay_plot_paths:
            advance_plot_progress()
        report_within_assay_control_plots(within_assay_plot_paths)
    else:
        emit_message("info", "Control plots skipped.")
        progress.update(1)


def save_normalization_outputs(data_with_if, measurement_columns, control_stats_by_assay_checked,
                         output_csv_path, plots_dir, should_plot, renderer=None):
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    data_with_if.to_csv(output_csv_path, index=False)
    save_control_pool(data_with_if, measurement_columns, control_stats_by_assay_checked,
                     output_csv_path.parent, plots_dir, save_plots=should_plot, renderer=renderer)


def run_normalization(args, run_output_dir, output_csv_path, plots_dir):
    emit_run_configuration(args, output_csv_path)
    stages = ["Load data", "Create output directory", "Filter organisms",
              "Find incomplete assays", "Detect measurements", "Compute assay controls",
              "Check within-assay outliers", "Compute organism controls", "Check across-assay outliers",
              "Compute imputation stats", "Generate control plots", "Apply assay filters",
              "Calculate inhibition", "Generate inhibition plot", "Save output"]
    total_stages = len(stages)
    progress = create_progress_bar(total_stages, format_stage_label(1, total_stages, stages[0]))
    data_with_if = None
    plot_step_count = 1
    render_executor = None
    renderer = BackgroundRenderer(None)
    try:
        data = pd.read_csv(args.input_csv)
        validate_input_columns(data)
        data = filter_selected_organisms(data, args.organism)
        progress.update(1)

        set_progress_stage(progress, 2, total_stages, stages[1])
        should_plot, plot_output_path = parse_plot_option(args.plot, plots_dir)
        if should_plot and min(MAX_RENDER_WORKERS, os.cpu_count() or 1) > 1:
            render_executor = ProcessPoolExecutor(max_workers=min(MAX_RENDER_WORKERS, os.cpu_count() or 1))
            renderer = BackgroundRenderer(render_executor)
        emit_message("info", "Output directory ready", lines=format_detail("Path", style_text(str(run_output_dir), TEXT_COLOURS["accent"])))
        progress.update(1)

        set_progress_stage(progress, 3, total_stages, stages[2])
        controls = CONTROL_TREATMENT_NAMES
        data_valid_organisms = filter_valid_organisms(data)
        if data_valid_organisms.empty:
            raise ValueError("No valid organisms remain after filtering for Growth Control and Sterility Control.")
        plot_step_count = 1 + data_valid_organisms["organism"].nunique() if should_plot else 1
        set_progress_total(progress, total_stages - 1 + plot_step_count)
        progress.update(1)

        set_progress_stage(progress, 4, total_stages, stages[3])
        assays_missing_controls_raw = find_assays_missing_controls(data_valid_organisms, controls)
        progress.update(1)

        set_progress_stage(progress, 5, total_stages, stages[4])
        measurement_columns = get_measurement_columns(data_valid_organisms, controls)
        if not measurement_columns:
            raise ValueError("No measurement columns matching the pattern A<number> were found in the input data.")
        progress.update(1)

        set_progress_stage(progress, 6, total_stages, stages[5])
        control_readings_by_assay, control_stats_by_assay_raw = calculate_control_stats_by_assay(
            data_valid_organisms,
            controls,
            measurement_columns,
        )
        progress.update(1)

        set_progress_stage(progress, 7, total_stages, stages[6])
        (
            control_readings_by_assay_checked,
            control_readings_by_assay_for_summary,
            control_stats_by_assay,
            control_stats_by_assay_comparison,
            assays_missing_controls,
        ) = compute_control_qc_tables(
            control_readings_by_assay,
            controls,
            control_stats_by_assay_raw,
            assays_missing_controls_raw,
            args.within_assay_outlier_controls,
        )
        progress.update(1)

        set_progress_stage(progress, 8, total_stages, stages[7])
        control_stats_by_organism_all = summarize_control_readings_by_organism(control_readings_by_assay_for_summary)
        progress.update(1)

        set_progress_stage(progress, 9, total_stages, stages[8])
        control_stats_by_assay_checked, control_outliers = perform_across_assay_outlier_check(
            control_stats_by_assay,
            controls,
        )
        progress.update(1)

        set_progress_stage(progress, 10, total_stages, stages[9])
        control_stats_by_organism = calculate_control_stats_by_organism_for_imputation(
            control_readings_by_assay_for_summary,
            controls,
            control_outliers,
        )
        control_stats_by_assay_imputed = impute_missing_control_stats_by_assay(
            assays_missing_controls,
            control_stats_by_assay,
            control_stats_by_organism)
        if args.outlier_controls == "impute":
            control_stats_by_assay_imputed = impute_outlier_control_stats_by_assay(
                control_stats_by_assay_imputed,
                control_outliers,
                control_stats_by_organism,
            )
        progress.update(1)

        set_progress_stage(progress, 11, total_stages, stages[10])
        generate_normalization_plots(
            should_plot, progress, stages, total_stages, plot_step_count,
            control_stats_by_assay_checked, control_stats_by_organism,
            control_stats_by_organism_all, control_readings_by_assay_checked,
            control_stats_by_assay_comparison, controls,
            args.within_assay_outlier_controls, plot_output_path,
            renderer=renderer,
        )

        set_progress_stage(progress, 12, total_stages, stages[11])
        data_for_analysis, control_stats_for_analysis = filter_analysis_inputs(
            data_valid_organisms, control_stats_by_assay,
            control_stats_by_assay_imputed, args.missing_controls,
            assays_missing_controls, args.outlier_controls, control_outliers)
        if data_for_analysis.empty:
            raise ValueError("No data remain after applying the selected assay exclusion rules.")
        progress.update(1)

        set_progress_stage(progress, 13, total_stages, stages[12])
        data_with_if, inhibition_columns, raw_inhibition_columns = add_inhibition_fractions(data_for_analysis,
                                                                                        control_stats_for_analysis,
                                                                                        measurement_columns)
        progress.update(1)

        set_progress_stage(progress, 14, total_stages, stages[13])
        if should_plot:
            plot_raw_inhibition_fractions(data_with_if,
                                        raw_inhibition_columns,
                                        controls,
                                        show_plot=False,
                                        plot_output=plot_output_path,
                                        renderer=renderer)
        else:
            emit_message("info", "Raw inhibition plot skipped.")
        progress.update(1)

        set_progress_stage(progress, 15, total_stages, stages[14])
        save_normalization_outputs(data_with_if, measurement_columns, control_stats_by_assay_checked,
                             output_csv_path, plots_dir, should_plot, renderer=renderer)
        progress.update(1)
        renderer.wait()
    finally:
        progress.close()
        if render_executor is not None:
            render_executor.shutdown()
    emit_message(
        "success",
        "Normalization complete",
        lines=[
            format_detail("Output", style_text(str(output_csv_path), TEXT_COLOURS["accent"])),
            format_detail("Rows", style_text(f"{len(data_with_if):,}", "1")),
            format_detail("Unique assays", style_text(f"{len(assay_keys(data_with_if)):,}", "1"))])


def main():
    args = parse_args()
    if not (isinstance(args.plot, str) and args.plot.lower() in PLOT_DISABLE_TOKENS):
        import matplotlib.pyplot
    run_output_dir = create_cwd_dated_run_directory(args.output_dir, DEFAULT_RUN_DIR_PREFIX,
                                                    DEFAULT_OUTPUT_DIR_NAME)
    directories = prepare_output_directories(run_output_dir, ["logs", "plots", "tables"])
    output_csv_path = resolve_output_paths(args.input_csv, args.output_csv, directories["tables"])

    with RunLogger(directories["logs"] / "run_log.txt"):
        write_run_parameters(args, directories["logs"], "normalization.py", TOOL_VERSION, {"output_dir": str(run_output_dir), "output_csv": str(output_csv_path)})
        try:
            run_normalization(args, run_output_dir, output_csv_path, directories["plots"])
        except Exception as error:
            emit_message("error", "Normalization failed", lines=format_detail("Reason", str(error)))
            traceback.print_exc()
            raise


if __name__ == "__main__":
    main()
