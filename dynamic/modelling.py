#!/usr/bin/env python3

import argparse
import math
import os
import signal
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import NamedTuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import expit, gammaln

from dynamic.calibration import (CONTROL_INHIBITION_SOURCES, ensure_dose_layout_columns,
                                 load_normalized_inhibition_data)
from dynamic.classification import (LABEL_VARIANT, group_treatment_replicates, build_analysis_treatment_curves,
                                    load_exact_layout_rows)
from dynamic.utils import (LogoArgumentParser, RunLogger, TEXT_COLOURS, create_dated_run_directory,
                           create_progress_bar, emit_message, format_detail, format_stage_label,
                           get_run_logo_lines, open_probability, parse_csv_list,
                           positive_float, positive_int, prepare_output_directories, require_columns,
                           set_progress_stage, style_text, write_run_parameters)


from dynamic import __version__ as TOOL_VERSION
DEFAULT_RUN_DIR_PREFIX = "modelling_results"
DEFAULT_FIT_LABEL = LABEL_VARIANT
DEFAULT_FIT_MODEL = "2pl"
DEFAULT_MIC_INHIBITION_THRESHOLD = 0.95
DEFAULT_BOOTSTRAP_SEED = 31
DEFAULT_N_BOOTSTRAPS = 500
DEFAULT_MAX_FIT_WORKERS = 30
DEFAULT_FIT_TIMEOUT_SECONDS = 900
DEFAULT_MIN_SLOPE = 0.01
DEFAULT_MAX_SLOPE = 30.0
LOGIT_EPSILON = 1e-3
TRANSITION_LOWER_PERCENTILE = 99
TRANSITION_UPPER_PERCENTILE = 1
START_SLOPE = np.array([0.25, 0.5, 1.0, 2.0, 5.0, 10.0])
DEFAULT_FIT_PLATEAU_BOUNDS = (0.0, 1.0)


class ModelSpec(NamedTuple):
    name: str
    prefix: str
    n_plateau_params: int  # 0 (2pl), 1 (3pl), 2 (4pl)

    @property
    def plateau_suffixes(self):
        if self.n_plateau_params == 0:
            return []
        suffixes = ["max_inhibition"]
        if self.n_plateau_params >= 2:
            suffixes.append("min_inhibition")
        return suffixes + ["effect_size"]


MODEL_SPECS = {
    "2pl": ModelSpec("2pl", "two_pl_fit", 0),
    "3pl": ModelSpec("3pl", "three_pl_fit", 1),
    "4pl": ModelSpec("4pl", "four_pl_fit", 2),
}
BOOTSTRAP_SUMMARY_COLUMNS = ["row_index", "n_fit_bootstraps_used"]
DYNAMIC_RESULTS_OUTPUT_NAME = "dynamic_results.csv"
DYNAMIC_RESULTS_COLUMNS = [
    "organism", "treatment", "dose_layout", "sequence", "mic_annotation", "mbc_annotation",
    "assay_id", "data_id", "replicate_index", "n_replicates",
    "concentration", "inhibition_fraction",
    "GC_imputed", "SC_imputed",
    "replicate_curve_outlier", "dose_response_label",
    "two_pl_signal_score", "two_pl_signal_score_p", "two_pl_signal_score_q",
    "fit_status",
    "fit_slope", "fit_slope_ci_low", "fit_slope_ci_high",
    "fit_midpoint", "fit_midpoint_ci_low", "fit_midpoint_ci_high",
    "continuous_mic", "continuous_mic_ci_low", "continuous_mic_ci_high",
    "discrete_mic", "discrete_mic_ci_low", "discrete_mic_ci_high",
    "mic_qc", "midpoint_qc", "slope_qc",
]
DYNAMIC_RESULTS_FIT_COLUMNS = [
    "fit_status",
    "fit_slope", "fit_slope_ci_low", "fit_slope_ci_high",
    "fit_midpoint", "fit_midpoint_ci_low", "fit_midpoint_ci_high",
    "continuous_mic", "continuous_mic_ci_low", "continuous_mic_ci_high",
    "discrete_mic", "discrete_mic_ci_low", "discrete_mic_ci_high",
    "mic_qc", "midpoint_qc", "slope_qc",
]

MISSING_INHIBITION_COLUMNS_MESSAGE = (
    "The normalized data are missing inhibition column(s) required by the classification output: "
    "{columns}. Use the same normalized file used for classification, "
    "or rerun classification for this dataset."
)
EMPTY_OBSERVED_ROWS_MESSAGE = "No exact-layout curves found for {organism}."


class FitContext(dict):

    def configure(self, values):
        self.clear()
        self.update(values)


FIT_CONTEXT = FitContext()


def _install_payload_fit_context(payload):
    payload_context = payload["fit_context"]
    if FIT_CONTEXT.get("_context_token") != payload_context["_context_token"]:
        FIT_CONTEXT.configure(payload_context)


def plateau_param_suffixes(fit_model):
    return MODEL_SPECS[fit_model].plateau_suffixes


def plateau_param_names(prefix, fit_model):
    return [f"{prefix}_{suffix}" for suffix in plateau_param_suffixes(fit_model)]


def clean_plateau_columns(fit_model):
    return [column for suffix in plateau_param_suffixes(fit_model)
            for column in (suffix, f"{suffix}_ci_low", f"{suffix}_ci_high")]


def fit_result_columns(prefix, fit_model):
    plateau_columns = [column for suffix in plateau_param_suffixes(fit_model)
                       for column in (f"{prefix}_{suffix}", f"{prefix}_{suffix}_ci_low", f"{prefix}_{suffix}_ci_high")]
    return [
        "organism", "treatment", "dose_layout", "dose_layout_full", "sequence",
        "n_fit_replicates", "fit_status", "n_fit_bootstraps",
        f"{prefix}_slope", f"{prefix}_slope_ci_low", f"{prefix}_slope_ci_high",
        f"{prefix}_midpoint", f"{prefix}_midpoint_ci_low", f"{prefix}_midpoint_ci_high",
        f"{prefix}_rss", f"{prefix}_rss_ci_low", f"{prefix}_rss_ci_high",
        *plateau_columns,
        "mic_inhibition_threshold",
        f"{prefix}_continuous_mic", f"{prefix}_continuous_mic_ci_low", f"{prefix}_continuous_mic_ci_high",
        f"{prefix}_discrete_mic", f"{prefix}_discrete_mic_ci_low", f"{prefix}_discrete_mic_ci_high",
        f"{prefix}_discrete_mic_relation", f"{prefix}_discrete_mic_value", f"{prefix}_discrete_mic_order",
        "mic_qc", "midpoint_qc", "slope_qc",
    ]


def fit_result_internal_columns(prefix, fit_model):
    return ["row_index", "treatment_index", *fit_result_columns(prefix, fit_model)]


def fit_result_column_renames(prefix, fit_model):
    renames = {}
    for suffix in ["slope", "slope_ci_low", "slope_ci_high", "midpoint", "midpoint_ci_low", "midpoint_ci_high", "rss", "rss_ci_low", "rss_ci_high"]:
        renames[f"{prefix}_{suffix}"] = f"fit_{suffix}"
    for suffix in ["continuous_mic", "continuous_mic_ci_low", "continuous_mic_ci_high", "discrete_mic", "discrete_mic_ci_low", "discrete_mic_ci_high", "discrete_mic_relation", "discrete_mic_value", "discrete_mic_order"]:
        renames[f"{prefix}_{suffix}"] = suffix
    for suffix in plateau_param_suffixes(fit_model):
        for column in (suffix, f"{suffix}_ci_low", f"{suffix}_ci_high"):
            renames[f"{prefix}_{column}"] = column
    return renames


def bootstrap_result_columns(prefix, fit_model):
    plateau_columns = [f"{prefix}_{suffix}" for suffix in plateau_param_suffixes(fit_model)]
    return [
        "row_index", "treatment_index", "organism", "treatment", "dose_layout", "dose_layout_full", "sequence",
        "bootstrap_index", "bootstrap_mode", "bootstrap_weight",
        f"{prefix}_slope", f"{prefix}_midpoint", f"{prefix}_rss",
        *plateau_columns,
        "mic_inhibition_threshold", f"{prefix}_continuous_mic",
        f"{prefix}_discrete_mic", f"{prefix}_discrete_mic_relation",
        f"{prefix}_discrete_mic_value", f"{prefix}_discrete_mic_order",
        "mic_qc", "midpoint_qc", "slope_qc",
    ]


def parse_args():
    parser = LogoArgumentParser(description="DynaMIC curve modelling and mic estimation", formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--normalized-data", required=True, help="CSV produced by normalization.py.")
    parser.add_argument("--classification-dir", required=True, help="Directory produced by classification.py.")
    parser.add_argument("--calibration-dir", required=True, help="Directory produced by calibration.py.")
    parser.add_argument("--output-dir", default=None, help="Directory for modelling outputs.")
    parser.add_argument("--organism", action="append", default=None, help="Organism to fit. Repeat to select multiple. Default: all classified organisms.")
    parser.add_argument("--fit-label", default=DEFAULT_FIT_LABEL, help="Dose-response label to fit. Default: dose-variant.")
    parser.add_argument("--fit-model", choices=list(MODEL_SPECS), default=DEFAULT_FIT_MODEL, help="Template model for curve fitting. Default: 2pl.")
    parser.add_argument("--min-slope", type=positive_float, default=DEFAULT_MIN_SLOPE, help="Minimum fitted slope. Default: 0.01.")
    parser.add_argument("--max-slope", type=positive_float, default=DEFAULT_MAX_SLOPE, help="Maximum fitted slope. Default: 30.")
    parser.add_argument("--mic-threshold", type=open_probability, default=DEFAULT_MIC_INHIBITION_THRESHOLD, help="Inhibition threshold for fitted mic. Default: 0.95.")
    parser.add_argument("--bootstrap", choices=["on", "off"], default="on", help="Turn bootstrap uncertainty on or off. Default: on.")
    parser.add_argument("--n-bootstraps", type=positive_int, default=DEFAULT_N_BOOTSTRAPS, help="Maximum bootstraps. Default: 500.")
    parser.add_argument("--max-workers", "--threads", dest="max_workers", type=positive_int, default=DEFAULT_MAX_FIT_WORKERS, help=f"Parallel fit processes. Default: {DEFAULT_MAX_FIT_WORKERS}.")
    parser.add_argument("--fit-timeout-seconds", type=positive_int, default=DEFAULT_FIT_TIMEOUT_SECONDS, help="Timeout per fitted curve. Default: 900.")
    args = parser.parse_args()
    if args.min_slope >= args.max_slope:
        parser.error("--min-slope must be less than --max-slope")
    return args


def load_manifest(manifest_path, selected_organisms=None):
    if not Path(manifest_path).exists():
        raise ValueError(f"Manifest was not found: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    require_columns(manifest.columns, ["organism", "organism_token"], "Manifest", manifest_path)
    manifest = ensure_dose_layout_columns(manifest)
    if selected_organisms:
        manifest = manifest[manifest["organism"].isin(selected_organisms)].copy()
    if manifest.empty:
        raise ValueError("No records matched the requested organism selection.")
    return manifest


def load_control_inhibition_measurements(calibration_dir, control_inhibition_path):
    with np.load(Path(calibration_dir) / control_inhibition_path) as data:
        available_keys = set(data.files)
        missing_keys = [key for key in CONTROL_INHIBITION_SOURCES if key not in available_keys]
        if missing_keys:
            raise ValueError(f"Control inhibition file is missing required array(s): {', '.join(missing_keys)}")
        return {key: data[key] for key in CONTROL_INHIBITION_SOURCES}


def load_classification_rows(classification_dir, classification_results_path):
    result = pd.read_csv(Path(classification_dir) / classification_results_path)
    require_columns(result.columns, ["organism", "treatment", "dose_response_label", "replicate_outlier_exclusion"], "Classification results", Path(classification_dir) / classification_results_path)
    return result


def add_internal_treatment_index(replicate_outliers, treatment_metadata):
    if replicate_outliers.empty or "treatment_index" in replicate_outliers.columns:
        return replicate_outliers
    require_columns(replicate_outliers.columns, ["treatment", "replicate_index", "replicate_curve_outlier"], "Replicate outlier table", "classification replicate outliers")
    treatment_to_index = dict(zip(treatment_metadata["treatment"], treatment_metadata["treatment_index"].astype(int)))
    replicate_outliers = replicate_outliers.copy()
    replicate_outliers["treatment_index"] = replicate_outliers["treatment"].map(treatment_to_index)
    if replicate_outliers["treatment_index"].isna().any():
        missing = sorted(replicate_outliers.loc[replicate_outliers["treatment_index"].isna(), "treatment"].astype(str).unique())
        raise ValueError(f"Replicate outlier table contains treatment(s) not found in the normalized data: {', '.join(missing[:5])}")
    replicate_outliers["treatment_index"] = replicate_outliers["treatment_index"].astype(int)
    return replicate_outliers


def load_observed_rows(normalized_data_path, organism, inhibition_columns, preloaded_data=None):
    return load_exact_layout_rows(
        normalized_data_path, organism, inhibition_columns,
        metadata_columns=["organism", "treatment", "data_id", "assay_id", "sequence", "mic", "mbc", "GC_imputed", "SC_imputed"],
        missing_columns_message=MISSING_INHIBITION_COLUMNS_MESSAGE,
        empty_message=EMPTY_OBSERVED_ROWS_MESSAGE,
        preloaded_data=preloaded_data,
    )


def detect_dilution_base(concentrations):
    values = np.sort(np.unique(np.asarray(concentrations, dtype=float)))
    ratios = values[1:] / values[:-1]
    ratios = ratios[np.isfinite(ratios) & (ratios > 1)]
    return float(np.median(ratios)) if ratios.size else 2.0


def dilution_step_logit_transform(value, dose_log_base):
    return float(np.log(value / (1.0 - value)) / np.log(dose_log_base))


def calculate_slope_resolution_fields(concentrations, control_inhibition_measurements):
    concentrations = np.asarray(concentrations, dtype=float)
    dose_log_base = detect_dilution_base(concentrations)
    log_concentrations = np.log(concentrations) / np.log(dose_log_base)
    growth_control_inhibition = control_inhibition_measurements["growth_control_inhibition"]
    sterility_control_inhibition = control_inhibition_measurements["sterility_control_inhibition"]
    lower_transition_if = float(np.nanpercentile(growth_control_inhibition, TRANSITION_LOWER_PERCENTILE))
    upper_transition_if = float(np.nanpercentile(sterility_control_inhibition, TRANSITION_UPPER_PERCENTILE))
    lower_transition_logit = dilution_step_logit_transform(np.clip(lower_transition_if, LOGIT_EPSILON, 1.0 - LOGIT_EPSILON), dose_log_base)
    upper_transition_logit = dilution_step_logit_transform(np.clip(upper_transition_if, LOGIT_EPSILON, 1.0 - LOGIT_EPSILON), dose_log_base)
    transition_constant = upper_transition_logit - lower_transition_logit
    dose_range_steps = float(log_concentrations.max() - log_concentrations.min())
    return {
        "dose_log_base": dose_log_base,
        "fit_log_concentrations": log_concentrations,
        "lower_transition_if": lower_transition_if,
        "upper_transition_if": upper_transition_if,
        "transition_constant_dilution_steps": transition_constant,
        "dose_range_steps": dose_range_steps,
        "minimum_measurable_slope": transition_constant / dose_range_steps,
        "maximum_measurable_slope": transition_constant,
    }


def build_base_start_matrix(fit_concentrations, dose_log_base):
    start_midpoints = np.array([fit_concentrations.min(), np.sqrt(fit_concentrations.min() * fit_concentrations.max()),
                                np.median(fit_concentrations), np.median(fit_concentrations),
                                fit_concentrations.max() / 2, fit_concentrations.max()])
    return np.column_stack([np.log(START_SLOPE), np.log(start_midpoints) / np.log(dose_log_base)])


def configure_fit_context(concentrations, slope_resolution, args, context_token=0):
    fit_concentrations = np.asarray(concentrations, dtype=float)
    midpoint_bounds = (fit_concentrations.min() / 4, fit_concentrations.max() * 4)
    dose_log_base = slope_resolution["dose_log_base"]
    fit_log_midpoint_bounds = np.log(midpoint_bounds) / np.log(dose_log_base)
    plateau_params = MODEL_SPECS[args.fit_model].n_plateau_params
    theta_bounds = (
        [np.log(args.min_slope), fit_log_midpoint_bounds[0]] + [DEFAULT_FIT_PLATEAU_BOUNDS[0]] * plateau_params,
        [np.log(args.max_slope), fit_log_midpoint_bounds[1]] + [DEFAULT_FIT_PLATEAU_BOUNDS[1]] * plateau_params,
    )
    base_starts = build_base_start_matrix(fit_concentrations, dose_log_base)
    FIT_CONTEXT.configure({
        "fit_concentrations": fit_concentrations,
        "fit_log_concentrations": slope_resolution["fit_log_concentrations"],
        "dose_log_base": dose_log_base,
        "scale_log": np.log(dose_log_base),
        "fit_model": args.fit_model,
        "fit_prefix": MODEL_SPECS[args.fit_model].prefix,
        "fit_plateau_params": plateau_params,
        "fit_mic_inhibition_threshold": float(args.mic_threshold),
        "n_fit_bootstraps": int(args.n_bootstraps),
        "fit_bootstrap_seed": int(DEFAULT_BOOTSTRAP_SEED),
        "fit_curve_timeout_seconds": int(args.fit_timeout_seconds),
        "fit_theta_bounds": theta_bounds,
        "fit_base_starts": base_starts,
        "fit_prepared_base_starts": (
            np.unique(np.round(np.clip(base_starts, theta_bounds[0], theta_bounds[1]), 10), axis=0)
            if plateau_params == 0 else None
        ),
        "minimum_measurable_slope": slope_resolution["minimum_measurable_slope"],
        "maximum_measurable_slope": slope_resolution["maximum_measurable_slope"],
        "run_bootstrap": args.bootstrap == "on",
        "_context_token": context_token,
    })


def compute_effect_size(slope, log_midpoint, front, back):
    if not (np.isfinite(slope) and np.isfinite(log_midpoint) and np.isfinite(front) and np.isfinite(back)):
        return np.nan
    log_concentrations = FIT_CONTEXT["fit_log_concentrations"]
    scale = FIT_CONTEXT["scale_log"] * slope
    return float((front - back) * (expit(scale * (log_concentrations.max() - log_midpoint)) - expit(scale * (log_concentrations.min() - log_midpoint))))


def plateau_fields(slope, log_midpoint, front, back):
    suffixes = plateau_param_suffixes(FIT_CONTEXT["fit_model"])
    if not suffixes:
        return {}
    prefix = FIT_CONTEXT["fit_prefix"]
    values = {"max_inhibition": float(front), "min_inhibition": float(back), "effect_size": compute_effect_size(slope, log_midpoint, front, back)}
    return {f"{prefix}_{suffix}": values[suffix] for suffix in suffixes}


def resolve_continuous_mic(slope, log_midpoint, front, back, inhibition_threshold):
    if not (np.isfinite(slope) and np.isfinite(log_midpoint) and np.isfinite(front) and np.isfinite(back) and slope > 0 and 0 < inhibition_threshold < 1 and front > back):
        return np.nan, "invalid"
    if back >= inhibition_threshold:
        return np.nan, "below"
    if front <= inhibition_threshold:
        return np.nan, "above"
    sigma_target = (inhibition_threshold - back) / (front - back)
    continuous_mic = float(FIT_CONTEXT["dose_log_base"] ** (log_midpoint + dilution_step_logit_transform(sigma_target, FIT_CONTEXT["dose_log_base"]) / slope))
    if not np.isfinite(continuous_mic):
        return np.nan, "invalid"
    return continuous_mic, "mic"


def calculate_discrete_mic(continuous_mic, censoring, concentrations):
    prefix = FIT_CONTEXT["fit_prefix"]
    concentrations = np.sort(np.asarray(concentrations, dtype=float))
    if censoring == "invalid":
        return {f"{prefix}_discrete_mic": "NA", f"{prefix}_discrete_mic_relation": "NA", f"{prefix}_discrete_mic_value": np.nan, f"{prefix}_discrete_mic_order": np.nan}
    if censoring == "below":
        return {f"{prefix}_discrete_mic": f"<{concentrations[0]:g}", f"{prefix}_discrete_mic_relation": "<", f"{prefix}_discrete_mic_value": float(concentrations[0]), f"{prefix}_discrete_mic_order": -1}
    if censoring == "above" or continuous_mic > concentrations[-1]:
        return {f"{prefix}_discrete_mic": f">{concentrations[-1]:g}", f"{prefix}_discrete_mic_relation": ">", f"{prefix}_discrete_mic_value": float(concentrations[-1]), f"{prefix}_discrete_mic_order": int(len(concentrations))}
    order = int(np.clip(np.searchsorted(concentrations, continuous_mic, side="left"), 0, len(concentrations) - 1))
    return {f"{prefix}_discrete_mic": f"{concentrations[order]:g}", f"{prefix}_discrete_mic_relation": "=", f"{prefix}_discrete_mic_value": float(concentrations[order]), f"{prefix}_discrete_mic_order": order}


def calculate_qc_fields(slope, log_midpoint, mic_relation):
    log_concentrations = FIT_CONTEXT["fit_log_concentrations"]
    return {
        "mic_qc": "measured" if mic_relation == "=" else "range_limited" if mic_relation in (">", "<") else "NA",
        "midpoint_qc": "measured" if np.isfinite(log_midpoint) and log_concentrations.min() <= log_midpoint <= log_concentrations.max() else "range_limited",
        "slope_qc": "measured" if np.isfinite(slope) and FIT_CONTEXT["minimum_measurable_slope"] <= slope <= FIT_CONTEXT["maximum_measurable_slope"] else "resolution_limited",
    }


def calculate_mic_fields(slope, log_midpoint, front, back, concentrations, inhibition_threshold):
    prefix = FIT_CONTEXT["fit_prefix"]
    continuous_mic, censoring = resolve_continuous_mic(slope, log_midpoint, front, back, inhibition_threshold)
    mic_fields = {"mic_inhibition_threshold": float(inhibition_threshold), f"{prefix}_continuous_mic": continuous_mic, **calculate_discrete_mic(continuous_mic, censoring, concentrations)}
    return {**mic_fields, **calculate_qc_fields(slope, log_midpoint, mic_fields[f"{prefix}_discrete_mic_relation"])}


def prepare_fit_starts(y, preferred_start=None):
    theta_lower, theta_upper = FIT_CONTEXT["fit_theta_bounds"]
    plateau_params = FIT_CONTEXT["fit_plateau_params"]
    if plateau_params == 0 and preferred_start is None:
        return FIT_CONTEXT["fit_prepared_base_starts"]
    starts = FIT_CONTEXT["fit_base_starts"]
    if plateau_params >= 1:
        starts = np.column_stack([starts, np.full(len(starts), float(np.clip(np.nanmax(y), 0.0, 1.0)))])
    if plateau_params >= 2:
        starts = np.column_stack([starts, np.full(len(starts), float(np.clip(np.nanmin(y), 0.0, 1.0)))])
    if preferred_start is not None:
        preferred_start = np.asarray(preferred_start, dtype=float)[:starts.shape[1]]
        if np.all(np.isfinite(preferred_start)) and preferred_start[0] > 0:
            starts = np.vstack([[np.log(preferred_start[0]), *preferred_start[1:]], starts])
    starts = np.clip(starts, theta_lower, theta_upper)
    return np.unique(np.round(starts, 10), axis=0)


def fit_curve(inhibition_values, preferred_start=None, starts=None):
    y = np.asarray(inhibition_values, dtype=float)
    log_concentrations = FIT_CONTEXT["fit_log_concentrations"]
    scale_log = FIT_CONTEXT["scale_log"]
    dose_log_base = FIT_CONTEXT["dose_log_base"]
    theta_bounds = FIT_CONTEXT["fit_theta_bounds"]
    plateau_params = FIT_CONTEXT["fit_plateau_params"]
    has_front = plateau_params >= 1
    has_back = plateau_params >= 2

    def residual(theta):
        front = theta[2] if has_front else 1.0
        back = theta[3] if has_back else 0.0
        return back + (front - back) * expit(scale_log * np.exp(theta[0]) * (log_concentrations - theta[1])) - y

    if starts is None:
        starts = prepare_fit_starts(y, preferred_start)
    best_rss, best_theta = np.inf, np.full(starts.shape[1], np.nan)
    for start in starts:
        try:
            fit = least_squares(residual, start, bounds=theta_bounds, method="trf", max_nfev=1000)
            rss = float(np.sum(fit.fun ** 2))
            if rss < best_rss:
                best_rss, best_theta = rss, fit.x
        except Exception:
            pass
    slope = np.exp(best_theta[0])
    log_midpoint = best_theta[1]
    front = best_theta[2] if has_front else 1.0
    back = best_theta[3] if has_back else 0.0
    return best_rss, float(slope), float(log_midpoint), float(dose_log_base ** log_midpoint) if np.isfinite(log_midpoint) else np.nan, float(front), float(back)


def enumerate_bootstrap_count_patterns(n_replicates):
    records = []

    def fill(prefix, remaining, slots):
        if slots == 1:
            records.append(prefix + [remaining])
        else:
            for count in range(remaining + 1):
                fill(prefix + [count], remaining - count, slots - 1)

    fill([], n_replicates, n_replicates)
    return np.asarray(records, dtype=np.int16)


def build_bootstrap_count_table(n_replicates, rng):
    n_exact = math.comb(2 * n_replicates - 1, n_replicates)
    if n_exact <= FIT_CONTEXT["n_fit_bootstraps"]:
        counts = enumerate_bootstrap_count_patterns(n_replicates)
        weights = np.rint(np.exp(gammaln(n_replicates + 1) - gammaln(counts + 1).sum(axis=1))).astype(np.int64)
        return counts, weights, "exact_unique"
    sampled = rng.multinomial(n_replicates, np.full(n_replicates, 1 / n_replicates), size=FIT_CONTEXT["n_fit_bootstraps"])
    counts, weights = np.unique(sampled, axis=0, return_counts=True)
    return counts.astype(np.int16), weights.astype(np.int64), "sampled_unique"


def set_timeout_alarm(seconds):
    def timeout_handler(signum, frame):
        raise TimeoutError(f"timed out after {seconds} seconds")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(int(seconds))


def clear_timeout_alarm():
    signal.alarm(0)


def make_primary_record(payload, fit_rss=np.nan, fit_slope=np.nan, fit_log_midpoint=np.nan, fit_midpoint=np.nan, fit_front=np.nan, fit_back=np.nan, status="failed"):
    prefix = FIT_CONTEXT["fit_prefix"]
    replicate_curves = np.asarray(payload["replicate_curves"], dtype=np.float32)
    record = {
        "row_index": int(payload["row_index"]),
        "treatment_index": int(payload["treatment_index"]),
        "organism": payload["organism"],
        "treatment": payload["treatment"],
        "n_fit_replicates": int(replicate_curves.shape[0]),
        "fit_status": status,
        f"{prefix}_slope": fit_slope,
        f"{prefix}_midpoint": fit_midpoint,
        f"{prefix}_rss": fit_rss,
        **plateau_fields(fit_slope, fit_log_midpoint, fit_front, fit_back),
        **calculate_mic_fields(fit_slope, fit_log_midpoint, fit_front, fit_back, FIT_CONTEXT["fit_concentrations"], FIT_CONTEXT["fit_mic_inhibition_threshold"]),
        "n_fit_bootstraps": int(FIT_CONTEXT["n_fit_bootstraps"]) if FIT_CONTEXT["run_bootstrap"] else 0,
    }
    for optional_key in ("dose_layout", "dose_layout_full", "sequence"):
        if optional_key in payload:
            record[optional_key] = payload[optional_key]
    return record


def fit_curve_payload(payload):
    _install_payload_fit_context(payload)
    row_index = int(payload["row_index"])
    treatment_index = int(payload["treatment_index"])
    try:
        set_timeout_alarm(FIT_CONTEXT["fit_curve_timeout_seconds"])
        prefix = FIT_CONTEXT["fit_prefix"]
        fit_rss, fit_slope, fit_log_midpoint, fit_midpoint, fit_front, fit_back = fit_curve(payload["mean_curve"])
        status = "ok" if np.isfinite(fit_slope) and np.isfinite(fit_log_midpoint) and np.isfinite(fit_front) and np.isfinite(fit_back) else "failed"
        fit_record = make_primary_record(payload, fit_rss, fit_slope, fit_log_midpoint, fit_midpoint, fit_front, fit_back, status)
        bootstrap_records = []
        bootstrap_summary = None

        replicate_curves = np.asarray(payload["replicate_curves"], dtype=np.float32)
        if FIT_CONTEXT["run_bootstrap"] and status == "ok" and replicate_curves.shape[0] >= 2:
            counts, weights, bootstrap_mode = build_bootstrap_count_table(replicate_curves.shape[0], np.random.default_rng(FIT_CONTEXT["fit_bootstrap_seed"] + row_index * 100000))
            counts_float = counts.astype(np.float32)
            optional_fields = {key: payload[key] for key in ("dose_layout", "dose_layout_full", "sequence") if key in payload}
            preferred_start = (fit_slope, fit_log_midpoint, fit_front, fit_back)
            bootstrap_starts = (prepare_fit_starts(payload["mean_curve"], preferred_start)
                               if FIT_CONTEXT["fit_plateau_params"] == 0 else None)
            for bootstrap_index, (count_vector, weight) in enumerate(zip(counts_float, weights)):
                bootstrap_curve = (count_vector @ replicate_curves) / replicate_curves.shape[0]
                bootstrap_rss, bootstrap_slope, bootstrap_log_midpoint, bootstrap_midpoint, bootstrap_front, bootstrap_back = fit_curve(bootstrap_curve, preferred_start=preferred_start, starts=bootstrap_starts)
                if np.isfinite(bootstrap_slope) and np.isfinite(bootstrap_log_midpoint):
                    bootstrap_records.append({
                        "row_index": row_index, "treatment_index": treatment_index,
                        "organism": payload["organism"], "treatment": payload["treatment"],
                        "bootstrap_index": bootstrap_index, "bootstrap_mode": bootstrap_mode, "bootstrap_weight": int(weight),
                        f"{prefix}_slope": bootstrap_slope, f"{prefix}_midpoint": bootstrap_midpoint, f"{prefix}_rss": bootstrap_rss,
                        **plateau_fields(bootstrap_slope, bootstrap_log_midpoint, bootstrap_front, bootstrap_back),
                        **calculate_mic_fields(bootstrap_slope, bootstrap_log_midpoint, bootstrap_front, bootstrap_back, FIT_CONTEXT["fit_concentrations"], FIT_CONTEXT["fit_mic_inhibition_threshold"]),
                        **optional_fields,
                    })
            if bootstrap_records:
                bootstrap_summary = {"row_index": row_index, "n_fit_bootstraps_used": int(weights.sum())}
        clear_timeout_alarm()
        return fit_record, bootstrap_records, bootstrap_summary
    except TimeoutError:
        clear_timeout_alarm()
        return make_primary_record(payload, status="timeout"), [], None
    except Exception as error:
        clear_timeout_alarm()
        return make_primary_record(payload, status=f"error:{type(error).__name__}"), [], None


def weighted_percentile(values, weights, percentiles):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        return np.full(len(np.atleast_1d(percentiles)), np.nan)
    order = np.argsort(values[valid])
    values, weights = values[valid][order], weights[valid][order]
    cutoffs = np.asarray(percentiles, dtype=float) / 100 * weights.sum()
    return values[np.clip(np.searchsorted(np.cumsum(weights), cutoffs, side="left"), 0, len(values) - 1)]


def discrete_mic_summary(labels, orders, weights):
    prefix = FIT_CONTEXT["fit_prefix"]
    label_values = labels.astype(str).to_numpy()
    order_values = pd.to_numeric(orders, errors="coerce").to_numpy(dtype=float)
    weight_values = np.asarray(weights, dtype=float)
    valid = ~np.isnan(order_values) & (weight_values > 0)
    if not valid.any():
        return {f"{prefix}_discrete_mic_ci_low": "NA", f"{prefix}_discrete_mic_ci_high": "NA"}
    sort_order = np.argsort(order_values[valid], kind="quicksort")
    sorted_labels = label_values[valid][sort_order]
    cumulative_weight = np.cumsum(weight_values[valid][sort_order])

    def pick(percentile):
        cutoff = percentile / 100 * cumulative_weight[-1]
        return sorted_labels[int(np.clip(np.searchsorted(cumulative_weight, cutoff, side="left"), 0, len(sorted_labels) - 1))]

    return {f"{prefix}_discrete_mic_ci_low": pick(2.5), f"{prefix}_discrete_mic_ci_high": pick(97.5)}


def summarize_bootstraps(bootstrap_results, bootstrap_run_summaries):
    prefix, fit_model = FIT_CONTEXT["fit_prefix"], FIT_CONTEXT["fit_model"]
    summary_records = []
    if not bootstrap_results.empty:
        bootstrap_run_summaries = bootstrap_run_summaries.set_index("row_index") if not bootstrap_run_summaries.empty else pd.DataFrame()
        for row_index, group in bootstrap_results.groupby("row_index", sort=False):
            weights = group["bootstrap_weight"].to_numpy(float)
            if row_index in bootstrap_run_summaries.index:
                run_summary = bootstrap_run_summaries.loc[row_index]
                n_bootstraps_used = int(run_summary["n_fit_bootstraps_used"])
            else:
                n_bootstraps_used = int(weights.sum())
            record = {"row_index": int(row_index), "n_fit_bootstraps_used": n_bootstraps_used}
            for column in [f"{prefix}_slope", f"{prefix}_midpoint", f"{prefix}_rss", f"{prefix}_continuous_mic", *plateau_param_names(prefix, fit_model)]:
                low, high = weighted_percentile(group[column], weights, [2.5, 97.5])
                record[f"{column}_ci_low"] = low
                record[f"{column}_ci_high"] = high
            record.update(discrete_mic_summary(group[f"{prefix}_discrete_mic"], group[f"{prefix}_discrete_mic_order"], weights))
            summary_records.append(record)
    return pd.DataFrame(summary_records)


def build_fit_payload(result, treatment_metadata, analysis_replicate_curves, observed_inhibition_curves, fit_label):
    treatment_to_index = dict(zip(treatment_metadata["treatment"], treatment_metadata["treatment_index"].astype(int)))
    fit_rows = result.index[result["dose_response_label"].eq(fit_label)].to_numpy()
    optional_columns = [column for column in ("dose_layout", "dose_layout_full", "sequence") if column in result.columns]
    fit_context = dict(FIT_CONTEXT)
    payload = []
    for row_index in fit_rows:
        treatment = result.loc[row_index, "treatment"]
        treatment_index = int(treatment_to_index[treatment])
        fit_record = {
            "row_index": int(row_index),
            "treatment_index": treatment_index,
            "organism": result.loc[row_index, "organism"],
            "replicate_curves": np.asarray(analysis_replicate_curves[treatment_index], dtype=np.float32),
            "mean_curve": np.asarray(observed_inhibition_curves[treatment_index], dtype=np.float32),
            "treatment": treatment,
            "fit_context": fit_context,
        }
        for column in optional_columns:
            fit_record[column] = result.loc[row_index, column]
        payload.append(fit_record)
    return payload


def estimate_payload_cost(payload):
    n_replicates = int(payload["replicate_curves"].shape[0])
    if FIT_CONTEXT["run_bootstrap"] and n_replicates >= 2:
        return 1 + min(math.comb(2 * n_replicates - 1, n_replicates), FIT_CONTEXT["n_fit_bootstraps"])
    return 1


def sort_existing_columns(table, columns):
    sort_columns = [column for column in columns if column in table.columns]
    if table.empty or not sort_columns:
        return table
    return table.sort_values(sort_columns).reset_index(drop=True)


def run_fits(fit_payload, executor):
    fit_records, bootstrap_records, bootstrap_summary_records = [], [], []
    prefix, fit_model = FIT_CONTEXT["fit_prefix"], FIT_CONTEXT["fit_model"]
    internal_columns = fit_result_internal_columns(prefix, fit_model)
    bootstrap_columns = bootstrap_result_columns(prefix, fit_model)
    if not fit_payload:
        return (pd.DataFrame(columns=internal_columns),
                pd.DataFrame(columns=bootstrap_columns),
                pd.DataFrame(columns=BOOTSTRAP_SUMMARY_COLUMNS))
    progress = create_progress_bar(len(fit_payload), f"{fit_model.upper()} fits", compact=True)

    def collect(result):
        fit_record, bootstrap_recs, bootstrap_summary = result
        fit_records.append(fit_record)
        bootstrap_records.extend(bootstrap_recs)
        if bootstrap_summary is not None:
            bootstrap_summary_records.append(bootstrap_summary)
        progress.update(1)

    try:
        if executor is None:
            for payload in fit_payload:
                collect(fit_curve_payload(payload))
        else:
            submission_order = sorted(fit_payload,
                                      key=lambda payload: (-estimate_payload_cost(payload), payload["row_index"]))
            futures = [executor.submit(fit_curve_payload, payload) for payload in submission_order]
            for future in as_completed(futures):
                collect(future.result())
    finally:
        progress.close()
    fit_results = pd.DataFrame(fit_records)
    bootstrap_results = pd.DataFrame(bootstrap_records)
    bootstrap_summaries = pd.DataFrame(bootstrap_summary_records)
    if fit_results.empty:
        fit_results = pd.DataFrame(columns=internal_columns)
    if bootstrap_results.empty:
        bootstrap_results = pd.DataFrame(columns=bootstrap_columns)
    if bootstrap_summaries.empty:
        bootstrap_summaries = pd.DataFrame(columns=BOOTSTRAP_SUMMARY_COLUMNS)
    fit_results = sort_existing_columns(fit_results, ["row_index"])
    bootstrap_results = sort_existing_columns(bootstrap_results, ["row_index", "bootstrap_index"])
    bootstrap_summaries = sort_existing_columns(bootstrap_summaries, ["row_index"])
    return fit_results, bootstrap_results, bootstrap_summaries


def merge_bootstrap_summary(fit_results, bootstrap_results, bootstrap_run_summaries):
    prefix, fit_model = FIT_CONTEXT["fit_prefix"], FIT_CONTEXT["fit_model"]
    bootstrap_summary = summarize_bootstraps(bootstrap_results, bootstrap_run_summaries)
    if bootstrap_summary.empty:
        bootstrap_summary = pd.DataFrame(columns=BOOTSTRAP_SUMMARY_COLUMNS)
    fit_results = fit_results.merge(bootstrap_summary, on="row_index", how="left")
    bootstraps_used = pd.to_numeric(fit_results["n_fit_bootstraps_used"], errors="coerce").to_numpy(float)
    bootstraps_default = pd.to_numeric(fit_results["n_fit_bootstraps"], errors="coerce").to_numpy(float)
    fit_results["n_fit_bootstraps"] = np.where(np.isfinite(bootstraps_used), bootstraps_used, bootstraps_default).astype(int)
    fit_results = fit_results.drop(columns=["n_fit_bootstraps_used"], errors="ignore")
    for column in [
        f"{prefix}_slope_ci_low", f"{prefix}_slope_ci_high",
        f"{prefix}_midpoint_ci_low", f"{prefix}_midpoint_ci_high",
        f"{prefix}_rss_ci_low", f"{prefix}_rss_ci_high",
        f"{prefix}_continuous_mic_ci_low", f"{prefix}_continuous_mic_ci_high",
        *[f"{name}_{bound}" for name in plateau_param_names(prefix, fit_model) for bound in ("ci_low", "ci_high")],
    ]:
        if column not in fit_results.columns:
            fit_results[column] = np.nan
    for column in [f"{prefix}_discrete_mic_ci_low", f"{prefix}_discrete_mic_ci_high"]:
        if column not in fit_results.columns:
            fit_results[column] = "NA"
    return sort_existing_columns(fit_results, ["organism", "dose_layout", "treatment"])


def prepare_user_table(table):
    table = table.drop(columns=["row_index", "treatment_index"], errors="ignore")
    if table.empty:
        return table
    leading_columns = [column for column in ["organism", "treatment", "dose_layout", "dose_layout_full", "sequence"] if column in table.columns]
    return table[leading_columns + [column for column in table.columns if column not in leading_columns]]


def prepare_fit_results_table(table):
    table = prepare_user_table(table)
    ordered_columns = [column for column in fit_result_columns(FIT_CONTEXT["fit_prefix"], FIT_CONTEXT["fit_model"]) if column in table.columns]
    table = table[ordered_columns + [column for column in table.columns if column not in ordered_columns]]
    return table.rename(columns=fit_result_column_renames(FIT_CONTEXT["fit_prefix"], FIT_CONTEXT["fit_model"]))


def build_dynamic_results_table(observed_rows, concentration_values,
                              treatment_replicates_curves, replicate_metadata,
                              replicate_outliers, classification_result, fit_results):
    n_doses = len(concentration_values)
    replicate_table = replicate_metadata.copy()
    for column in ["dose_layout", "sequence", "assay_id", "data_id"]:
        if column not in replicate_table.columns:
            replicate_table[column] = classification_result[column].iloc[0] if column in classification_result.columns else pd.NA

    optional_metadata_columns = [column for column in ["GC_imputed", "SC_imputed", "mic", "mbc"] if column in observed_rows.columns]
    if optional_metadata_columns and "data_id" in replicate_table.columns and "data_id" in observed_rows.columns:
        metadata_key_columns = [column for column in ["organism", "treatment", "assay_id", "data_id"]
                                if column in replicate_table.columns and column in observed_rows.columns]
        optional_metadata = observed_rows[metadata_key_columns + optional_metadata_columns].drop_duplicates(metadata_key_columns)
        replicate_table = replicate_table.merge(optional_metadata, on=metadata_key_columns, how="left", sort=False)
    replicate_table = replicate_table.rename(columns={"mic": "mic_annotation", "mbc": "mbc_annotation"})
    for column in ["GC_imputed", "SC_imputed"]:
        if column not in replicate_table.columns:
            replicate_table[column] = pd.NA

    if replicate_outliers.empty:
        replicate_table["replicate_curve_outlier"] = False
    else:
        outlier_columns = ["treatment_index", "replicate_index", "replicate_curve_outlier"]
        replicate_table = replicate_table.merge(
            replicate_outliers[[column for column in outlier_columns if column in replicate_outliers.columns]],
            on=["treatment_index", "replicate_index"],
            how="left",
            sort=False,
        )
        replicate_table["replicate_curve_outlier"] = replicate_table["replicate_curve_outlier"].fillna(False).astype(bool)

    classification_table = classification_result.reset_index().rename(columns={"index": "row_index"})
    classification_columns = [
        "row_index", "organism", "treatment", "dose_response_label",
        "two_pl_signal_score", "two_pl_signal_score_p", "two_pl_signal_score_q",
    ]
    merge_keys = ["organism", "treatment"]
    if "dose_layout" in classification_table.columns and "dose_layout" in replicate_table.columns:
        classification_columns.insert(3, "dose_layout")
        merge_keys.append("dose_layout")
    replicate_table = replicate_table.merge(
        classification_table[classification_columns],
        on=merge_keys,
        how="left",
        sort=False,
    )

    fit_model = FIT_CONTEXT["fit_model"]
    dynamic_results_columns = DYNAMIC_RESULTS_COLUMNS + clean_plateau_columns(fit_model)
    fit_table = fit_results.rename(columns=fit_result_column_renames(FIT_CONTEXT["fit_prefix"], fit_model))
    fit_columns = ["row_index"] + [column for column in DYNAMIC_RESULTS_FIT_COLUMNS + clean_plateau_columns(fit_model) if column in fit_table.columns]
    replicate_table = replicate_table.merge(fit_table[fit_columns], on="row_index", how="left", sort=False)
    replicate_table["fit_status"] = replicate_table["fit_status"].fillna("not_fit")

    replicate_curves = np.vstack(treatment_replicates_curves)
    inhibition_values = replicate_curves.reshape(-1)
    used_concentrations = pd.notna(inhibition_values)
    dynamic_results = replicate_table.loc[
        np.repeat(replicate_table.index.to_numpy(), n_doses)[used_concentrations]
    ].reset_index(drop=True)
    dynamic_results["concentration"] = np.tile(concentration_values, len(replicate_table))[used_concentrations]
    dynamic_results["inhibition_fraction"] = inhibition_values[used_concentrations]

    for column in dynamic_results_columns:
        if column not in dynamic_results.columns:
            dynamic_results[column] = pd.NA
    return dynamic_results[dynamic_results_columns]


def run_modelling(args, output_dir, directories):
    tables_dir = directories["tables"]
    classification_manifest = load_manifest(Path(args.classification_dir) / "classification_manifest.csv", args.organism)
    calibration_manifest = load_manifest(Path(args.calibration_dir) / "calibration_manifest.csv", args.organism)
    require_columns(classification_manifest.columns, ["organism", "organism_token", "dose_layout", "dose_layout_token", "inhibition_columns", "concentration_values", "classification_results_path", "replicate_outliers_path"], "Classification manifest", Path(args.classification_dir) / "classification_manifest.csv")
    require_columns(calibration_manifest.columns, ["organism", "dose_layout", "control_inhibition_path"], "Calibration manifest", Path(args.calibration_dir) / "calibration_manifest.csv")
    calibration_keys = ["organism", "dose_layout"]
    if calibration_manifest.duplicated(calibration_keys).any():
        duplicated = calibration_manifest.loc[calibration_manifest.duplicated(calibration_keys, keep=False), calibration_keys].drop_duplicates()
        raise ValueError("Calibration manifest has duplicate organism-dose-layout records: " + "; ".join(duplicated.astype(str).agg(" | ".join, axis=1)))
    calibration_by_set = calibration_manifest.set_index(calibration_keys).to_dict("index")
    output_records = []
    dynamic_result_tables = []
    emit_message("info", "Modelling run", lines=[
        *get_run_logo_lines(),
        format_detail("Version", TOOL_VERSION),
        format_detail("Normalized data", style_text(str(args.normalized_data), TEXT_COLOURS["accent"])),
        format_detail("Classification", style_text(str(args.classification_dir), TEXT_COLOURS["accent"])),
        format_detail("Calibration", style_text(str(args.calibration_dir), TEXT_COLOURS["accent"])),
        format_detail("Output", style_text(str(output_dir), TEXT_COLOURS["accent"])),
        format_detail("Bootstrap", args.bootstrap),
        format_detail("MIC threshold", args.mic_threshold),
    ])

    total_stages = max(1, 4 * len(classification_manifest))
    progress = create_progress_bar(total_stages, format_stage_label(1, total_stages, "Fit organisms"))
    stage_number = 1

    def complete_stage(next_stage=None):
        nonlocal stage_number
        progress.update(1)
        stage_number += 1
        if next_stage is not None and stage_number <= total_stages:
            set_progress_stage(progress, stage_number, total_stages, next_stage)

    normalized_data = load_normalized_inhibition_data(
        args.normalized_data,
        metadata_columns=("organism", "treatment", "data_id", "assay_id", "sequence",
                          "mic", "mbc", "GC_imputed", "SC_imputed"))
    fit_executor = None
    if args.max_workers > 1:
        fit_executor = ProcessPoolExecutor(max_workers=min(args.max_workers, os.cpu_count() or 1))
    try:
        for context_token, row in enumerate(classification_manifest.itertuples(index=False)):
            organism = row.organism
            organism_token = row.organism_token
            dose_layout = row.dose_layout
            dose_layout_full = row.dose_layout_full
            dose_layout_token = row.dose_layout_token
            calibration_record = calibration_by_set.get((organism, dose_layout))
            if calibration_record is None:
                raise ValueError(f"Calibration file missing for {organism} dose layout: {dose_layout}")
            record_token = f"{organism_token}_{dose_layout_token}"
            set_progress_stage(progress, stage_number, total_stages, f"{organism}: load classified curves")
            emit_message("info", f"Modelling {organism}", lines=format_detail("Dose layout", dose_layout))
            inhibition_columns = parse_csv_list(row.inhibition_columns)
            concentration_values = np.asarray(parse_csv_list(row.concentration_values), dtype=float)
            classification_result = load_classification_rows(args.classification_dir, row.classification_results_path)
            if "dose_layout" not in classification_result.columns:
                classification_result["dose_layout"] = dose_layout
            if "dose_layout_full" not in classification_result.columns:
                classification_result["dose_layout_full"] = dose_layout_full
            replicate_outliers = pd.read_csv(Path(args.classification_dir) / row.replicate_outliers_path)
            complete_stage(f"{organism}: rebuild analysis curves")

            observed_rows = load_observed_rows(args.normalized_data, organism, inhibition_columns, preloaded_data=normalized_data)
            treatment_metadata, treatment_replicates_curves, replicate_metadata = group_treatment_replicates(observed_rows, inhibition_columns)
            treatment_metadata["dose_layout"] = dose_layout
            treatment_metadata["dose_layout_full"] = dose_layout_full
            replicate_metadata["dose_layout"] = dose_layout
            replicate_metadata["dose_layout_full"] = dose_layout_full
            replicate_outliers = add_internal_treatment_index(replicate_outliers, treatment_metadata)
            analysis_replicate_curves, observed_inhibition_curves = build_analysis_treatment_curves(
                treatment_replicates_curves,
                replicate_outliers,
                bool(classification_result["replicate_outlier_exclusion"].iloc[0]),
            )
            control_inhibition_measurements = load_control_inhibition_measurements(args.calibration_dir, calibration_record["control_inhibition_path"])
            slope_resolution = calculate_slope_resolution_fields(concentration_values, control_inhibition_measurements)
            configure_fit_context(concentration_values, slope_resolution, args, context_token=context_token)
            fit_payload = build_fit_payload(classification_result, treatment_metadata, analysis_replicate_curves, observed_inhibition_curves, args.fit_label)
            complete_stage(f"{organism}: fit {args.fit_model.upper()} models")

            fit_results, bootstrap_results, bootstrap_summaries = run_fits(fit_payload, fit_executor)
            if not fit_results.empty:
                fit_results = merge_bootstrap_summary(fit_results, bootstrap_results, bootstrap_summaries)
            dynamic_result_tables.append(build_dynamic_results_table(
                observed_rows,
                concentration_values,
                treatment_replicates_curves,
                replicate_metadata,
                replicate_outliers,
                classification_result,
                fit_results,
            ))
            complete_stage(f"{organism}: save fit outputs")

            qc_excluded_fields = {"fit_log_concentrations", "transition_constant_dilution_steps"}
            qc_summary = pd.DataFrame([{key: value for key, value in slope_resolution.items() if key not in qc_excluded_fields} | {
                "organism": organism,
                "dose_layout": dose_layout,
                "dose_layout_full": dose_layout_full,
                "fit_model": args.fit_model,
                "mic_inhibition_threshold": args.mic_threshold,
                "run_bootstrap": args.bootstrap == "on",
                "n_fit_bootstraps": args.n_bootstraps,
            }])
            record_prefix = f"{record_token}_{MODEL_SPECS[args.fit_model].prefix}"
            fit_results_path = tables_dir / f"{record_prefix}_results.csv"
            bootstrap_results_path = tables_dir / f"{record_prefix}_bootstrap_results.csv"
            qc_summary_path = tables_dir / f"{record_prefix}_qc_summary.csv"
            prepare_fit_results_table(fit_results).to_csv(fit_results_path, index=False)
            prepare_user_table(bootstrap_results).to_csv(bootstrap_results_path, index=False)
            qc_summary.to_csv(qc_summary_path, index=False)
            output_records.append({
                "organism": organism,
                "organism_token": organism_token,
                "dose_layout": dose_layout,
                "dose_layout_full": dose_layout_full,
                "dose_layout_token": dose_layout_token,
                "fit_results_path": str(fit_results_path.relative_to(output_dir)),
                "bootstrap_results_path": str(bootstrap_results_path.relative_to(output_dir)),
                "fit_qc_summary_path": str(qc_summary_path.relative_to(output_dir)),
            })
            emit_message("info", "Fit summary", lines=[
                format_detail("Fitted curves", f"{len(fit_results):,}"),
                format_detail("OK fits", f"{int(fit_results['fit_status'].eq('ok').sum()) if not fit_results.empty else 0:,}"),
            ])
            complete_stage("Fit organisms")
    finally:
        progress.close()
        if fit_executor is not None:
            fit_executor.shutdown()

    dynamic_results_path = tables_dir / DYNAMIC_RESULTS_OUTPUT_NAME
    dynamic_results = (
        pd.concat(dynamic_result_tables, ignore_index=True)
        if dynamic_result_tables else pd.DataFrame(columns=DYNAMIC_RESULTS_COLUMNS + clean_plateau_columns(args.fit_model))
    )
    dynamic_results.to_csv(dynamic_results_path, index=False)
    pd.DataFrame(output_records).to_csv(output_dir / "modelling_manifest.csv", index=False)
    emit_message("success", "Modelling complete", lines=[
        format_detail("Output", style_text(str(output_dir), TEXT_COLOURS["accent"])),
        format_detail("Manifest", "modelling_manifest.csv"),
        format_detail("DynaMIC results", f"tables/{DYNAMIC_RESULTS_OUTPUT_NAME}"),
    ])


def main():
    args = parse_args()
    output_dir = create_dated_run_directory(args.output_dir, args.normalized_data, DEFAULT_RUN_DIR_PREFIX)
    directories = prepare_output_directories(output_dir, ["logs", "tables"])
    with RunLogger(directories["logs"] / "run_log.txt"):
        write_run_parameters(args, directories["logs"], "modelling.py", TOOL_VERSION, {"output_dir": str(output_dir)})
        try:
            run_modelling(args, output_dir, directories)
        except Exception as error:
            emit_message("error", "Modelling failed", lines=format_detail("Reason", str(error)))
            traceback.print_exc()
            raise


if __name__ == "__main__":
    main()
