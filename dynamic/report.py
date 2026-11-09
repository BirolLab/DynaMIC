#!/usr/bin/env python3

import argparse
import json
import math
from importlib import resources
from pathlib import Path
import re
import traceback

import numpy as np
import pandas as pd

from dynamic.utils import (LogoArgumentParser, TEXT_COLOURS, emit_message, format_detail,
                           get_run_logo_lines, style_text)


from dynamic import __version__ as TOOL_VERSION
REPORT_OUTPUT_NAME = "Report.html"
ASSET_DIR = resources.files(__package__) / "assets"
PLOTLY_ASSET_PATH = ASSET_DIR / "plotly" / "plotly.min.js"
ICON_ASSET_DIR = ASSET_DIR / "icons"
ICON_ASSET_PATHS = {
    "download": ICON_ASSET_DIR / "download.svg",
    "move": ICON_ASSET_DIR / "move.svg",
    "save_figure": ICON_ASSET_DIR / "save_figure.svg",
    "select": ICON_ASSET_DIR / "select.svg",
    "table": ICON_ASSET_DIR / "table.svg",
    "zoom_out": ICON_ASSET_DIR / "zoom-out.svg",
}
DYNAMIC_RESULTS_PATH = Path("modelling") / "tables" / "dynamic_results.csv"
CONTROL_POOL_PATH = Path("normalization") / "tables" / "control_pool.csv"
NORMALIZATION_TABLES_PATH = Path("normalization") / "tables"
DEFAULT_FIT_MODEL = "2pl"
DEFAULT_MIC_INHIBITION_THRESHOLD = 0.95
DEFAULT_Q_THRESHOLD = 0.05
SIGNAL_CLASSES = ["dose-variant", "inverse dose-variant", "dose-invariant"]
GROWTH_CONTROL = "Growth Control"
STERILITY_CONTROL = "Sterility Control"
CLASS_COLOURS = ["#65C083", "#D77FB3", "#8F8BE8"]
CONTROL_QC_COLOURS = {GROWTH_CONTROL: "#F2A678", STERILITY_CONTROL: "#65C083"}
CONTROL_POOL_COLOURS = {GROWTH_CONTROL: "#F2A678", STERILITY_CONTROL: "#65C083"}
RAW_INHIBITION_BAND_COLOURS = {"<0": "#D77FB3", "[0,1]": "#65C083", ">1": "#F2A678"}
THRESHOLD_KEY_SEPARATOR = "\u0000"
CALIBRATION_MANIFEST_PATH = Path("calibration") / "calibration_manifest.csv"
NULL_SCORE_KEYS = ("no_inhibition_null", "full_inhibition_null")
RUN_PARAMETER_SCRIPT_PRIORITY = ["normalization.py", "calibration.py", "classification.py", "modelling.py", "dynamic.py"]
CONTROL_INDEX = {GROWTH_CONTROL: 0, STERILITY_CONTROL: 1}


def parse_args():
    parser = LogoArgumentParser(description="DynaMIC interactive run report", formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True, help="Completed DynaMIC run directory (containing normalization/ and modelling/).")
    parser.add_argument("--output", default=REPORT_OUTPUT_NAME, help=f"Report filename or path; relative paths resolve inside the run directory. Default: {REPORT_OUTPUT_NAME}.")
    return parser.parse_args()



def load_run_parameters(run_dir):
    records = {}
    for path in sorted(Path(run_dir).rglob("run_parameters.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        records.setdefault(record.get("script"), record)
    merged, meta = {}, {}
    for script in RUN_PARAMETER_SCRIPT_PRIORITY:
        record = records.get(script)
        if record:
            merged.update(record.get("parameters") or {})
            meta = {"script": script, "tool_version": record.get("tool_version"),
                    "started_at": record.get("started_at")}
    return {
        "fit_model": merged.get("fit_model") or DEFAULT_FIT_MODEL,
        "mic_threshold": float(merged.get("mic_threshold") or DEFAULT_MIC_INHIBITION_THRESHOLD),
        "q_threshold": float(merged.get("q_threshold") or DEFAULT_Q_THRESHOLD),
        "input": merged.get("input") or merged.get("input_csv") or "",
        "tool_version": meta.get("tool_version") or TOOL_VERSION,
        "started_at": meta.get("started_at") or "",
    }


def locate_run_files(run_dir):
    run_dir = Path(run_dir)
    dynamic_results_path = run_dir / DYNAMIC_RESULTS_PATH
    control_pool_path = run_dir / CONTROL_POOL_PATH
    normalized_candidates = sorted((run_dir / NORMALIZATION_TABLES_PATH).glob("*_with_if.csv"))
    missing = [str(path) for path in (dynamic_results_path, control_pool_path) if not path.exists()]
    if not normalized_candidates:
        missing.append(str(run_dir / NORMALIZATION_TABLES_PATH / "*_with_if.csv"))
    if missing:
        raise ValueError("Not a completed DynaMIC run directory; missing: " + ", ".join(missing))
    return dynamic_results_path, control_pool_path, normalized_candidates[0]


def _clean(value, digits=4):
    return None if pd.isna(value) else round(float(value), digits)


def _clean_text(value):
    return None if pd.isna(value) else str(value)


def _interval(row, column):
    return [_clean(row[column]), _clean(row.get(f"{column}_ci_low")), _clean(row.get(f"{column}_ci_high"))]


def extract_treatments(dynamic_results):
    concentration_grids, treatments = [], []
    grid_index = {}
    plateau_columns = [column for column in ("max_inhibition", "min_inhibition") if column in dynamic_results.columns]
    for (organism, treatment, dose_layout), rows in dynamic_results.groupby(["organism", "treatment", "dose_layout"], sort=False, dropna=False):
        first = rows.iloc[0]
        concentrations = sorted(rows["concentration"].unique())
        grid_key = tuple(concentrations)
        if grid_key not in grid_index:
            grid_index[grid_key] = len(concentration_grids)
            concentration_grids.append([_clean(concentration) for concentration in concentrations])

        curve = rows.pivot(index="replicate_index", columns="concentration", values="inhibition_fraction").reindex(columns=concentrations)
        outlier_by_replicate = rows.groupby("replicate_index")["replicate_curve_outlier"].first().reindex(curve.index)
        inlier_rows = rows[~rows["replicate_curve_outlier"].astype(bool)]
        effect_rows = inlier_rows if not inlier_rows.empty else rows
        concentration_means = effect_rows.groupby("concentration")["inhibition_fraction"].mean()
        signed_change = concentration_means.get(concentrations[-1], np.nan) - concentration_means.get(concentrations[0], np.nan)
        mean_inhibition = concentration_means.mean()
        q_value = float(first["two_pl_signal_score_q"])

        record = {
            "treatment": str(treatment),
            "organism": str(organism),
            "doseLayout": _clean_text(dose_layout),
            "cls": SIGNAL_CLASSES.index(first["dose_response_label"]) if first["dose_response_label"] in SIGNAL_CLASSES else 2,
            "x": _clean(signed_change),
            "meanInhibition": _clean(mean_inhibition),
            "y": _clean(-np.log10(max(q_value, 1e-300))),
            "q": _clean(q_value, 6),
            "score": _clean(first["two_pl_signal_score"], 6),
            "cc": _interval(first, "continuous_mic"),
            "qc": {"mic": _clean_text(first["mic_qc"]), "midpoint": _clean_text(first["midpoint_qc"]),
                   "slope": _clean_text(first["slope_qc"])},
            "dc": [_clean_text(first["discrete_mic"]), _clean_text(first["discrete_mic_ci_low"]), _clean_text(first["discrete_mic_ci_high"])],
            "grid": grid_index[grid_key],
            "reps": [[_clean(value) for value in replicate] for replicate in curve.to_numpy()],
            "out": [bool(flag) for flag in outlier_by_replicate.fillna(False)],
        }
        if first["fit_status"] == "ok" and pd.notna(first["fit_slope"]):
            record["fit"] = {
                "slope": _interval(first, "fit_slope"),
                "midpoint": _interval(first, "fit_midpoint"),
                **{column: _interval(first, column) for column in plateau_columns},
            }
        treatments.append(record)
    return treatments, concentration_grids


def _threshold_key(organism, dose_layout):
    return f"{organism}{THRESHOLD_KEY_SEPARATOR}{dose_layout or ''}"


def _null_score_threshold(run_dir, organism, dose_layout):
    manifest_path = Path(run_dir) / CALIBRATION_MANIFEST_PATH
    group_label = f"{organism} ({dose_layout or 'default dose layout'})"
    if not manifest_path.exists():
        raise ValueError(f"{group_label} has no responsive treatments and no calibration manifest at {manifest_path} for the null-score fallback.")
    manifest = pd.read_csv(manifest_path)
    rows = manifest[manifest["organism"].astype(str).eq(str(organism)) & manifest["dose_layout"].astype(str).eq(str(dose_layout))]
    if rows.empty:
        raise ValueError(f"Calibration manifest has no record for {group_label}; cannot derive its decision threshold.")
    scores_path = manifest_path.parent / rows.iloc[0]["null_scores_path"]
    if not scores_path.exists():
        raise ValueError(f"Null scores file for {group_label} is missing: {scores_path}")
    with np.load(scores_path) as scores:
        return float(max(np.percentile(scores[key], 95) for key in NULL_SCORE_KEYS))


def compute_decision_thresholds(treatments, run_dir):
    groups = {}
    for treatment in treatments:
        score = treatment.get("score")
        if not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError(f"Treatment {treatment['organism']}/{treatment['treatment']} has no finite two_pl_signal_score; regenerate dynamic_results.csv.")
        key = _threshold_key(treatment["organism"], treatment["doseLayout"])
        bucket = groups.setdefault(key, {"responsive": [], "organism": treatment["organism"], "dose_layout": treatment["doseLayout"]})
        if treatment["cls"] != 2:
            bucket["responsive"].append(score)

    thresholds = {}
    for key, bucket in groups.items():
        if bucket["responsive"]:
            threshold = float(min(bucket["responsive"]))
        else:
            threshold = _null_score_threshold(run_dir, bucket["organism"], bucket["dose_layout"])
        if not (isinstance(threshold, float) and math.isfinite(threshold) and threshold > 0):
            raise ValueError(f"Decision threshold for group {key!r} is not a positive finite number: {threshold!r}")
        thresholds[key] = round(threshold, 6)
    return thresholds


def extract_normalization_checks(control_pool, normalized_data):
    control_rows = control_pool[control_pool["treatment"].isin([GROWTH_CONTROL, STERILITY_CONTROL])].copy()
    control_rows["measurement"] = pd.to_numeric(control_rows["measurement"], errors="coerce")
    control_rows = control_rows.dropna(subset=["measurement"])
    within_flag = control_rows["within_assay_mZ_score"].fillna(False).astype(bool)
    across_flag = control_rows["across_assay_mZ_score"].fillna(False).astype(bool)

    raw_inhibition_columns = [column for column in normalized_data.columns if column.startswith("RI")]
    experimental = normalized_data[~normalized_data["treatment"].isin([GROWTH_CONTROL, STERILITY_CONTROL])]

    checks = {}
    for organism in sorted(set(control_rows["organism"].dropna()) | set(experimental["organism"].dropna())):
        organism_controls = control_rows[control_rows["organism"].eq(organism)]
        organism_within = within_flag[organism_controls.index]
        organism_across = across_flag[organism_controls.index]

        across = (organism_controls.assign(flag=organism_across)
                  .groupby(["assay_id", "treatment"])
                  .agg(median=("measurement", "median"), flag=("flag", "any"))
                  .reset_index().sort_values(["treatment", "assay_id"]))
        inliers = organism_controls[~organism_within & ~organism_across]

        raw = experimental[experimental["organism"].eq(organism)]
        raw_concentrations, raw_values = [], []
        for column in raw_inhibition_columns:
            values = pd.to_numeric(raw[column], errors="coerce").dropna()
            raw_concentrations.extend([float(column[2:])] * len(values))
            raw_values.extend(round(value, 4) for value in values)

        checks[organism] = {
            "within": {
                "assay": [int(value) for value in organism_controls["assay_id"]],
                "control": [CONTROL_INDEX[treatment] for treatment in organism_controls["treatment"]],
                "value": [round(value, 4) for value in organism_controls["measurement"]],
                "flag": [bool(value) for value in organism_within],
            },
            "across": {
                "assay": [int(value) for value in across["assay_id"]],
                "control": [CONTROL_INDEX[treatment] for treatment in across["treatment"]],
                "value": [_clean(value) for value in across["median"]],
                "flag": [bool(value) for value in across["flag"]],
            },
            "inlier": {
                "control": [CONTROL_INDEX[treatment] for treatment in inliers["treatment"]],
                "value": [round(value, 4) for value in inliers["measurement"]],
            },
            "raw": {"concentration": raw_concentrations, "value": raw_values},
        }
    return checks


def build_payload(run_dir):
    parameters = load_run_parameters(run_dir)
    dynamic_results_path, control_pool_path, normalized_path = locate_run_files(run_dir)
    dynamic_results = pd.read_csv(dynamic_results_path, low_memory=False)
    control_pool = pd.read_csv(control_pool_path)
    normalized_data = pd.read_csv(normalized_path, low_memory=False)

    treatments, concentration_grids = extract_treatments(dynamic_results)
    thresholds = compute_decision_thresholds(treatments, run_dir)
    checks = extract_normalization_checks(control_pool, normalized_data)
    return {
        "meta": {
            "input": Path(parameters["input"]).name if parameters["input"] else "",
            "startedAt": parameters["started_at"],
            "toolVersion": parameters["tool_version"],
            "fitModel": parameters["fit_model"],
            "micThreshold": parameters["mic_threshold"],
            "qThreshold": parameters["q_threshold"],
            "runDir": Path(run_dir).name,
        },
        "classes": SIGNAL_CLASSES,
        "classColours": CLASS_COLOURS,
        "controls": [GROWTH_CONTROL, STERILITY_CONTROL],
        "controlQcColours": [CONTROL_QC_COLOURS[GROWTH_CONTROL], CONTROL_QC_COLOURS[STERILITY_CONTROL]],
        "controlPoolColours": [CONTROL_POOL_COLOURS[GROWTH_CONTROL], CONTROL_POOL_COLOURS[STERILITY_CONTROL]],
        "rawBandColours": RAW_INHIBITION_BAND_COLOURS,
        "concentrationGrids": concentration_grids,
        "treatments": treatments,
        "organisms": sorted({treatment["organism"] for treatment in treatments}),
        "thresholds": thresholds,
        "normalization": checks,
    }



def _inline_icon(path):
    svg = path.read_text(encoding="utf-8")
    svg = svg[svg.index("<svg"):].strip()
    svg = re.sub(r'\s(?:width|height)="[^"]*"', "", svg)
    svg = re.sub(
        r"<svg\b",
        '<svg class="ui-icon" aria-hidden="true" focusable="false"',
        svg,
        count=1,
    )
    return re.sub(
        r'\b(fill|stroke)="(?!none\b)[^"]*"',
        r'\1="currentColor"',
        svg,
        flags=re.IGNORECASE,
    )


def build_report_html(payload):
    plotly_js = PLOTLY_ASSET_PATH.read_text(encoding="utf-8")
    data_json = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    html = (_REPORT_TEMPLATE
            .replace("__PLOTLY_JS__", plotly_js)
            .replace("__PAYLOAD__", data_json))
    for name, path in ICON_ASSET_PATHS.items():
        html = html.replace(f"__{name.upper()}_ICON__", _inline_icon(path))
    return html


def write_report(run_dir, output):
    run_dir = Path(run_dir)
    output_path = Path(output)
    if not output_path.is_absolute():
        output_path = run_dir / output_path
    payload = build_payload(run_dir)
    output_path.write_text(build_report_html(payload), encoding="utf-8")
    return output_path, payload


_REPORT_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DynaMIC Report</title>
<script>__PLOTLY_JS__</script>
<style>
:root {
  --bg:#FAFBFC; --surface:#FFFFFF; --surface2:#F4F5F7; --ink:#1B1C1E; --muted:#6E7176;
  --line:#E8EAEC; --hair:#F0F1F3; --grid:#F3F4F6; --accent:#1B1C1E; --button-gap:8px;
  --sans:-apple-system, BlinkMacSystemFont, "SF Pro Text", "SF Pro Display", "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  --mono:-apple-system, BlinkMacSystemFont, "SF Pro Text", "SF Pro Display", "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}
html, body { margin:0; height:100%; background:var(--bg); color:var(--ink); -webkit-font-smoothing:antialiased; overflow:hidden; }
input::placeholder { color:var(--muted); opacity:.75; }
input[type="search"]::-webkit-search-cancel-button { -webkit-appearance:none; appearance:none; }
@keyframes fade { from { opacity:0; } to { opacity:1; } }
::-webkit-scrollbar { width:10px; height:10px; }
::-webkit-scrollbar-thumb { background:#D3D6DA; border-radius:6px; border:3px solid var(--surface); }
::-webkit-scrollbar-track { background:transparent; }
.plotly-notifier { position:absolute !important; top:10px !important; right:20px !important; left:auto !important; z-index:40 !important; font-family:var(--sans) !important; }
.plotly-notifier .notifier-note { appearance:none; cursor:pointer; display:flex !important; align-items:center !important; justify-content:center !important; background:#FFFFFF !important; color:var(--muted) !important; border:1px solid var(--line) !important; border-radius:7px !important; line-height:1 !important; padding:4px !important; min-width:0 !important; max-width:none !important; box-shadow:0 8px 22px -8px rgba(28,28,26,.3) !important; }
.plotly-notifier .notifier-note .ui-icon { width:22px; height:22px; transition:color .15s, transform .15s; }
.plotly-notifier .notifier-note:hover { color:var(--ink) !important; border-color:#D3D6DA !important; }
.plotly-notifier .notifier-note:hover .ui-icon { transform:scale(1.06); }
.plotly-notifier .notifier-note:active .ui-icon { transform:scale(.94); }
.plotly-notifier .notifier-note:focus-visible { outline:2px solid var(--ink); outline-offset:2px; }
#app { height:100%; min-width:1080px; display:flex; flex-direction:column; font-family:var(--sans); font-size:13.5px; line-height:1.5; color:var(--ink); background:var(--bg); animation:fade .4s ease; }
.topbar { height:52px; flex:none; display:flex; align-items:center; gap:20px; padding:0 18px; background:var(--surface); border-bottom:1px solid var(--line); }
.brand { font-weight:700; font-size:16px; letter-spacing:-.01em; }
.tabs { display:flex; gap:2px; align-self:stretch; }
.tabbtn { appearance:none; border:0; background:none; cursor:pointer; font-family:var(--sans); font-size:13px; font-weight:590; padding:0 14px; color:var(--muted); border-bottom:2px solid transparent; margin-bottom:-1px; white-space:nowrap; letter-spacing:-.006em; }
.tabbtn:hover { color:var(--ink); }
.tabbtn.active { color:var(--ink); border-bottom-color:var(--ink); }
#search-wrap { position:relative; width:130px; transition:width .25s cubic-bezier(.4,0,.2,1); }
#search-field { display:flex; align-items:center; gap:5px; width:100%; box-sizing:border-box; background:var(--surface2); border:1px solid transparent; border-radius:7px; padding:3px 30px 3px 7px; transition:background .15s, border-color .15s; cursor:text; }
#search-field:focus-within { border-color:var(--ink); background:var(--surface); }
#search-chips { display:flex; align-items:center; gap:4px; flex:0 1 auto; max-width:calc(100% - 58px); overflow:hidden; }
.schip { display:inline-flex; align-items:center; gap:3px; background:var(--surface); border:1px solid var(--line); border-radius:999px; padding:0 3px 0 8px; font-size:11px; line-height:18px; font-weight:600; white-space:nowrap; flex:none; }
.schip-x { appearance:none; border:0; background:none; cursor:pointer; color:var(--muted); font-size:10px; line-height:1; padding:2px 3px; }
.schip-more { background:var(--surface2); border:1px solid var(--line); border-radius:999px; padding:0 7px; font-size:10.5px; line-height:18px; color:var(--muted); flex:none; }
.schip > span:first-child { max-width:104px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; display:inline-block; vertical-align:bottom; }
#global-search { flex:1 1 40px; min-width:40px; width:auto; border:0; background:none; color:var(--ink); padding:1.5px 0; font-family:var(--sans); font-size:12px; outline:none; }
.match-check { background:var(--surface2); border:1px solid var(--line); border-radius:999px; color:var(--ink); font-size:9.5px; line-height:1; padding:3px 6px; flex:none; }
#search-slash { position:absolute; right:9px; top:50%; transform:translateY(-50%); font-family:var(--mono); font-size:10px; color:var(--muted); background:var(--surface); border:1px solid var(--line); border-radius:5px; padding:1px 6px; pointer-events:none; }
#search-clear { position:absolute; right:7px; top:50%; transform:translateY(-50%); appearance:none; border:0; background:none; cursor:pointer; color:var(--ink); font-size:13px; font-weight:600; line-height:1; padding:2px 4px; transition:color .15s; }
#search-matches { position:absolute; top:36px; left:0; right:0; background:var(--surface); border:1px solid var(--line); border-radius:10px; box-shadow:0 12px 32px -12px rgba(26,26,24,.25); max-height:280px; overflow-y:auto; z-index:60; }
.match-row { display:flex; align-items:center; gap:9px; padding:7px 12px; cursor:pointer; font-size:12.5px; border-bottom:1px solid var(--hair); }
.match-row:hover { background:var(--surface2); }
#main-grid { flex:1; min-height:0; display:grid; grid-template-columns:236px minmax(0,1fr) 0px; }
.sidebar { border-right:1px solid var(--line); background:var(--bg); overflow-y:auto; padding:12px; display:flex; flex-direction:column; gap:12px; }
.scard { background:var(--surface); border:1px solid var(--line); border-radius:10px; padding:12px 14px; }
.scard-title { font-size:10px; font-weight:600; letter-spacing:.14em; text-transform:uppercase; color:var(--muted); margin-bottom:8px; }
.cls-row { padding:6px 8px 7px; margin:0 -8px; border-radius:7px; cursor:pointer; transition:opacity .15s; }
.cls-row:hover { background:#F4F5F7; }
.cls-line { display:flex; align-items:center; gap:8px; font-size:12.5px; letter-spacing:-.006em; }
.cls-dot { width:8px; height:8px; border-radius:50%; flex:none; }
.cls-count { font-family:var(--mono); font-size:11px; color:var(--muted); }
.cls-bar { height:2px; border-radius:1px; background:var(--hair); margin:6px 0 1px 16px; overflow:hidden; }
.cls-bar > div { height:100%; border-radius:1px; }
.organism-row { display:flex; align-items:center; gap:8px; padding:5px 8px 5px 10px; margin:0 -8px; border-radius:7px; cursor:pointer; font-size:12.5px; letter-spacing:-.006em; color:var(--ink); transition:background .15s, color .15s; }
.organism-count { font-family:var(--mono); font-size:11px; }
#run-card, #qcf-card { background:var(--surface); border:1px solid var(--line); border-radius:10px; overflow:hidden; padding:0; }
#run-toggle, #qcf-toggle { appearance:none; width:100%; border:0; background:none; cursor:pointer; display:flex; align-items:center; gap:8px; padding:11px 14px; font-family:var(--sans); font-size:10px; font-weight:600; letter-spacing:.14em; text-transform:uppercase; color:var(--muted); transition:color .15s; }
#run-chev, #qcf-chev { font-size:15px; line-height:1; transition:transform .2s; }
#run-body, #qcf-body { max-height:0px; opacity:0; overflow:hidden; transition:max-height .35s cubic-bezier(.4,0,.2,1), opacity .3s ease; padding:0 14px; }
.qcf-hint { color:var(--muted); font-size:11.5px; padding:2px 0 6px; }
.qcf-row { display:flex; align-items:center; justify-content:space-between; gap:10px; padding:6px 0; font-size:12.5px; border-bottom:1px solid var(--hair); }
.qcf-switch { width:28px; height:16px; border-radius:999px; background:var(--line); position:relative; cursor:pointer; flex:none; transition:background .18s; }
.qcf-switch i { position:absolute; top:2px; left:2px; width:12px; height:12px; border-radius:50%; background:#FFFFFF; box-shadow:0 1px 2px rgba(28,28,26,.25); transition:left .18s; }
.qcf-switch[data-on="1"] { background:var(--ink); }
.qcf-switch[data-on="1"] i { left:14px; }
.meta-row { display:flex; justify-content:space-between; gap:10px; padding:3px 0; font-size:12px; border-bottom:1px solid var(--hair); }
.meta-row span:first-child { color:var(--muted); }
.meta-row span:last-child { font-size:12px; font-weight:500; text-align:right; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; font-variant-numeric:tabular-nums; }
.canvas { position:relative; min-width:0; display:flex; flex-direction:column; overflow:hidden; }
.view { flex:1; min-height:0; display:flex; flex-direction:column; }
.view-head { flex:none; display:flex; align-items:center; flex-wrap:wrap; gap:var(--button-gap); padding:13px 20px 9px; }
.view-title { font-size:14.5px; font-weight:650; }
.hbtn { appearance:none; cursor:pointer; font-family:var(--sans); font-size:12px; font-weight:590; padding:6px 13px; border-radius:8px; border:1px solid var(--line); background:var(--surface); color:var(--muted); white-space:nowrap; transition:background .15s, border-color .15s, color .15s; }
.iconbtn { box-sizing:border-box; display:inline-flex; align-items:center; justify-content:center; width:31px; height:29px; padding:0; line-height:1; }
.ui-icon { display:block; width:17px; height:17px; flex:none; color:inherit; pointer-events:none; transition:transform .15s ease; }
.iconbtn:hover { color:var(--ink) !important; border-color:#D3D6DA !important; }
.iconbtn:hover .ui-icon { color:var(--ink); transform:scale(1.06); }
.iconbtn:active .ui-icon { transform:scale(.94); }
.iconbtn:focus-visible { outline:2px solid var(--ink); outline-offset:2px; }
#drawer-btn.active { background:var(--surface2); color:var(--ink); border-color:#D3D6DA; }
.mean-toggle { box-sizing:border-box; appearance:none; height:29px; display:inline-flex; align-items:center; gap:7px; padding:0 8px 0 10px; border:1px solid var(--line); border-radius:8px; background:var(--surface); color:var(--muted); cursor:pointer; font-family:var(--sans); font-size:11.5px; font-weight:600; white-space:nowrap; user-select:none; transition:background .15s, border-color .15s, color .15s; }
.mean-toggle:hover { color:var(--ink); border-color:#D3D6DA; font-weight:650; }
.mean-toggle.active { background:#F0EFFB; color:var(--ink); border-color:#C4C6EC; font-weight:650; }
.mean-toggle-track { position:relative; width:28px; height:16px; flex:none; border-radius:999px; background:var(--line); transition:background .18s; }
.mean-toggle-track i { position:absolute; top:2px; left:2px; width:12px; height:12px; border-radius:50%; background:#FFFFFF; box-shadow:0 1px 2px rgba(28,28,26,.25); transition:left .18s, transform .15s; }
.mean-toggle.active .mean-toggle-track { background:linear-gradient(90deg,#C4C6EC,#302B78); }
.mean-toggle.active .mean-toggle-track i { left:14px; }
.mean-toggle:focus-visible { outline:2px solid var(--ink); outline-offset:2px; }
.mean-toggle:active .mean-toggle-track i { transform:scale(.94); }
.button-glyph { display:block; pointer-events:none; transition:transform .15s ease; }
.xbtn:hover, .qc-close:hover, .navbtn:hover { color:var(--ink); border-color:#D3D6DA; font-weight:700; }
.xbtn:hover .button-glyph, .qc-close:hover .button-glyph, .navbtn:hover .button-glyph { transform:scale(1.06); }
.xbtn:active .button-glyph, .qc-close:active .button-glyph, .navbtn:active .button-glyph { transform:scale(.94); }
.xbtn:focus-visible, .qc-close:focus-visible, .navbtn:focus-visible { outline:2px solid var(--ink); outline-offset:2px; }
.plot-actions { display:flex; align-items:center; justify-content:flex-end; gap:var(--button-gap); flex:none; }
.plot-mode-controls { display:inline-flex; align-items:center; gap:var(--button-gap); }
.plot-mode-btn { transition:background .15s, border-color .15s, color .15s; }
.plot-mode-btn:hover { color:var(--ink); border-color:var(--ink); }
.plot-mode-btn.active { background:var(--surface2); color:var(--ink); border-color:#D3D6DA; }
[data-interaction-mode="inspect"] .nsewdrag { cursor:pointer !important; }
[data-interaction-mode="zoom"] .nsewdrag { cursor:crosshair !important; }
[data-interaction-mode="pan"] .nsewdrag { cursor:grab !important; }
[data-interaction-mode="pan"] .nsewdrag:active { cursor:grabbing !important; }
#main-grid.inspector-snapped #inspector .plot-mode-btn[data-interaction-mode="pan"] { display:none !important; }
.inspector-plot-actions { padding:2px 8px 0; }
#ov-stage { flex:1; min-height:0; position:relative; overflow:hidden; }
#ov-scatter { position:absolute; inset:0 10px 6px 10px; transition:inset .3s cubic-bezier(.4,0,.2,1); }
#resize-notch { position:absolute; right:0; top:50%; transform:translateY(-50%); width:17px; height:70px; cursor:grab; z-index:31; display:none; align-items:center; justify-content:center; gap:3px; }
#resize-notch .grip { position:relative; width:1.5px; height:22px; background:rgba(27,28,30,.18); border-radius:1px; pointer-events:none; transition:background .15s; }
#drawer { position:absolute; left:10px; right:10px; bottom:0; height:322px; background:var(--surface); border:1px solid var(--line); border-bottom:0; border-radius:12px 12px 0 0; box-shadow:0 -16px 40px -24px rgba(28,28,26,.35); transform:translateY(102%); transition:transform .3s cubic-bezier(.4,0,.2,1); display:flex; flex-direction:column; z-index:30; overflow:hidden; }
.drawer-head { flex:none; display:flex; align-items:center; gap:var(--button-gap); padding:9px 18px; border-bottom:1px solid var(--hair); }
.drawer-title { font-size:13px; font-weight:630; letter-spacing:-.006em; }
#table-count { font-family:var(--mono); font-size:10.5px; color:var(--muted); background:var(--surface2); border-radius:999px; padding:2px 9px; }
#table-search { background:var(--surface2); color:var(--ink); border:1px solid var(--line); border-radius:8px; padding:5px 26px 5px 11px; font-family:var(--sans); font-size:12px; width:190px; outline:none; box-sizing:border-box; transition:border-color .15s; }
#table-search:focus { border-color:var(--ink); }
#table-clear { position:absolute; right:5px; top:50%; transform:translateY(-50%); appearance:none; border:0; background:none; cursor:pointer; color:var(--ink); font-size:12px; font-weight:600; line-height:1; padding:2px 4px; transition:color .15s; }
.xbtn { box-sizing:border-box; appearance:none; cursor:pointer; display:inline-flex; align-items:center; justify-content:center; width:31px; height:29px; padding:0; border:1px solid var(--line); border-radius:7px; background:var(--surface); color:var(--muted); font-size:13px; font-weight:600; line-height:1; transition:background .15s, border-color .15s, color .15s; }
#table-csv .ui-icon { width:20px; height:20px; }
#table-scroll { flex:1; min-height:0; overflow:auto; }
.tgrid { display:grid; grid-template-columns:minmax(104px,1fr) minmax(72px,.7fr) minmax(84px,.7fr) minmax(104px,1fr) minmax(82px,.7fr) minmax(112px,.85fr) minmax(74px,.7fr) minmax(150px,1.2fr) minmax(112px,.9fr) minmax(150px,1.2fr) minmax(150px,1.2fr); min-width:1348px; gap:0 12px; }
#table-head { position:sticky; top:0; z-index:2; background:var(--surface); padding:7px 18px 5px; color:var(--ink); font-family:var(--mono); font-size:10px; font-weight:700; letter-spacing:.1em; text-transform:uppercase; border-bottom:1px solid var(--line); user-select:none; }
#table-head div { cursor:pointer; white-space:nowrap; }
#table-head div:hover { color:var(--ink); }
.trow { padding:5.5px 18px; font-size:12.5px; border-bottom:1px solid var(--hair); cursor:pointer; font-variant-numeric:tabular-nums; align-items:baseline; }
.tcell-treatment { font-weight:600; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.tcell-organism { color:var(--muted); font-family:var(--mono); font-size:11.5px; }
.tcell-doselayout { color:var(--muted); font-family:var(--mono); font-size:11.5px; }
.tcell-cls { display:flex; align-items:center; gap:7px; white-space:nowrap; overflow:hidden; }
.tcell-num { text-align:center; font-family:var(--mono); font-size:11.5px; white-space:nowrap; }
.pager { position:sticky; bottom:0; left:0; background:var(--surface); border-top:1px solid var(--hair); display:flex; align-items:center; gap:var(--button-gap); padding:6px 18px; }
#page-label { color:var(--muted); font-family:var(--mono); font-size:10.5px; font-variant-numeric:tabular-nums; white-space:nowrap; }
.pagebtn { appearance:none; cursor:pointer; font-family:var(--sans); font-size:11.5px; font-weight:590; padding:3px 12px; border-radius:7px; border:1px solid var(--line); background:var(--surface); color:var(--ink); white-space:nowrap; transition:background .15s, border-color .15s; }
.pagebtn:hover { background:var(--surface2); }
#ov-info { position:absolute; right:10px; bottom:10px; z-index:25; width:18px; height:18px; border-radius:50%; border:1px solid var(--line); background:var(--surface); color:var(--muted); font-size:11px; font-weight:650; font-family:var(--sans); display:flex; align-items:center; justify-content:center; cursor:default; opacity:.5; box-shadow:0 1px 4px rgba(28,28,26,.08); transition:opacity .15s, color .15s, border-color .15s, bottom .3s cubic-bezier(.4,0,.2,1); }
#ov-info:hover { opacity:1; color:var(--ink); border-color:var(--ink); }
.qc-scroll { flex:1; min-height:0; overflow-y:auto; padding:4px 20px 18px; }
.qc-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px; }
.qc-card { background:var(--surface); border:1px solid var(--line); border-radius:12px; padding:13px 16px; display:flex; flex-direction:column; }
.qc-head { display:flex; align-items:center; gap:var(--button-gap); }
.qc-title { min-width:0; font-size:13px; font-weight:650; line-height:1.25; }
.qc-hint { color:var(--muted); font-size:11.5px; margin:1px 0 6px; }
.pngbtn { appearance:none; cursor:pointer; font-family:var(--sans); font-size:11px; font-weight:600; padding:4px 10px; border-radius:7px; border:1px solid var(--line); background:var(--surface); color:var(--muted); transition:background .15s, border-color .15s, color .15s; }
.pngbtn.iconbtn { padding:0; }
.qc-close { box-sizing:border-box; appearance:none; cursor:pointer; display:inline-flex; align-items:center; justify-content:center; width:31px; height:29px; padding:0; border:1px solid var(--line); background:var(--surface); border-radius:7px; color:var(--muted); font-size:13px; font-weight:600; line-height:1; transition:background .15s, color .15s, border-color .15s; }
.qc-box { width:100%; margin:auto 0; overflow-x:hidden; }
#inspector { border-left:1px solid var(--line); background:var(--surface); overflow:hidden; min-width:0; position:relative; }
#insp-edge { position:absolute; left:0; top:0; bottom:0; width:8px; cursor:col-resize; z-index:20; display:none; }
#insp-body { width:100%; min-width:300px; height:100%; display:none; flex-direction:column; overflow-y:auto; }
.insp-head { flex:none; padding:14px 18px 10px; border-bottom:1px solid var(--hair); }
.insp-title-row { display:flex; align-items:center; gap:var(--button-gap); }
#insp-treatment { font-size:16.5px; font-weight:700; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.navbtn { box-sizing:border-box; appearance:none; cursor:pointer; border:1px solid var(--line); background:var(--surface); border-radius:7px; color:var(--muted); font-size:18px; line-height:1; width:31px; height:29px; display:flex; align-items:center; justify-content:center; padding:0; transition:background .15s, border-color .15s, color .15s; }
#insp-close { font-size:13px; font-weight:600; padding:0; }
.insp-meta { display:flex; align-items:center; gap:8px; margin-top:3px; white-space:nowrap; }
#insp-dot { width:8px; height:8px; border-radius:50%; }
#insp-cls { flex:none; font-size:12px; color:var(--muted); }
#insp-organism { min-width:0; overflow:hidden; text-overflow:ellipsis; font-family:var(--mono); font-size:11px; color:var(--muted); }
#insp-pos { flex:none; font-family:var(--mono); font-size:10.5px; color:var(--muted); }
.insp-section { font-size:10px; font-weight:600; letter-spacing:.12em; text-transform:uppercase; color:var(--muted); padding:12px 0 3px; }
.kv-row { display:flex; gap:12px; padding:6.5px 0; border-bottom:1px solid var(--hair); align-items:baseline; }
.kv-row > div:first-child { color:var(--muted); white-space:nowrap; width:138px; flex:none; font-size:12px; }
.kv-row > div:last-child { font-variant-numeric:tabular-nums; font-family:var(--mono); font-size:12px; min-width:0; white-space:nowrap; }
.kv-sub { color:var(--muted); font-size:10.5px; }
.iv-sub { color:var(--muted); font-size:10.5px; }
.qc-dot { width:7px; height:7px; border-radius:50%; display:inline-block; margin-left:8px; vertical-align:1px; cursor:default; }
.kv-dotslot { display:inline-block; width:15px; }
.kv-dotslot .qc-dot { margin-left:0; }
#insp-multi { overflow-x:auto; padding:6px 14px 16px; display:none; }
.cgrid { display:grid; grid-template-columns:110px repeat(6,minmax(64px,1fr)); gap:0 10px; }
.chead { padding:6px 4px; color:var(--ink); font-family:var(--mono); font-size:9.5px; font-weight:700; letter-spacing:.08em; text-transform:uppercase; border-bottom:1px solid var(--line); }
.crow { padding:6.5px 4px; font-size:11.5px; border-bottom:1px solid var(--hair); font-variant-numeric:tabular-nums; align-items:baseline; }
.crow .cnum { text-align:right; font-family:var(--mono); font-size:11px; white-space:nowrap; }
.right { text-align:right; }
</style>
</head>
<body>
<div id="app">
  <div class="topbar">
    <span class="brand">DynaMIC</span>
    <div class="tabs">
      <button class="tabbtn active" data-tab="overview">Overview</button>
      <button class="tabbtn" data-tab="qc">Normalization QC</button>
    </div>
    <div style="flex:1;"></div>
    <div id="search-wrap">
      <div id="search-field">
        <div id="search-chips" style="display:none;"></div>
        <input id="global-search" type="search" placeholder="Search…" autocomplete="off">
      </div>
      <span id="search-slash">/</span>
      <button id="search-clear" data-tip="Clear search" aria-label="Clear search" style="display:none;">✕</button>
      <div id="search-matches" style="display:none;"></div>
    </div>
  </div>
  <div id="main-grid">
    <div class="sidebar">
      <div class="scard" data-overview-only><div class="scard-title">Classes</div><div id="classes-list"></div></div>
      <div class="scard"><div class="scard-title">Organisms</div><div id="organisms-list"></div></div>
      <div id="qcf-card" data-overview-only>
        <button id="qcf-toggle">QC Labels<span style="flex:1;"></span><span id="qcf-chev">▾</span></button>
        <div id="qcf-body">
          <div class="qcf-hint">Exclude treatments with resolution limited fit estimates.</div>
          <div class="qcf-row"><span>Continuous MIC</span><div class="qcf-switch" data-kind="mic" data-on="0"><i></i></div></div>
          <div class="qcf-row"><span>Slope</span><div class="qcf-switch" data-kind="slope" data-on="0"><i></i></div></div>
          <div class="qcf-row"><span>Midpoint</span><div class="qcf-switch" data-kind="midpoint" data-on="0"><i></i></div></div>
          <div style="height:10px;"></div>
        </div>
      </div>
      <div id="run-card" data-overview-only>
        <button id="run-toggle">Run Information<span style="flex:1;"></span><span id="run-chev">▾</span></button>
        <div id="run-body"><div id="run-items"></div><div style="height:10px;"></div></div>
      </div>
    </div>
    <div class="canvas">
      <div id="view-overview" class="view">
        <div class="view-head">
          <div style="flex:1;"></div>
          <button id="mean-colour-toggle" class="mean-toggle" type="button" role="switch" aria-checked="false" aria-label="Colour points by Mean Inhibition" data-tip="Colour points by Mean Inhibition">
            <span>Mean Inhibition</span>
            <span class="mean-toggle-track"><i></i></span>
          </button>
          <button id="drawer-btn" class="hbtn iconbtn" data-tip="Show treatment table" aria-label="Show treatment table" aria-expanded="false">__TABLE_ICON__</button>
          <div class="plot-actions">
            <div class="plot-mode-controls" data-mode-controls="ov-scatter">
              <button class="hbtn iconbtn plot-mode-btn" data-mode-plot="ov-scatter" data-interaction-mode="zoom" data-tip="Select area to zoom" aria-label="Select area to zoom" aria-pressed="false">__SELECT_ICON__</button>
              <button class="hbtn iconbtn plot-mode-btn" data-mode-plot="ov-scatter" data-interaction-mode="pan" data-tip="Pan plot" aria-label="Pan plot" aria-pressed="false" style="display:none;">__MOVE_ICON__</button>
            </div>
            <button class="hbtn iconbtn" data-plot="ov-scatter" data-name="effect-vs-evidence" data-tip="Save overview plot as PNG" aria-label="Save overview plot as PNG">__SAVE_FIGURE_ICON__</button>
          </div>
        </div>
        <div id="ov-stage">
          <div id="ov-scatter"></div>
          <div id="resize-notch">
            <svg width="17" height="70" viewBox="0 0 17 70" style="position:absolute; inset:0; pointer-events:none; overflow:visible;"><path d="M17 0 C17 6 13 9 10 10 C4 12 0 16 0 24 L0 46 C0 54 4 58 10 60 C13 61 17 64 17 70 Z" fill="#FFFFFF" stroke="#E8EAEC" stroke-width="1" style="transition:fill .15s;"></path></svg>
            <span class="grip"></span>
            <span class="grip"></span>
          </div>
          <div id="ov-info" data-tip="Hold Ctrl/⌘ + Click<br>to select multiple data points" data-tip-above="1">i</div>
          <div id="drawer">
            <div class="drawer-head">
              <span class="drawer-title">Treatment table</span>
              <span id="table-count"></span>
              <div style="flex:1;"></div>
              <div style="position:relative;">
                <input id="table-search" type="search" placeholder="Search…" autocomplete="off">
                <button id="table-clear" data-tip="Clear filter" aria-label="Clear filter" style="display:none;">✕</button>
              </div>
              <button id="table-csv" class="pngbtn iconbtn" data-tip="Download the filtered table as CSV" aria-label="Download the filtered table as CSV">__DOWNLOAD_ICON__</button>
              <button id="drawer-close" class="xbtn" data-tip="Close table" aria-label="Close table"><span class="button-glyph">✕</span></button>
            </div>
            <div id="table-scroll">
              <div id="table-head" class="tgrid"></div>
              <div id="table-rows"></div>
              <div class="pager">
                <span id="page-label"></span>
                <div style="flex:1;"></div>
                <button id="prev-btn" class="pagebtn">← Prev</button>
                <button id="next-btn" class="pagebtn">Next →</button>
              </div>
            </div>
          </div>
        </div>
      </div>
      <div id="view-qc" class="view" style="display:none;">
        <div class="view-head">
          <div class="view-title" id="qc-title"></div>
          <div style="flex:1;"></div>
        </div>
        <div class="qc-scroll">
          <div class="qc-grid">
            <div class="qc-card">
              <div class="qc-head">
                <div class="qc-title" id="rc-within-title"></div>
                <div style="flex:1;"></div>
                <div class="plot-actions">
                  <div class="plot-mode-controls" data-mode-controls="rc-within" style="display:none;">
                    <button class="pngbtn iconbtn plot-mode-btn" data-mode-plot="rc-within" data-interaction-mode="zoom" data-tip="Select area to zoom" aria-label="Select area to zoom" aria-pressed="false">__SELECT_ICON__</button>
                    <button class="pngbtn iconbtn plot-mode-btn" data-mode-plot="rc-within" data-interaction-mode="pan" data-tip="Pan plot" aria-label="Pan plot" aria-pressed="false" style="display:none;">__MOVE_ICON__</button>
                  </div>
                  <button class="pngbtn iconbtn" data-plot="rc-within" data-name="within-assay" data-tip="Save within-assay control plot as PNG" aria-label="Save within-assay control plot as PNG">__SAVE_FIGURE_ICON__</button>
                  <button id="rc-within-close" class="qc-close" data-tip="Back to all assays" aria-label="Back to all assays" style="display:none;"><span class="button-glyph">✕</span></button>
                </div>
              </div>
              <div class="qc-hint" id="rc-within-hint"></div>
              <div class="qc-box" id="rc-within-box" style="height:340px;"><div id="rc-within" style="width:100%;"></div></div>
            </div>
            <div class="qc-card">
              <div class="qc-head">
                <div class="qc-title">Across-Assay Control Outliers</div>
                <div style="flex:1;"></div>
                <div class="plot-actions">
                  <div class="plot-mode-controls" data-mode-controls="rc-across">
                    <button class="pngbtn iconbtn plot-mode-btn" data-mode-plot="rc-across" data-interaction-mode="zoom" data-tip="Select area to zoom" aria-label="Select area to zoom" aria-pressed="false">__SELECT_ICON__</button>
                    <button class="pngbtn iconbtn plot-mode-btn" data-mode-plot="rc-across" data-interaction-mode="pan" data-tip="Pan plot" aria-label="Pan plot" aria-pressed="false" style="display:none;">__MOVE_ICON__</button>
                  </div>
                  <button class="pngbtn iconbtn" data-plot="rc-across" data-name="across-assay" data-tip="Save across-assay control plot as PNG" aria-label="Save across-assay control plot as PNG">__SAVE_FIGURE_ICON__</button>
                </div>
              </div>
              <div class="qc-hint"></div>
              <div class="qc-box" style="height:300px; overflow-y:hidden;"><div id="rc-across" style="width:100%; height:100%;"></div></div>
            </div>
            <div class="qc-card">
              <div class="qc-head">
                <div class="qc-title">Raw Inhibition Fractions</div>
                <div style="flex:1;"></div>
                <div class="plot-actions">
                  <div class="plot-mode-controls" data-mode-controls="rc-raw">
                    <button class="pngbtn iconbtn plot-mode-btn" data-mode-plot="rc-raw" data-interaction-mode="zoom" data-tip="Select area to zoom" aria-label="Select area to zoom" aria-pressed="false">__SELECT_ICON__</button>
                    <button class="pngbtn iconbtn plot-mode-btn" data-mode-plot="rc-raw" data-interaction-mode="pan" data-tip="Pan plot" aria-label="Pan plot" aria-pressed="false" style="display:none;">__MOVE_ICON__</button>
                  </div>
                  <button class="pngbtn iconbtn" data-plot="rc-raw" data-name="raw-inhibition" data-tip="Save raw inhibition plot as PNG" aria-label="Save raw inhibition plot as PNG">__SAVE_FIGURE_ICON__</button>
                </div>
              </div>
              <div class="qc-hint"></div>
              <div class="qc-box" style="height:300px; overflow-y:hidden;"><div id="rc-raw" style="width:100%; height:100%;"></div></div>
            </div>
            <div class="qc-card">
              <div class="qc-head">
                <div class="qc-title">Inlier Control Measurements</div>
                <div style="flex:1;"></div>
                <div class="plot-actions">
                  <div class="plot-mode-controls" data-mode-controls="rc-inlier">
                    <button class="pngbtn iconbtn plot-mode-btn" data-mode-plot="rc-inlier" data-interaction-mode="zoom" data-tip="Select area to zoom" aria-label="Select area to zoom" aria-pressed="false">__SELECT_ICON__</button>
                    <button class="pngbtn iconbtn plot-mode-btn" data-mode-plot="rc-inlier" data-interaction-mode="pan" data-tip="Pan plot" aria-label="Pan plot" aria-pressed="false" style="display:none;">__MOVE_ICON__</button>
                  </div>
                  <button class="pngbtn iconbtn" data-plot="rc-inlier" data-name="inlier-controls" data-tip="Save inlier control plot as PNG" aria-label="Save inlier control plot as PNG">__SAVE_FIGURE_ICON__</button>
                </div>
              </div>
              <div class="qc-hint"></div>
              <div class="qc-box" style="height:300px; overflow-y:hidden;"><div id="rc-inlier" style="width:100%; height:100%;"></div></div>
            </div>
          </div>
        </div>
      </div>
    </div>
    <div id="inspector">
      <div id="insp-edge"></div>
      <div id="insp-body">
        <div class="insp-head">
          <div class="insp-title-row">
            <div id="insp-treatment"></div>
            <div style="flex:1;"></div>
            <button id="insp-prev" class="navbtn" data-tip="Previous treatment (←)" aria-label="Previous treatment"><span class="button-glyph">‹</span></button>
            <button id="insp-next" class="navbtn" data-tip="Next treatment (→)" aria-label="Next treatment"><span class="button-glyph">›</span></button>
            <button class="pngbtn iconbtn" data-plot="ov-curve" data-name="inhibition-curve" data-tip="Save inhibition curve as PNG" aria-label="Save inhibition curve as PNG">__SAVE_FIGURE_ICON__</button>
            <button id="insp-close" class="navbtn" data-tip="Close (Esc)" aria-label="Close dose-response panel"><span class="button-glyph">✕</span></button>
          </div>
          <div class="insp-meta">
            <i id="insp-dot"></i>
            <span id="insp-cls"></span>
            <span id="insp-organism"></span>
            <span style="flex:1;"></span>
            <span id="insp-pos"></span>
          </div>
        </div>
        <div style="flex:none; padding:10px 10px 0;">
          <div id="ov-curve" style="width:100%; height:250px;"></div>
          <div class="plot-actions inspector-plot-actions">
            <div class="plot-mode-controls" data-mode-controls="ov-curve">
              <button class="pngbtn iconbtn plot-mode-btn" data-mode-plot="ov-curve" data-interaction-mode="zoom" data-tip="Select area to zoom" aria-label="Select area to zoom" aria-pressed="false">__SELECT_ICON__</button>
              <button class="pngbtn iconbtn plot-mode-btn" data-mode-plot="ov-curve" data-interaction-mode="pan" data-tip="Pan plot" aria-label="Pan plot" aria-pressed="false" style="display:none;">__MOVE_ICON__</button>
            </div>
          </div>
        </div>
        <div id="insp-single" style="padding:2px 18px 20px;">
          <div class="insp-section">Evidence</div>
          <div id="evidence-rows"></div>
          <div id="fit-section">
            <div class="insp-section" style="padding-top:18px;">Fit parameters</div>
            <div id="fit-rows"></div>
          </div>
        </div>
        <div id="insp-multi">
          <div style="min-width:520px;">
            <div class="cgrid chead"><div>Treatment</div><div class="right">q</div><div class="right">Δ change</div><div class="right">Cont. MIC</div><div class="right">Disc. MIC</div><div class="right">Slope</div><div class="right">Midpoint</div></div>
            <div id="compare-rows"></div>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>
<script>
const DATA = __PAYLOAD__;
(function () {
"use strict";
const ZOOM_OUT_ICON = `__ZOOM_OUT_ICON__`;
const C = Object.freeze({
  SIDEBAR: 236, INSP_W: 470, MIN_W: 300, PLOT_MIN: 340, PER: 50, MARKER: 10,
  FLAG: "#C0392B", SNAP_OPEN: 40, SNAP_CLOSE: 80, CLOSE_AT: 235, CURVE_N: 140, BAND_N: 100,
  CMP: ["#0072B2", "#D55E00", "#2E7D32", "#7B1FA2", "#C2185B", "#00838F", "#5D4037", "#455A64"],
  MEAN_COLOURS: [[0, "#D5D8EC"], [0.25, "#B9BCE7"], [0.5, "#8F8BE8"], [0.75, "#655FCD"], [1, "#302B78"]],
  SHORT: ["variant", "inverse", "invariant"],
  OV_IDS: ["ov-scatter", "ov-curve"], QC_IDS: ["rc-within", "rc-across", "rc-raw", "rc-inlier"],
});
const D = DATA;
D.classes = D.classes.map(c => (c && c.length) ? c.charAt(0).toUpperCase() + c.slice(1) : c);
const THR = D.thresholds;
const QC_ORGANISMS = Object.keys(D.normalization).sort();
const ORGANISM_TREATMENT_COUNTS = D.treatments.reduce((counts, u) => {
  counts[u.organism] = (counts[u.organism] || 0) + 1;
  return counts;
}, {});
const DEFAULT_QC_ORGANISM = QC_ORGANISMS.reduce((best, organism) =>
  !best || ORGANISM_TREATMENT_COUNTS[organism] > (ORGANISM_TREATMENT_COUNTS[best] || 0) ? organism : best, "");
const state = {
  view: "overview", organism: "", qcOrganism: DEFAULT_QC_ORGANISM, cls: [true, true, true],
  q: "", showMatches: false, search: "", sortKey: "q", sortDesc: true, page: 0, meanColour: false,
  sel: null, selSet: [], drawer: false, runOpen: false, qcZoomAssay: null,
  qcOpen: false, qcFilters: { mic: false, slope: false, midpoint: false },
};
const el = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"]/g, ch => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[ch]));
let orderCache = null;
let chartSig = null, layoutSig, lastView, inspW = C.INSP_W;
let animRaf = null, fastAnim = false, resizePending = false, resizeQueued = false;

function fmt(v, sig) {
  if (v === null || v === undefined || Number.isNaN(v)) return "NA";
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a >= 1e4 || a < 1e-3) return v.toExponential(2);
  return String(parseFloat(v.toPrecision(sig || 4)));
}
const fmtConcentration = v => String(parseFloat(v.toPrecision(6)));
function interval(t, sig) {
  if (!t || t[0] === null) return { v: "NA", sub: "" };
  const sub = (t[1] !== null || t[2] !== null) ? `[${fmt(t[1], sig)} – ${fmt(t[2], sig)}]` : "";
  return { v: fmt(t[0], sig), sub };
}
function ivText(t, sig) { const r = interval(t, sig); return r.sub ? r.v + " " + r.sub : r.v; }
function dcText(dc) {
  if (dc[0] === null) return "NA";
  const sub = (dc[1] !== null || dc[2] !== null) ? ` [${dc[1] ?? "NA"} – ${dc[2] ?? "NA"}]` : "";
  return dc[0] + sub;
}
function ivHtml(t, sig) {
  const r = interval(t, sig);
  return r.sub ? esc(r.v) + ' <span class="iv-sub">' + esc(r.sub) + "</span>" : esc(r.v);
}
function dcHtml(dc) {
  if (dc[0] === null) return "NA";
  const sub = (dc[1] !== null || dc[2] !== null) ? `[${dc[1] ?? "NA"} – ${dc[2] ?? "NA"}]` : "";
  return esc(dc[0]) + (sub ? ' <span class="iv-sub">' + esc(sub) + "</span>" : "");
}
function hexToRgba(hex, a) {
  const h = hex.replace("#", "");
  const n = parseInt(h.length === 3 ? h.split("").map(c => c + c).join("") : h, 16);
  return "rgba(" + ((n >> 16) & 255) + "," + ((n >> 8) & 255) + "," + (n & 255) + "," + a + ")";
}
const QC_NOTES = {
  mic: { measured: "MIC lands within the tested concentration range",
              range_limited: "MIC lies beyond the tested concentrations", NA: "not applicable" },
  midpoint: { measured: "midpoint lies within the tested concentration range",
            range_limited: "midpoint lies beyond the tested concentrations" },
  slope: { measured: "slope is inside the resolvable range",
               resolution_limited: "slope fell outside the measurable range" },
};
function qcVisible(u, filters) {
  const active = filters || state.qcFilters;
  const failing = kind => active[kind] && u.qc && u.qc[kind] && u.qc[kind] !== "measured";
  return !(failing("mic") || failing("slope") || failing("midpoint"));
}
function qcDot(u, kind) {
  const flag = u.qc ? u.qc[kind] : null;
  if (!flag) return "";
  const note = (QC_NOTES[kind] || {})[flag] || "";
  const label = (kind === "mic" ? "mic_qc" : kind + "_qc") + " · " + flag + (note ? " — " + note : "");
  return `<i class="qc-dot" data-tip="${esc(label)}" style="background:${flag === "measured" ? "#65C083" : "#C0392B"};"></i>`;
}
const cssVar = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const groupKey = u => u.organism + "\u0000" + (u.doseLayout ?? "");
const scatterY = u => Math.max(u.score / THR[groupKey(u)], 0.03);
const stratumGrouped = () => state.sortKey === "q";
const jitter = i => { const s = Math.sin(i * 127.1) * 43758.5453; return (s - Math.floor(s)) - 0.5; };

function baseLayout(o) {
  return Object.assign({
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: cssVar("--ink"), family: cssVar("--sans"), size: 12 },
    hovermode: "closest", hoverdistance: 14,
    margin: { l: 54, r: 16, t: 12, b: 44 },
    legend: { orientation: "h", y: 1.05, x: 0, font: { size: 12.5, color: cssVar("--muted") } },
  }, o || {});
}
function qcLayout(o) {
  return baseLayout(Object.assign({ legend: bottomLegend(), margin: { l: 54, r: 16, t: 12, b: 82 } }, o));
}
function ax(o) {
  return Object.assign({
    gridcolor: cssVar("--grid"), zerolinecolor: cssVar("--grid"), linecolor: "#D9DBDE",
    ticks: "outside", ticklen: 4, tickcolor: "rgba(0,0,0,0)",
    tickfont: { family: cssVar("--mono"), size: 12, color: cssVar("--muted") },
  }, o || {});
}
function axTitle(text, size) {
  return { text: "<b>" + text + "</b>", font: { family: cssVar("--sans"), size: size || 12, color: cssVar("--ink") }, standoff: 10 };
}
function bottomLegend() {
  return { orientation: "h", x: 0.5, xanchor: "center", yref: "container", y: 0.005, yanchor: "bottom",
           font: { size: 12.5, color: cssVar("--muted"), family: cssVar("--sans") } };
}
const plotConfig = () => ({ displayModeBar: false, responsive: true, showTips: false, doubleClick: "reset+autosize" });

const interactionDragMode = mode => mode === "zoom" ? "zoom" : (mode === "pan" ? "pan" : false);
function syncPlotInteraction(plot) {
  if (!plot) return;
  const enabled = plot._interactionEnabled !== false;
  const mode = plot._interactionMode || "inspect";
  plot.dataset.interactionMode = mode;
  const controls = document.querySelector('[data-mode-controls="' + plot.id + '"]');
  if (!controls) return;
  controls.style.display = enabled ? "inline-flex" : "none";
  controls.querySelectorAll(".plot-mode-btn").forEach(btn => {
    const active = btn.dataset.interactionMode === mode;
    btn.classList.toggle("active", active);
    btn.setAttribute("aria-pressed", active ? "true" : "false");
    if (btn.dataset.interactionMode === "pan") btn.style.display = plot._viewIsZoomed ? "" : "none";
  });
}
function setPlotInteractionMode(plot, mode) {
  if (!plot) return;
  if (!["inspect", "zoom", "pan"].includes(mode)) mode = "inspect";
  if (mode === "pan" && !plot._viewIsZoomed) mode = "inspect";
  plot._interactionMode = mode;
  syncPlotInteraction(plot);
  if (plot._fullLayout && window.Plotly)
    Plotly.relayout(plot, { dragmode: interactionDragMode(mode) }).catch(() => {});
}
function interactionLayout(plot, layout, revision) {
  if (plot._interactionRevision !== revision) {
    dismissNotifier(plot);
    plot._interactionRevision = revision;
    plot._interactionMode = "inspect";
    plot._viewIsZoomed = false;
  }
  plot._interactionEnabled = true;
  layout.uirevision = revision;
  layout.dragmode = interactionDragMode(plot._interactionMode || "inspect");
  syncPlotInteraction(plot);
  return layout;
}
function disablePlotInteraction(plot) {
  if (!plot) return;
  dismissNotifier(plot);
  plot._interactionEnabled = false;
  plot._interactionMode = "inspect";
  plot._viewIsZoomed = false;
  plot.dataset.interactionMode = "inspect";
  plot._zoomReset = null;
  if (plot.removeAllListeners) {
    plot.removeAllListeners("plotly_relayout");
    plot.removeAllListeners("plotly_doubleclick");
  }
  syncPlotInteraction(plot);
}

function makeTip(host) {
  let tip = host.querySelector(":scope > .ui-tip");
  if (!tip) {
    tip = document.createElement("div");
    tip.className = "ui-tip";
    Object.assign(tip.style, { position: "absolute", zIndex: "70", pointerEvents: "none",
      background: "#FFFFFF", color: cssVar("--ink"), border: "1px solid " + cssVar("--line"),
      fontSize: "11.5px", fontFamily: cssVar("--sans"), lineHeight: "1.5",
      padding: "5px 10px", borderRadius: "7px", whiteSpace: "nowrap",
      boxShadow: "0 8px 22px -8px rgba(28,28,26,.3)", opacity: "0", transition: "opacity .12s",
      top: "0px", left: "0px" });
    if (getComputedStyle(host).position === "static") host.style.position = "relative";
    host.appendChild(tip);
  }
  return tip;
}
function showTip(host, html, x, y) {
  const tip = makeTip(host);
  tip.innerHTML = html;
  tip.style.whiteSpace = "nowrap";
  tip.style.maxWidth = "";
  requestAnimationFrame(() => {
    let tw = tip.offsetWidth, th = tip.offsetHeight;
    const avail = host.clientWidth - 8;
    if (tw > avail && avail > 80) {
      tip.style.whiteSpace = "normal";
      tip.style.maxWidth = avail + "px";
      tw = tip.offsetWidth; th = tip.offsetHeight;
    }
    let nx = x, ny = Math.max(2, y);
    if (nx + tw > host.clientWidth - 4) nx = Math.max(2, host.clientWidth - tw - 4);
    if (ny + th > host.clientHeight - 2) ny = Math.max(2, host.clientHeight - th - 2);
    tip.style.left = nx + "px"; tip.style.top = ny + "px";
    tip.style.opacity = "1";
  });
}
function hideTip(host) {
  const tip = host.querySelector(":scope > .ui-tip");
  if (tip) tip.style.opacity = "0";
}
function bindCustomHover(plot, textFn) {
  plot._tipTextFn = textFn;
  if (!plot.on) return;
  if (plot.removeAllListeners) { plot.removeAllListeners("plotly_hover"); plot.removeAllListeners("plotly_unhover"); }
  if (!plot._mouseTracked) {
    plot._mouseTracked = true;
    plot.addEventListener("mousemove", e => {
      const bb = plot.getBoundingClientRect();
      const s = plot.offsetWidth ? bb.width / plot.offsetWidth : 1;
      plot._mx = (e.clientX - bb.left) / (s || 1);
      plot._my = (e.clientY - bb.top) / (s || 1);
    });
  }
  plot.on("plotly_hover", ev => {
    const pt = ev.points && ev.points.length ? ev.points[0] : null;
    if (!pt || !plot._tipTextFn) return;
    const html = plot._tipTextFn(pt);
    if (!html) return;
    let x, y;
    try {
      x = pt.xaxis._offset + pt.xaxis.d2p(pt.x);
      y = pt.yaxis._offset + pt.yaxis.d2p(pt.y);
    } catch (err) { return; }
    if (!isFinite(x) || !isFinite(y)) return;
    if (plot._mx !== undefined && Math.hypot(x - plot._mx, y - plot._my) > 42) return;
    showTip(plot, html, x + 16, y + 16);
  });
  plot.on("plotly_unhover", () => hideTip(plot));
}

function tableOrder() {
  const order = filteredOrder();
  if (state.selSet.length <= 1) return order;
  const selected = new Set(state.selSet);
  const inView = order.filter(i => selected.has(i));
  const extra = state.selSet.filter(i => order.indexOf(i) < 0);
  return inView.concat(extra);
}

function filteredOrder() {
  if (orderCache) return orderCache;
  const q = state.search.trim().toLowerCase();
  const out = [];
  D.treatments.forEach((u, i) => {
    if (!state.cls[u.cls]) return;
    if (state.organism && u.organism !== state.organism) return;
    if (!qcVisible(u)) return;
    if (q && u.treatment.toLowerCase().indexOf(q) < 0 && u.organism.toLowerCase().indexOf(q) < 0) return;
    out.push(i);
  });
  const desc = state.sortDesc ? -1 : 1;
  const sortValue = (u, key) => {
    switch (key) {
      case "treatment": return u.treatment;
      case "organism": return u.organism;
      case "cls": return u.cls;
      case "cc": return u.cc[0];
      case "dc": return u.dc[0];
      case "slope": return u.fit ? u.fit.slope[0] : null;
      case "midpoint": return u.fit ? u.fit.midpoint[0] : null;
      default: return u[key];
    }
  };
  const groupByStratum = stratumGrouped();
  const cmpValue = (a, b) => {
    const va = sortValue(D.treatments[a], state.sortKey), vb = sortValue(D.treatments[b], state.sortKey);
    if (va === null || va === undefined) return 1;
    if (vb === null || vb === undefined) return -1;
    if (typeof va === "string") return va.localeCompare(vb) * -desc;
    return (va - vb) * desc;
  };
  out.sort((a, b) => {
    if (groupByStratum) {
      const ga = groupKey(D.treatments[a]), gb = groupKey(D.treatments[b]);
      if (ga !== gb) return ga < gb ? -1 : 1;
    }
    return cmpValue(a, b) || (a - b);
  });
  orderCache = out;
  return out;
}

function renderSearchChips() {
  const box = el("search-chips");
  if (!state.selSet.length) { box.style.display = "none"; box.innerHTML = ""; return; }
  box.style.display = "flex";
  box.innerHTML = state.selSet.map(i =>
    `<span class="schip"><span>${esc(D.treatments[i].treatment)}</span><button class="schip-x" data-i="${i}" aria-label="Deselect">✕</button></span>`).join("");
  const fits = () => box.scrollWidth <= box.clientWidth + 1;
  if (fits()) return;
  const more = document.createElement("span");
  more.className = "schip-more";
  box.appendChild(more);
  let hidden = 0;
  let chips = box.querySelectorAll(".schip");
  while (!fits() && chips.length > 1) {
    chips[chips.length - 1].remove();
    hidden += 1;
    more.textContent = "+" + hidden;
    chips = box.querySelectorAll(".schip");
  }
}

const R = {
  tabs() {
    const inQc = state.view === "qc";
    document.querySelectorAll(".tabbtn").forEach(b => {
      b.classList.toggle("active", b.dataset.tab === state.view);
      b.style.color = ""; b.style.borderColor = ""; b.style.fontWeight = "";
    });
    document.querySelectorAll("[data-overview-only]").forEach(card => { card.style.display = inQc ? "none" : ""; });
    el("search-wrap").style.display = inQc ? "none" : "";
    el("view-overview").style.display = inQc ? "none" : "flex";
    el("view-qc").style.display = inQc ? "flex" : "none";
  },
  meanToggle() {
    el("mean-colour-toggle").classList.toggle("active", state.meanColour);
    el("mean-colour-toggle").setAttribute("aria-checked", state.meanColour ? "true" : "false");
  },
  classes() {
    const counts = D.classes.map((c, i) => D.treatments.reduce((n, u) => n + (u.cls === i && (!state.organism || u.organism === state.organism) && qcVisible(u) ? 1 : 0), 0));
    const maxCount = Math.max(...counts, 1);
    el("classes-list").innerHTML = D.classes.map((c, i) => {
      const dot = D.classColours[i];
      const w = Math.max(2, Math.round(100 * counts[i] / maxCount));
      return `<div class="cls-row" data-idx="${i}" data-tip="Click to show / hide — double-click to solo" style="opacity:${state.cls[i] ? "1" : ".38"};">
        <div class="cls-line"><i class="cls-dot" style="background:${dot}; box-shadow:0 0 0 3px ${hexToRgba(dot, 0.14)};"></i>
        <span style="white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">${esc(c)}</span><span style="flex:1;"></span>
        <span class="cls-count">${counts[i].toLocaleString()}</span></div>
        <div class="cls-bar"><div style="width:${w}%; background:${dot};"></div></div></div>`;
    }).join("");
  },
  organisms() {
    const inQc = state.view === "qc";
    const organismCounts = {};
    let visibleTotal = 0;
    D.treatments.forEach(u => {
      if (!inQc && !qcVisible(u)) return;
      visibleTotal += 1;
      organismCounts[u.organism] = (organismCounts[u.organism] || 0) + 1;
    });
    const rows = (inQc ? QC_ORGANISMS : D.organisms).map(p => ({ id: p, label: p, count: organismCounts[p] || 0 }));
    if (!inQc) rows.unshift({ id: "", label: "All organisms", count: visibleTotal });
    const selected = inQc ? state.qcOrganism : state.organism;
    el("organisms-list").innerHTML = rows.map(p => {
      const on = selected === p.id;
      return `<div class="organism-row" data-organism="${esc(p.id)}" data-full="${esc(p.label)}" data-on="${on ? "1" : "0"}" style="background:${on ? "#EBEDF0" : "transparent"}; font-weight:${on ? "600" : "400"};">
        <span class="organism-name" style="white-space:nowrap; overflow:hidden; text-overflow:ellipsis; min-width:0;">${esc(p.label)}</span><span style="flex:1;"></span>
        <span class="organism-count" style="color:${on ? "var(--ink)" : "var(--muted)"};">${p.count.toLocaleString()}</span></div>`;
    }).join("");
    el("organisms-list").querySelectorAll(".organism-row").forEach(row => {
      const name = row.querySelector(".organism-name");
      if (name && name.scrollWidth > name.clientWidth + 1) row.dataset.tip = row.dataset.full;
      else row.removeAttribute("data-tip");
    });
  },
  qcCard() {
    el("qcf-chev").style.transform = "rotate(" + (state.qcOpen ? "180deg" : "0deg") + ")";
    el("qcf-body").style.maxHeight = state.qcOpen ? "190px" : "0px";
    el("qcf-body").style.opacity = state.qcOpen ? "1" : "0";
    document.querySelectorAll("#qcf-body .qcf-switch").forEach(sw => { sw.dataset.on = state.qcFilters[sw.dataset.kind] ? "1" : "0"; });
  },
  runInfo() {
    el("run-chev").style.transform = "rotate(" + (state.runOpen ? "180deg" : "0deg") + ")";
    el("run-body").style.maxHeight = state.runOpen ? "230px" : "0px";
    el("run-body").style.opacity = state.runOpen ? "1" : "0";
  },
  search() {
    const gq = state.q.trim().toLowerCase();
    const hasChips = state.selSet.length > 0;
    el("search-slash").style.display = (state.q.length === 0 && !hasChips) ? "" : "none";
    el("search-clear").style.display = state.q.length > 0 ? "" : "none";
    if (state.q || hasChips) el("search-wrap").style.width = "min(280px, 26vw)";
    else if (document.activeElement !== el("global-search")) el("search-wrap").style.width = "130px";
    renderSearchChips();
    let hits = [];
    if (state.showMatches && gq) {
      for (let i = 0; i < D.treatments.length && hits.length < 40; i++) {
        const u = D.treatments[i];
        if (qcVisible(u) && u.treatment.toLowerCase().indexOf(gq) >= 0)
          hits.push({ i, treatment: u.treatment, organism: u.organism, color: D.classColours[u.cls], y: "−log₁₀ " + fmt(u.y, 3), on: state.selSet.indexOf(i) >= 0 });
      }
    }
    const box = el("search-matches");
    box.style.display = hits.length ? "block" : "none";
    box.innerHTML = hits.map(mt =>
      `<div class="match-row" data-i="${mt.i}"><i style="width:7px; height:7px; border-radius:50%; background:${mt.color}; flex:none;"></i>
       <b style="font-weight:600;">${esc(mt.treatment)}</b><span style="color:var(--muted); font-family:var(--mono); font-size:11px;">${esc(mt.organism)}</span>
       <span style="flex:1;"></span>${mt.on ? '<span class="match-check">✓</span>' : ""}<span style="color:var(--muted); font-family:var(--mono); font-size:11px;">${mt.y}</span></div>`).join("");
  },
  table() {
    const order = tableOrder();
    const maxPage = Math.max(0, Math.ceil(order.length / C.PER) - 1);
    const page = Math.min(state.page, maxPage);
    const headers = [
      { key: "treatment", label: "Treatment", align: "left" }, { key: "organism", label: "Organism", align: "left" },
      { key: "doseLayout", label: "Dose layout", align: "left" },
      { key: "cls", label: "Class", align: "left" }, { key: "x", label: "Δ inhibition", align: "center" },
      { key: "meanInhibition", label: "Mean Inhibition", align: "center" },
      { key: "q", label: "q", align: "center" },
      { key: "cc", label: "Cont. MIC", align: "center" }, { key: "dc", label: "Disc. MIC", align: "center" },
      { key: "slope", label: "Slope", align: "center" }, { key: "midpoint", label: "Midpoint", align: "center" },
    ];
    el("table-head").innerHTML = headers.map(h =>
      `<div data-key="${h.key}" style="text-align:${h.align};">${h.label} ${state.sortKey === h.key ? (state.sortDesc ? "↓" : "↑") : ""}</div>`).join("");
    const restricted = state.selSet.length > 1;
    el("table-rows").innerHTML = order.slice(page * C.PER, page * C.PER + C.PER).map(i => {
      const u = D.treatments[i];
      const on = !restricted && state.selSet.indexOf(i) >= 0;
      return `<div class="tgrid trow" data-i="${i}" data-on="${on ? "1" : "0"}" style="background:${on ? "#E9EBEE" : "transparent"};">
        <div class="tcell-treatment">${esc(u.treatment)}</div><div class="tcell-organism">${esc(u.organism)}</div>
        <div class="tcell-doselayout">${esc(u.doseLayout ?? "")}</div>
        <div class="tcell-cls"><i style="width:7px; height:7px; border-radius:50%; background:${D.classColours[u.cls]}; flex:none;"></i><span style="overflow:hidden; text-overflow:ellipsis; font-size:12px;">${C.SHORT[u.cls]}</span></div>
        <div class="tcell-num">${fmt(u.x, 3)}</div><div class="tcell-num">${fmt(u.meanInhibition, 3)}</div><div class="tcell-num">${fmt(u.q, 2)}</div>
        <div class="tcell-num">${ivHtml(u.cc, 3)}</div><div class="tcell-num">${dcHtml(u.dc)}</div>
        <div class="tcell-num">${u.fit ? ivHtml(u.fit.slope, 3) : "NA"}</div><div class="tcell-num">${u.fit ? ivHtml(u.fit.midpoint, 3) : "NA"}</div></div>`;
    }).join("");
    el("table-count").textContent = order.length.toLocaleString() + " treatments";
    const drawerLabel = state.drawer ? "Hide treatment table" : "Show treatment table (" + order.length.toLocaleString() + " treatments)";
    el("drawer-btn").dataset.tip = drawerLabel;
    el("drawer-btn").setAttribute("aria-label", drawerLabel);
    el("drawer-btn").setAttribute("aria-expanded", state.drawer ? "true" : "false");
    el("page-label").textContent = order.length === 0 ? "No treatments match — clear the search or re-enable classes."
      : (page * C.PER + 1).toLocaleString() + "–" + Math.min(order.length, (page + 1) * C.PER).toLocaleString() + " of " + order.length.toLocaleString();
    el("prev-btn").disabled = page === 0;
    el("next-btn").disabled = page >= maxPage;
    el("table-clear").style.display = state.search.length > 0 ? "" : "none";
  },
  drawerStyles() {
    el("drawer").style.transform = "translateY(" + (state.drawer ? "0%" : "102%") + ")";
    el("ov-scatter").style.inset = "0 10px " + (state.drawer ? "330px" : "6px") + " 10px";
    el("drawer-btn").classList.toggle("active", state.drawer);
    el("ov-info").style.bottom = state.drawer ? "336px" : "10px";
  },
  inspector() {
    const set = state.selSet;
    const u = state.sel !== null ? D.treatments[state.sel] : null;
    const hasSel = set.length > 0 && !!u;
    const single = set.length === 1 && !!u;
    el("insp-body").style.display = hasSel ? "flex" : "none";
    el("insp-edge").style.display = hasSel ? "block" : "none";
    el("resize-notch").style.display = (hasSel && state.view === "overview") ? "flex" : "none";
    el("insp-single").style.display = single ? "" : "none";
    el("insp-multi").style.display = set.length > 1 ? "block" : "none";
    el("insp-prev").style.display = single ? "" : "none";
    el("insp-next").style.display = single ? "" : "none";
    if (!hasSel) return;
    if (set.length > 1) {
      el("insp-treatment").textContent = "Multiple Comparison";
      el("insp-cls").textContent = "";
      el("insp-dot").style.background = "transparent";
      el("insp-organism").textContent = "";
      el("insp-pos").textContent = "";
      el("compare-rows").innerHTML = set.map((ui, k) => {
        const uu = D.treatments[ui];
        return `<div class="cgrid crow"><div style="display:flex; align-items:center; gap:6px; min-width:0;"><i style="width:8px; height:8px; border-radius:50%; background:${C.CMP[k % C.CMP.length]}; flex:none;"></i><span style="font-weight:600; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">${esc(uu.treatment)}</span></div>
          <div class="cnum">${fmt(uu.q, 3)}</div><div class="cnum">${fmt(uu.x, 3)}</div><div class="cnum">${fmt(uu.cc[0], 3)}${qcDot(uu, "mic")}</div>
          <div class="cnum">${uu.dc[0] === null ? "NA" : esc(uu.dc[0])}</div>
          <div class="cnum">${uu.fit ? fmt(uu.fit.slope[0], 3) : "NA"}${qcDot(uu, "slope")}</div><div class="cnum">${uu.fit ? fmt(uu.fit.midpoint[0], 3) : "NA"}${qcDot(uu, "midpoint")}</div></div>`;
      }).join("");
      return;
    }
    el("insp-treatment").textContent = u.treatment;
    el("insp-cls").textContent = D.classes[u.cls];
    el("insp-dot").style.background = D.classColours[u.cls];
    el("insp-organism").textContent = "· " + u.organism + (u.doseLayout ? " · " + u.doseLayout : "");
    const order = filteredOrder();
    const pos = order.indexOf(state.sel);
    el("insp-pos").textContent = pos >= 0 ? (pos + 1).toLocaleString() + " / " + order.length.toLocaleString() : "";
    const kv = d => `<div class="kv-row"><div><span class="kv-dotslot">${d.dot || ""}</span>${d.k}</div><div>${d.v} <span class="kv-sub">${d.sub}</span></div></div>`;
    const evidenceRows = [
      { k: "q", v: fmt(u.q, 3), sub: "" },
      { k: "Dynamic Range", v: fmt(u.x, 3), sub: "" },
    ];
    if (u.cc[0] !== null) evidenceRows.push(Object.assign({ k: "Continuous MIC", dot: qcDot(u, "mic") }, interval(u.cc)));
    if (u.dc[0] !== null) evidenceRows.push({ k: "Discrete MIC", v: u.dc[0], sub: (u.dc[1] !== null || u.dc[2] !== null) ? `[${u.dc[1] ?? "NA"} – ${u.dc[2] ?? "NA"}]` : "" });
    el("evidence-rows").innerHTML = evidenceRows.map(kv).join("");
    el("fit-section").style.display = u.fit ? "block" : "none";
    if (u.fit) {
      const rows = [
        Object.assign({ k: "Slope", dot: qcDot(u, "slope") }, interval(u.fit.slope)),
        Object.assign({ k: "Midpoint", dot: qcDot(u, "midpoint") }, interval(u.fit.midpoint)),
      ];
      if (u.fit.max_inhibition) rows.push(Object.assign({ k: "Upper plateau" }, interval(u.fit.max_inhibition)));
      if (u.fit.min_inhibition) rows.push(Object.assign({ k: "Lower plateau" }, interval(u.fit.min_inhibition)));
      el("fit-rows").innerHTML = rows.map(kv).join("");
    } else el("fit-rows").innerHTML = "";
  },
  qcChrome() {
    const organism = state.qcOrganism;
    el("qc-title").textContent = organism ? "Normalization QC — " + organism : "Normalization QC";
    const chk = D.normalization[organism];
    const zoom = state.qcZoomAssay;
    el("rc-within-title").textContent = zoom === null ? "Within-Assay Control Outliers" : "Within-Assay Control Outliers — assay " + zoom;
    el("rc-within-hint").textContent = zoom === null ? "Select Plots to Enlarge" : "";
    el("rc-within-close").style.display = zoom === null ? "none" : "";
    const controls = document.querySelector('[data-mode-controls="rc-within"]');
    if (controls) controls.style.display = zoom === null ? "none" : "inline-flex";
    const box = el("rc-within-box");
    box.style.overflowY = zoom === null ? "auto" : "hidden";
    el("rc-within").style.height = (zoom === null && chk) ? withinFacetMeta(chk).height + "px" : "340px";
  },
};
const DEPS = {
  view: ["tabs", "organisms", "inspector", "qcChrome"],
  organism: ["classes", "organisms", "table", "inspector"],
  qcOrganism: ["organisms", "qcChrome"],
  cls: ["classes", "table", "inspector"],
  q: ["search"], showMatches: ["search"],
  meanColour: ["meanToggle"],
  search: ["table", "inspector"], sortKey: ["table", "inspector"], sortDesc: ["table", "inspector"], page: ["table"],
  sel: ["table", "inspector", "search"], selSet: ["table", "inspector", "search"],
  drawer: ["table", "drawerStyles"],
  runOpen: ["runInfo"],
  qcOpen: ["qcCard"],
  qcFilters: ["qcCard", "classes", "organisms", "table", "inspector", "search"],
  qcZoomAssay: ["qcChrome"],
};
const ORDER_KEYS = ["search", "organism", "cls", "sortKey", "sortDesc", "qcFilters"];

function setState(patch) {
  const changed = Object.keys(patch);
  Object.assign(state, patch);
  if (changed.some(k => ORDER_KEYS.indexOf(k) >= 0)) orderCache = null;
  const run = new Set();
  changed.forEach(k => (DEPS[k] || []).forEach(name => run.add(name)));
  run.forEach(name => R[name]());
  afterUpdate();
}
function afterUpdate() {
  const sig = JSON.stringify([state.view, state.organism, state.qcOrganism, state.cls, state.sel, state.selSet, state.qcZoomAssay, state.qcFilters, state.meanColour]);
  if (sig !== chartSig) {
    chartSig = sig;
    requestAnimationFrame(renderVisibleCharts);
  }
  const ls = JSON.stringify([state.sel === null, state.drawer, state.view]);
  const viewChanged = lastView !== undefined && lastView !== state.view;
  lastView = state.view;
  if (ls !== layoutSig) {
    const first = layoutSig === undefined;
    layoutSig = ls;
    if (first) setInspectorWidth(inspectorTarget());
    else { fastAnim = viewChanged; animateLayout(); }
  }
}

function inspectorTarget() {
  return (state.sel !== null && state.view === "overview") ? (inspW || C.INSP_W) : 0;
}
function syncInspectorSnap(w) {
  const grid = el("main-grid");
  if (!grid) return;
  const snapped = state.sel !== null && state.view === "overview" &&
    w > grid.offsetWidth - C.SIDEBAR - C.PLOT_MIN + 1;
  const wasSnapped = grid.classList.contains("inspector-snapped");
  grid.classList.toggle("inspector-snapped", snapped);
  if (snapped && !wasSnapped) setPlotInteractionMode(el("ov-curve"), "inspect");
}
function setInspectorWidth(w) {
  el("main-grid").style.gridTemplateColumns = C.SIDEBAR + "px minmax(0,1fr) " + w + "px";
  syncInspectorSnap(w);
}
function currentInspectorWidth() {
  const parts = getComputedStyle(el("main-grid")).gridTemplateColumns.trim().split(/\s+/);
  return parseFloat(parts[parts.length - 1]) || 0;
}
function animateLayout() {
  const from = currentInspectorWidth(), to = inspectorTarget();
  if (animRaf) cancelAnimationFrame(animRaf);
  const fast = fastAnim;
  fastAnim = false;
  const dist = Math.abs(to - from);
  const dur = fast ? Math.max(170, Math.min(280, dist * 0.45)) : Math.max(340, Math.min(780, dist * 1.05));
  const ease = fast ? (t => 1 - Math.pow(1 - t, 3)) : (t => 0.5 * (1 - Math.cos(Math.PI * t)));
  const t0 = performance.now();
  const tick = () => {
    const p = Math.min(1, (performance.now() - t0) / dur);
    setInspectorWidth(from + (to - from) * ease(p));
    if (!fast) resizeVisible();
    if (p < 1) animRaf = requestAnimationFrame(tick);
    else {
      animRaf = null;
      setInspectorWidth(to);
      resizeQueued = true;
      if (!resizePending) { resizeQueued = false; resizeVisible(); }
    }
  };
  animRaf = requestAnimationFrame(tick);
}
function resizeVisible() {
  syncInspectorSnap(currentInspectorWidth());
  if (resizePending) { resizeQueued = true; return; }
  const ids = state.view === "overview" ? C.OV_IDS : C.QC_IDS;
  const ps = [];
  ids.forEach(id => {
    const plot = el(id);
    if (!plot || !plot._fullLayout) return;
    const w = plot.clientWidth, h = plot.clientHeight;
    if (!w || !h) return;
    if (Math.abs((plot._fullLayout.width || 0) - w) < 1 && Math.abs((plot._fullLayout.height || 0) - h) < 1) return;
    try { ps.push(Plotly.relayout(plot, { width: w, height: h })); } catch (err) {}
  });
  if (ps.length) {
    resizePending = true;
    Promise.all(ps).catch(() => {}).then(() => {
      resizePending = false;
      const w = el("rc-within");
      if (w && w._facetAssays && state.view === "qc" && state.qcZoomAssay === null)
        buildFacetOverlays(w, w._facetAssays);
      if (resizeQueued) { resizeQueued = false; resizeVisible(); }
    });
  }
}

function renderVisibleCharts() {
  if (!window.Plotly) return;
  if (state.view === "overview") {
    renderScatter();
    if (state.sel !== null) renderCurve();
  } else renderNormalization();
}

function renderScatter() {
  const plot = el("ov-scatter");
  if (!plot) return;
  const traces = [];
  const meanValue = i => D.treatments[i].meanInhibition ?? -Infinity;
  const addTrace = (name, idx, marker) => traces.push({
    name, type: "scattergl", mode: "markers",
    x: idx.map(i => D.treatments[i].x), y: idx.map(i => scatterY(D.treatments[i])),
    customdata: idx.map(i => { const u = D.treatments[i]; return [i, u.treatment, u.organism, fmt(u.x), fmt(u.q, 3), D.classes[u.cls], fmt(scatterY(u), 3), fmt(u.meanInhibition, 3)]; }),
    marker, hoverinfo: "none",
  });
  if (state.meanColour) {
    const idx = [];
    D.treatments.forEach((u, i) => {
      if (state.cls[u.cls] && (!state.organism || u.organism === state.organism) && qcVisible(u)) idx.push(i);
    });
    idx.sort((a, b) => meanValue(a) - meanValue(b));
    addTrace("Mean inhibition", idx, { color: idx.map(i => D.treatments[i].meanInhibition), coloraxis: "coloraxis", size: C.MARKER,
      opacity: state.selSet.length ? 0.42 : 0.82, line: { width: 0.5, color: cssVar("--surface") } });
  } else D.classes.forEach((cls, ci) => {
    if (!state.cls[ci]) return;
    const idx = [];
    D.treatments.forEach((u, i) => { if (u.cls === ci && (!state.organism || u.organism === state.organism) && qcVisible(u)) idx.push(i); });
    addTrace(cls, idx, { color: D.classColours[ci], size: C.MARKER, opacity: state.selSet.length ? 0.35 : 0.75,
      line: { width: 0.5, color: cssVar("--surface") } });
  });
  const ringIdx = state.selSet.slice();
  if (state.meanColour) ringIdx.sort((a, b) => meanValue(a) - meanValue(b));
  if (ringIdx.length) {
    traces.push({ name: "selected", type: "scattergl", mode: "markers", showlegend: false,
      x: ringIdx.map(si => D.treatments[si].x), y: ringIdx.map(si => scatterY(D.treatments[si])),
      customdata: ringIdx.map(si => { const u = D.treatments[si]; return [si, u.treatment, u.organism, fmt(u.x), fmt(u.q, 3), D.classes[u.cls] + " · selected", fmt(scatterY(u), 3), fmt(u.meanInhibition, 3)]; }),
      marker: { color: state.meanColour ? ringIdx.map(si => D.treatments[si].meanInhibition) : ringIdx.map(si => D.classColours[D.treatments[si].cls]),
                coloraxis: state.meanColour ? "coloraxis" : undefined, size: C.MARKER + 5, opacity: 1,
                line: { width: 2, color: cssVar("--ink") } },
      hoverinfo: "none" });
  }
  const layout = baseLayout({
    showlegend: false,
    uirevision: "scatter-" + state.organism,
    margin: { l: 54, r: state.meanColour ? 92 : 16, t: 12, b: 44 },
    xaxis: ax({ title: axTitle("Dynamic Range", 14.5), zeroline: false }),
    yaxis: ax({ type: "log", dtick: 1, title: axTitle("Signal Score / Significance Threshold", 14.5), zeroline: false }),
    shapes: [
      { type: "line", xref: "paper", x0: 0, x1: 1, y0: 1, y1: 1, yref: "y", line: { color: "#8B9096", width: 1.4, dash: "dash" } },
      { type: "line", yref: "paper", x0: 0, x1: 0, y0: 0, y1: 1, line: { color: cssVar("--grid"), width: 1 } },
    ],
    annotations: [{ xref: "paper", x: 1, xanchor: "right", yref: "y", y: 0, yanchor: "bottom", yshift: 3,
      text: "q ≤ " + fmt(D.meta.qThreshold, 3), showarrow: false,
      font: { family: cssVar("--sans"), size: 12.5, color: "#8B9096" } }],
  });
  if (state.meanColour) layout.coloraxis = {
    cmin: 0, cmax: 1, colorscale: C.MEAN_COLOURS,
    colorbar: { thickness: 11, len: 0.66, x: 1.02, outlinewidth: 0,
      tickvals: [0, 0.25, 0.5, 0.75, 1], tickfont: { family: cssVar("--mono"), size: 11.5, color: cssVar("--muted") },
      title: { text: "<b>Mean<br>Inhibition</b>", side: "right", font: { family: cssVar("--sans"), size: 12, color: cssVar("--ink") } } },
  };
  Plotly.react(plot, traces, interactionLayout(plot, layout, "scatter-" + state.organism), plotConfig());
  wireZoomDismiss(plot, { "xaxis.autorange": true, "yaxis.autorange": true });
  if (plot.removeAllListeners) plot.removeAllListeners("plotly_click");
  if (plot.on) plot.on("plotly_click", ev => {
    if (!(ev.points && ev.points.length && ev.points[0].customdata)) return;
    selectUnit(ev.points[0].customdata[0], !!(ev.event && (ev.event.ctrlKey || ev.event.metaKey)));
  });
  if (!plot._firstSized) {
    plot._firstSized = true;
    setTimeout(resizeVisible, 80);
    setTimeout(resizeVisible, 450);
  }
  bindCustomHover(plot, pt => {
    const c = pt.customdata;
    if (!c) return null;
    return "<b>" + esc(c[1]) + "</b> · " + esc(c[2]) + "<br>" + esc(c[5]) + "<br>Dynamic range " + c[3] + " · mean inhibition " + c[7] + " · q " + c[4] + " · signal score / significance threshold " + c[6];
  });
}

function fittedCurve(u, concentrations) {
  const f = u.fit;
  const slope = f.slope[0], midpoint = f.midpoint[0];
  const front = f.max_inhibition ? f.max_inhibition[0] : 1.0;
  const back = f.min_inhibition ? f.min_inhibition[0] : 0.0;
  if ([slope, midpoint, front, back].some(v => v === null)) return null;
  const lo = Math.log(concentrations[0]), hi = Math.log(concentrations[concentrations.length - 1]);
  const x = [], y = [];
  for (let i = 0; i <= C.CURVE_N; i++) {
    const lv = Math.exp(lo + (hi - lo) * i / C.CURVE_N);
    x.push(lv); y.push(back + (front - back) / (1 + Math.pow(midpoint / lv, slope)));
  }
  return { x, y };
}
function fittedBand(u, concentrations) {
  const f = u.fit;
  const params = [f.slope, f.midpoint, f.max_inhibition || [1, null, null], f.min_inhibition || [0, null, null]];
  if (params.some(p => !p || p[0] === null)) return null;
  const ranges = params.map(p => [
    (p[1] === null || p[1] === undefined) ? p[0] : p[1],
    (p[2] === null || p[2] === undefined) ? p[0] : p[2],
  ]);
  if (ranges.every(r => r[0] === r[1])) return null;
  const loA = Math.log(concentrations[0]), hiA = Math.log(concentrations[concentrations.length - 1]);
  const x = [], lo = [], hi = [];
  for (let i = 0; i <= C.BAND_N; i++) { x.push(Math.exp(loA + (hiA - loA) * i / C.BAND_N)); lo.push(Infinity); hi.push(-Infinity); }
  for (let m = 0; m < 16; m++) {
    const st = ranges[0][m & 1], ce = ranges[1][(m >> 1) & 1], fr = ranges[2][(m >> 2) & 1], bk = ranges[3][(m >> 3) & 1];
    for (let i = 0; i <= C.BAND_N; i++) {
      const y = bk + (fr - bk) / (1 + Math.pow(ce / x[i], st));
      if (y < lo[i]) lo[i] = y;
      if (y > hi[i]) hi[i] = y;
    }
  }
  return { xx: x.concat(x.slice().reverse()), yy: hi.concat(lo.slice().reverse()) };
}
function fitTraces(u, concentrations, color, legendgroup) {
  const out = [];
  if (u.cls !== 0 || !u.fit) return out;
  const band = fittedBand(u, concentrations);
  if (band) out.push({ type: "scatter", x: band.xx, y: band.yy, fill: "toself",
    fillcolor: hexToRgba(color, 0.12), line: { width: 0 }, hoverinfo: "skip", showlegend: false, legendgroup });
  const curve = fittedCurve(u, concentrations);
  if (curve) out.push({ name: legendgroup ? legendgroup + " fit" : "fit", type: "scatter", mode: "lines",
    x: curve.x, y: curve.y, legendgroup, showlegend: false, line: { color, width: legendgroup ? 2 : 2.2 }, hoverinfo: "skip" });
  return out;
}
function curveLayout(concentrations) {
  return baseLayout({
    margin: { l: 46, r: 12, t: 10, b: 40 },
    xaxis: ax({ type: "log", title: axTitle("Concentration"), tickvals: concentrations, ticktext: concentrations.map(fmtConcentration) }),
    yaxis: ax({ title: axTitle("Inhibition Fraction"), range: [-0.05, 1.05] }),
    showlegend: false,
  });
}
function renderCurve() {
  const set = state.selSet.length ? state.selSet : (state.sel !== null ? [state.sel] : []);
  if (!set.length) return;
  const plot = el("ov-curve");
  if (!plot) return;
  let traces = [], concentrations = null;
  if (set.length === 1) {
    const u = D.treatments[set[0]];
    concentrations = D.concentrationGrids[u.grid];
    const dots = { in: { x: [], y: [] }, out: { x: [], y: [] } };
    u.reps.forEach((rep, r) => {
      const b = u.out[r] ? dots.out : dots.in;
      rep.forEach((v, li) => { if (v !== null) { b.x.push(concentrations[li]); b.y.push(v); } });
    });
    traces.push({ name: "replicate", type: "scatter", mode: "markers", x: dots.in.x, y: dots.in.y,
      marker: { color: D.classColours[u.cls], size: 6.5, opacity: 0.8 }, hoverinfo: "none" });
    if (dots.out.x.length)
      traces.push({ name: "outlier", type: "scatter", mode: "markers", x: dots.out.x, y: dots.out.y,
        marker: { color: D.classColours[u.cls], size: 6.5, opacity: 0.35, symbol: "circle-open" }, hoverinfo: "none" });
    traces = traces.concat(fitTraces(u, concentrations, D.classColours[0], null));
  } else {
    set.forEach((ui, k) => {
      const u = D.treatments[ui];
      const lv = D.concentrationGrids[u.grid];
      if (!concentrations || lv.length > concentrations.length) concentrations = lv;
      const xs = [], ys = [];
      u.reps.forEach(rep => rep.forEach((v, li) => { if (v !== null) { xs.push(lv[li]); ys.push(v); } }));
      const col = C.CMP[k % C.CMP.length];
      traces.push({ name: u.treatment, type: "scatter", mode: "markers", x: xs, y: ys, legendgroup: u.treatment,
        marker: { color: col, size: 5.5, opacity: 0.75 }, hoverinfo: "none" });
      traces = traces.concat(fitTraces(u, lv, col, u.treatment));
    });
  }
  const layout = interactionLayout(plot, curveLayout(concentrations), "curve-" + set.join(","));
  Plotly.react(plot, traces, layout, plotConfig());
  wireZoomDismiss(plot, { "xaxis.autorange": true, "yaxis.autorange": false, "yaxis.range": [-0.05, 1.05] });
  bindCustomHover(plot, pt => {
    const nm = pt.data.name || "";
    if (nm === "fit" || nm.endsWith(" fit")) return null;
    const base = "concentration " + fmtConcentration(pt.x) + "<br>inhibition " + pt.y.toFixed(3);
    if (nm === "replicate") return base;
    if (nm === "outlier") return base + " <span style=\"color:var(--muted)\">(outlier)</span>";
    return "<b>" + esc(nm) + "</b><br>" + base;
  });
}

function splitControlPoints(check, assay) {
  const xs = [[], []], ys = [[], []], fx = [], fy = [], ft = [];
  check.assay.forEach((bb, i) => {
    if (bb !== assay) return;
    const a = check.control[i], v = check.value[i];
    if (check.flag[i]) { fx.push(a + jitter(i) * 0.3); fy.push(v); ft.push(D.controls[a]); }
    else { xs[a].push(a + jitter(i) * 0.3); ys[a].push(v); }
  });
  return { xs, ys, fx, fy, ft };
}
function controlScatter(check, colours) {
  const assays = [...new Set(check.assay)].sort((a, b) => a - b);
  const pos = new Map(assays.map((b, i) => [b, i]));
  const traces = D.controls.map((control, ai) => {
    const x = [], y = [], text = [];
    check.control.forEach((a, i) => {
      if (a === ai && !check.flag[i]) {
        x.push(pos.get(check.assay[i]) + (ai ? 0.16 : -0.16) + jitter(i) * 0.14);
        y.push(check.value[i]); text.push("assay " + check.assay[i]);
      }
    });
    return { name: control, type: "scattergl", mode: "markers", x, y, text,
      marker: { color: colours[ai], size: 6, opacity: 0.75 }, hoverinfo: "none" };
  });
  const x = [], y = [], text = [];
  check.control.forEach((a, i) => {
    if (check.flag[i]) {
      x.push(pos.get(check.assay[i]) + (a ? 0.16 : -0.16) + jitter(i) * 0.14);
      y.push(check.value[i]); text.push(D.controls[a] + " · assay " + check.assay[i]);
    }
  });
  if (x.length) traces.push({ name: "Flagged Outlier", type: "scattergl", mode: "markers", x, y, text,
    marker: { color: C.FLAG, size: 6.5, symbol: "circle-open", line: { width: 1.4, color: C.FLAG } },
    hoverinfo: "none" });
  return { traces, assays };
}
function assayAxis(assays) {
  const angled = assays.length > 14;
  const title = axTitle("Assay ID");
  title.standoff = angled ? 6 : 10;
  return ax({ tickvals: assays.map((b, i) => i), ticktext: assays.map(String),
    title, tickangle: angled ? 45 : 0, automargin: true });
}
function withinFacetMeta(checks) {
  const assays = [...new Set(checks.within.assay)].sort((a, b) => a - b);
  const cols = Math.min(4, Math.max(2, Math.ceil(Math.sqrt(assays.length * 1.2))));
  const rows = Math.max(1, Math.ceil(assays.length / cols));
  return { assays, cols, rows, height: Math.max(340, rows * 112 + 64) };
}
function renderWithinFacets(checks) {
  const { assays, cols, rows } = withinFacetMeta(checks);
  const check = checks.within;
  const yMin = Math.min(...check.value), yMax = Math.max(...check.value);
  const pad = ((yMax - yMin) || 1) * 0.14;
  const traces = [], owners = [];
  const layout = baseLayout({ showlegend: true, legend: bottomLegend(), margin: { l: 56, r: 42, t: 22, b: 58 } });
  layout.grid = { rows, columns: cols, pattern: "independent", xgap: 0.12, ygap: rows > 2 ? 0.22 : 0.16 };
  const annotations = [];
  let flagLegendShown = false;
  assays.forEach((b, k) => {
    const sfx = k === 0 ? "" : String(k + 1);
    const { xs, ys, fx, fy } = splitControlPoints(check, b);
    D.controls.forEach((control, ai) => {
      traces.push({ type: "scatter", mode: "markers", x: xs[ai], y: ys[ai],
        xaxis: "x" + sfx, yaxis: "y" + sfx, hoverinfo: "skip",
        name: control, legendgroup: control, showlegend: k === 0,
        marker: { color: D.controlQcColours[ai], size: 3.2, opacity: 0.65 } });
      owners.push(b);
    });
    if (fx.length) {
      traces.push({ type: "scatter", mode: "markers", x: fx, y: fy,
        xaxis: "x" + sfx, yaxis: "y" + sfx, hoverinfo: "skip",
        name: "Flagged Outlier", legendgroup: "flagged", showlegend: !flagLegendShown,
        marker: { color: C.FLAG, size: 2.6, opacity: 0.9, symbol: "circle-open", line: { width: 0.9, color: C.FLAG } } });
      flagLegendShown = true;
      owners.push(b);
    }
    layout["xaxis" + sfx] = { range: [-0.8, 1.8], tickvals: [0, 1], ticktext: ["A", "B"], showticklabels: false,
      tickfont: { family: cssVar("--sans"), size: 10.5, color: cssVar("--muted") }, ticks: "",
      gridcolor: "rgba(0,0,0,0)", zeroline: false, showline: false, fixedrange: true };
    layout["yaxis" + sfx] = { range: [yMin - pad, yMax + pad], showticklabels: k % cols === 0,
      title: k % cols === 0 ? { text: "<b>Measurement</b>", font: { size: 10.5, family: cssVar("--sans"), color: cssVar("--ink") }, standoff: 5 } : undefined,
      tickfont: { family: cssVar("--mono"), size: 10.5, color: cssVar("--muted") }, ticks: "",
      gridcolor: cssVar("--grid"), zeroline: false, showline: false, fixedrange: true };
    annotations.push({ xref: "x" + sfx + " domain", yref: "y" + sfx + " domain", x: 0.5, y: 1.02, yanchor: "bottom",
      showarrow: false, text: "<b>" + b + "</b>", font: { size: 11, color: cssVar("--ink"), family: cssVar("--sans") } });
  });
  layout.annotations = annotations;
  const plot = el("rc-within");
  if (!plot) return;
  disablePlotInteraction(plot);
  layout.dragmode = false;
  layout.height = plot.offsetHeight || undefined;
  layout.width = plot.offsetWidth || undefined;
  plot._facetOwners = owners;
  plot._facetAssays = assays;
  Plotly.react(plot, traces, layout, plotConfig()).then(() => buildFacetOverlays(plot, assays));
}
function facetTraceGroups(plot) {
  const out = [];
  plot.querySelectorAll(".scatterlayer .trace").forEach(g => {
    let idx = null;
    try { idx = g.__data__[0].trace.index; } catch (err) {}
    if (idx !== null && idx !== undefined) out.push([idx, g]);
  });
  return out;
}
function setFacetDim(plot, keepB) {
  if (!plot._facetOwners) return;
  facetTraceGroups(plot).forEach(([idx, g]) => {
    const keep = keepB === null || plot._facetOwners[idx] === keepB;
    g.style.transition = "opacity .3s ease, filter .3s ease";
    g.style.opacity = keep ? "1" : "0.22";
    g.style.filter = keep ? "none" : "grayscale(1)";
  });
}
function buildFacetOverlays(plot, assays) {
  plot.querySelectorAll(".facet-hit").forEach(n => n.remove());
  const fl = plot._fullLayout;
  if (!fl || !fl._size) return;
  const sz = fl._size;
  plot.style.position = "relative";
  assays.forEach((b, k) => {
    const sfx = k === 0 ? "" : String(k + 1);
    const xa = fl["xaxis" + sfx], ya = fl["yaxis" + sfx];
    if (!xa || !ya || !xa.domain || !ya.domain) return;
    const d = document.createElement("div");
    d.className = "facet-hit";
    Object.assign(d.style, {
      position: "absolute", zIndex: "6", cursor: "pointer",
      left: (sz.l + xa.domain[0] * sz.w) + "px",
      top: (sz.t + (1 - ya.domain[1]) * sz.h) + "px",
      width: ((xa.domain[1] - xa.domain[0]) * sz.w) + "px",
      height: ((ya.domain[1] - ya.domain[0]) * sz.h) + "px",
      border: "1px solid " + cssVar("--line"), borderRadius: "5px",
      transition: "border-color .12s",
    });
    d.onmouseenter = () => { d.style.borderColor = cssVar("--ink"); setFacetDim(plot, b); };
    d.onmouseleave = () => { d.style.borderColor = cssVar("--line"); setFacetDim(plot, null); };
    d.onclick = () => setState({ qcZoomAssay: b });
    plot.appendChild(d);
  });
}
function renderWithinZoom(checks, b) {
  const { xs, ys, fx, fy, ft } = splitControlPoints(checks.within, b);
  const traces = D.controls.map((control, ai) => ({ name: control, type: "scatter", mode: "markers",
    x: xs[ai], y: ys[ai], marker: { color: D.controlQcColours[ai], size: 7, opacity: 0.8 }, hoverinfo: "none" }));
  if (fx.length) traces.push({ name: "Flagged Outlier", type: "scatter", mode: "markers", x: fx, y: fy, text: ft,
    marker: { color: C.FLAG, size: 7.5, symbol: "circle-open", line: { width: 1.5, color: C.FLAG } },
    hoverinfo: "none" });
  const plot = el("rc-within");
  if (!plot) return;
  plot._facetAssays = null;
  plot.querySelectorAll(".facet-hit").forEach(n => n.remove());
  const clearDim = () => plot.querySelectorAll(".scatterlayer .trace").forEach(g => {
    g.style.opacity = ""; g.style.filter = ""; g.style.transition = "";
  });
  clearDim();
  const layout = interactionLayout(plot, qcLayout({
    height: plot.offsetHeight || 340,
    margin: { l: 54, r: 16, t: 12, b: 78 },
    xaxis: ax({ tickvals: [0, 1], ticktext: D.controls, range: [-0.7, 1.7] }),
    yaxis: ax({ title: axTitle("Measurement") }),
  }), "within-" + state.qcOrganism + "-" + b);
  Plotly.react(plot, traces, layout, plotConfig()).then(clearDim);
  wireZoomDismiss(plot, { "xaxis.autorange": false, "xaxis.range": [-0.7, 1.7], "yaxis.autorange": true });
  bindCustomHover(plot, pt =>
    (pt.data.name === "Flagged Outlier" ? "<b>Flagged Outlier</b>" + (pt.text ? " · " + esc(pt.text) : "") : "<b>" + esc(pt.data.name) + "</b>") +
    " · assay " + b + "<br>" + pt.y.toFixed(4));
}
function renderNormalization() {
  const checks = D.normalization[state.qcOrganism];
  if (!checks) return;
  const cfg = plotConfig();
  if (state.qcZoomAssay !== null) renderWithinZoom(checks, state.qcZoomAssay);
  else renderWithinFacets(checks);

  const across = controlScatter(checks.across, D.controlQcColours);
  const acrossEl = el("rc-across");
  if (acrossEl) {
    const layout = interactionLayout(acrossEl, qcLayout({
      xaxis: assayAxis(across.assays), yaxis: ax({ title: axTitle("Control median") }) }), "across-" + state.qcOrganism);
    Plotly.react(acrossEl, across.traces, layout, cfg);
    wireZoomDismiss(acrossEl, { "xaxis.autorange": true, "yaxis.autorange": true });
    bindCustomHover(acrossEl, pt => "<b>" + esc(pt.data.name) + "</b>" + (pt.text ? " · " + esc(pt.text) : "") + "<br>" + pt.y.toFixed(4));
  }

  const bands = { "<0": v => v < 0, "[0,1]": v => v >= 0 && v <= 1, ">1": v => v > 1 };
  const rawConcentrations = [...new Set(checks.raw.concentration)].sort((a, b) => a - b);
  const rawTraces = Object.keys(bands).map(band => {
    const x = [], y = [];
    checks.raw.value.forEach((v, i) => { if (bands[band](v)) { x.push(checks.raw.concentration[i] * (1 + jitter(i) * 0.16)); y.push(v); } });
    return { name: band, type: "scattergl", mode: "markers", x, y,
      marker: { color: D.rawBandColours[band], size: 4, opacity: 0.45 }, hoverinfo: "none" };
  });
  const rawEl = el("rc-raw");
  if (rawEl) {
    const layout = interactionLayout(rawEl, qcLayout({
      xaxis: ax({ type: "log", title: axTitle("Concentration"), tickvals: rawConcentrations, ticktext: rawConcentrations.map(fmtConcentration) }),
      yaxis: ax({ title: axTitle("Raw inhibition fraction") }) }), "raw-" + state.qcOrganism);
    Plotly.react(rawEl, rawTraces, layout, cfg);
    wireZoomDismiss(rawEl, { "xaxis.autorange": true, "yaxis.autorange": true });
    bindCustomHover(rawEl, pt => "raw inhibition " + pt.y.toFixed(3));
  }

  const inlierTraces = D.controls.map((control, ai) => {
    const values = checks.inlier.value.filter((v, i) => checks.inlier.control[i] === ai);
    const x = values.map((v, i) => ai + jitter(i) * 0.3);
    return { name: control + " (N=" + values.length + ")", type: "scattergl", mode: "markers", x, y: values,
      marker: { color: D.controlPoolColours[ai], size: 5, opacity: 0.6 }, hoverinfo: "none" };
  });
  const medianShapes = D.controls.map((control, ai) => {
    const v = checks.inlier.value.filter((x2, i) => checks.inlier.control[i] === ai).sort((a, b) => a - b);
    if (!v.length) return null;
    const m = v[Math.floor((v.length - 1) / 2)];
    return { type: "line", x0: ai - 0.34, x1: ai + 0.34, y0: m, y1: m, line: { color: cssVar("--ink"), width: 2 } };
  }).filter(Boolean);
  const inlEl = el("rc-inlier");
  if (inlEl) {
    const layout = interactionLayout(inlEl, qcLayout({
      xaxis: ax({ tickvals: [0, 1], ticktext: D.controls, range: [-0.6, 1.6] }),
      yaxis: ax({ title: axTitle("Measurement") }), shapes: medianShapes }), "inlier-" + state.qcOrganism);
    Plotly.react(inlEl, inlierTraces, layout, cfg);
    wireZoomDismiss(inlEl, { "xaxis.autorange": false, "xaxis.range": [-0.6, 1.6], "yaxis.autorange": true });
    bindCustomHover(inlEl, pt => "<b>" + esc(pt.data.name) + "</b><br>" + pt.y.toFixed(4));
  }
}

function selectUnit(i, additive, keepView) {
  let set = state.selSet.slice();
  if (additive) {
    const k = set.indexOf(i);
    if (k >= 0) set.splice(k, 1); else set.push(i);
  } else set = [i];
  const sel = set.length ? set[set.length - 1] : null;
  const patch = { sel, selSet: set };
  if (!keepView) patch.view = "overview";
  const drawerOpen = state.drawer && sel !== null;
  if (drawerOpen) {
    const selected = new Set(set);
    const displayOrder = set.length > 1 ? filteredOrder().filter(i => selected.has(i)) : filteredOrder();
    const pos = displayOrder.indexOf(sel);
    if (pos >= 0) patch.page = Math.floor(pos / C.PER);
  }
  setState(patch);
  if (drawerOpen) {
    setTimeout(() => {
      const row = document.querySelector('#table-rows [data-i="' + sel + '"]');
      if (!row) return;
      const sc = el("table-scroll");
      if (sc && sc.clientHeight < sc.scrollHeight)
        sc.scrollTop = Math.max(0, row.offsetTop - sc.offsetTop - sc.clientHeight / 2 + row.offsetHeight / 2);
    }, 90);
  }
}
function clearSelection() {
  inspW = C.INSP_W;
  setState({ sel: null, selSet: [] });
}
function step(d) {
  const order = filteredOrder();
  if (!order.length) return;
  const cur = order.indexOf(state.sel);
  const next = cur < 0 ? 0 : Math.min(order.length - 1, Math.max(0, cur + d));
  setState({ sel: order[next], selSet: [order[next]] });
}

function dragResize(e) {
  e.preventDefault();
  const grid = el("main-grid");
  const notch = el("resize-notch");
  const npath = notch ? notch.querySelector("path") : null;
  if (notch) {
    if (npath) { npath._pf = npath.style.fill || "#FFFFFF"; npath.style.fill = "#F4F5F7"; }
    notch.style.cursor = "grabbing";
    notch.querySelectorAll(".grip").forEach(g => {
      g._pbg = getComputedStyle(g).backgroundColor;
      g.style.background = "rgba(27,28,30,.30)";
    });
  }
  document.body.style.cursor = "grabbing";
  document.body.style.userSelect = "none";
  let targetW = inspW || C.INSP_W;
  let snapped = targetW > (grid.offsetWidth - C.SIDEBAR - C.PLOT_MIN);
  let anchorX = e.clientX, anchorW = targetW;
  let raf = null, closed = false;
  const cleanup = () => {
    document.removeEventListener("mousemove", move);
    document.removeEventListener("mouseup", up);
    document.body.style.cursor = "";
    document.body.style.userSelect = "";
    if (notch) {
      if (npath && npath._pf) npath.style.fill = npath._pf;
      notch.style.cursor = "grab";
      notch.querySelectorAll(".grip").forEach(g => { if (g._pbg) g.style.background = g._pbg; });
    }
    if (raf) { cancelAnimationFrame(raf); raf = null; }
  };
  const apply = () => {
    raf = null;
    if (animRaf) { cancelAnimationFrame(animRaf); animRaf = null; }
    inspW = targetW;
    setInspectorWidth(targetW);
    resizeVisible();
  };
  const move = ev => {
    if (closed) return;
    const rect = grid.getBoundingClientRect();
    const scale = grid.offsetWidth ? rect.width / grid.offsetWidth : 1;
    const total = grid.offsetWidth;
    const maxNormal = total - C.SIDEBAR - C.PLOT_MIN;
    const full = total - C.SIDEBAR;
    let w = anchorW + (anchorX - ev.clientX) / (scale || 1);
    if (!snapped && w < C.CLOSE_AT) {
      closed = true;
      cleanup();
      clearSelection();
      return;
    }
    const wasSnapped = snapped;
    if (snapped) {
      if (w < full - C.SNAP_CLOSE) { snapped = false; w = maxNormal; }
    } else if (w > maxNormal + C.SNAP_OPEN) snapped = true;
    w = snapped ? full : Math.max(C.MIN_W, Math.min(maxNormal, w));
    if (snapped !== wasSnapped) {
      anchorX = ev.clientX; anchorW = w;
      if (raf) { cancelAnimationFrame(raf); raf = null; }
      targetW = w;
      inspW = w;
      animateLayout();
      return;
    }
    if (w === targetW) return;
    targetW = w;
    if (!raf) raf = requestAnimationFrame(apply);
  };
  const up = () => { cleanup(); if (!closed && !animRaf) apply(); };
  document.addEventListener("mousemove", move);
  document.addEventListener("mouseup", up);
}

function initButtonHover() {
  document.addEventListener("mouseover", e => {
    const b = e.target && e.target.closest ? e.target.closest("button") : null;
    if (!b || b.matches(".tabbtn, .iconbtn, .xbtn, .qc-close, .navbtn, .notifier-note, .mean-toggle") || (e.relatedTarget && b.contains(e.relatedTarget))) return;
    const cs = getComputedStyle(b);
    b._pc = cs.color; b._pb = cs.borderColor; b._pw = cs.fontWeight;
    const hc = "var(--ink)";
    b.style.color = hc;
    if (cs.borderStyle !== "none" && parseFloat(cs.borderWidth) > 0) b.style.borderColor = hc;
    b.style.fontWeight = "650";
  });
  document.addEventListener("mouseout", e => {
    const b = e.target && e.target.closest ? e.target.closest("button") : null;
    if (!b || b.matches(".tabbtn, .iconbtn, .xbtn, .qc-close, .navbtn, .notifier-note, .mean-toggle") || (e.relatedTarget && b.contains(e.relatedTarget))) return;
    if (b._pc) b.style.color = b._pc;
    if (b._pb) b.style.borderColor = b._pb;
    if (b._pw) b.style.fontWeight = b._pw;
  });
}
function initKeys() {
  window.addEventListener("keydown", e => {
    const tag = (e.target && e.target.tagName) || "";
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
    if (e.key === "/" && state.view === "overview") {
      e.preventDefault();
      el("global-search").focus();
      return;
    }
    if (e.key === "Escape") {
      if (document.activeElement && document.activeElement.tagName === "BUTTON") document.activeElement.blur();
      if (state.sel !== null || state.drawer) {
        inspW = C.INSP_W;
        setState({ sel: null, selSet: [], drawer: false });
      }
      return;
    }
    if (state.sel === null) return;
    if (e.key === "ArrowRight") step(1);
    else if (e.key === "ArrowLeft") step(-1);
  });
}
function downloadTableCsv() {
  const csvField = v => /[",\n]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v;
  const header = ["Treatment", "Organism", "Dose layout", "Class", "Delta inhibition", "Mean Inhibition", "q",
                  "Cont. MIC", "Disc. MIC", "Slope", "Midpoint"];
  const lines = [header.join(",")];
  tableOrder().forEach(i => {
    const u = D.treatments[i];
    lines.push([
      u.treatment, u.organism, u.doseLayout ?? "", C.SHORT[u.cls], fmt(u.x, 3), fmt(u.meanInhibition, 3), fmt(u.q, 2),
      ivText(u.cc, 3), dcText(u.dc),
      u.fit ? ivText(u.fit.slope, 3) : "NA",
      u.fit ? ivText(u.fit.midpoint, 3) : "NA",
    ].map(v => csvField(String(v))).join(","));
  });
  const blob = new Blob([lines.join("\n") + "\n"], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "dynamic-treatment-table.csv";
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}
function dismissNotifier(plot) {
  document.querySelectorAll(".plotly-notifier .notifier-note").forEach(note => {
    if (!plot || note._zoomPlot === plot) note.remove();
  });
}
function notifierHost(plot) {
  const fallback = (state.view === "overview" ? el("ov-stage") : document.querySelector(".canvas")) || document.body;
  const host = (plot && plot.parentElement) || fallback;
  if (getComputedStyle(host).position === "static") host.style.position = "relative";
  return host;
}
function showZoomToast(plot) {
  const host = notifierHost(plot);
  let toast = document.querySelector(".plotly-notifier");
  if (!toast) { toast = document.createElement("div"); toast.className = "plotly-notifier"; }
  if (toast.parentElement !== host) host.appendChild(toast);
  toast.querySelectorAll(".notifier-note").forEach(note => note.remove());
  const note = document.createElement("button");
  note.type = "button";
  note.className = "notifier-note";
  note.setAttribute("aria-label", "Reset plot view");
  note.dataset.tip = "Reset plot view";
  note._zoomPlot = plot;
  note.innerHTML = ZOOM_OUT_ICON;
  note.addEventListener("click", () => resetPlotZoom(plot));
  toast.appendChild(note);
}
function resetPlotZoom(plot) {
  if (!plot || !window.Plotly || !plot._zoomReset) return;
  plot._resettingZoom = true;
  const done = () => {
    plot._resettingZoom = false;
    plot._viewIsZoomed = false;
    setPlotInteractionMode(plot, "inspect");
    dismissNotifier(plot);
  };
  Plotly.relayout(plot, plot._zoomReset).then(done).catch(done);
}
function wireZoomDismiss(plot, reset) {
  if (!plot || !plot.on) return;
  plot._zoomReset = reset;
  if (plot.removeAllListeners) {
    plot.removeAllListeners("plotly_relayout");
    plot.removeAllListeners("plotly_doubleclick");
  }
  plot.on("plotly_relayout", ev => {
    if (!ev) return;
    if (plot._resettingZoom) { dismissNotifier(plot); return; }
    const keys = Object.keys(ev);
    if (keys.some(key => key.indexOf("autorange") >= 0 && ev[key])) {
      requestAnimationFrame(() => {
        plot._viewIsZoomed = false;
        setPlotInteractionMode(plot, "inspect");
        dismissNotifier(plot);
      });
    } else if (keys.some(key => key.indexOf(".range") >= 0)) {
      plot._viewIsZoomed = true;
      syncPlotInteraction(plot);
      showZoomToast(plot);
      if (plot._interactionMode === "zoom")
        requestAnimationFrame(() => setPlotInteractionMode(plot, "pan"));
    }
  });
  plot.on("plotly_doubleclick", () => requestAnimationFrame(() => {
    plot._viewIsZoomed = false;
    setPlotInteractionMode(plot, "inspect");
    dismissNotifier(plot);
  }));
}
function exportPlot(btn) {
  const plot = el(btn.dataset.plot);
  if (!plot || !window.Plotly) return;
  const clear = { paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)" };
  Plotly.relayout(plot, { paper_bgcolor: "#FFFFFF", plot_bgcolor: "#FFFFFF" })
    .then(() => Plotly.downloadImage(plot, { format: "png", filename: "dynamic-" + btn.dataset.name, scale: 2, width: plot.offsetWidth, height: plot.offsetHeight }))
    .then(() => Plotly.relayout(plot, clear))
    .catch(() => Plotly.relayout(plot, clear));
}

function wire() {
  document.querySelectorAll(".tabbtn").forEach(b =>
    b.addEventListener("click", () => setState({ view: b.dataset.tab, showMatches: false })));
  document.addEventListener("click", e => {
    const b = e.target && e.target.closest ? e.target.closest("button[data-plot]") : null;
    if (b) exportPlot(b);
  });
  document.querySelectorAll(".plot-mode-btn").forEach(btn => {
    btn.addEventListener("mousedown", e => e.stopPropagation());
    btn.addEventListener("click", e => {
      e.preventDefault();
      e.stopPropagation();
      const plot = el(btn.dataset.modePlot);
      const requested = btn.dataset.interactionMode;
      setPlotInteractionMode(plot, plot && plot._interactionMode === requested ? "inspect" : requested);
    });
  });
  el("mean-colour-toggle").addEventListener("click", () => setState({ meanColour: !state.meanColour }));
  const gs = el("global-search");
  const expand = () => { el("search-wrap").style.width = "min(280px, 26vw)"; };
  gs.addEventListener("input", () => { expand(); setState({ q: gs.value, showMatches: true }); });
  gs.addEventListener("focus", () => { expand(); setState({ q: gs.value, showMatches: true }); });
  gs.addEventListener("blur", () => setTimeout(() => {
    if (!state.q && !state.selSet.length) el("search-wrap").style.width = "130px";
    setState({ showMatches: false });
  }, 150));
  el("search-clear").addEventListener("mousedown", e => {
    e.preventDefault();
    gs.value = "";
    setState({ q: "", showMatches: false });
  });
  el("search-matches").addEventListener("mousedown", e => {
    e.preventDefault();
    const row = e.target.closest("[data-i]");
    if (row) {
      gs.value = "";
      setState({ q: "", showMatches: false });
      selectUnit(+row.dataset.i, true);
    }
  });
  el("search-chips").addEventListener("click", e => {
    const x = e.target.closest(".schip-x");
    if (x) selectUnit(+x.dataset.i, true, true);
  });
  el("classes-list").addEventListener("click", e => {
    const row = e.target.closest("[data-idx]");
    if (!row) return;
    const i = +row.dataset.idx;
    if (e.detail >= 2) setState({ cls: state.cls.map((v, j) => j === i) });
    else { const c = state.cls.slice(); c[i] = !c[i]; setState({ cls: c }); }
  });
  const TIP_HOST_IDS = ["classes-list", "organisms-list", "insp-single", "insp-multi",
                        "ov-stage", "insp-body"];
  const tipHostFor = node => {
    for (const id of TIP_HOST_IDS) {
      const host = el(id);
      if (host && host.contains(node)) return host;
    }
    return document.body;
  };
  const wireHoverTips = host => {
    host.addEventListener("mousemove", e => {
      const row = e.target.closest("[data-tip]");
      if (!row || tipHostFor(row) !== host) return;
      if (row._tipTimer) clearTimeout(row._tipTimer);
      const r = host.getBoundingClientRect();
      let x = e.clientX - r.left + 12, y = e.clientY - r.top - 34;
      if (row.dataset.tipAbove) {
        const rowRect = row.getBoundingClientRect();
        x = rowRect.left - r.left - 8;
        y = rowRect.top - r.top - 58;
      }
      hideTip(host);
      row._tipTimer = setTimeout(() => showTip(host, row.dataset.tip, x, y), 300);
    });
    host.addEventListener("mouseout", e => {
      const row = e.target.closest("[data-tip]");
      if (!row || tipHostFor(row) !== host) return;
      if (e.relatedTarget && row.contains(e.relatedTarget)) return;
      if (row._tipTimer) clearTimeout(row._tipTimer);
      hideTip(host);
    });
  };
  [...TIP_HOST_IDS.map(el).filter(Boolean), document.body].forEach(wireHoverTips);
  el("organisms-list").addEventListener("click", e => {
    const row = e.target.closest("[data-organism]");
    if (!row) return;
    if (state.view === "qc") setState({ qcOrganism: row.dataset.organism, qcZoomAssay: null });
    else setState({ organism: row.dataset.organism, qcZoomAssay: null });
  });
  el("organisms-list").addEventListener("mouseover", e => {
    const row = e.target.closest("[data-organism]");
    if (row && row.dataset.on !== "1") row.style.background = "#F4F5F7";
  });
  el("organisms-list").addEventListener("mouseout", e => {
    const row = e.target.closest("[data-organism]");
    if (row && !(e.relatedTarget && row.contains(e.relatedTarget)))
      row.style.background = row.dataset.on === "1" ? "#EBEDF0" : "transparent";
  });
  el("run-toggle").addEventListener("click", () => {
    const opening = !state.runOpen;
    setState(opening ? { runOpen: true, qcOpen: false } : { runOpen: false });
  });
  el("qcf-toggle").addEventListener("click", () => {
    const opening = !state.qcOpen;
    setState(opening ? { qcOpen: true, runOpen: false } : { qcOpen: false });
  });
  el("qcf-body").addEventListener("click", e => {
    const sw = e.target.closest(".qcf-switch");
    if (!sw) return;
    const kind = sw.dataset.kind;
    const next = Object.assign({}, state.qcFilters, { [kind]: !state.qcFilters[kind] });
    const patch = { qcFilters: next };
    const kept = state.selSet.filter(i => qcVisible(D.treatments[i], next));
    if (kept.length !== state.selSet.length) {
      patch.selSet = kept;
      patch.sel = kept.length ? kept[kept.length - 1] : null;
      if (patch.sel === null) inspW = C.INSP_W;
    }
    setState(patch);
  });
  el("drawer-btn").addEventListener("click", () => setState({ drawer: !state.drawer }));
  el("drawer-close").addEventListener("click", () => setState({ drawer: false }));
  const ts = el("table-search");
  ts.addEventListener("input", () => setState({ search: ts.value }));
  el("table-clear").addEventListener("click", () => { ts.value = ""; setState({ search: "" }); });
  el("table-csv").addEventListener("click", downloadTableCsv);
  el("table-head").addEventListener("click", e => {
    const h = e.target.closest("[data-key]");
    if (!h) return;
    const key = h.dataset.key;
    setState({ sortKey: key, sortDesc: state.sortKey === key ? !state.sortDesc : true });
  });
  el("table-rows").addEventListener("click", e => {
    const row = e.target.closest("[data-i]");
    if (row) selectUnit(+row.dataset.i, !!(e.ctrlKey || e.metaKey));
  });
  el("table-rows").addEventListener("mouseover", e => {
    const row = e.target.closest("[data-i]");
    if (row && row.dataset.on !== "1") row.style.background = "#F4F5F7";
  });
  el("table-rows").addEventListener("mouseout", e => {
    const row = e.target.closest("[data-i]");
    if (row && !(e.relatedTarget && row.contains(e.relatedTarget)))
      row.style.background = row.dataset.on === "1" ? "#E9EBEE" : "transparent";
  });
  el("prev-btn").addEventListener("click", () => setState({ page: Math.max(0, state.page - 1) }));
  el("next-btn").addEventListener("click", () => setState({ page: state.page + 1 }));
  el("insp-prev").addEventListener("click", () => step(-1));
  el("insp-next").addEventListener("click", () => step(1));
  el("insp-close").addEventListener("click", clearSelection);
  el("insp-edge").addEventListener("mousedown", dragResize);
  el("resize-notch").addEventListener("mousedown", dragResize);
  el("rc-within-close").addEventListener("click", () => setState({ qcZoomAssay: null }));
  window.addEventListener("resize", resizeVisible);
}

function boot() {
  el("run-items").innerHTML = [
    { k: "Input", v: D.meta.input },
    { k: "Fit model", v: D.meta.fitModel },
    { k: "MIC thr.", v: String(D.meta.micThreshold) },
    { k: "q thr.", v: String(D.meta.qThreshold) },
    { k: "Version", v: D.meta.toolVersion },
  ].filter(x => x.v).map(m =>
    `<div class="meta-row"><span>${esc(m.k)}</span><span data-tip="${esc(m.v)}">${esc(m.v)}</span></div>`).join("");
  wire();
  initButtonHover();
  initKeys();
  Object.keys(R).forEach(name => R[name]());
  afterUpdate();
}
document.addEventListener("DOMContentLoaded", boot);
})();
</script>
</body>
</html>
"""


def main():
    args = parse_args()
    emit_message("info", "Report generation", lines=[
        *get_run_logo_lines(),
        format_detail("Version", TOOL_VERSION),
        format_detail("Run directory", style_text(str(args.run_dir), TEXT_COLOURS["accent"])),
    ])
    try:
        output_path, payload = write_report(args.run_dir, args.output)
    except Exception as error:
        emit_message("error", "Report generation failed", lines=format_detail("Reason", str(error)))
        traceback.print_exc()
        raise
    emit_message("success", "Report written", lines=[
        format_detail("Report", style_text(str(output_path), TEXT_COLOURS["accent"])),
        format_detail("Treatments", f"{len(payload['treatments']):,}"),
        format_detail("Organisms", f"{len(payload['organisms']):,}"),
    ])


if __name__ == "__main__":
    main()
