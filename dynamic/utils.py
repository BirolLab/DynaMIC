#!/usr/bin/env python3

import argparse
import csv
from datetime import datetime
import importlib
import json
import multiprocessing
import os
from pathlib import Path
import re
import sys

from dynamic import ASCII_logo, USE_COLOUR

try:
    from tqdm.auto import tqdm
except ImportError:
    class tqdm:

        def __init__(self, total=None, desc="", unit="it", **kwargs):
            self.total = total
            self.n = 0
            self.desc = desc
            self.unit = unit
            if self.desc:
                self._print()

        def set_description(self, desc):
            self.desc = desc
            self._print()

        def set_description_str(self, desc):
            self.set_description(desc)

        def update(self, n=1):
            self.n += n
            self._print()

        def refresh(self):
            self._print()

        def close(self):
            pass

        @staticmethod
        def write(message=""):
            print(message)

        def _print(self):
            if self.total is None:
                print(f"{self.desc}: {self.n}")
            else:
                print(f"{self.desc}: {self.n}/{self.total} {self.unit}")


ANSI_ESCAPE_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
PROGRESS_LINE_RE = re.compile(
    r"^\s*(?:"
    r"\[?\d{2}/\d{2}\]?.*\d{1,3}%.*\d+/\d+"
    r"|2PL (?:curve )?fits\s+.*\d{1,3}%.*\d+/\d+"
    r"|.*\s+\d{1,3}%.*\d+/\d+"
    r")"
)
PLOT_DPI = 300
TEXT_COLOURS = {"accent": "38;2;96;165;250", "muted": "38;2;148;163;184", "info": "38;2;147;197;253",
                "stage": "38;2;196;181;253", "warn": "38;2;251;191;36",
                "success": "38;2;110;231;183", "error": "38;2;248;113;113"}
TAG_STYLES = {"info": ("INFO", "97", "48;2;30;41;59"), "warn": ("WARN", "97", "48;2;120;53;15"),
              "stage": ("INFO", "97", "48;2;88;28;135"),
              "success": ("DONE", "97", "48;2;6;78;59"), "error": ("FAIL", "97", "48;2;127;29;29")}
SUPPRESS_RUN_LOGO = os.getenv("DYNAMIC_SUPPRESS_STAGE_LOGO") == "1"


def get_run_logo_lines():
    return [] if SUPPRESS_RUN_LOGO else list(ASCII_logo)


def style_text(text, *codes):
    if not USE_COLOUR or not codes:
        return text
    return f"\033[{';'.join(codes)}m{text}\033[0m"


def console_write(message="", log_message=None):
    if log_message is not None and hasattr(sys.stdout, "write_log_message"):
        sys.stdout.suppress_logging = True
        try:
            tqdm.write(message)
        finally:
            sys.stdout.suppress_logging = False
        sys.stdout.write_log_message(log_message)
        return
    tqdm.write(message)


def format_detail(label, value):
    return f"{style_text(f'{label}:', '1', TEXT_COLOURS['muted'])} {value}"


def format_bullet(text):
    return f"{style_text('•', '1', '97')} {text}"


def shorten_text(text, max_length=54):
    text = str(text)
    if len(text) <= max_length:
        return text
    return f"{text[:max_length - 1]}…"


def format_message_block(kind, title, lines=None, items=None, max_items=None):
    tag_label, tag_fg, tag_bg = TAG_STYLES[kind]
    title_colour = TEXT_COLOURS[kind]
    block = [f"{style_text(f' {tag_label} ', '1', tag_fg, tag_bg)} {style_text(title, '1', title_colour)}"]
    block.extend(f"       {line}" for line in ([lines] if isinstance(lines, str) else (lines or [])))
    item_list = [items] if isinstance(items, str) else (items or [])
    visible_items = item_list if max_items is None else item_list[:max_items]
    block.extend(f"       {format_bullet(item)}" for item in visible_items)
    if max_items is not None and len(item_list) > max_items:
        block.append(f"       {format_bullet(f'{len(item_list) - max_items:,} additional records omitted from this summary.')}")
    return "\n".join(block)


def emit_message(kind, title, lines=None, items=None, max_items=12):
    console_block = format_message_block(kind, title, lines=lines, items=items, max_items=max_items)
    item_count = 1 if isinstance(items, str) else len(items or [])
    truncated = max_items is not None and item_count > max_items
    log_block = format_message_block(kind, title, lines=lines, items=items) if truncated else console_block
    console_write(console_block, log_message=log_block)


def format_stage_label(stage_number, total_stages, stage_name):
    return f"{stage_number:02d}/{total_stages:02d}  {shorten_text(stage_name)}"


def set_progress_stage(progress, stage_number, total_stages, stage_name):
    progress.set_description_str(format_stage_label(stage_number, total_stages, stage_name))


def set_progress_total(progress, total_steps):
    progress.total = total_steps
    refresh_progress = getattr(progress, "refresh", None)
    if callable(refresh_progress):
        refresh_progress()


def create_progress_bar(total, initial_stage, compact=False):
    if compact:
        progress_kwargs = {"total": total, "desc": initial_stage, "unit": "fit", "dynamic_ncols": True,
                           "leave": False, "file": sys.stdout, "mininterval": 0.5, "smoothing": 0.08,
                           "bar_format": "{desc:<10} {percentage:3.0f}%  {bar:20}  {n_fmt}/{total_fmt}"}
    else:
        progress_kwargs = {"total": total, "desc": initial_stage, "unit": "step", "dynamic_ncols": True,
                           "leave": False, "file": sys.stdout, "mininterval": 0.25, "smoothing": 0.08,
                           "bar_format": "{desc:<62} {percentage:3.0f}%  {bar:30}  {n_fmt}/{total_fmt}  {elapsed}<{remaining}"}
    try:
        return tqdm(colour="#22c55e", **progress_kwargs)
    except TypeError:
        return tqdm(**progress_kwargs)


class LogoArgumentParser(argparse.ArgumentParser):

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("add_help", False)
        super().__init__(*args, **kwargs)
        self.add_argument("--help", action="help", help="Show this message and exit.")
        self.add_argument("--h", action="help", help=argparse.SUPPRESS)

    def format_help(self):
        return "\n".join(ASCII_logo) + "\n\n" + super().format_help()

    def error(self, message):
        emit_message("error", "Invalid command line", lines=message)
        raise SystemExit(2)


def strip_terminal_control(text):
    return ANSI_ESCAPE_RE.sub("", text)


def is_progress_line(line):
    return bool(PROGRESS_LINE_RE.match(line.strip()))


class TeeStream:

    def __init__(self, stream, log_handle, stream_name):
        self.stream = stream
        self.log_handle = log_handle
        self.stream_name = stream_name
        self.log_buffer = ""
        self.suppress_logging = False

    def write(self, text):
        self.stream.write(text)
        if not self.suppress_logging:
            self._write_clean_log(text)

    def flush(self):
        self.stream.flush()
        if self.log_buffer.strip():
            self._write_log_line(self.log_buffer)
            self.log_buffer = ""
        self.log_handle.flush()

    def isatty(self):
        return self.stream.isatty()

    @property
    def encoding(self):
        return getattr(self.stream, "encoding", "utf-8")

    def _write_clean_log(self, text):
        text = strip_terminal_control(text).replace("\r", "\n")
        if not text:
            return

        parts = text.split("\n")
        parts[0] = self.log_buffer + parts[0]
        for line in parts[:-1]:
            self._write_log_line(line)
        self.log_buffer = parts[-1]

    def write_log_message(self, text):
        if self.log_buffer.strip():
            self._write_log_line(self.log_buffer)
            self.log_buffer = ""
        for line in strip_terminal_control(text).replace("\r", "\n").split("\n"):
            self._write_log_line(line)

    def _write_log_line(self, line):
        line = line.rstrip()
        if not line.strip() or is_progress_line(line):
            return
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        prefix = f"[{timestamp}]"
        if self.stream_name == "stderr":
            prefix = f"{prefix} STDERR"
        self.log_handle.write(f"{prefix} {line}\n")


class RunLogger:

    def __init__(self, log_path):
        self.log_path = Path(log_path)
        self.log_handle = None
        self.stdout = None
        self.stderr = None

    def __enter__(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_handle = self.log_path.open("w", encoding="utf-8")
        self.stdout, self.stderr = sys.stdout, sys.stderr
        sys.stdout = TeeStream(self.stdout, self.log_handle, "stdout")
        sys.stderr = TeeStream(self.stderr, self.log_handle, "stderr")
        return self

    def __exit__(self, exc_type, exc, traceback):
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout, sys.stderr = self.stdout, self.stderr
        self.log_handle.close()
        return False


def preload_fork_worker_modules(*module_names):
    if module_names and multiprocessing.get_start_method() == "fork":
        for module_name in module_names:
            importlib.import_module(module_name)


class BackgroundRenderer:

    def __init__(self, executor=None):
        self.executor = executor
        self.futures = []

    def submit(self, render_function, *args, **kwargs):
        if self.executor is None:
            render_function(*args, **kwargs)
        else:
            self.futures.append(self.executor.submit(render_function, *args, **kwargs))

    def wait(self):
        try:
            for future in self.futures:
                future.result()
        finally:
            self.futures = []


def prepare_output_directories(run_dir, folder_names):
    directories = {name: Path(run_dir) / name for name in folder_names}
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    return directories


def unique_directory_path(path):
    path = Path(path)
    if not path.exists():
        return path

    suffix = 2
    while True:
        candidate = path.with_name(f"{path.name}_{suffix:02d}")
        if not candidate.exists():
            return candidate
        suffix += 1


def create_unique_directory(path):
    path = unique_directory_path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def create_dated_run_directory(output_dir, base_path, prefix):
    if output_dir is not None:
        return create_unique_directory(output_dir)
    date_stamp = datetime.now().strftime("%Y-%m-%d")
    return create_unique_directory(Path(base_path).parent / f"{prefix}_{date_stamp}")


def create_cwd_dated_run_directory(output_dir, prefix, parent_name="output"):
    if output_dir is not None:
        return create_unique_directory(output_dir)
    date_stamp = datetime.now().strftime("%Y-%m-%d")
    return create_unique_directory(Path.cwd() / parent_name / f"{prefix}_{date_stamp}")


def get_dose_labels(inhibition_columns):
    return [str(column)[1:] if str(column).startswith("I") else str(column) for column in inhibition_columns]


def format_dose_layout_full(inhibition_columns):
    return ", ".join(get_dose_labels(inhibition_columns))


def ensure_unique_filename(path):
    path = Path(path)
    if not path.exists():
        return path

    suffix = 2
    while True:
        candidate = path.with_name(f"{path.stem}_{suffix:02d}{path.suffix}")
        if not candidate.exists():
            return candidate
        suffix += 1


def parse_csv_list(value):
    return [item for item in str(value).split(",") if item != ""]


def slugify_filename_token(value, default="item"):
    token = re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_")
    return token or default


def save_plot_figure(fig, plot_path, dpi=PLOT_DPI):
    plot_path = Path(plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=dpi, bbox_inches="tight", pad_inches=0.10)
    return plot_path


def _json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def _csv_ready(value):
    return json.dumps(value) if isinstance(value, (list, dict)) else value


def write_run_parameters(args, logs_dir, script_name, tool_version, extra=None):
    started_at = datetime.now().isoformat(timespec="seconds")
    parameters = {key: _json_ready(value) for key, value in vars(args).items()}
    extra = {key: _json_ready(value) for key, value in (extra or {}).items()}
    record = {
        "script": script_name,
        "tool_version": tool_version,
        "started_at": started_at,
        "parameters": parameters,
        "extra": extra,
    }

    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    json_path = logs_dir / "run_parameters.json"
    csv_path = logs_dir / "run_parameters.csv"
    json_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["parameter", "value"])
        writer.writerow(["script", script_name])
        writer.writerow(["tool_version", tool_version])
        writer.writerow(["started_at", started_at])
        writer.writerows((key, _csv_ready(value)) for key, value in parameters.items())
        writer.writerows((key, _csv_ready(value)) for key, value in extra.items())

    return json_path, csv_path


def require_columns(headers, required_columns, file_label, path):
    missing = sorted(set(required_columns) - set(headers))
    if missing:
        raise ValueError(f"{file_label} is missing required column(s): {', '.join(missing)}. File: {path}")


def _parse_number(value, cast, type_message):
    try:
        return cast(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(type_message) from error


def positive_int(value):
    parsed = _parse_number(value, int, "must be an integer")
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


def auto_or_positive_int(value):
    if value == "auto":
        return value
    parsed = _parse_number(value, int, "must be 'auto' or a positive integer")
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be 'auto' or a positive integer")
    return parsed


def nonnegative_int(value):
    parsed = _parse_number(value, int, "must be an integer")
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return parsed


def positive_float(value):
    parsed = _parse_number(value, float, "must be a number")
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


def probability(value):
    parsed = _parse_number(value, float, "must be a number")
    if not 0 <= parsed <= 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def open_probability(value):
    parsed = _parse_number(value, float, "must be a number")
    if not 0 < parsed < 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed
