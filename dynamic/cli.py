#!/usr/bin/env python3

import difflib
import importlib
import sys

from dynamic import ASCII_logo, __version__

COMMANDS = {
    "run": ("dynamic.dynamic", "Run the complete DynaMIC pipeline"),
    "normalize": ("dynamic.normalization", "Normalize optical density to inhibition fractions"),
    "calibrate": ("dynamic.calibration", "Build the organism-specific empirical nulls"),
    "classify": ("dynamic.classification", "Classify dilution series against the nulls"),
    "model": ("dynamic.modelling", "Fit dose-response curves and estimate MICs"),
    "report": ("dynamic.report", "Build the interactive HTML report for a run"),
    "test": ("dynamic.test", "Verify the installation against the bundled dataset"),
}

USAGE = "\n".join([
    *ASCII_logo,
    "",
    "usage: dynamic <command> [options]",
    "       dynamic [options]            same as 'dynamic run'",
    "",
    "commands:",
    *[f"  {name:<10} {description}" for name, (_, description) in COMMANDS.items()],
    "",
    "  --version   show the DynaMIC version and exit",
    "  --help      show this message; 'dynamic <command> --help' for a command",
])


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)

    if argv and argv[0] in ("--version", "--v"):
        print(f"DynaMIC {__version__}")
        return 0
    if not argv or argv[0] in ("--help", "--h", "help"):
        print(USAGE)
        return 0

    command = "run" if argv[0].startswith("-") else argv[0]
    if command not in COMMANDS:
        close = difflib.get_close_matches(command, COMMANDS, n=1)
        print(f"dynamic: unknown command '{command}'"
              + (f"; did you mean '{close[0]}'?" if close else ""), file=sys.stderr)
        print(f"\n{USAGE}", file=sys.stderr)
        return 2
    if not argv[0].startswith("-"):
        argv = argv[1:]

    module_name = COMMANDS[command][0]
    sys.argv = [f"dynamic {command}", *argv]
    try:
        return importlib.import_module(module_name).main() or 0
    except KeyboardInterrupt:
        print("\ndynamic: interrupted", file=sys.stderr)
        return 130
    except BrokenPipeError:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
