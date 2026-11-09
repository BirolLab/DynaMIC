__version__ = "1.0.0"

import ctypes
import os
import sys

PR_SET_NAME = 15

if sys.platform == "linux":
    try:
        ctypes.CDLL("libc.so.6").prctl(PR_SET_NAME, b"DynaMIC", 0, 0, 0)
    except (OSError, AttributeError):
        pass

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

USE_COLOUR = sys.stdout.isatty() and os.getenv("TERM") != "dumb" and os.getenv("NO_COLOR") is None

COLOURS = {
    "W": "244;247;250", "G": "0;176;80", "0": "37;187;106", "1": "25;183;97",
    "2": "87;201;140", "3": "62;194;123", "4": "131;214;171", "5": "124;212;166",
    "6": "157;222;189", "7": "200;234;220", "8": "186;230;210", "9": "242;247;249"
}

LOGO = (
    ("⣿⡿⠿⢿⣷⣦", "WWWWWW"),
    ("⣿⡇  ⢹⣿⡇⣿⣷ ⢰⣿⠇⣿⣷⡿⢿⣿⡄⠰⠿⠿⢿⣿⡄", "WW  WWWWW WWWWWWWWWWWWWWW"),
    ("⣿⡇  ⣸⣿⠇⠘⣿⣇⣿⡟ ⣿⣿ ⢸⣿⡇⢠⣶⡶⢾⣿⡇", "WW  WWWWWWWW WW WWWWWWWWW"),
    ("⣿⣷⣶⣾⡿⠋  ⢹⣿⣿⠁ ⣿⣿ ⢸⣿⡇⠸⣿⣦⡾⣿⡇", "WWWWWW  WWWW WW WWWWWWWWW"),
    ("⣶⡆      ⣰⣿⠇    ⢀⣤⣶⣶⣶⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷", "WW      WWW    GGGGGGGGGGGGGGGGG"),
    ("⣿⡇      ⠈⠁    ⣴⣿⠟⠉       ⣀⣠⣤⣶⣿⠿⠟", "WW      WW    GGGG       GGGGGGG"),
    ("⣿⡇           ⣸⣿⠏        ⣿⣿⣿⣏⣉", "WW           GGG        GGGGG"),
    ("⣿⡇          ⢠⣿⡟          ⠈⠙⠛⠿⢿⣶⣦", "WW          001          GGGGGGG"),
    ("⣿⡇          ⣾⣿⠁         ⠿⠿⠿⠿⠿⠿⠿⠿", "WW          223         GGGGGGGG"),
    ("⣿⡇         ⢰⣿⡏          ⣤⣤⣤⣤⣤⣤⣤⣤", "WW         445          GGGGGGGG"),
    ("⣿⡇         ⣾⣿           ⠛⠛⠛⠛⠛⠛⠛⠛", "WW         66           GGGGGGGG"),
    ("⣿⡇        ⣼⣿⠇           ⢠⣴⣾⣿⣿⣷⣦⡀", "WW        778           GGGGGGGG"),
    ("⣿⡇      ⢀⣼⣿⠏            ⣿⡟⠉  ⠉⢻⣿", "WW      WWW9            GGG  GGG"),
    ("⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣶⣶⣶⣶⣶⣶⣶⣶⣶⣶⣶⣶⡆⢿⣧    ⣼⡿", "WWWWWWWWWWWWWWWWWWWWWWWWGG    GG"),
)


def render_logo():
    lines = []
    for glyphs, tints in LOGO:
        out, active = [], None
        for glyph, tint in zip(glyphs, tints):
            if USE_COLOUR and tint != " " and tint != active:
                out.append(f"\x1b[38;2;{COLOURS[tint]}m")
                active = tint
            out.append(glyph)
        lines.append("".join(out) + ("\x1b[0m" if active else ""))
    return tuple(lines)


ASCII_logo = render_logo()
