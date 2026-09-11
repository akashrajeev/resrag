from __future__ import annotations

import os
import re
from io import StringIO
from pathlib import Path

from dotenv import dotenv_values

_ENV_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$")


def load_resrag_env(path: str | Path = ".env") -> list[int]:
    """Load a .env file while ignoring malformed lines without noisy parser warnings.

    Values already present in the process environment take precedence. The return
    value contains 1-based line numbers that were ignored so the UI can surface a
    concise configuration hint when needed.
    """
    env_path = Path(path)
    if not env_path.exists() or not env_path.is_file():
        return []

    valid_lines: list[str] = []
    invalid_lines: list[int] = []

    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []

    for line_number, raw_line in enumerate(lines, start=1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            valid_lines.append(raw_line)
            continue
        if _ENV_LINE.match(raw_line):
            valid_lines.append(raw_line)
        else:
            invalid_lines.append(line_number)

    # Every remaining non-comment line has KEY=VALUE syntax, so python-dotenv can
    # parse it without emitting the repeated "could not parse statement" warnings.
    values = dotenv_values(stream=StringIO("\n".join(valid_lines)))
    for key, value in values.items():
        if value is not None and key not in os.environ:
            os.environ[key] = value

    return invalid_lines
