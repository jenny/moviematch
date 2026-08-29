#!/usr/bin/env python
"""Dev server with a debounced reloader — one restart per *burst* of edits.

Why this exists
---------------
`uvicorn --reload` restarts once per saved file. Its `--reload-delay` helps, but it
is a rate limit, not a quiet-gap timer: the reloader sleeps N seconds between checks,
so the first save of a burst still restarts immediately and a lone save can sit for
up to N seconds (measured: six saves over six seconds cost six restarts by default,
two with `--reload-delay 5`).

This runner uses watchfiles' own timings instead, which uvicorn never exposes:

    step      how long to wait for quiet before reporting a batch  ("stop typing")
    debounce  the ceiling — report anyway after this long of unbroken editing

Two regimes, both measured:

  * You pause longer than `step` — one restart, about `step` after your last save.
    Save four files in two seconds, get one restart.
  * You never pause (an agent writing file after file) — the ceiling takes over and
    restarts arrive about every `debounce` + `step`, so ~17s at the defaults.

That matters here because every restart re-runs the FastAPI lifespan, which warms
the embedding model and the TMDB connection (see api/app.py).

Usage
-----
    python tools/dev.py                    # http://127.0.0.1:8000
    python tools/dev.py --port 8001
    python tools/dev.py --step 500         # snappier, more restarts

Run it instead of `uvicorn --reload`, not alongside it: this process *is* the
reloader, so the uvicorn it spawns is deliberately started without `--reload`.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from watchfiles import DefaultFilter, run_process

# Project root, resolved from this file so the script works from any cwd.
ROOT = Path(__file__).resolve().parent.parent

# Restart only after this many ms of no file changes ("wait until I stop typing").
DEFAULT_STEP_MS = 2000
# ...but never sit on a pending change longer than this, however long the burst runs.
# Only bites while genuinely new changes keep arriving; then batches land roughly
# every debounce + step (measured: 15000 + 2000 → ~18s). Repeated saves of the *same*
# file dedupe, so they stop counting as new and `step` ends the batch instead.
DEFAULT_DEBOUNCE_MS = 15_000
# Ignore changes for this many seconds after a start, so a stray save during the
# model warm-up doesn't kill a server that hasn't finished booting.
DEFAULT_GRACE_S = 3.0

# Only files whose contents the running server actually reads at startup.
# .html and .json are here because api/app.py reads app.html, admin.html,
# login.html and hints.json once in the lifespan — editing them needs a restart.
WATCHED_SUFFIXES = (".py", ".html", ".json")

# DefaultFilter.ignore_dirs covers .git, __pycache__, .venv, .pytest_cache and
# friends, but not this project's own heavy or irrelevant directories. Passing
# ignore_dirs replaces the defaults, so splat them back in.
IGNORED_DIRS = (
    *DefaultFilter.ignore_dirs,
    "venv",  # forbidden by CLAUDE.md, and huge
    "data",  # raw TMDB dumps
    "embeddings",  # ChromaDB binaries
    "logs",  # the server writes here itself — watching it would loop
    "local",  # scratch notes
    "tests",  # editing a test shouldn't restart the server
    ".vscode",
)

logger = logging.getLogger("dev")

# Counted in the callback rather than taken from run_process's return value, so the
# summary means exactly one thing: batches of changes that triggered a restart.
_restarts = 0


def _relative(path: str) -> str:
    """Path relative to the project root, or absolute if it lies outside it."""
    try:
        return str(Path(path).relative_to(ROOT))
    except ValueError:  # pragma: no cover - only for paths outside the repo
        return path


class DevFilter(DefaultFilter):
    """DefaultFilter, narrowed to the file types that require a server restart."""

    def __init__(self) -> None:
        super().__init__(ignore_dirs=IGNORED_DIRS)

    def __call__(self, change, path: str) -> bool:
        # Same shape as watchfiles' own PythonFilter: extension check first (cheap),
        # then the inherited directory/pattern rules.
        return path.endswith(WATCHED_SUFFIXES) and super().__call__(change, path)


def on_changes(changes) -> None:
    """Log the batch that is about to cause a restart.

    watchfiles calls this once per batch, immediately before it restarts the target,
    so the count here is the number of restarts and the paths explain each one.
    """
    global _restarts
    _restarts += 1
    paths = sorted(_relative(path) for _, path in changes)
    shown = ", ".join(paths[:5])
    if len(paths) > 5:
        shown += f", +{len(paths) - 5} more"
    logger.info("restart #%d — %d file(s) changed: %s", _restarts, len(paths), shown)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default="8000")
    parser.add_argument(
        "--step",
        type=int,
        default=DEFAULT_STEP_MS,
        metavar="MS",
        help=f"ms of quiet before restarting (default: {DEFAULT_STEP_MS})",
    )
    parser.add_argument(
        "--debounce",
        type=int,
        default=DEFAULT_DEBOUNCE_MS,
        metavar="MS",
        help=f"max ms to hold a pending change (default: {DEFAULT_DEBOUNCE_MS})",
    )
    parser.add_argument(
        "--grace",
        type=float,
        default=DEFAULT_GRACE_S,
        metavar="S",
        help=f"seconds to ignore changes after a start (default: {DEFAULT_GRACE_S})",
    )
    parser.add_argument(
        "--target",
        default=None,
        help="command to run instead of uvicorn (for testing the reloader itself)",
    )
    args = parser.parse_args()

    # uvicorn needs the project root on sys.path to import api.app, and run_process
    # inherits this process's cwd for the command it spawns.
    os.chdir(ROOT)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s dev: %(message)s",
        datefmt="%H:%M:%S",
    )
    # watchfiles logs its own "N changes detected" line for the same batch our
    # callback reports, with less detail. Keep one line per restart, not two.
    logging.getLogger("watchfiles").setLevel(logging.WARNING)

    # No --reload: this process is the reloader.
    target = args.target or f"uvicorn api.app:app --host {args.host} --port {args.port}"

    logger.info("watching %s", ROOT)
    logger.info(
        "step=%dms debounce=%dms grace=%.1fs — restarts after %.1fs of quiet, "
        "or every ~%.0fs while editing without a pause",
        args.step,
        args.debounce,
        args.grace,
        args.step / 1000,
        (args.debounce + args.step) / 1000,
    )
    logger.info("target: %s", target)

    try:
        run_process(
            ROOT,
            target=target,
            target_type="command",
            watch_filter=DevFilter(),
            callback=on_changes,
            step=args.step,
            debounce=args.debounce,
            grace_period=args.grace,
        )
    except KeyboardInterrupt:
        pass  # Ctrl-C is the normal way to stop; don't dump a traceback.

    logger.info("stopped after %d restart(s)", _restarts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
