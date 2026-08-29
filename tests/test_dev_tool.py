"""Static checks for `tools/dev.py` — the debounced dev-server runner.

Every failure mode guarded here is *silent*. A dropped suffix means the reloader
quietly stops noticing a file type you edit, and a step at or above the debounce
means the quiet-gap wait never happens at all. Neither raises — the server just
stops restarting when you expect it to.
"""

import importlib.util
from pathlib import Path

import pytest
from watchfiles import Change

ROOT = Path(__file__).resolve().parent.parent


def _load_dev_module():
    """Load tools/dev.py by path — `tools/` is a script directory, not a package."""
    spec = importlib.util.spec_from_file_location("dev_tool", ROOT / "tools" / "dev.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dev = _load_dev_module()


@pytest.mark.parametrize(
    "relative_path",
    [
        "config.py",
        "main.py",
        "api/routes/search.py",
        "app.html",  # read once in the lifespan, so an edit needs a restart
        "admin.html",
        "hints.json",
    ],
)
def test_watched_files_trigger_a_restart(relative_path):
    assert dev.DevFilter()(Change.modified, str(ROOT / relative_path))


@pytest.mark.parametrize(
    "relative_path",
    [
        "README.md",  # wrong suffix
        ".env",
        "venv/lib/python3.11/site-packages/x.py",
        "data/movies.json",
        "embeddings/chroma.sqlite3",
        "logs/app.py",  # the server writes here; watching it would loop
        "tests/test_api.py",  # editing a test shouldn't restart the server
        "local/scratch.py",
        "__pycache__/config.cpython-311.pyc",
    ],
)
def test_ignored_paths_do_not_trigger_a_restart(relative_path):
    assert not dev.DevFilter()(Change.modified, str(ROOT / relative_path))


def test_step_stays_below_the_debounce_ceiling():
    # If step >= debounce, the ceiling always fires first and the "wait for quiet"
    # behaviour this tool exists for never happens. Measured: with distinct files
    # changing non-stop, batches arrive every debounce + step, so the ceiling needs
    # real headroom above the quiet gap to stay a ceiling and not the main timer.
    assert dev.DEFAULT_STEP_MS < dev.DEFAULT_DEBOUNCE_MS
