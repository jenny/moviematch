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


def _documented_values():
    """Every default a reader plans around, including the two derived figures."""
    return (
        str(dev.DEFAULT_STEP_MS),
        str(dev.DEFAULT_DEBOUNCE_MS),
        str(dev.DEFAULT_GRACE_S),
        dev.DEFAULT_HOST,
        dev.DEFAULT_PORT,
        # Quiet gap in seconds — "restarts 2 seconds after your last save".
        str(dev.DEFAULT_STEP_MS // 1000),
        # Non-stop editing, debounce + step in seconds — the "~17s" figure.
        str((dev.DEFAULT_DEBOUNCE_MS + dev.DEFAULT_STEP_MS) // 1000),
    )


def _docstring_settings_section() -> str:
    """Just the Settings block, so a value elsewhere in the docstring can't count.

    Searching the whole docstring would let this test pass on a table that had lost
    a row: the Usage examples already contain 127.0.0.1 and 8000.
    """
    doc = dev.__doc__
    return doc[doc.index("Settings") : doc.index("Usage", doc.index("Settings"))]


def _readme_dev_runner_section() -> str:
    """Just the debounced-runner part of README.md, for the same reason."""
    readme = (ROOT / "README.md").read_text()
    start = readme.index("`--reload` restarts the server once per saved file")
    return readme[start : readme.index("## API", start)]


@pytest.mark.parametrize("value", _documented_values())
def test_docstring_settings_table_matches_the_constants(value):
    # The table is hand-written prose, so a changed constant silently makes it lie.
    assert value in _docstring_settings_section(), (
        f"tools/dev.py Settings table no longer documents {value!r}"
    )


@pytest.mark.parametrize("value", _documented_values())
def test_readme_documents_the_same_defaults(value):
    # README.md is where a human meets this tool first; it repeats every default,
    # including the prose figures derived from step and debounce.
    assert value in _readme_dev_runner_section(), (
        f"README.md no longer documents {value!r}"
    )


def test_step_stays_below_the_debounce_ceiling():
    # If step >= debounce, the ceiling always fires first and the "wait for quiet"
    # behaviour this tool exists for never happens. Measured: with distinct files
    # changing non-stop, batches arrive every debounce + step, so the ceiling needs
    # real headroom above the quiet gap to stay a ceiling and not the main timer.
    assert dev.DEFAULT_STEP_MS < dev.DEFAULT_DEBOUNCE_MS
