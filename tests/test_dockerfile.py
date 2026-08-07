"""Static checks on the embedding-model pre-bake in the Dockerfile.

The image downloads the SentenceTransformers weights at build time so a container cold
start never pulls ~418MB from the HuggingFace Hub. Three things have to stay true for
that to hold, and each fails in a way that is easy to ship without noticing:

  - The pinned model must match config.MODEL_NAME. The Dockerfile can't read config.py
    (it is copied after the prefetch, on purpose — see below), so the name is duplicated.
    On drift, HF_HUB_OFFLINE=1 turns the mismatch into a startup crash in production.
  - The prefetch must stay ABOVE `COPY . .`, or every app-code edit invalidates the
    418MB layer and each deploy re-downloads the weights.
  - HF_HUB_OFFLINE=1 must come AFTER the prefetch, or it blocks the very download it
    exists to make unnecessary, failing the build.

No Docker build is required — these parse the Dockerfile as text.
"""

import re
from pathlib import Path

import pytest

import config

DOCKERFILE = Path(__file__).resolve().parent.parent / "Dockerfile"


@pytest.fixture(scope="module")
def dockerfile() -> str:
    return DOCKERFILE.read_text()


def _line_index(dockerfile: str, pattern: str) -> int:
    """Return the 0-based line number of the first line matching `pattern`."""
    for i, line in enumerate(dockerfile.splitlines()):
        if re.search(pattern, line):
            return i
    raise AssertionError(f"no line in Dockerfile matches {pattern!r}")


def test_pinned_model_matches_config(dockerfile: str) -> None:
    """The ARG default is a duplicate of config.MODEL_NAME — keep them in sync."""
    match = re.search(r"^ARG\s+EMBEDDING_MODEL=(\S+)", dockerfile, re.M)
    assert match, "ARG EMBEDDING_MODEL not found in Dockerfile"
    assert match.group(1) == config.MODEL_NAME, (
        f"Dockerfile pins EMBEDDING_MODEL={match.group(1)!r} but "
        f"config.MODEL_NAME is {config.MODEL_NAME!r}. With HF_HUB_OFFLINE=1 this "
        f"mismatch crashes the app at startup instead of downloading the model."
    )


def test_prefetch_uses_the_arg(dockerfile: str) -> None:
    """The prefetch must interpolate the ARG, not re-hardcode a name past the guard."""
    prefetch = re.search(
        r"^RUN python -c .*SentenceTransformer\((.*?)\)", dockerfile, re.M
    )
    assert prefetch, "model prefetch RUN not found in Dockerfile"
    assert "${EMBEDDING_MODEL}" in prefetch.group(1), (
        "prefetch should load '${EMBEDDING_MODEL}' so test_pinned_model_matches_config "
        f"actually guards it; got {prefetch.group(1)!r}"
    )


def test_prefetch_precedes_app_code_copy(dockerfile: str) -> None:
    """Layer-cache guard: app-code edits must not invalidate the 418MB model layer."""
    prefetch = _line_index(dockerfile, r"SentenceTransformer\(")
    copy_all = _line_index(dockerfile, r"^COPY \. \.")
    assert prefetch < copy_all, (
        "the model prefetch RUN must come before `COPY . .`, otherwise every change to "
        "app code busts the cached layer and re-downloads the weights on each deploy"
    )


def test_hub_offline_set_after_prefetch(dockerfile: str) -> None:
    """HF_HUB_OFFLINE=1 blocks Hub access, so it must not precede the prefetch."""
    offline = _line_index(dockerfile, r"^ENV\s+HF_HUB_OFFLINE=1")
    prefetch = _line_index(dockerfile, r"SentenceTransformer\(")
    assert offline > prefetch, (
        "ENV HF_HUB_OFFLINE=1 must be set after the prefetch RUN — set earlier it "
        "blocks the download it is meant to make unnecessary, breaking the build"
    )
