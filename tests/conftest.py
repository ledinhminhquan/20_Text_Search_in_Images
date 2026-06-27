"""Shared pytest fixtures — force offline, isolate artifacts under a tmp dir."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("MPLBACKEND", "Agg")


@pytest.fixture(scope="session", autouse=True)
def _artifacts(tmp_path_factory):
    d = tmp_path_factory.mktemp("imgtext_artifacts")
    os.environ["IMGTEXT_ARTIFACTS_DIR"] = str(d)
    os.environ["HF_HOME"] = str(d / "hf")
    return d


@pytest.fixture
def cfg():
    from imgtextsearch.config import AppConfig
    c = AppConfig()
    c.data.use_hf = False
    return c
