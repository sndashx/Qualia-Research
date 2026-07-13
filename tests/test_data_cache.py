"""Tests for the on-disk dataset cache layer (bead fae9a030).

Coverage:

- Cache key derivation is stable, sensitive to every field, and includes
  the dataset version.
- :class:`SyntheticColoredShapes` is deterministic across constructions
  and emits the right tensor shapes / dtypes.
- :class:`CachedDataset` writes a ``.pt`` file on first construction
  (cache MISS) and reloads it on the second construction (cache HIT) —
  measured with :attr:`CachedDataset.stats`.
- The cache key changes when the preprocessing config changes, which forces
  a MISS even when the on-disk file is still there.
- ``rebuild=True`` overwrites the existing cache file.
- :func:`make_dataloader` reads ``cfg.data`` and returns a working DataLoader.
- The training entrypoint logs cache hit / miss + timing to stdout and to
  the run's tracker.
- The ``--rebuild-cache`` CLI flag forces cache regeneration.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from qualia.data import (
    DATASET_VERSION,
    CachedDataset,
    SyntheticColoredShapes,
    compute_cache_key,
    default_cache_root,
    make_dataloader,
    make_dataset,
)
from qualia.data import __init__ as _qualia_data  # noqa: F401  (import side-effects)

pytestmark = pytest.mark.smoke


CONFIG_DIR = str(Path(__file__).resolve().parent.parent / "configs")


# ---------------------------------------------------------------------------
# Cache key
# ---------------------------------------------------------------------------


def test_cache_key_is_stable_for_same_inputs() -> None:
    k1 = compute_cache_key(
        dataset_name="stochastic_colored_shapes",
        seed=0,
        n_samples=16,
        image_size=32,
        preprocessing_cfg={"flip": True},
    )
    k2 = compute_cache_key(
        dataset_name="stochastic_colored_shapes",
        seed=0,
        n_samples=16,
        image_size=32,
        preprocessing_cfg={"flip": True},
    )
    assert k1 == k2
    assert len(k1) == 32


def test_cache_key_changes_with_seed_and_size() -> None:
    base = dict(dataset_name="stochastic_colored_shapes", seed=0, n_samples=16, image_size=32)
    base_key = compute_cache_key(**base)
    assert compute_cache_key(**{**base, "seed": 1}) != base_key
    assert compute_cache_key(**{**base, "n_samples": 32}) != base_key
    assert compute_cache_key(**{**base, "image_size": 64}) != base_key


def test_cache_key_changes_with_preprocessing_cfg() -> None:
    base = compute_cache_key(
        dataset_name="stochastic_colored_shapes",
        seed=0,
        n_samples=16,
        image_size=32,
        preprocessing_cfg={"rotate_deg": 0},
    )
    rotated = compute_cache_key(
        dataset_name="stochastic_colored_shapes",
        seed=0,
        n_samples=16,
        image_size=32,
        preprocessing_cfg={"rotate_deg": 90},
    )
    assert base != rotated


def test_cache_key_changes_with_dataset_version() -> None:
    base = compute_cache_key(
        dataset_name="stochastic_colored_shapes",
        seed=0,
        n_samples=16,
        image_size=32,
        dataset_version=DATASET_VERSION,
    )
    bumped = compute_cache_key(
        dataset_name="stochastic_colored_shapes",
        seed=0,
        n_samples=16,
        image_size=32,
        dataset_version="999",
    )
    assert base != bumped


# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------


def test_synthetic_dataset_is_deterministic() -> None:
    a = SyntheticColoredShapes(n_samples=8, image_size=16, seed=42)
    b = SyntheticColoredShapes(n_samples=8, image_size=16, seed=42)
    for i in range(8):
        ia, _ = a[i]
        ib, _ = b[i]
        assert torch.equal(ia, ib)


def test_synthetic_dataset_shapes_and_dtypes(tmp_path: Path) -> None:
    ds = SyntheticColoredShapes(n_samples=4, image_size=16, seed=0)
    assert len(ds) == 4
    img, labels = ds[0]
    assert img.shape == (3, 16, 16)
    assert img.dtype == torch.float32
    assert torch.isfinite(img).all()
    assert set(labels.keys()) == {"shape", "color", "hue", "brightness", "agency"}
    assert labels["shape"].dtype == torch.long
    assert labels["color"].dtype == torch.long
    assert labels["hue"].dtype == torch.float32
    assert labels["brightness"].dtype == torch.float32
    assert labels["agency"].dtype == torch.float32


# ---------------------------------------------------------------------------
# CachedDataset
# ---------------------------------------------------------------------------


def _build_factory(n_samples: int, image_size: int, seed: int, preproc: dict | None = None):
    def _factory():
        return SyntheticColoredShapes(
            n_samples=n_samples,
            image_size=image_size,
            seed=seed,
            preprocessing_cfg=preproc,
        )

    return _factory


def test_cached_dataset_miss_then_hit(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    log_lines: list[str] = []

    def logger(msg: str) -> None:
        log_lines.append(msg)

    factory = _build_factory(n_samples=8, image_size=16, seed=1)
    first = CachedDataset(factory=factory, cache_root=cache_root, logger=logger)
    assert first.stats.hit is False
    assert first.stats.build_seconds >= 0.0
    assert first.stats.load_seconds == 0.0
    assert (cache_root / f"{first.cache_key}.pt").exists()
    assert any("cache MISS" in m for m in log_lines)

    second = CachedDataset(factory=factory, cache_root=cache_root, logger=logger)
    assert second.stats.hit is True
    assert second.stats.load_seconds >= 0.0
    assert second.stats.build_seconds == 0.0
    assert second.cache_key == first.cache_key
    assert any("cache HIT" in m for m in log_lines)

    # Samples returned from the cached dataset must be identical to those
    # from a freshly-built inner dataset (no in-place mutation, no copy bug).
    fresh = factory()
    for i in range(len(fresh)):
        cached_img, cached_lbl = second[i]
        fresh_img, fresh_lbl = fresh[i]
        assert torch.equal(cached_img, fresh_img)
        for k in cached_lbl:
            assert torch.equal(cached_lbl[k], fresh_lbl[k])


def test_cached_dataset_key_changes_with_preprocessing(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    factory_v1 = _build_factory(n_samples=4, image_size=16, seed=2, preproc={"v": 1})
    factory_v2 = _build_factory(n_samples=4, image_size=16, seed=2, preproc={"v": 2})

    first = CachedDataset(factory=factory_v1, cache_root=cache_root)
    second = CachedDataset(factory=factory_v2, cache_root=cache_root)
    assert first.cache_key != second.cache_key
    # Both files should now exist on disk.
    assert (cache_root / f"{first.cache_key}.pt").exists()
    assert (cache_root / f"{second.cache_key}.pt").exists()
    # And the second construction is a cache MISS (different key).
    assert second.stats.hit is False


def test_cached_dataset_rebuild_flag_forces_miss(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    factory = _build_factory(n_samples=4, image_size=16, seed=3)

    first = CachedDataset(factory=factory, cache_root=cache_root)
    assert first.stats.hit is False
    # Sanity: a second call without rebuild would HIT.
    second = CachedDataset(factory=factory, cache_root=cache_root)
    assert second.stats.hit is True

    # But rebuild=True forces a fresh MISS and overwrites the file.
    rebuilt = CachedDataset(factory=factory, cache_root=cache_root, rebuild=True)
    assert rebuilt.stats.hit is False
    assert rebuilt.cache_key == first.cache_key
    assert (cache_root / f"{rebuilt.cache_key}.pt").exists()


def test_cached_dataset_explicit_key_skips_probe(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    factory = _build_factory(n_samples=4, image_size=16, seed=4)
    probe = factory()
    explicit_key = probe.cache_key

    ds = CachedDataset(factory=factory, cache_root=cache_root, cache_key=explicit_key)
    assert ds.cache_key == explicit_key
    assert ds.stats.hit is False
    # And the second call hits the cache using the same explicit key.
    ds2 = CachedDataset(factory=factory, cache_root=cache_root, cache_key=explicit_key)
    assert ds2.stats.hit is True


# ---------------------------------------------------------------------------
# make_dataloader / make_dataset
# ---------------------------------------------------------------------------


def test_make_dataset_reads_cfg_data(tmp_path: Path) -> None:
    cfg = OmegaConf.create(
        {
            "seed": 7,
            "data": {
                "name": "stochastic_colored_shapes",
                "n_samples": 12,
                "image_size": 16,
                "batch_size": 4,
                "num_workers": 0,
                "shuffle": False,
                "rebuild_cache": True,
                "preprocessing": {"flip": False},
            },
        }
    )
    ds = make_dataset(cfg, cache_root=tmp_path / "cache")
    assert len(ds) == 12
    assert "shape" in ds.label_names()


def test_make_dataloader_yields_batches(tmp_path: Path) -> None:
    cfg = OmegaConf.create(
        {
            "seed": 8,
            "data": {
                "name": "stochastic_colored_shapes",
                "n_samples": 12,
                "image_size": 16,
                "batch_size": 4,
                "num_workers": 0,
                "shuffle": False,
                "rebuild_cache": True,
                "preprocessing": {},
            },
        }
    )
    ds, loader = make_dataloader(cfg, cache_root=tmp_path / "cache")
    assert len(loader) == 3
    images, labels = next(iter(loader))
    assert images.shape == (4, 3, 16, 16)
    assert set(labels.keys()) == {"shape", "color", "hue", "brightness", "agency"}


def test_default_cache_root_respects_qualia_cache_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("QUALIA_CACHE", str(tmp_path / "envcache"))
    root = default_cache_root()
    assert root == tmp_path / "envcache"


# ---------------------------------------------------------------------------
# Integration: training entrypoint emits cache hit/miss and honors --rebuild-cache
# ---------------------------------------------------------------------------


def _run_train_cli(
    args: list[str], env_extra: dict[str, str] | None = None
) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["QUALIA_TRACKER"] = "none"  # offline / no tensorboard
    env.setdefault("PYTHONPATH", "src")
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, "-m", "qualia.train.train", *args],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        cwd=str(Path(__file__).resolve().parent.parent),
    )


def test_train_entrypoint_logs_cache_status(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    env = {"QUALIA_CACHE": str(cache_dir)}
    overrides = [
        "data.n_samples=8",
        "data.image_size=16",
        "data.batch_size=4",
        "+run.name=cache-status",
    ]

    first = _run_train_cli(overrides, env_extra=env)
    assert first.returncode == 0, first.stderr
    assert "cache MISS" in first.stdout
    assert "data_cache_hit=0" in first.stdout

    second = _run_train_cli(overrides, env_extra=env)
    assert second.returncode == 0, second.stderr
    assert "cache HIT" in second.stdout
    assert "data_cache_hit=1" in second.stdout

    miss_marker = "data_cache_hit=0"
    hit_marker = "data_cache_hit=1"
    assert miss_marker in first.stdout
    assert hit_marker in second.stdout


def test_train_entrypoint_rebuild_cache_flag(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    env = {"QUALIA_CACHE": str(cache_dir)}
    base = [
        "data.n_samples=8",
        "data.image_size=16",
        "data.batch_size=4",
        "+run.name=rebuild-flag",
    ]

    prime = _run_train_cli(base, env_extra=env)
    assert prime.returncode == 0, prime.stderr
    assert "cache MISS" in prime.stdout

    rebuilt = _run_train_cli(["--rebuild-cache", *base], env_extra=env)
    assert rebuilt.returncode == 0, rebuilt.stderr
    assert "cache MISS" in rebuilt.stdout
    assert "data_cache_hit=0" in rebuilt.stdout


def test_train_entrypoint_rebuild_on_preprocessing_change(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    env = {"QUALIA_CACHE": str(cache_dir)}

    base_pre = [
        "data.n_samples=8",
        "data.image_size=16",
        "data.batch_size=4",
        "+run.name=cfg-change",
    ]

    first = _run_train_cli([*base_pre, "+data.preprocessing={flip: false}"], env_extra=env)
    assert first.returncode == 0, first.stderr
    assert "cache MISS" in first.stdout

    second = _run_train_cli([*base_pre, "+data.preprocessing={flip: false}"], env_extra=env)
    assert second.returncode == 0, second.stderr
    assert "cache HIT" in second.stdout

    third = _run_train_cli([*base_pre, "+data.preprocessing={flip: true}"], env_extra=env)
    assert third.returncode == 0, third.stderr
    assert "cache MISS" in third.stdout


# ---------------------------------------------------------------------------
# Hydra config: defaults are wired up correctly
# ---------------------------------------------------------------------------


def test_hydra_config_exposes_data_cache_fields() -> None:
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="baseline")
        assert cfg.data.name == "stochastic_colored_shapes"
        assert cfg.data.n_samples == 256
        assert cfg.data.image_size == 32
        assert cfg.data.batch_size == 16
        assert cfg.data.rebuild_cache is False
        # preprocessing can be either a dict or an empty DictConfig.
        assert cfg.data.preprocessing is not None
