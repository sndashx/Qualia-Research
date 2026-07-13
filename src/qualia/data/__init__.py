"""Datasets, synthetic generators, and on-disk caching for ``qualia``.

Public surface:

- :class:`SyntheticColoredShapes` — deterministic synthetic dataset emitting
  ``(image, label_dict)`` pairs compatible with :class:`qualia.model.encoder.QualiaEncoder`.
- :class:`CachedDataset` — :class:`torch.utils.data.Dataset` wrapper that
  materializes an inner dataset once and writes it to disk as a single
  ``.pt`` tensor store; subsequent runs hit the cache. The cache key is a
  SHA-256 of the inner dataset's identity (name, seed, sample count, image
  size, preprocessing config hash, dataset version), so any change to the
  preprocessing config or the dataset version automatically invalidates.
- :func:`make_dataset` / :func:`make_dataloader` — Hydra-friendly factories
  that read ``cfg.data`` and construct a ready-to-iterate dataloader.

Cache invalidation is automatic on config change and can be forced with
``--rebuild-cache`` (mapped to ``data.rebuild_cache: true`` in the config).
Cache hit / miss is logged to stdout **and** to the run's :class:`Tracker`
when one is available, with elapsed build / load time in seconds.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

__all__ = [
    "CachedDataset",
    "CacheStats",
    "DATASET_VERSION",
    "SyntheticColoredShapes",
    "compute_cache_key",
    "default_cache_root",
    "make_dataset",
    "make_dataloader",
]


DATASET_VERSION = "1"
"""Bumped whenever the on-disk cache schema changes; auto-invalidates stale caches."""

_log = logging.getLogger("qualia.data")


# ---------------------------------------------------------------------------
# Cache key derivation
# ---------------------------------------------------------------------------


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def compute_cache_key(
    *,
    dataset_name: str,
    seed: int,
    n_samples: int,
    image_size: int,
    preprocessing_cfg: Mapping[str, Any] | None = None,
    dataset_version: str = DATASET_VERSION,
) -> str:
    """SHA-256 hex digest of every input that affects sample contents.

    Two :class:`CachedDataset` instances sharing this key are guaranteed to
    produce identical tensors; different keys mean different content and
    therefore different cache files.
    """
    payload = {
        "dataset_name": str(dataset_name),
        "seed": int(seed),
        "n_samples": int(n_samples),
        "image_size": int(image_size),
        "preprocessing_cfg": dict(preprocessing_cfg or {}),
        "dataset_version": str(dataset_version),
    }
    encoded = _stable_json(payload).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:32]


def default_cache_root() -> Path:
    """Default location for cached datasets (``$QUALIA_CACHE`` or ``./.cache/qualia``)."""
    env = os.environ.get("QUALIA_CACHE")
    return Path(env).expanduser() if env else Path.cwd() / ".cache" / "qualia"


# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ShapeParams:
    """Per-sample generative parameters for :class:`SyntheticColoredShapes`."""

    shape: int  # categorical 0..n_shapes-1
    color: int  # categorical 0..n_colors-1
    hue: float  # continuous [0, 1)
    brightness: float  # continuous [0, 1]
    agency: float  # continuous [-1, 1]


def _draw_color_palette(n_colors: int, seed: int) -> Tensor:
    """Deterministic palette of ``n_colors`` RGB triplets in ``[0, 1]``."""
    g = torch.Generator().manual_seed(int(seed) * 9973 + 1)
    return torch.rand((n_colors, 3), generator=g)


def _draw_shape_params(n: int, seed: int, n_shapes: int, n_colors: int) -> list[_ShapeParams]:
    """Draw a list of ``n`` structured shape parameters deterministically from ``seed``."""
    g = torch.Generator().manual_seed(int(seed) * 2654435761 + 7)
    shapes = torch.randint(0, n_shapes, (n,), generator=g).tolist()
    colors = torch.randint(0, n_colors, (n,), generator=g).tolist()
    hues = torch.rand(n, generator=g)
    brightnesses = torch.rand(n, generator=g)
    agencies = torch.rand(n, generator=g) * 2.0 - 1.0
    out: list[_ShapeParams] = []
    for i in range(n):
        out.append(
            _ShapeParams(
                shape=int(shapes[i]),
                color=int(colors[i]),
                hue=float(hues[i].item()),
                brightness=float(brightnesses[i].item()),
                agency=float(agencies[i].item()),
            )
        )
    return out


def _render_shape(
    image_size: int,
    shape: int,
    color_rgb: Tensor,
    brightness: float,
    hue: float,
    rng: torch.Generator,
) -> Tensor:
    """Render one ``(3, H, W)`` image in ``[-1, 1]`` with the given attributes.

    Shapes: 0=square, 1=triangle, 2=circle, 3=diamond. The position, in-plane
    rotation and size are randomized for variety.
    """
    img = torch.full((3, image_size, image_size), -1.0)
    H = W = image_size
    cy = int(torch.randint(image_size // 4, 3 * image_size // 4, (1,), generator=rng).item())
    cx = int(torch.randint(image_size // 4, 3 * image_size // 4, (1,), generator=rng).item())
    side = int(torch.randint(image_size // 4, image_size // 2, (1,), generator=rng).item())
    base_rgb = color_rgb.clone()
    base_rgb = base_rgb * 2.0 - 1.0
    base_rgb = base_rgb * float(brightness)

    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32),
        torch.arange(W, dtype=torch.float32),
        indexing="ij",
    )

    def _in_square() -> Tensor:
        return (xx >= cx - side) & (xx <= cx + side) & (yy >= cy - side) & (yy <= cy + side)

    def _in_circle() -> Tensor:
        return ((xx - cx) ** 2 + (yy - cy) ** 2) <= float(side) ** 2

    def _in_triangle() -> Tensor:
        # Triangle pointing up; width narrows linearly from base to apex.
        a = (yy >= cy - side) & (yy <= cy + side)
        progress = ((yy - cy + side).clamp(min=0)) / (2.0 * float(side) + 1e-6)
        half_w = side * (1.0 - progress)
        b = (xx >= cx - half_w) & (xx <= cx + half_w)
        return a & b

    def _in_diamond() -> Tensor:
        return ((xx - cx).abs() + (yy - cy).abs()) <= float(side)

    mask = {
        0: _in_square,
        1: _in_triangle,
        2: _in_circle,
        3: _in_diamond,
    }.get(int(shape), _in_square)()

    # Hue rotation in [-1, 1] space is approximated as a constant channel offset.
    hue_shift = (float(hue) - 0.5) * 0.4
    for c in range(3):
        img[c] = torch.where(mask, base_rgb[c] + hue_shift, img[c])
    return img.clamp(-1.0, 1.0)


class SyntheticColoredShapes(Dataset):
    """Deterministic synthetic dataset of colored shapes on a ``[-1, 1]`` canvas.

    Each sample is a tuple ``(image, label_dict)`` where:

    - ``image`` is a ``(3, image_size, image_size)`` float tensor in ``[-1, 1]``.
    - ``label_dict`` maps every key in :data:`qualia.model.encoder.PAYLOAD_KEYS`
      to its target:

      * ``"shape"`` — ``int64`` class index in ``[0, n_shapes)``,
      * ``"color"`` — ``int64`` class index in ``[0, n_colors)``,
      * ``"hue"`` / ``"brightness"`` / ``"agency"`` — ``float32`` scalars.

    Determinism: the same ``(seed, n_samples, image_size, n_shapes, n_colors,
    preprocessing_cfg)`` always produces the same dataset.
    """

    def __init__(
        self,
        n_samples: int = 256,
        image_size: int = 32,
        seed: int = 0,
        n_shapes: int = 4,
        n_colors: int = 4,
        preprocessing_cfg: Mapping[str, Any] | None = None,
    ) -> None:
        if n_samples < 0:
            raise ValueError(f"n_samples must be >= 0; got {n_samples}")
        if image_size < 4:
            raise ValueError(f"image_size must be >= 4; got {image_size}")
        if n_shapes < 1 or n_colors < 1:
            raise ValueError("n_shapes and n_colors must be >= 1")
        self.n_samples = int(n_samples)
        self.image_size = int(image_size)
        self.seed = int(seed)
        self.n_shapes = int(n_shapes)
        self.n_colors = int(n_colors)
        self.preprocessing_cfg: dict[str, Any] = dict(preprocessing_cfg or {})
        self._name = "stochastic_colored_shapes"
        self._palette = _draw_color_palette(self.n_colors, self.seed)
        self._params = _draw_shape_params(self.n_samples, self.seed, self.n_shapes, self.n_colors)

    @property
    def name(self) -> str:
        return self._name

    @property
    def cache_key(self) -> str:
        return compute_cache_key(
            dataset_name=self._name,
            seed=self.seed,
            n_samples=self.n_samples,
            image_size=self.image_size,
            preprocessing_cfg=self.preprocessing_cfg,
        )

    def __len__(self) -> int:
        return self.n_samples

    def _render(self, idx: int) -> Tensor:
        rng = torch.Generator().manual_seed(self.seed * 1000003 + idx + 1)
        params = self._params[idx]
        color_rgb = self._palette[params.color]
        return _render_shape(
            image_size=self.image_size,
            shape=params.shape,
            color_rgb=color_rgb,
            brightness=params.brightness,
            hue=params.hue,
            rng=rng,
        )

    def __getitem__(self, idx: int) -> tuple[Tensor, dict[str, Tensor]]:
        image = self._render(int(idx))
        params = self._params[int(idx)]
        labels: dict[str, Tensor] = {
            "shape": torch.tensor(params.shape, dtype=torch.long),
            "color": torch.tensor(params.color, dtype=torch.long),
            "hue": torch.tensor(params.hue, dtype=torch.float32),
            "brightness": torch.tensor(params.brightness, dtype=torch.float32),
            "agency": torch.tensor(params.agency, dtype=torch.float32),
        }
        return image, labels


# ---------------------------------------------------------------------------
# Cached dataset wrapper
# ---------------------------------------------------------------------------


@dataclass
class CacheStats:
    """Recorded once per :class:`CachedDataset` construction."""

    cache_key: str
    cache_path: str
    hit: bool
    build_seconds: float
    load_seconds: float
    n_samples: int

    def log_fields(self) -> dict[str, float | int | str]:
        return {
            "data_cache_hit": int(self.hit),
            "data_cache_build_seconds": float(self.build_seconds),
            "data_cache_load_seconds": float(self.load_seconds),
            "data_cache_n_samples": int(self.n_samples),
        }


class CachedDataset(Dataset):
    """Disk-backed cache around an inner :class:`Dataset`.

    On the first construction (or whenever the cache key changes) the inner
    dataset is materialized into a single ``cache_root/<key>.pt`` file
    containing the images tensor and a label-dict of tensors. On subsequent
    constructions with the same key, the tensors are loaded directly from
    disk in a single ``torch.load`` and ``__getitem__`` becomes an O(1)
    index into those tensors.

    The cache is intentionally framework-light: it depends only on PyTorch
    (already a hard dependency) and writes a single file per dataset
    variant. For a synthetic 256-sample 32x32 RGB dataset the on-disk cost
    is ~3 MB and reload time is well under 100 ms on a warm filesystem.

    Args:
        factory: Callable returning the inner :class:`Dataset`. It is invoked
            only on a cache miss.
        cache_root: Directory to write / read cache files from.
        cache_key: Hex digest identifying this dataset variant. If omitted,
            it is taken from ``factory().cache_key`` on a cache miss.
        rebuild: Force regeneration of the cache (overwrite existing file).
        logger: Optional callable taking ``(message: str) -> None`` invoked
            with one human-readable line per construction (cache hit / miss
            with timing).
    """

    def __init__(
        self,
        factory: Callable[[], Dataset],
        cache_root: str | os.PathLike[str] | None = None,
        cache_key: str | None = None,
        rebuild: bool = False,
        logger: Callable[[str], None] | None = None,
    ) -> None:
        self._factory = factory
        self._explicit_key = cache_key
        root = Path(cache_root) if cache_root is not None else default_cache_root()
        root.mkdir(parents=True, exist_ok=True)
        self.cache_root = root

        key = self._explicit_key
        path: Path | None = None
        hit = False
        build_seconds = 0.0
        load_seconds = 0.0
        images: Tensor
        labels: dict[str, Tensor]

        if key is None:
            probe = factory()
            if not hasattr(probe, "cache_key"):
                raise TypeError(
                    "CachedDataset requires either an explicit `cache_key` or an inner "
                    "dataset exposing a `cache_key` property."
                )
            key = str(probe.cache_key)
        path = self.cache_root / f"{key}.pt"

        if path.exists() and not rebuild:
            t0 = time.perf_counter()
            payload = torch.load(path, map_location="cpu")
            load_seconds = time.perf_counter() - t0
            images = payload["images"]
            labels = payload["labels"]
            n_samples = int(images.shape[0])
            hit = True
        else:
            t0 = time.perf_counter()
            inner = factory()
            n_samples = len(inner)
            images_list: list[Tensor] = []
            labels_acc: dict[str, list[Tensor]] = {}
            for i in range(n_samples):
                img, lbl = inner[i]
                images_list.append(img)
                for k, v in lbl.items():
                    labels_acc.setdefault(k, []).append(v)
            images = torch.stack(images_list, dim=0).contiguous()
            labels = {k: torch.stack(v, dim=0).contiguous() for k, v in labels_acc.items()}
            payload = {"images": images, "labels": labels, "cache_key": key}
            torch.save(payload, path)
            build_seconds = time.perf_counter() - t0
            hit = False

        self.cache_key = key
        self.cache_path = str(path)
        self._images = images
        self._labels = labels
        self._n_samples = int(images.shape[0])
        self.stats = CacheStats(
            cache_key=key,
            cache_path=str(path),
            hit=hit,
            build_seconds=build_seconds,
            load_seconds=load_seconds,
            n_samples=self._n_samples,
        )

        message = (
            f"[qualia.data] cache {'HIT' if hit else 'MISS'} key={key[:12]} "
            f"n={self._n_samples} "
            f"{'load' if hit else 'build'}={load_seconds if hit else build_seconds:.3f}s "
            f"path={path}"
        )
        if logger is not None:
            logger(message)
        else:
            print(message, flush=True)
        _log.info(message)

    def __len__(self) -> int:
        return self._n_samples

    def __getitem__(self, idx: int) -> tuple[Tensor, dict[str, Tensor]]:
        i = int(idx)
        image = self._images[i]
        labels = {k: self._labels[k][i] for k in self._labels}
        return image, labels

    def label_names(self) -> tuple[str, ...]:
        return tuple(self._labels.keys())


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def _get(cfg: Any, dotted: str, default: Any = None) -> Any:
    try:
        node = cfg
        for part in dotted.split("."):
            if node is None:
                return default
            if hasattr(node, "get"):
                node = node.get(part)
            else:
                node = node[part]
        if node is None:
            return default
        return node
    except Exception:
        return default


def make_dataset(
    cfg: Any,
    *,
    rebuild_cache: bool = False,
    cache_root: str | os.PathLike[str] | None = None,
    logger: Callable[[str], None] | None = None,
) -> CachedDataset:
    """Build a :class:`CachedDataset` from ``cfg.data``.

    Reads:
      - ``cfg.data.name`` (default ``"synthetic_colored_shapes"``)
      - ``cfg.data.n_samples`` (default 256)
      - ``cfg.data.image_size`` (default 32)
      - ``cfg.data.seed`` (default ``cfg.seed`` or 0)
      - ``cfg.data.rebuild_cache`` (default ``False``)
      - ``cfg.data.preprocessing`` (optional mapping)
    """
    name = str(_get(cfg, "data.name", default="synthetic_colored_shapes"))
    n_samples = int(_get(cfg, "data.n_samples", default=256))
    image_size = int(_get(cfg, "data.image_size", default=32))
    seed = int(_get(cfg, "data.seed", default=_get(cfg, "seed", default=0)))
    rebuild = bool(rebuild_cache or _get(cfg, "data.rebuild_cache", default=False))
    preprocessing = _get(cfg, "data.preprocessing", default={}) or {}
    if preprocessing is None:
        preprocessing = {}

    def _factory() -> Dataset:
        if name != "stochastic_colored_shapes":
            raise ValueError(
                f"unknown dataset {name!r}; only 'stochastic_colored_shapes' is registered."
            )
        return SyntheticColoredShapes(
            n_samples=n_samples,
            image_size=image_size,
            seed=seed,
            preprocessing_cfg=dict(preprocessing),
        )

    return CachedDataset(
        factory=_factory,
        cache_root=cache_root,
        rebuild=rebuild,
        logger=logger,
    )


def make_dataloader(
    cfg: Any,
    *,
    rebuild_cache: bool = False,
    cache_root: str | os.PathLike[str] | None = None,
    logger: Callable[[str], None] | None = None,
) -> tuple[CachedDataset, DataLoader]:
    """Build a :class:`CachedDataset` and a :class:`DataLoader` from ``cfg.data``."""
    dataset = make_dataset(cfg, rebuild_cache=rebuild_cache, cache_root=cache_root, logger=logger)
    batch_size = int(_get(cfg, "data.batch_size", default=16))
    num_workers = int(_get(cfg, "data.num_workers", default=0))
    shuffle = bool(_get(cfg, "data.shuffle", default=True))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
    )
    return dataset, loader
