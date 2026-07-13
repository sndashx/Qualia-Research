"""Synthetic datasets with reportable attributes.

Provides ``ColoredShapesDataset`` — a tiny CPU-friendly dataset of percepts
with structured, ground-truth phenomenal attributes (``shape``, ``color``,
``hue``, ``brightness``, ``agency``) that the eval suite uses to:

  * drive the encoder / phenomenal-state pipeline,
  * provide held-out stimuli for the introspection accuracy eval,
  * provide labeled "what does this percept look like?" targets that the
    self-report's ``content`` slot should be able to recover.

Each sample is a ``(percept_vector, attributes)`` pair. The percept vector is
a small (e.g. 16-dim) real-valued feature built deterministically from the
attributes so that the encoder / state modules see structured, repeatable
inputs across runs (the spec asks the suite to run end-to-end on CPU).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

SHAPE_NAMES: tuple[str, ...] = ("circle", "square", "triangle")
COLOR_NAMES: tuple[str, ...] = ("red", "green", "blue")


@dataclass
class ColoredShapeSample:
    """A single synthetic sample.

    Attributes:
        percept: ``(percept_dim,)`` real-valued percept feature vector.
        shape_idx: Integer index into ``SHAPE_NAMES``.
        color_idx: Integer index into ``COLOR_NAMES``.
        hue: Continuous hue in ``[0, 1)``.
        brightness: Scalar in ``[0, 1]``.
        agency: Scalar in ``[0, 1]`` representing self-causation.
    """

    percept: Tensor
    shape_idx: int
    color_idx: int
    hue: float
    brightness: float
    agency: float

    def attributes(self) -> dict[str, float | int]:
        return {
            "shape": self.shape_idx,
            "color": self.color_idx,
            "hue": self.hue,
            "brightness": self.brightness,
            "agency": self.agency,
        }


class ColoredShapesDataset:
    """Deterministic dataset of colored-shape percepts with ground-truth attributes.

    Args:
        num_samples: Number of samples in the dataset.
        percept_dim: Dimensionality of the returned percept vectors.
        seed: RNG seed; the dataset is fully deterministic in ``(num_samples,
            percept_dim, seed)`` so that repeat runs produce identical
            percepts/attributes.
        include_agency: If False, agency is fixed to 0.5 for every sample. The
            eval suite uses this to verify downstream grounding on tasks where
            the agency signal is uninformative.
    """

    def __init__(
        self,
        num_samples: int = 64,
        percept_dim: int = 16,
        seed: int = 0,
        include_agency: bool = True,
    ) -> None:
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive; got {num_samples}")
        if percept_dim <= 0:
            raise ValueError(f"percept_dim must be positive; got {percept_dim}")

        self.num_samples = num_samples
        self.percept_dim = percept_dim
        self.seed = seed
        self.include_agency = include_agency

        samples: list[ColoredShapeSample] = []
        for i in range(num_samples):
            samples.append(self._build_sample(index=i))
        self._samples: list[ColoredShapeSample] = samples

    def _build_sample(self, index: int) -> ColoredShapeSample:
        # Use a per-sample generator so the (num_samples, percept_dim, seed)
        # tuple fully determines every sample.
        gen = torch.Generator().manual_seed(self.seed * 1_000_003 + index)
        shape_idx = int(torch.randint(0, len(SHAPE_NAMES), (1,), generator=gen).item())
        color_idx = int(torch.randint(0, len(COLOR_NAMES), (1,), generator=gen).item())
        hue = float(torch.rand(1, generator=gen).item())
        brightness = float(torch.rand(1, generator=gen).item())
        agency = float(torch.rand(1, generator=gen).item()) if self.include_agency else 0.5

        percept = self._encode_percept(
            shape_idx=shape_idx,
            color_idx=color_idx,
            hue=hue,
            brightness=brightness,
            agency=agency,
            gen=gen,
        )
        return ColoredShapeSample(
            percept=percept,
            shape_idx=shape_idx,
            color_idx=color_idx,
            hue=hue,
            brightness=brightness,
            agency=agency,
        )

    def _encode_percept(
        self,
        *,
        shape_idx: int,
        color_idx: int,
        hue: float,
        brightness: float,
        agency: float,
        gen: torch.Generator,
    ) -> Tensor:
        """Build a percept vector that is a deterministic function of attributes.

        Layout (percept_dim >= 8; remaining dims are Gaussian noise seeded by
        the sample index, so different samples look different while the
        attribute-related coordinates stay clean):

        - [0]: sin(2*pi*hue)
        - [1]: cos(2*pi*hue)
        - [2]: brightness
        - [3]: agency
        - [4..6]: one-hot shape (3 dims)
        - [7..9]: one-hot color (3 dims)
        - [10:]: Gaussian noise (deterministic given the per-sample generator)
        """
        coords = torch.zeros(self.percept_dim)
        coords[0] = math.sin(2.0 * math.pi * hue)
        coords[1] = math.cos(2.0 * math.pi * hue)
        coords[2] = brightness
        coords[3] = agency
        if self.percept_dim >= 4 + len(SHAPE_NAMES):
            coords[4 + shape_idx] = 1.0
        if self.percept_dim >= 4 + len(SHAPE_NAMES) + len(COLOR_NAMES):
            base = 4 + len(SHAPE_NAMES)
            coords[base + color_idx] = 1.0
        if self.percept_dim > 4 + len(SHAPE_NAMES) + len(COLOR_NAMES):
            noise_dim = self.percept_dim - (4 + len(SHAPE_NAMES) + len(COLOR_NAMES))
            noise = torch.randn(noise_dim, generator=gen)
            coords[4 + len(SHAPE_NAMES) + len(COLOR_NAMES) :] = noise
        return coords

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> ColoredShapeSample:
        if not 0 <= index < self.num_samples:
            raise IndexError(f"index {index} out of range [0, {self.num_samples})")
        return self._samples[index]

    def batched_percepts(self, indices: Sequence[int] | None = None) -> Tensor:
        """Return a ``(len(indices), percept_dim)`` tensor of percepts.

        If ``indices`` is None, returns every sample in order.
        """
        if indices is None:
            indices = list(range(self.num_samples))
        return torch.stack([self._samples[i].percept for i in indices], dim=0)

    def attribute_table(self, indices: Sequence[int] | None = None) -> dict[str, list[float | int]]:
        """Return a dict-of-lists of attributes for the given sample indices."""
        if indices is None:
            indices = list(range(self.num_samples))
        table: dict[str, list[float | int]] = {
            "shape": [],
            "color": [],
            "hue": [],
            "brightness": [],
            "agency": [],
        }
        for i in indices:
            sample = self._samples[i]
            table["shape"].append(sample.shape_idx)
            table["color"].append(sample.color_idx)
            table["hue"].append(sample.hue)
            table["brightness"].append(sample.brightness)
            table["agency"].append(sample.agency)
        return table


__all__ = [
    "ColoredShapeSample",
    "ColoredShapesDataset",
    "COLOR_NAMES",
    "SHAPE_NAMES",
]
