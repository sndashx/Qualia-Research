"""Training entrypoint (scaffold placeholder).

The full training loop is implemented in a downstream bead. This stub exists so
that `make train-toy` runs cleanly and Hydra config-loading is verifiable.
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print("[qualia.train.train] scaffold placeholder")
    print(f"  workspace_dim={cfg.workspace_dim}")
    print(f"  payload_slots={cfg.payload_slots}")
    print(f"  train.steps={cfg.train.steps}")
    print("  (training loop is implemented in a downstream bead)")


if __name__ == "__main__":
    main()
