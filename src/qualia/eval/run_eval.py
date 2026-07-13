"""Evaluation entrypoint (scaffold placeholder).

The full eval suite is implemented in a downstream bead. This stub exists so
that `make eval-toy` runs cleanly.
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print("[qualia.eval.run_eval] scaffold placeholder")
    print(f"  eval.report_consistency_threshold={cfg.eval.report_consistency_threshold}")
    print(f"  eval.introspection_accuracy_threshold={cfg.eval.introspection_accuracy_threshold}")
    print("  (eval suite is implemented in a downstream bead)")


if __name__ == "__main__":
    main()
