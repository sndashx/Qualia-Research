"""Run metadata capture (git SHA, hostname, CLI argv, env snapshot)."""

from __future__ import annotations

import os
import platform
import socket
import subprocess
import sys
from typing import Any


def _safe_run(cmd: list[str], cwd: str | None = None, timeout: float = 5.0) -> str | None:
    try:
        out = subprocess.check_output(cmd, cwd=cwd, stderr=subprocess.DEVNULL, timeout=timeout)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None
    return out.decode("utf-8", errors="replace").strip()


def git_sha(cwd: str | None = None) -> str | None:
    return _safe_run(["git", "rev-parse", "HEAD"], cwd=cwd)


def git_branch(cwd: str | None = None) -> str | None:
    return _safe_run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd)


def git_dirty(cwd: str | None = None) -> bool:
    out = _safe_run(["git", "status", "--porcelain"], cwd=cwd)
    return bool(out)


def hostname() -> str:
    try:
        return socket.gethostname()
    except OSError:
        return "unknown"


def platform_info() -> str:
    return f"{platform.system()} {platform.release()} ({platform.machine()})"


def python_info() -> str:
    return f"{sys.implementation.name} {sys.version.split()[0]}"


def torch_info() -> dict[str, str]:
    try:
        import torch
    except ImportError:
        return {"available": "false"}
    return {
        "version": torch.__version__,
        "cuda_available": str(torch.cuda.is_available()),
        "cuda_device_count": str(torch.cuda.device_count()) if torch.cuda.is_available() else "0",
    }


def cli_argv() -> list[str]:
    return list(sys.argv)


def env_subset(
    keys: tuple[str, ...] = ("CUDA_VISIBLE_DEVICES", "PYTHONHASHSEED", "OMP_NUM_THREADS")
) -> dict[str, str]:
    return {k: os.environ[k] for k in keys if k in os.environ}


def capture_metadata(cfg: Any, cwd: str | None = None) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "git_sha": git_sha(cwd),
        "git_branch": git_branch(cwd),
        "git_dirty": git_dirty(cwd) if git_sha(cwd) is not None else None,
        "hostname": hostname(),
        "platform": platform_info(),
        "python": python_info(),
        "torch": torch_info(),
        "argv": cli_argv(),
        "env": env_subset(),
    }
    try:
        from omegaconf import OmegaConf

        meta["config"] = OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        meta["config"] = None
    return meta
