"""Lightweight system-metrics sampling (CPU, RAM, GPU).

Returns a flat dict of scalar samples that the tracker can log alongside
training metrics. CPU/RAM are sampled via psutil; GPU via NVML when CUDA is
available. Designed to be called periodically (e.g. every ``log_every`` steps)
without dominating the training step time.
"""

from __future__ import annotations


def _try_import_psutil():
    try:
        import psutil
    except ImportError:
        return None
    return psutil


def sample() -> dict[str, float]:
    metrics: dict[str, float] = {}

    psutil = _try_import_psutil()
    if psutil is not None:
        try:
            metrics["system/cpu_percent"] = float(psutil.cpu_percent(interval=None))
            mem = psutil.virtual_memory()
            metrics["system/ram_used_mb"] = float(mem.used) / (1024 * 1024)
            metrics["system/ram_total_mb"] = float(mem.total) / (1024 * 1024)
            metrics["system/ram_percent"] = float(mem.percent)
        except Exception:
            pass

    try:
        import torch

        if torch.cuda.is_available():
            try:
                for i in range(torch.cuda.device_count()):
                    free, total = torch.cuda.mem_get_info(i)
                    metrics[f"system/gpu{i}_mem_used_mb"] = (total - free) / (1024 * 1024)
                    metrics[f"system/gpu{i}_mem_total_mb"] = total / (1024 * 1024)
            except Exception:
                pass
            try:
                util = torch.cuda.utilization(0)
                metrics["system/gpu_util_percent"] = float(util)
            except (AttributeError, RuntimeError):
                pass
    except ImportError:
        pass

    return metrics
