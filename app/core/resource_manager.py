from __future__ import annotations

from dataclasses import dataclass


def _kv_cache_gb(n_ctx: int, hidden_size: int = 3072, bytes_per_weight: int = 2) -> float:
    return (n_ctx * hidden_size * 2 * bytes_per_weight) / (1024**3)


@dataclass
class ResourceEstimate:
    required_ram_gb: float
    required_vram_gb: float


class ResourceManager:
    RAM_SAFETY = 1.3
    VRAM_SAFETY = 1.2

    @staticmethod
    def estimate(required_ram_gb: float, required_vram_gb: float) -> ResourceEstimate:
        return ResourceEstimate(required_ram_gb=required_ram_gb, required_vram_gb=required_vram_gb)

    @staticmethod
    def estimate_for_models(model_sizes_gb: list[float], n_ctx_values: list[int]) -> ResourceEstimate:
        if not model_sizes_gb:
            return ResourceEstimate(required_ram_gb=0.0, required_vram_gb=0.0)

        total_weights = sum(model_sizes_gb)
        max_ctx = max(n_ctx_values) if n_ctx_values else 8192
        kv_cache = _kv_cache_gb(max_ctx)

        required_ram = total_weights + kv_cache + 1.0
        required_vram = total_weights * 0.7 + kv_cache
        return ResourceEstimate(required_ram_gb=required_ram, required_vram_gb=required_vram)

    @classmethod
    def can_run(
        cls,
        available_ram_gb: float,
        available_vram_gb: float,
        estimate: ResourceEstimate,
    ) -> bool:
        return (
            available_ram_gb > estimate.required_ram_gb * cls.RAM_SAFETY
            and available_vram_gb > estimate.required_vram_gb * cls.VRAM_SAFETY
        )


    @classmethod
    def check_safety_coefficients(
        cls,
        available_ram_gb: float,
        available_vram_gb: float,
        required_ram_gb: float = 4.0,
        required_vram_gb: float = 2.0,
    ) -> bool:
        estimate = ResourceEstimate(required_ram_gb=required_ram_gb, required_vram_gb=required_vram_gb)
        return cls.can_run(available_ram_gb, available_vram_gb, estimate)
