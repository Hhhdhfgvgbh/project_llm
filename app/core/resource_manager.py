from __future__ import annotations

from dataclasses import dataclass


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
