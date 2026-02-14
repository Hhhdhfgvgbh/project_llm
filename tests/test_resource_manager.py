from app.core.resource_manager import ResourceEstimate, ResourceManager


def test_check_safety_coefficients() -> None:
    assert ResourceManager.check_safety_coefficients(available_ram_gb=16, available_vram_gb=8)
    assert not ResourceManager.check_safety_coefficients(available_ram_gb=2, available_vram_gb=1)


def test_can_run_uses_coefficients() -> None:
    estimate = ResourceEstimate(required_ram_gb=4, required_vram_gb=2)
    assert ResourceManager.can_run(available_ram_gb=6, available_vram_gb=3, estimate=estimate)
    assert not ResourceManager.can_run(available_ram_gb=5, available_vram_gb=3, estimate=estimate)
