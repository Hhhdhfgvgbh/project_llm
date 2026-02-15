from app.core.resource_manager import ResourceEstimate, ResourceManager


def test_check_safety_coefficients() -> None:
    assert ResourceManager.check_safety_coefficients(available_ram_gb=16, available_vram_gb=8)
    assert not ResourceManager.check_safety_coefficients(available_ram_gb=2, available_vram_gb=1)


def test_can_run_uses_coefficients() -> None:
    estimate = ResourceEstimate(required_ram_gb=4, required_vram_gb=2)
    assert ResourceManager.can_run(available_ram_gb=6, available_vram_gb=3, estimate=estimate)
    assert not ResourceManager.can_run(available_ram_gb=5, available_vram_gb=3, estimate=estimate)


def test_estimate_for_models_scales_with_size_and_ctx() -> None:
    small = ResourceManager.estimate_for_models([2.0], [4096])
    big = ResourceManager.estimate_for_models([6.0, 3.0], [16384, 8192])

    assert small.required_ram_gb > 0
    assert small.required_vram_gb > 0
    assert big.required_ram_gb > small.required_ram_gb
    assert big.required_vram_gb > small.required_vram_gb
