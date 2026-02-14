from pathlib import Path

import pytest

from app.config.loader import ConfigLoader
from app.config.validation import RuntimeValidator


def test_models_config_parses(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "llama3_q5.gguf").write_text("x", encoding="utf-8")

    path = tmp_path / "models.yaml"
    path.write_text(
        """
version: 1
models_directory: "{models_dir}"
models:
  llama3_q5:
    file: "llama3_q5.gguf"
    quantization: "Q5_K_M"
    required_for_base: true
""".format(models_dir=models_dir),
        encoding="utf-8",
    )

    config = ConfigLoader.load_models_config(path)
    report = RuntimeValidator.validate_model_files(config)

    assert report.ok


def test_pipeline_requires_input_from_after_first(tmp_path: Path) -> None:
    path = tmp_path / "pipeline.yaml"
    path.write_text(
        """
version: 1
base_pipeline:
  stages:
    - id: stage1
      type: single
      model: llama3_q5
    - id: stage2
      type: single
      model: mistral_q4
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        ConfigLoader.load_pipeline_config(path)
