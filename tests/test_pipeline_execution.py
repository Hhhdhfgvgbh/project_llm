from pathlib import Path

from app.core.executors import BasePipelineExecutor


def _write_models(tmp_path: Path, llama_strip_reasoning: bool = True) -> Path:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "llama3_q5.gguf").write_text("x", encoding="utf-8")
    (models_dir / "mistral_q4.gguf").write_text("x", encoding="utf-8")

    models_cfg = tmp_path / "models.yaml"
    models_cfg.write_text(
        f"""
version: 1
models_directory: "{models_dir}"
models:
  llama3_q5:
    file: "llama3_q5.gguf"
    quantization: "Q5_K_M"
    required_for_base: true
    strip_reasoning: {str(llama_strip_reasoning).lower()}
  mistral_q4:
    file: "mistral_q4.gguf"
    quantization: "Q4_K_M"
    required_for_base: false
""",
        encoding="utf-8",
    )
    return models_cfg


def _write_pipeline(tmp_path: Path) -> Path:
    pipeline_cfg = tmp_path / "pipeline.yaml"
    pipeline_cfg.write_text(
        """
version: 1
base_pipeline:
  stages:
    - id: stage1
      type: single
      model: llama3_q5
      system_prompt: "Analyze"
      output_mode: question_and_answer

    - id: stage2
      type: multi
      models: [llama3_q5, mistral_q4]
      system_prompt: "Alternatives"
      aggregation:
        type: concat
      input_from: stage1

    - id: stage3
      type: single
      model: llama3_q5
      input_from: [stage1, stage2]
""",
        encoding="utf-8",
    )
    return pipeline_cfg


def test_base_pipeline_executor_runs(tmp_path: Path) -> None:
    models = _write_models(tmp_path)
    pipeline = _write_pipeline(tmp_path)

    result = BasePipelineExecutor(str(models), str(pipeline)).run("hello")
    assert result.steps[0].stage_id == "stage1"
    assert result.steps[1].stage_id == "stage2"
    assert result.steps[2].stage_id == "stage3"
    assert "generated response" in result.final_output


def test_stage_can_forward_question_and_answer(tmp_path: Path) -> None:
    models = _write_models(tmp_path)
    pipeline = _write_pipeline(tmp_path)

    context = BasePipelineExecutor(str(models), str(pipeline)).build()
    result = context.engine.run(context.pipeline, user_input="hello", session_enabled=False)
    stage2_prompt_echo = result.steps[1].model_outputs["llama3_q5"]
    assert "Question:" in stage2_prompt_echo
    assert "Answer:" in stage2_prompt_echo


def test_stage_instructions_are_sent_to_model_but_not_forwarded(tmp_path: Path) -> None:
    models = _write_models(tmp_path)
    pipeline = tmp_path / "pipeline_instructions.yaml"
    pipeline.write_text(
        """
version: 1
base_pipeline:
  stages:
    - id: stage1
      type: single
      model: llama3_q5
      instructions: "PRIVATE INSTRUCTIONS"
      output_mode: question_and_answer
    - id: stage2
      type: single
      model: llama3_q5
      input_from: stage1
""",
        encoding="utf-8",
    )

    context = BasePipelineExecutor(str(models), str(pipeline)).build()
    result = context.engine.run(context.pipeline, user_input="hello", session_enabled=False)

    assert "PRIVATE INSTRUCTIONS" in result.steps[0].aggregated_output
    assert "Question: hello" in result.steps[1].aggregated_output
    assert "Question: PRIVATE INSTRUCTIONS" not in result.steps[1].aggregated_output


def test_reasoning_blocks_are_stripped_when_model_enabled(tmp_path: Path, monkeypatch) -> None:
    models = _write_models(tmp_path, llama_strip_reasoning=True)
    pipeline = tmp_path / "pipeline_reasoning.yaml"
    pipeline.write_text(
        """
version: 1
base_pipeline:
  stages:
    - id: stage1
      type: single
      model: llama3_q5
""",
        encoding="utf-8",
    )

    context = BasePipelineExecutor(str(models), str(pipeline)).build()

    def fake_generate(model_name: str, prompt: str, system_prompt: str = "", runtime=None) -> str:
        return "<think>hidden</think>\nVisible answer"

    monkeypatch.setattr(context.engine.model_wrapper, "generate", fake_generate)
    result = context.engine.run(context.pipeline, user_input="hello", session_enabled=False)
    assert result.final_output == "Visible answer"


def test_reasoning_blocks_are_kept_when_model_disabled(tmp_path: Path, monkeypatch) -> None:
    models = _write_models(tmp_path, llama_strip_reasoning=False)
    pipeline = tmp_path / "pipeline_reasoning.yaml"
    pipeline.write_text(
        """
version: 1
base_pipeline:
  stages:
    - id: stage1
      type: single
      model: llama3_q5
""",
        encoding="utf-8",
    )

    context = BasePipelineExecutor(str(models), str(pipeline)).build()

    def fake_generate(model_name: str, prompt: str, system_prompt: str = "", runtime=None) -> str:
        return "<think>hidden</think>\nVisible answer"

    monkeypatch.setattr(context.engine.model_wrapper, "generate", fake_generate)
    result = context.engine.run(context.pipeline, user_input="hello", session_enabled=False)
    assert "<think>hidden</think>" in result.final_output


def test_translate_stage_uses_specialized_model_method(tmp_path: Path, monkeypatch) -> None:
    models = _write_models(tmp_path)
    pipeline = tmp_path / "pipeline_translate.yaml"
    pipeline.write_text(
        """
version: 1
base_pipeline:
  stages:
    - id: translate
      type: translate
      model: llama3_q5
      source_language: Russian
      target_language: English
""",
        encoding="utf-8",
    )

    context = BasePipelineExecutor(str(models), str(pipeline)).build()

    called = {}

    def fake_translate(model_name: str, text: str, source_language: str, target_language: str, runtime=None) -> str:
        called["payload"] = (model_name, text, source_language, target_language)
        return "translated"

    monkeypatch.setattr(context.engine.model_wrapper, "generate_translategemma", fake_translate)
    result = context.engine.run(context.pipeline, user_input="Привет", session_enabled=False)

    assert called["payload"] == ("llama3_q5", "Привет", "Russian", "English")
    assert result.final_output == "translated"
