from __future__ import annotations

import time
from pathlib import Path

import psutil
import streamlit as st

from app.config.loader import ConfigLoader
from app.config.schemas import AggregationType, PipelineConfig, StageOutputMode
from app.core.aggregation_engine import AggregationEngine
from app.core.model_registry import ModelRegistry
from app.core.model_wrapper import ModelWrapper
from app.core.pipeline_engine import PipelineEngine
from app.core.resource_manager import ResourceManager
from app.core.session_manager import SessionManager


CONFIG_DIR = Path("config")
MODELS_PATH = CONFIG_DIR / "models.yaml"
PIPELINE_PATH = CONFIG_DIR / "pipeline.yaml"


@st.cache_data(show_spinner=False)
def load_configs() -> tuple[object | None, PipelineConfig | None, list[str]]:
    warnings: list[str] = []

    models_cfg = None
    pipeline_cfg = None

    if MODELS_PATH.exists():
        try:
            models_cfg = ConfigLoader.load_models_config(MODELS_PATH)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Ошибка models.yaml: {exc}")
    else:
        warnings.append("config/models.yaml не найден")

    if PIPELINE_PATH.exists():
        try:
            pipeline_cfg = ConfigLoader.load_pipeline_config(PIPELINE_PATH)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Ошибка pipeline.yaml: {exc}")
    else:
        warnings.append("config/pipeline.yaml не найден")

    return models_cfg, pipeline_cfg, warnings


def build_engine(models_cfg: object) -> tuple[PipelineEngine, ModelRegistry]:
    registry = ModelRegistry(models_cfg).build()
    engine = PipelineEngine(
        registry=registry,
        model_wrapper=ModelWrapper(),
        aggregation_engine=AggregationEngine(
            synthesis_callback=lambda model, prompt: f"[SYNTHESIS:{model}]\n{prompt}"
        ),
        resource_manager=ResourceManager(),
        session_manager=SessionManager(),
    )
    overrides = st.session_state.get("strip_reasoning_overrides", {})
    for name, value in overrides.items():
        if name in registry.models:
            registry.models[name].strip_reasoning = bool(value)
    return engine, registry


def build_manual_pipeline_from_ui(models_cfg: object, fallback_pipeline: PipelineConfig | None) -> PipelineConfig:
    available_models = sorted(models_cfg.models.keys())
    if not available_models:
        raise ValueError("Нет доступных моделей в models.yaml")

    default_stages = len(fallback_pipeline.base_pipeline.stages) if fallback_pipeline else 2
    stages_count = st.number_input("Количество стадий", min_value=1, max_value=8, value=default_stages, step=1)

    stage_payloads: list[dict] = []
    stage_ids: list[str] = []

    for idx in range(int(stages_count)):
        st.markdown(f"### Стадия {idx + 1}")
        stage_id = st.text_input("ID стадии", value=f"manual_stage_{idx + 1}", key=f"m_id_{idx}")
        stage_type = st.selectbox("Тип стадии", ["single", "multi"], key=f"m_type_{idx}")
        show_prompt_key = f"m_show_prompt_{idx}"
        show_instructions_key = f"m_show_instructions_{idx}"
        st.session_state.setdefault(show_prompt_key, False)
        st.session_state.setdefault(show_instructions_key, False)

        prompt_btn_label = "➖ Скрыть System prompt" if st.session_state[show_prompt_key] else "➕ Добавить System prompt"
        instructions_btn_label = (
            "➖ Скрыть Instructions" if st.session_state[show_instructions_key] else "➕ Добавить Instructions"
        )

        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button(prompt_btn_label, key=f"m_prompt_btn_{idx}", use_container_width=True):
                st.session_state[show_prompt_key] = not st.session_state[show_prompt_key]
        with btn_col2:
            if st.button(instructions_btn_label, key=f"m_instructions_btn_{idx}", use_container_width=True):
                st.session_state[show_instructions_key] = not st.session_state[show_instructions_key]

        system_prompt = ""
        if st.session_state[show_prompt_key]:
            system_prompt = st.text_area("System prompt", key=f"m_prompt_{idx}", height=80)
        instructions = ""
        if st.session_state[show_instructions_key]:
            instructions = st.text_area("Instructions (необязательно)", key=f"m_instructions_{idx}", height=80)

        col1, col2, col3 = st.columns(3)
        with col1:
            temperature = st.number_input("temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.05, key=f"m_t_{idx}")
            top_p = st.number_input("top_p", min_value=0.0, max_value=1.0, value=0.95, step=0.05, key=f"m_p_{idx}")
        with col2:
            max_tokens = st.number_input("max_tokens", min_value=1, max_value=8192, value=1024, step=32, key=f"m_mt_{idx}")
            n_ctx = st.number_input("n_ctx", min_value=256, max_value=65536, value=8192, step=256, key=f"m_ctx_{idx}")
        with col3:
            n_gpu_layers = st.number_input("n_gpu_layers", min_value=-1, max_value=512, value=0, step=1, key=f"m_gpu_{idx}")
            threads = st.number_input("threads", min_value=1, max_value=128, value=8, step=1, key=f"m_th_{idx}")

        generation = {
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_tokens": int(max_tokens),
            "n_ctx": int(n_ctx),
            "n_gpu_layers": int(n_gpu_layers),
            "threads": int(threads),
        }

        output_mode = st.selectbox(
            "Что передавать в следующую стадию",
            options=[item.value for item in StageOutputMode],
            format_func=lambda value: "Только ответ" if value == StageOutputMode.ANSWER_ONLY.value else "Вопрос + ответ",
            key=f"m_output_mode_{idx}",
        )

        stage_data: dict = {
            "id": stage_id.strip() or f"manual_stage_{idx + 1}",
            "type": stage_type,
            "system_prompt": system_prompt,
            "instructions": instructions,
            "output_mode": output_mode,
            "generation": generation,
        }

        if idx > 0:
            input_count = st.number_input(
                "Сколько входов объединить",
                min_value=1,
                max_value=len(stage_ids),
                value=1,
                step=1,
                key=f"m_input_count_{idx}",
            )
            selected_inputs: list[str] = []
            for input_idx in range(int(input_count)):
                selected = st.selectbox(
                    f"Выход берём из #{input_idx + 1}",
                    options=stage_ids,
                    index=min(input_idx, len(stage_ids) - 1),
                    key=f"m_input_{idx}_{input_idx}",
                )
                selected_inputs.append(selected)

            deduped_inputs = list(dict.fromkeys(selected_inputs))
            stage_data["input_from"] = deduped_inputs[0] if len(deduped_inputs) == 1 else deduped_inputs

        if stage_type == "single":
            stage_data["model"] = st.selectbox("Модель", options=available_models, key=f"m_model_{idx}")
        else:
            selected = st.multiselect(
                "Модели",
                options=available_models,
                default=available_models[:2],
                key=f"m_models_{idx}",
            )
            stage_data["models"] = selected

            agg_type = st.selectbox(
                "Aggregation",
                options=[item.value for item in AggregationType],
                key=f"m_agg_type_{idx}",
            )
            agg_payload = {"type": agg_type}
            if agg_type in (AggregationType.SYNTHESIS.value, AggregationType.CUSTOM_TEMPLATE.value):
                agg_payload["synthesis_model"] = st.selectbox(
                    "Synthesis model",
                    options=available_models,
                    key=f"m_agg_model_{idx}",
                )
            if agg_type == AggregationType.CUSTOM_TEMPLATE.value:
                agg_payload["template"] = st.text_area(
                    "Template",
                    value="A={{model_1}}\n\nB={{model_2}}",
                    key=f"m_agg_tpl_{idx}",
                )
            stage_data["aggregation"] = agg_payload

        stage_payloads.append(stage_data)
        stage_ids.append(stage_data["id"])

    payload = {
        "version": 1,
        "base_pipeline": {
            "execution_mode": "sequential",
            "stages": stage_payloads,
        },
    }
    return PipelineConfig.model_validate(payload)


def render_sidebar(models_cfg: object | None, pipeline_cfg: PipelineConfig | None, warnings: list[str]) -> None:
    with st.sidebar:
        st.title("⚙️ Настройки")

        if st.button("🔄 Перезагрузить конфиги"):
            st.cache_data.clear()
            st.rerun()

        if models_cfg is not None:
            st.success(f"✅ Модели в конфиге: {len(models_cfg.models)}")
        else:
            st.error("❌ models.yaml не загружен")

        if pipeline_cfg is not None:
            st.success(f"✅ Стадий в base pipeline: {len(pipeline_cfg.base_pipeline.stages)}")
        else:
            st.warning("⚠️ pipeline.yaml не загружен")

        for warning in warnings:
            st.warning(f"⚠️ {warning}")

        st.divider()
        ram_gb = psutil.virtual_memory().available / (1024**3)
        st.metric("Доступно RAM", f"{ram_gb:.1f} GB")

        vram_gb = st.number_input("Оценка доступной VRAM (GB)", min_value=0.0, value=8.0, step=0.5)
        safety_ok = ResourceManager.check_safety_coefficients(ram_gb, vram_gb)
        st.success("✅ Ресурсы в норме") if safety_ok else st.error("❌ Недостаточно ресурсов")


def render_base_tab(models_cfg: object | None, pipeline_cfg: PipelineConfig | None) -> None:
    st.header("Базовый режим (фиксированный pipeline.yaml)")

    selected_pipeline_name = "base_pipeline"
    if pipeline_cfg is not None:
        pipeline_names = pipeline_cfg.list_pipelines()
        selected_pipeline_name = st.selectbox(
            "Пайплайн",
            options=pipeline_names,
            format_func=lambda value: "base_pipeline (по умолчанию)" if value == "base_pipeline" else value,
        )

    text_input = st.text_area("Входной текст", height=200, placeholder="Вставьте запрос или документ...")

    run_base = st.button("▶️ Запустить базовый пайплайн", type="primary", use_container_width=True)

    if not run_base:
        return

    if not text_input.strip():
        st.warning("Введите текст для обработки")
        return

    if models_cfg is None or pipeline_cfg is None:
        st.error("Невозможно запустить: отсутствуют валидные конфиги")
        return

    with st.spinner("Проверка и выполнение пайплайна..."):
        try:
            engine, registry = build_engine(models_cfg)
            if registry.base_mode_blocked:
                st.error("Base mode заблокирован: отсутствуют обязательные модели")
                return

            selected_pipeline = pipeline_cfg.get_pipeline(selected_pipeline_name)
            run_pipeline = pipeline_cfg.model_copy(update={"base_pipeline": selected_pipeline})

            start = time.perf_counter()
            result = engine.run(run_pipeline, user_input=text_input)
            elapsed = time.perf_counter() - start
            st.success("Готово")
            st.metric("Время полного ответа", f"{elapsed:.2f} сек")

            st.subheader("Промежуточные результаты")
            for step in result.steps:
                with st.expander(f"Этап {step.stage_id} ({step.stage_type})"):
                    for model_name, output in step.model_outputs.items():
                        st.markdown(f"**{model_name}**")
                        st.code(output[:1500])
                    st.markdown("**Aggregated output**")
                    st.write(step.aggregated_output)

            st.subheader("Финальный результат")
            st.write(result.final_output)
            st.download_button(
                "Скачать final.txt",
                result.final_output,
                file_name="final.txt",
                mime="text/plain",
            )
        except Exception as exc:  # noqa: BLE001
            st.exception(exc)


def render_manual_tab(models_cfg: object | None, pipeline_cfg: PipelineConfig | None) -> None:
    st.header("Ручной режим")
    st.info("Визуальный конструктор: настраивайте стадии, модели, параметры генерации, агрегацию и объединение входов из нескольких этапов.")

    text_input = st.text_area("Вход для manual режима", key="manual_input", height=160)

    if models_cfg is None:
        st.error("Невозможно запустить manual mode: отсутствует валидный models.yaml")
        return

    try:
        manual_pipeline = build_manual_pipeline_from_ui(models_cfg, pipeline_cfg)
        with st.expander("Предпросмотр pipeline (JSON)"):
            st.json(manual_pipeline.model_dump())
    except Exception as exc:  # noqa: BLE001
        st.error("Ошибка в конфигурации manual pipeline")
        st.exception(exc)
        return

    run_manual = st.button("▶️ Запустить в manual режиме", use_container_width=True)
    if not run_manual:
        return

    if not text_input.strip():
        st.warning("Введите текст для manual режима")
        return

    with st.spinner("Выполнение manual режима..."):
        try:
            engine, _ = build_engine(models_cfg)
            start = time.perf_counter()
            result = engine.run(manual_pipeline, user_input=text_input)
            elapsed = time.perf_counter() - start
            st.success("Manual run завершён")
            st.metric("Время полного ответа", f"{elapsed:.2f} сек")

            st.subheader("Промежуточные результаты")
            for step in result.steps:
                with st.expander(f"Этап {step.stage_id} ({step.stage_type})"):
                    for model_name, output in step.model_outputs.items():
                        st.markdown(f"**{model_name}**")
                        st.code(output[:1500])
                    st.markdown("**Aggregated output**")
                    st.write(step.aggregated_output)

            st.subheader("Финальный результат")
            st.write(result.final_output)
        except Exception as exc:  # noqa: BLE001
            st.exception(exc)


def render_history_tab() -> None:
    st.header("История сессий")
    manager = SessionManager()
    sessions = manager.list_sessions()

    if not sessions:
        st.info("Пока нет сохранённых сессий")
        return

    for item in sessions:
        with st.expander(f"{item.id} | mode={item.mode} | {item.created_at}"):
            st.write(f"Этапов: {len(item.stages)}")
            if st.button("Открыть", key=f"open_{item.id}"):
                details = manager.load_session(item.id)
                st.subheader("Final output")
                st.write(details.final_output or "(empty)")
                st.subheader("Stage artifacts")
                st.json(details.stages)


def render_models_tab(models_cfg: object | None) -> None:
    st.header("Статус моделей")
    if models_cfg is None:
        st.error("models.yaml не загружен")
        return

    if "strip_reasoning_overrides" not in st.session_state:
        st.session_state["strip_reasoning_overrides"] = {}

    _, registry = build_engine(models_cfg)
    overrides: dict[str, bool] = st.session_state["strip_reasoning_overrides"]

    st.caption("Переключатели ниже меняют очистку рассуждений только для текущей сессии UI.")
    for model in registry.list_models():
        overrides.setdefault(model.name, model.strip_reasoning)
        col1, col2, col3 = st.columns([3, 2, 2])
        with col1:
            st.write(f"**{model.name}** — {model.path}")
            st.caption(model.description or "Описание не задано")
        with col2:
            st.write(f"quant: {model.quantization} | ctx: {model.generation.n_ctx}")
        with col3:
            toggled = st.toggle(
                "Очистка рассуждений",
                value=overrides[model.name],
                key=f"strip_reasoning_toggle_{model.name}",
            )
            overrides[model.name] = toggled

    if registry.warnings:
        st.warning("Некоторые модели недоступны:")
        for warning in registry.warnings:
            st.write(f"- {warning}")


def main() -> None:
    st.set_page_config(
        page_title="Project LLM Orchestrator",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🧠 Project LLM Orchestrator")
    models_cfg, pipeline_cfg, warnings = load_configs()
    render_sidebar(models_cfg, pipeline_cfg, warnings)

    tab_base, tab_manual, tab_history, tab_models = st.tabs(
        ["🛠 Базовый режим", "🧩 Ручной режим", "📖 История", "📊 Модели"]
    )

    with tab_base:
        render_base_tab(models_cfg, pipeline_cfg)

    with tab_manual:
        render_manual_tab(models_cfg, pipeline_cfg)

    with tab_history:
        render_history_tab()

    with tab_models:
        render_models_tab(models_cfg)


if __name__ == "__main__":
    main()
