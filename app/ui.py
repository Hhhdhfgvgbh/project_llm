from __future__ import annotations

from pathlib import Path

import psutil
import streamlit as st

from app.config.loader import ConfigLoader
from app.config.schemas import PipelineConfig
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
    return engine, registry


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

            result = engine.run(pipeline_cfg, user_input=text_input)
            st.success("Готово")

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
    st.info(
        "Пока manual mode использует текущий pipeline.yaml как шаблон. "
        "Следующий шаг — визуальный stage-builder (добавление/удаление/параметры)."
    )

    text_input = st.text_area("Вход для manual режима", key="manual_input", height=160)
    run_manual = st.button("▶️ Запустить в manual режиме", use_container_width=True)

    if not run_manual:
        return

    if models_cfg is None or pipeline_cfg is None:
        st.error("Невозможно запустить manual mode: отсутствуют валидные конфиги")
        return

    if not text_input.strip():
        st.warning("Введите текст для manual режима")
        return

    with st.spinner("Выполнение manual режима..."):
        try:
            engine, _ = build_engine(models_cfg)
            result = engine.run(pipeline_cfg, user_input=text_input)
            st.success("Manual run завершён")
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

    _, registry = build_engine(models_cfg)
    for model in registry.list_models():
        col1, col2, col3 = st.columns([3, 2, 1])
        with col1:
            st.write(f"**{model.name}** — {model.path}")
        with col2:
            st.write(f"quant: {model.quantization} | ctx: {model.generation.n_ctx}")
        with col3:
            st.write("✅")

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
