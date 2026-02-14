# project_llm

Локальное приложение для оркестрации multi-stage пайплайнов поверх нескольких GGUF-моделей.

## Реализовано

- Строгие схемы `models.yaml` и `pipeline.yaml` (версия, типы стадий, агрегация, проверки ссылок).
- Runtime-валидация наличия моделей и проверки целостности пайплайна.
- `ModelRegistry` с объединением `defaults + overrides` и блокировкой Base Mode при отсутствии required-моделей.
- `ModelWrapper` (стабильный runtime-API), `PipelineEngine` (исполнение стадий), `BasePipelineExecutor` и `ManualPipelineExecutor`.
- `AggregationEngine`: `concat`, `majority_vote`, `synthesis`, `custom_template`.
- `ResourceManager` c safety-коэффициентами RAM/VRAM.
- `SessionManager` с versioned stage-артефактами (`stageX_vN`) и `meta.json`.
- CLI запуск пайплайнов.
- Streamlit UI для Base/Manual/History/Models вкладок.

## Структура

- `app/config` — схемы/загрузка/валидация конфигов.
- `app/core` — реестр, wrapper, движок, executors, агрегация, сессии, ресурсы.
- `tests` — тесты валидации, агрегации, исполнения пайплайна и хранения сессий.

## CLI

```bash
python -m app.cli \
  --models config/models.example.yaml \
  --pipeline config/pipeline.example.yaml \
  --mode base \
  --input "Сравни подходы" \
  --json
```

## Запуск тестов

```bash
python -m pip install -e .[dev]
pytest
```


## UI (Streamlit)

```bash
streamlit run app/ui.py
```
