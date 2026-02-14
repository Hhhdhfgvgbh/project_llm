# project_llm

Стартовый каркас локального Python-приложения для работы с несколькими GGUF-моделями.

## Что уже реализовано

- Строгие схемы и валидация `models.yaml` и `pipeline.yaml` через Pydantic.
- Runtime-проверки наличия модельных файлов и ссылочной целостности pipeline.
- `ModelRegistry` с объединением `defaults + overrides`.
- `AggregationEngine` для `concat`, `majority_vote`, `synthesis`, `custom_template`.
- `ResourceManager` c safety-коэффициентами RAM/VRAM.
- `SessionManager` с versioned stage-структурой и `meta.json`.

## Запуск тестов

```bash
python -m pip install -e .[dev]
pytest
```
