# project_llm

Локальное приложение для оркестрации multi-stage пайплайнов поверх нескольких GGUF-моделей.

> **Важно:** `ModelWrapper` использует `llama-cpp-python` runtime. Для локального запуска нужны совместимые бинари/драйверы (CPU или CUDA-сборка).

---

## 1) Что умеет проект уже сейчас

- Валидирует `models.yaml` и `pipeline.yaml` по строгим схемам.
- Проверяет, какие модели реально доступны на диске.
- Запускает pipeline по стадиям (single/multi) с агрегацией.
- Сохраняет историю прогонов в `sessions/`.
- Даёт CLI и Streamlit UI для запуска.
- Имеет тесты и CI workflow для GitHub.

---

## 2) Быстрый старт (локально)

### Шаг 1. Клонировать репозиторий

```bash
git clone <REPO_URL>
cd project_llm
```

### Шаг 2. Подготовить окружение

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .[dev]
```

### Шаг 3. Создать рабочие конфиги и папки

```bash
./scripts/bootstrap.sh
```

Скрипт создаст:
- `config/models.yaml` из примера;
- `config/pipeline.yaml` из примера;
- папки `models/` и `sessions/`.

### Шаг 4. Положить модели

Скопируйте `.gguf` файлы в `./models` и проверьте, что имена файлов совпадают с `config/models.yaml`.

### Шаг 5. Запустить UI

```bash
streamlit run app/ui.py
```

Или CLI:

```bash
./scripts/run_cli_example.sh
```

---

## 3) Запуск через Docker

```bash
docker compose up --build
```

Откройте: `http://localhost:8501`

Используются volume:
- `./config -> /app/config`
- `./models -> /app/models`
- `./sessions -> /app/sessions`

---

## 4) Подробная структура проекта (максимально просто)

Ниже разбор **каждого важного файла** и зачем он нужен.

### Корень

- `.gitignore` — исключает `__pycache__`, `*.pyc` и т.п.
- `pyproject.toml` — зависимости, пакетная конфигурация, точка входа CLI.
- `Dockerfile` — контейнер для запуска Streamlit UI.
- `docker-compose.yml` — удобный запуск контейнера с volume для моделей/сессий.
- `README.md` — эта инструкция.

### Папка `.github/`

- `.github/workflows/ci.yml` — GitHub Actions: установка зависимостей, compile check, pytest.
- `.github/dependabot.yml` — авто-PR для обновления pip и GitHub Actions зависимостей.
- `.github/pull_request_template.md` — шаблон PR, чтобы не забывать проверки.
- `.github/ISSUE_TEMPLATE/bug_report.yml` — шаблон баг-репорта.
- `.github/ISSUE_TEMPLATE/feature_request.yml` — шаблон запроса функционала.

### Папка `app/`

- `app/__init__.py` — маркер Python-пакета.
- `app/cli.py` — запуск pipeline из командной строки.
- `app/ui.py` — Streamlit интерфейс (Base/Manual/History/Models).

#### `app/config/`

- `app/config/__init__.py` — пакет config.
- `app/config/schemas.py` — Pydantic-схемы для `models.yaml` и `pipeline.yaml`.
- `app/config/loader.py` — чтение YAML + валидация в схемы.
- `app/config/validation.py` — runtime-проверки (наличие файлов, ссылки моделей).

#### `app/core/`

- `app/core/__init__.py` — пакет core.
- `app/core/model_registry.py` — формирует runtime-реестр моделей, фильтрует отсутствующие.
- `app/core/model_wrapper.py` — слой работы с моделью (сейчас mock, потом сюда llama.cpp).
- `app/core/aggregation.py` — concat/vote/synthesis/custom-template агрегация.
- `app/core/aggregation_engine.py` — shim-импорт для совместимости пути `aggregation_engine`.
- `app/core/resource_manager.py` — проверки RAM/VRAM с safety-коэффициентами.
- `app/core/session_manager.py` — хранение шагов, final.txt, чтение истории сессий.
- `app/core/pipeline_engine.py` — основной исполнитель стадий pipeline.
- `app/core/executors.py` — orchestration для base/manual режима.

### Папка `config/`

- `config/models.example.yaml` — пример конфигурации моделей.
- `config/pipeline.example.yaml` — пример base pipeline.
- `config/models.yaml` — рабочий конфиг (создаётся bootstrap-скриптом).
- `config/pipeline.yaml` — рабочий конфиг (создаётся bootstrap-скриптом).

### Папка `scripts/`

- `scripts/bootstrap.sh` — инициализация проекта после clone.
- `scripts/run_ui.sh` — быстрый запуск UI.
- `scripts/run_cli_example.sh` — пример запуска CLI.

### Папка `tests/`

- `tests/test_config_validation.py` — проверки схем и загрузки конфигов.
- `tests/test_aggregation.py` — проверки агрегаторов.
- `tests/test_pipeline_execution.py` — smoke execution pipeline.
- `tests/test_resource_manager.py` — checks safety-коэффициентов.
- `tests/test_session_manager.py` — запись/чтение сессий.

### Папки данных

- `models/` — сюда кладутся `.gguf`.
- `sessions/` — здесь хранится история запусков.

---

## 5) Конфиги

### `config/models.yaml`

Главные поля:
- `models_directory` — путь к папке моделей;
- `defaults` — параметры по умолчанию;
- `models` — словарь моделей с `file`, `quantization`, `required_for_base`, `overrides`.

### `config/pipeline.yaml`

Главные поля:
- `base_pipeline.stages[]` — список стадий;
- стадия `single` — одно поле `model`;
- стадия `multi` — `models[]` + `aggregation`;
- `input_from` — связь с предыдущим этапом (строка) или объединение нескольких предыдущих этапов (список).
- `output_mode` — что передавать в следующий этап: `answer_only` или `input_plus_answer`.

---

## 6) Что ещё не хватает до полноценного production

Ниже список честно, без «магии»:

1. **Production-hardening inference backend**
   - Базовый runtime уже работает через `llama-cpp-python`.
   - Нужны более детальная обработка ошибок (CUDA OOM и т.п.) и fallback-стратегии.

2. **Manual Builder UX-polish**
   - Визуальный конструктор стадий уже есть (single/multi, generation, aggregation).
   - Можно добавить drag-and-drop, сохранение пресетов и более компактный layout.

3. **Точность оценки памяти**
   - Добавлена оценка на основе размера модели + KV cache (n_ctx).
   - Для production желательно учитывать больше метаданных (слои, kv precision, offload profile).

4. **Полная ветвистая версия стадий (DAG rerun)**
   - Базовые versioned артефакты есть.
   - Нужно полноценное parent-child ветвление rerun с UI-выбором ветки.

5. **Набор интеграционных тестов с реальными GGUF**
   - Нужны e2e тесты с маленькой моделью для CI nightly.

6. **Релизный процесс**
   - Можно добавить tagging/changelog/release workflow.

---

## 7) Как добавить недостающее (коротко)

- **llama backend**: усилить обработку ошибок/таймаутов и добавить режимы graceful fallback.
- **manual builder**: добавить drag-and-drop порядок стадий, сохранение/загрузку пользовательских шаблонов.
- **DAG rerun**: в `SessionManager` хранить `run_id`, `parent_run_id` и отдельный индекс веток.
- **релизы**: добавить workflow, который на `tag` собирает wheel/docker image.

---

## 8) Финальная установка на отдельной машине (короткая инструкция)

1. Установить Python 3.10+ (или Docker).
2. `git clone` репо, перейти в папку.
3. `python -m venv .venv && source .venv/bin/activate`.
4. `python -m pip install -e .[dev]`.
5. `./scripts/bootstrap.sh`.
6. Положить `.gguf` в `models/`.
7. Отредактировать `config/models.yaml` под реальные имена файлов.
8. Запустить `streamlit run app/ui.py`.
9. Проверить вкладку **Статус моделей** (все нужные модели должны быть доступны).

---

## 9) Полезные команды

```bash
# Проверка синтаксиса
python -m compileall app tests

# Тесты
pytest -q

# UI
streamlit run app/ui.py

# CLI пример
python -m app.cli --models config/models.yaml --pipeline config/pipeline.yaml --mode base --input "Привет" --json
```
