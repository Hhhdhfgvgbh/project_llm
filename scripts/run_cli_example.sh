#!/usr/bin/env bash
set -euo pipefail
python -m app.cli \
  --models config/models.yaml \
  --pipeline config/pipeline.yaml \
  --mode base \
  --input "Пример запроса" \
  --json
