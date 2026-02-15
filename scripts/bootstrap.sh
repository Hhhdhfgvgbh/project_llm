#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p config models sessions

if [ ! -f config/models.yaml ]; then
  cp config/models.example.yaml config/models.yaml
  echo "Created config/models.yaml from example"
fi

if [ ! -f config/pipeline.yaml ]; then
  cp config/pipeline.example.yaml config/pipeline.yaml
  echo "Created config/pipeline.yaml from example"
fi

if [ ! -f config/cleaning_rules.yaml ]; then
  cp config/cleaning_rules.example.yaml config/cleaning_rules.yaml
  echo "Created config/cleaning_rules.yaml from example"
fi

echo "Bootstrap completed."
echo "1) Put your *.gguf models into ./models"
echo "2) Edit config/models.yaml file names"
echo "3) Run: streamlit run app/ui.py"
