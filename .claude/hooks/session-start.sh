#!/bin/bash
# Старт сессии Claude Code on the web: зависимости и сводка состояния проекта.
# Вывод попадает в контекст новой сессии: она сразу видит, что есть, а чего нет (данные, результаты, план).
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "${CLAUDE_PROJECT_DIR:-$(dirname "$0")/../..}"

# зависимости (идемпотентно; состояние контейнера кэшируется после хука)
python -m pip install -q -r requirements.txt ruff >/dev/null 2>&1 || python -m pip install -r requirements.txt ruff

echo 'export PYTHONPATH="."' >> "${CLAUDE_ENV_FILE:-/dev/null}"

echo "=== Проект: читать PLAN.md, INSIGHTS.md, HYPOTHESES.md, MORNING.md ==="
python -m tp.status || true
if [ ! -d data/raw/candles_1h ]; then
  echo "ДАННЫХ НЕТ (data/raw не в git): python -m tp.fetch universe && python -m tp.fetch all  (~40 мин, топ-100);"
  echo "затем python -m tp.fetch universe --top 400 && python -m tp.fetch all --universe 101-400  (~1.5 ч)"
fi
