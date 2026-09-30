#!/bin/bash
# Плановая проверка раз в час: одна строка состояния, коммит новых результатов. Без моделей.
cd "$(dirname "$0")/.."
f=$(tail -1 logs/fetch_101_400.log 2>/dev/null | cut -c1-40)
n=$(tail -1 logs/night.log 2>/dev/null)
git add results data/universe*.json logs/night.log 2>/dev/null
git commit -qm "Heartbeat: night run progress

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Q7KwXVrUwmATHAX2JAxacY" >/dev/null 2>&1 && git push -q origin claude/upbeat-meitner-6rvrzu 2>/dev/null
pgrep -f scripts/night.sh >/dev/null && r=идёт || r=остановлен
echo "ночь: $r | $n | выгрузка: $f"
