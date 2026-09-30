#!/bin/bash
# Плановая проверка раз в час: одна строка состояния, коммит новых результатов, перезапуск
# фоновых процессов, если контейнер перезапускался. Без моделей.
cd "$(dirname "$0")/.."
pgrep -f "tp.collect snapshots" >/dev/null || \
  nohup setsid python -m tp.collect snapshots --rpm 20 >> logs/snapshots.log 2>&1 < /dev/null &
pgrep -f scripts/api_queue.sh >/dev/null || \
  nohup setsid bash scripts/api_queue.sh > /dev/null 2>&1 < /dev/null &
grep -q "выгрузка других рынков завершена" logs/night.log || pgrep -f scripts/night_fetch.sh >/dev/null || \
  nohup setsid bash scripts/night_fetch.sh > /dev/null 2>&1 < /dev/null &
grep -q "ГОТОВО" logs/night.log || pgrep -f scripts/night.sh >/dev/null || \
  nohup setsid bash scripts/night.sh >> logs/night.log 2>&1 < /dev/null &
q=$(tail -1 logs/api_queue.log 2>/dev/null | cut -c1-60)
n=$(tail -1 logs/night.log 2>/dev/null)
git add results data/universe*.json logs/night.log tp scripts 2>/dev/null
git commit -qm "Heartbeat: night run progress

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Q7KwXVrUwmATHAX2JAxacY" >/dev/null 2>&1 && git push -q origin claude/upbeat-meitner-6rvrzu 2>/dev/null
pgrep -f scripts/night.sh >/dev/null && r=идёт || r=стоит
s=$(find data/raw/snap -name "*.parquet" -mmin -15 2>/dev/null | wc -l)
echo "ночь: $r | $n | очередь API: $q | снимков за 15 мин: $s | диск: $(du -sh data/raw | cut -f1)"
