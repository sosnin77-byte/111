#!/bin/bash
# Ночная работа без моделей: только код. Лог: logs/night.log. Результаты коммитит и пушит сам.
cd "$(dirname "$0")/.."
log() { echo "[$(date -u +%H:%M)] $*"; }
commit() {
  git add results data/universe*.json 2>/dev/null
  git commit -qm "Night run: $1

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Q7KwXVrUwmATHAX2JAxacY" 2>/dev/null
  for i in 1 2 3 4; do git push -q origin claude/upbeat-meitner-6rvrzu 2>/dev/null && break; sleep $((2**i)); done
  log "commit: $1"
}
log "жду окончания выгрузки 101-400"
while ! grep -q "\[2400/2400" logs/fetch_101_400.log; do sleep 60; done
log "1/6 дописываю хвост топ-100"
python -m tp.fetch all --universe 100 > logs/fetch_tail_100.log 2>&1
log "2/6 первичный отбор 11 сетапов по 5 когортам"
python -m tp.screen --pool main --cohorts --jobs 3 --runs 50 > logs/screen_main_cohorts.log 2>&1
commit "screening of main setups by cohorts"
log "3/6 первичный отбор гипотез (готовые файлы tp/hyp) по когортам"
python -m tp.screen --pool hyp --cohorts --jobs 3 --runs 50 > logs/screen_hyp_cohorts.log 2>&1
commit "screening of implemented hypotheses by cohorts"
log "4/6 перебор с контролем сетапов на OI после исправления"
python -m tp.research --setups squeeze_fuel squeeze_breakout oi_breakout > logs/research_oi_fix.log 2>&1
commit "research with controls after OI fix"
log "5/6 обратная инженерия на 400 монетах"
python -m tp.reverse discover --universe 400 > logs/reverse_400.log 2>&1 && \
python -m tp.reverse validate --universe 400 --top 15 --runs 30 >> logs/reverse_400.log 2>&1
commit "reverse engineering on 400 coins"
log "6/6 обратная инженерия топ-100 заново (OI в контрактах)"
python -m tp.reverse discover --universe 100 > logs/reverse_100_oi_fix.log 2>&1 && \
python -m tp.reverse validate --universe 100 --top 15 --runs 30 >> logs/reverse_100_oi_fix.log 2>&1
commit "reverse engineering top-100 with OI in contracts"
log "ГОТОВО"
