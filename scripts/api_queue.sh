#!/bin/bash
# Ночная очередь выгрузок API без моделей (аудит API, docs/reports/api_audit.md).
# Идёт после night_fetch.sh, по одному процессу за раз: лимит 600 в минуту общий на все процессы.
# Лог: logs/api_queue.log, итоги этапов — в logs/night.log.
cd "$(dirname "$0")/.."
log() { echo "[$(date -u +%H:%M)] очередь API: $*" >> logs/night.log; }
# сделанные шаги помнит в logs/queue_done.txt: после перезапуска контейнера не повторяет их
step() {
  grep -qxF "$1" logs/queue_done.txt 2>/dev/null && return
  log "$1"; local t="$1"; shift
  "$@" >> logs/api_queue.log 2>&1 && [ "${t#дозапись}" = "$t" ] && echo "$t" >> logs/queue_done.txt
}
while ! grep -q "выгрузка других рынков завершена" logs/night.log; do sleep 60; done
F="python -m tp.fetch all --rpm 560 --workers 12"
step "контекст: макро, ETF, onchain" python -m tp.collect context
step "события ликвидаций с ценой, 5 бирж, 400 монет" python -m tp.collect events --universe 400
step "свечи 5m: начало с 2026-03-05" $F --universe 400 --sets candles_5m --backfill
step "бары ликвидаций по биржам" $F --universe 400 --sets liq_binancef_5m liq_bybitf_5m liq_okx_5m
step "фандинг 6 бирж" $F --universe 400 --sets fund_okx fund_hyperliquid fund_aster fund_bitget fund_gate fund_htx
step "OI всех бирж 1h" $F --universe 400 --sets oi_all_1h
step "толпа: top_account и другие биржи" $F --universe 400 --sets ls_topacc ls_bybitf ls_okx ls_gate
step "CVD других бирж 1h" $F --universe 400 --sets cvd_bybitf_1h cvd_okx_1h cvd_coinbase_1h
step "OI всех бирж 5m" $F --universe 400 --sets oi_all_5m
step "спот 5m" $F --universe 400 --sets spot_5m
step "свечи 1m топ-100" $F --universe 100 --sets candles_1m
step "свечи 1m ранги 101-400" $F --universe 101-400 --sets candles_1m
# дальше до утра: дописывать хвосты всего скачанного раз в 2 часа
while true; do
  step "дозапись хвостов базовых наборов 400" $F --universe 400
  step "дозапись событий ликвидаций" python -m tp.collect events --universe 400
  sleep 7200
done
