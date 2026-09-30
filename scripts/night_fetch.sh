#!/bin/bash
# Ночная выгрузка других рынков (спот Binance, Bybit, Hyperliquid, Coinbase, фандинг Bybit) на 400 монет.
cd "$(dirname "$0")/.."
while ! grep -q "\[2400/2400" logs/fetch_101_400.log; do sleep 60; done
python -m tp.fetch all --universe 400 --sets spot_1h bybitf_1h hl_1h cb_1h funding_bybitf > logs/fetch_ext_400.log 2>&1
echo "[$(date -u +%H:%M)] выгрузка других рынков завершена" >> logs/night.log
