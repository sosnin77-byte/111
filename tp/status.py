"""Состояние проекта одной командой: данные, результаты по направлениям, реестр гипотез, план.

    python -m tp.status

Нужна, чтобы ничего не терялось: каждое направление работы оставляет файлы в results/ или
docs/, и здесь видно, что есть, чего нет и что ещё не доведено до воронки проверки.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
RES = ROOT / "results"

# направление -> (где лежат результаты, как подключено к воронке)
STREAMS = [
    ("Основные 11 сетапов", "results/*.json, results/screen/100/", "screen → research (контроль)"),
    ("100 гипотез", "tp/hyp/b01…b10.py", "check → screen --pool hyp --cohorts → research"),
    ("Обратная инженерия", "results/reverse/", "discover → validate (screen span) → full (research)"),
    ("Супертрейдер", "results/trader/, .claude/agents/knowledge/supertrader_lessons.md",
     "правила → HYPOTHESES.md → код → screen"),
    ("Изобретатель (ТРИЗ)", "results/inventor/", "идеи → HYPOTHESES.md → код → screen"),
    ("Цикл трейдер + изобретатель", ".claude/workflows/duo-cycle.js, tp/hyp/bduo_*.py", "screen против базы"),
    ("Миссия изобретателя", ".claude/workflows/inventor-mission.js", "план → код → screen"),
    ("Гайд TRADER.PRO", "results/guide/ (после воркфлоу)", "идеи → HYPOTHESES.md → код → screen"),
    ("Маркет данные TRADER.PRO", "docs/reports/markets_study.md", "идеи → HYPOTHESES.md → код → screen"),
    ("Аудит проекта", "results/audit/ (после воркфлоу)", "правки в движок, данные, методику"),
    ("Карта заработка", "docs/reports/ (после воркфлоу)", "направления → PLAN.md"),
]


def data_status() -> list[str]:
    out = []
    uni = {}
    for n in (100, 400):
        f = ROOT / "data" / ("universe.json" if n == 100 else f"universe_{n}.json")
        if f.exists():
            uni[n] = json.loads(f.read_text())["symbols"]
    for d in sorted(RAW.glob("*")) if RAW.exists() else []:
        have = {p.stem for p in d.glob("*.parquet")}
        cov = "  ".join(f"топ-{n}: {len(have & set(s))}/{len(s)}" for n, s in uni.items())
        out.append(f"  {d.name:12s} {len(have):4d} монет   {cov}")
    return out or ["  данных нет — python -m tp.fetch universe && python -m tp.fetch all (см. PLAN.md)"]


def results_status() -> list[str]:
    out = []
    for name, where, funnel in STREAMS:
        files = []
        for part in where.split(","):
            part = part.strip().split(" ")[0]
            files += [p for p in ROOT.glob(part.replace("…", "*")) if p.exists()] if part else []
        mark = "есть" if files else "нет"
        out.append(f"  [{mark:4s}] {name:30s} {where:55s} воронка: {funnel}")
    screens = sorted(p.parent.name for p in (RES / "screen").glob("*/*.json"))
    if screens:
        from collections import Counter
        out.append("  первичный отбор по наборам монет: " +
                   ", ".join(f"{k}: {v}" for k, v in sorted(Counter(screens).items())))
    return out


def registry_status() -> list[str]:
    f = ROOT / "HYPOTHESES.md"
    if not f.exists():
        return ["  нет HYPOTHESES.md"]
    rows = [ln for ln in f.read_text().splitlines() if ln.startswith("| ") and not ln.startswith("| ---")
            and not ln.startswith("| Имя") and not ln.startswith("| id") and not ln.startswith("| Правило")]
    keys = ["кандидат", "отбор", "к проверке", "доработать", "идея", "код", "отброшена", "отсеяна", "на грани",
            "противоречиво", "разобрать"]
    cnt = {k: sum(k in r.lower() for r in rows) for k in keys}
    return [f"  строк в реестре: {len(rows)}; " + ", ".join(f"{k}: {v}" for k, v in cnt.items() if v)]


def plan_status() -> list[str]:
    f = ROOT / "PLAN.md"
    if not f.exists():
        return []
    s = f.read_text()
    done = len(re.findall(r"^- \[x\]", s, re.M))
    todo = re.findall(r"^- \[ \] (.+)$", s, re.M)
    return [f"  выполнено: {done}, открыто: {len(todo)}"] + [f"   · {t[:120]}" for t in todo]


def main():
    print("ДАННЫЕ (data/raw, не в git)")
    print("\n".join(data_status()))
    print("\nНАПРАВЛЕНИЯ И РЕЗУЛЬТАТЫ")
    print("\n".join(results_status()))
    print("\nРЕЕСТР ГИПОТЕЗ")
    print("\n".join(registry_status()))
    print("\nПЛАН")
    print("\n".join(plan_status()))


if __name__ == "__main__":
    main()
