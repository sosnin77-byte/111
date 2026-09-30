"""100 гипотетических сетапов для массовой проверки (сверх 11 основных из tp/setups.py).

Сетапы разложены по файлам b01.py … b10.py, в каждом список SETUPS из 10 штук.
Имена вида h001_<кратко>, номер сквозной. Каталог с логикой — HYPOTHESES.md.

    python -m tp.hyp.check b01            # проверка файла: ошибки, заглядывание вперёд, частота
    python -m tp.research --pool hyp      # перебор параметров по всем гипотезам
"""
from __future__ import annotations

import importlib
import pkgutil

HYP = []
for _m in sorted(m.name for m in pkgutil.iter_modules(__path__) if m.name.startswith("b")):
    HYP.extend(importlib.import_module(f"{__name__}.{_m}").SETUPS)

for _s in HYP:
    if not _s.doc:
        _s.doc = " ".join((_s.fn.__doc__ or "").split())

BY_NAME = {s.name: s for s in HYP}
