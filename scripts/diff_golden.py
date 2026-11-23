#!/usr/bin/env python3
"""Детальное сравнение golden-статистики с текущим прогоном для списка документов."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))
from test_estimate import run_doc  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
GOLDEN = REPO / "tests/estimate/golden"

for name in sys.argv[1:]:
    pdf = REPO / "tests/data/real" / name
    snap = run_doc(pdf)
    gold = json.loads((GOLDEN / (pdf.stem + ".json")).read_text())
    print("=" * 20, name)
    for k in sorted(set(gold) | set(snap)):
        g, s = gold.get(k), snap.get(k)
        if g == s:
            continue
        if k in ("warnings", "errors"):
            from collections import Counter
            cg, cs = Counter(g or []), Counter(s or [])
            print(f"  {k}: −{dict(cg - cs)} +{dict(cs - cg)}")
        elif k == "sections_detail":
            for i, (go, so) in enumerate(zip(g, s)):
                if go != so:
                    print(f"  sections[{i}]: {go} → {so}")
            if len(g) != len(s):
                print(f"  секций: {len(g)} → {len(s)}")
        else:
            print(f"  {k}: {g} → {s}")
