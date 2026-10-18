"""openfootball/football.json — calendário + placar (domínio público, GitHub).

Fonte reserva: tem as 38 rodadas, mas atrasa algumas rodadas para lançar
placares e só tem o Brasileirão a partir de 2025.
"""
from __future__ import annotations

import re

import pandas as pd

from ..config import team
from .base import get_json

URL = "https://raw.githubusercontent.com/openfootball/football.json/master/{season}/br.1.json"


def _round(label: str) -> int:
    digits = re.findall(r"\d+", str(label))
    return int(digits[-1]) if digits else 0


def load(season: int) -> pd.DataFrame:
    data = get_json(URL.format(season=season))
    rows = []
    for m in data["matches"]:
        score = m.get("score")
        ft = score.get("ft") if isinstance(score, dict) else score
        rows.append({
            "round": _round(m["round"]),
            "date": pd.Timestamp(m["date"]),
            "home": team(m["team1"]), "away": team(m["team2"]),
            "hg": ft[0] if ft else pd.NA, "ag": ft[1] if ft else pd.NA,
        })
    df = pd.DataFrame(rows)
    df[["hg", "ag"]] = df[["hg", "ag"]].astype("Int64")
    df["source"] = "openfootball"
    return df
