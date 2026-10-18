"""openfootball/football.json — calendário completo + placares (domínio público).

Prós: calendário das 38 rodadas com o número da rodada; hospedado no GitHub
(raramente cai, sem limite prático). Contras: só placar (sem chutes/cartões)
e às vezes atrasa alguns dias para lançar a rodada.
"""
from __future__ import annotations

import re

import pandas as pd

from ..config import team
from .base import get_json

URL = "https://raw.githubusercontent.com/openfootball/football.json/master/{season}/{league}.json"


def _round(label: str) -> int:
    """'Matchday 7' -> 7. Playoffs da Championship ('Semifinals') -> 0."""
    digits = re.findall(r"\d+", str(label))
    return int(digits[-1]) if digits else 0


def load(season: str, league: str = "en.1") -> pd.DataFrame:
    data = get_json(URL.format(season=season, league=league))
    rows = []
    for m in data["matches"]:
        score = m.get("score")
        # O arquivo usa dois formatos: {"ft": [2, 1], "ht": [...]} ou só [2, 1].
        ft = score.get("ft") if isinstance(score, dict) else score
        rows.append({
            "season": season,
            "league": league,
            "round": _round(m["round"]),
            "date": pd.Timestamp(m["date"]),
            "home": team(m["team1"]),
            "away": team(m["team2"]),
            "hg": ft[0] if ft else pd.NA,
            "ag": ft[1] if ft else pd.NA,
        })
    df = pd.DataFrame(rows)
    df[["hg", "ag"]] = df[["hg", "ag"]].astype("Int64")
    df["source"] = "openfootball"
    return df
