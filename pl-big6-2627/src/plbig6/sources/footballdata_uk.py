"""football-data.co.uk — CSV gratuito com placar + estatísticas + odds.

É a fonte mais rica de graça: chutes (HS/AS), chutes no alvo (HST/AST),
escanteios (HC/AC), faltas, cartões e odds de várias casas. Não tem número
da rodada, por isso a rodada vem SEMPRE do calendário congelado (data.py).
"""
from __future__ import annotations

import io

import pandas as pd

from ..config import team
from .base import get_text

URL = "https://www.football-data.co.uk/mmz4281/{code}/{div}.csv"

# Nome no CSV -> nome no nosso DataFrame
STAT_COLS = {
    "HS": "h_shots", "AS": "a_shots", "HST": "h_sot", "AST": "a_sot",
    "HC": "h_corners", "AC": "a_corners", "HF": "h_fouls", "AF": "a_fouls",
    "HY": "h_yellow", "AY": "a_yellow", "HR": "h_red", "AR": "a_red",
    "HTHG": "ht_hg", "HTAG": "ht_ag", "Referee": "referee",
    # Odds médias de mercado (fechamento quando existir, senão abertura)
    "AvgCH": "odds_h", "AvgCD": "odds_d", "AvgCA": "odds_a",
}
_OPENING_FALLBACK = {"AvgH": "odds_h", "AvgD": "odds_d", "AvgA": "odds_a"}


def season_code(season: str) -> str:
    """'2026-27' -> '2627' (formato da URL do site)."""
    a, b = season.split("-")
    return a[2:] + b[-2:]


def load(season: str, div: str = "E0") -> pd.DataFrame:
    raw = get_text(URL.format(code=season_code(season), div=div))
    # O arquivo às vezes vem com BOM e linhas vazias no fim.
    df = pd.read_csv(io.StringIO(raw.lstrip("﻿"))).dropna(subset=["HomeTeam"])
    out = pd.DataFrame({
        "season": season,
        "date": pd.to_datetime(df["Date"], dayfirst=True),
        "home": df["HomeTeam"].map(team),
        "away": df["AwayTeam"].map(team),
        "hg": df["FTHG"].astype("Int64"),
        "ag": df["FTAG"].astype("Int64"),
    })
    for src, dst in STAT_COLS.items():
        if src in df:
            out[dst] = df[src]
    for src, dst in _OPENING_FALLBACK.items():
        if dst not in out and src in df:
            out[dst] = df[src]
    out["source"] = "football-data.co.uk"
    return out
