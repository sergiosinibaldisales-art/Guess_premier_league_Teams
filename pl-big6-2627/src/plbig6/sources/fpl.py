"""API oficial (e gratuita) do Fantasy Premier League.

Usamos para dados de JOGADOR: minutos, gols, assistências, xG/xA
(expected_goals / expected_assists), status de lesão e chance de jogar.
É isso que permite transformar a escalação do dia em ajuste de força.
Não exige login para estes dois endpoints.
"""
from __future__ import annotations

import pandas as pd

from ..config import team
from .base import get_json

BOOTSTRAP = "https://fantasy.premierleague.com/api/bootstrap-static/"
FIXTURES = "https://fantasy.premierleague.com/api/fixtures/"

POSITIONS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


def players() -> pd.DataFrame:
    data = get_json(BOOTSTRAP, max_age_hours=3)
    teams = {t["id"]: team(t["name"]) for t in data["teams"]}
    df = pd.DataFrame(data["elements"])
    num = ["minutes", "goals_scored", "assists", "expected_goals",
           "expected_assists", "expected_goal_involvements",
           "expected_goals_conceded", "starts"]
    for c in num:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df["team"] = df["team"].map(teams)
    df["pos"] = df["element_type"].map(POSITIONS)
    df["name"] = (df["first_name"] + " " + df["second_name"]).str.strip()
    keep = ["id", "team", "pos", "web_name", "name", "status", "news",
            "chance_of_playing_next_round"] + [c for c in num if c in df]
    return df[keep]


def fixtures() -> pd.DataFrame:
    """Jogos com a rodada ATUAL do FPL.

    Atenção: o FPL MUDA a rodada ('event') de um jogo adiado. Por isso não
    usamos esta rodada para alocar resultados — só a do calendário congelado.
    """
    data = get_json(FIXTURES, max_age_hours=3)
    teams = {t["id"]: team(t["name"]) for t in get_json(BOOTSTRAP)["teams"]}
    df = pd.DataFrame(data)
    return pd.DataFrame({
        "fpl_event": df["event"],
        "kickoff": pd.to_datetime(df["kickoff_time"], utc=True),
        "home": df["team_h"].map(teams),
        "away": df["team_a"].map(teams),
        "hg": df["team_h_score"].astype("Int64"),
        "ag": df["team_a_score"].astype("Int64"),
        "finished": df["finished"],
    })
