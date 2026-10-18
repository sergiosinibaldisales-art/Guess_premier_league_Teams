"""Estatísticas descritivas: tabela, forma, mando, confronto direto."""
from __future__ import annotations

import pandas as pd

STATS = ("shots", "sot", "corners", "possession", "fouls")


def long_format(matches: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por TIME por jogo (em vez de uma por jogo).

    Facilita quase toda estatística: "gols pró" é sempre a coluna gf, não
    importa se o time jogou em casa ou fora.
    """
    played = matches[matches["hg"].notna()].copy()
    rows = []
    for side, opp, pfx, opfx in (("home", "away", "h_", "a_"), ("away", "home", "a_", "h_")):
        d = pd.DataFrame({
            "round": played["round"], "date": played["date_played"],
            "team": played[side], "opponent": played[opp],
            "venue": "casa" if side == "home" else "fora",
            "gf": played["hg" if side == "home" else "ag"].astype(int),
            "ga": played["ag" if side == "home" else "hg"].astype(int),
        })
        for stat in STATS:
            if pfx + stat in played:
                d[stat] = played[pfx + stat]
                d[stat + "_contra"] = played[opfx + stat]
        rows.append(d)
    df = pd.concat(rows, ignore_index=True)
    df["pts"] = (3 * (df.gf > df.ga) + (df.gf == df.ga)).astype(int)
    df["res"] = df.apply(lambda r: "V" if r.gf > r.ga else ("E" if r.gf == r.ga else "D"), axis=1)
    return df.sort_values(["team", "date"])


def table(matches: pd.DataFrame, venue: str | None = None) -> pd.DataFrame:
    """Classificação com o desempate da CBF: pontos, vitórias, saldo, gols pró."""
    lf = long_format(matches)
    if venue:
        lf = lf[lf["venue"] == venue]
    t = lf.groupby("team").agg(
        J=("pts", "size"), V=("res", lambda s: (s == "V").sum()),
        E=("res", lambda s: (s == "E").sum()), D=("res", lambda s: (s == "D").sum()),
        GP=("gf", "sum"), GC=("ga", "sum"), Pts=("pts", "sum"))
    t["SG"] = t["GP"] - t["GC"]
    t["Aprov%"] = (100 * t["Pts"] / (3 * t["J"])).round(1)
    t = t.sort_values(["Pts", "V", "SG", "GP"], ascending=False)
    t.insert(0, "Pos", range(1, len(t) + 1))
    t["Forma"] = lf.groupby("team")["res"].apply(lambda s: "".join(s.tail(5)))
    return t


def team_profile(matches: pd.DataFrame, team: str) -> pd.DataFrame:
    """Médias por jogo de um time, separadas por mando."""
    lf = long_format(matches)
    lf = lf[lf["team"] == team]
    num = lf.select_dtypes("number").drop(columns=["round"])
    out = num.groupby(lf["venue"]).mean().T
    out["geral"] = num.mean()
    return out.round(2)


def head_to_head(history: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    h = history[((history.home == a) & (history.away == b)) |
                ((history.home == b) & (history.away == a))]
    return h.sort_values("date")[["date", "home", "hg", "ag", "away"]]
