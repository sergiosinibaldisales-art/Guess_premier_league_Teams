"""API pública da ESPN (placar do dia) — estatísticas de jogo e odds.

Por jogo encerrado: finalizações, no alvo, escanteios, posse, faltas.
Por jogo futuro: odds de 1X2 e de +/-2,5 gols da DraftKings (quando a ESPN publica).
A consulta é por DATA (não aceita intervalo), então buscamos um dia por vez
e o cache guarda os dias passados por muito tempo (não mudam mais).
"""
from __future__ import annotations

import pandas as pd

from ..config import team
from .base import SourceUnavailable, get_json

URL = "https://site.api.espn.com/apis/site/v2/sports/soccer/bra.1/scoreboard?dates={d:%Y%m%d}"
_STATS = {"totalShots": "shots", "shotsOnTarget": "sot", "wonCorners": "corners",
          "possessionPct": "possession", "foulsCommitted": "fouls"}


def _american_to_decimal(ml) -> float | None:
    """Odd americana (+275 / -155) -> decimal (3.75 / 1.65)."""
    try:
        ml = float(str(ml).replace("+", ""))
    except (TypeError, ValueError):
        return None
    if ml == 0:
        return None
    return 1 + ml / 100 if ml > 0 else 1 + 100 / abs(ml)


def _line(block: dict, side: str):
    x = block.get(side) or {}
    return (x.get("close") or x.get("open") or {}).get("odds")


def day(d: pd.Timestamp) -> pd.DataFrame:
    old = pd.Timestamp(d) < pd.Timestamp.today().normalize() - pd.Timedelta(days=3)
    data = get_json(URL.format(d=pd.Timestamp(d)), max_age_hours=24 * 365 if old else 2)
    rows = []
    for e in data.get("events", []):
        c = e["competitions"][0]
        side = {x["homeAway"]: x for x in c["competitors"]}
        done = c["status"]["type"].get("completed", False)
        row = {"home": team(side["home"]["team"]["displayName"]),
               "away": team(side["away"]["team"]["displayName"]),
               "hg": int(side["home"]["score"]) if done else pd.NA,
               "ag": int(side["away"]["score"]) if done else pd.NA,
               "date": pd.Timestamp(e["date"]).tz_convert("America/Sao_Paulo")
                         .tz_localize(None).normalize()}
        for pfx, k in (("h_", "home"), ("a_", "away")):
            for s in side[k].get("statistics", []):
                if s["name"] in _STATS:
                    row[pfx + _STATS[s["name"]]] = pd.to_numeric(s["displayValue"], errors="coerce")
        odds = (c.get("odds") or [None])[0] or {}
        ml = odds.get("moneyline") or {}
        for k, col in (("home", "h"), ("draw", "d"), ("away", "a")):
            # "close" = linha mais recente; antes do jogo é a odd atual
            row[f"odd_mkt_{col}"] = _american_to_decimal(_line(ml, k))
        if odds.get("overUnder") == 2.5:
            row["odd_mkt_over25"] = _american_to_decimal(_line(odds.get("total") or {}, "over"))
            row["odd_mkt_under25"] = _american_to_decimal(_line(odds.get("total") or {}, "under"))
        rows.append(row)
    return pd.DataFrame(rows)


def load_days(days) -> pd.DataFrame:
    """Vários dias. Buscamos também o dia seguinte: a ESPN usa outro fuso e
    um jogo às 21h30 de Brasília pode aparecer na data do dia seguinte."""
    want = sorted({pd.Timestamp(x).normalize() for x in days} |
                  {pd.Timestamp(x).normalize() + pd.Timedelta(days=1) for x in days})
    frames = []
    for d in want:
        try:
            frames.append(day(d))
        except SourceUnavailable as e:      # um dia falhando não derruba os outros
            print(f"[aviso] ESPN {d:%d/%m}: {e}")
    frames = [f for f in frames if len(f)]
    if not frames:
        return pd.DataFrame(columns=["home", "away", "hg", "ag", "date"])
    df = pd.concat(frames, ignore_index=True).drop_duplicates(["home", "away"], keep="last")
    df[["hg", "ag"]] = df[["hg", "ag"]].astype("Int64")
    df["source"] = "ESPN"
    return df
