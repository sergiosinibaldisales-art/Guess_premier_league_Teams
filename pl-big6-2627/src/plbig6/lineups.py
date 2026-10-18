"""Transforma a escalação do dia em ajuste de força do time.

O modelo Dixon-Coles mede a força do TIME "típico". Se no dia o time joga
sem o artilheiro, a força real é menor. Aqui comparamos o XI informado
com o XI "habitual" (os 11 com mais minutos na temporada) usando dados do
FPL:

  ataque: soma de xGI/90 (gols + assistências esperados por 90 min) dos
          jogadores de linha. XI de hoje / XI habitual = multiplicador.
  defesa: média de xGC/90 (gols esperados sofridos com o jogador em campo)
          do goleiro e defensores. Maior que o habitual = defesa pior.

É uma HEURÍSTICA, não um modelo validado: por isso encolhemos o ajuste pela
metade (SHRINK) e limitamos a ±20%. Se o FPL estiver fora do ar, você pode
escrever o ajuste à mão nas colunas ajuste_ataque / ajuste_defesa.

Formato do arquivo data/escalacoes/rodada_XX.csv (uma linha por titular):
    time,jogador[,ajuste_ataque,ajuste_defesa]
    Arsenal,Raya
    Arsenal,Saliba
    ...
"""
from __future__ import annotations

import difflib
import unicodedata

import numpy as np
import pandas as pd

from .config import LINEUPS, team
from .sources import SourceUnavailable, fpl

SHRINK = 0.5
CLIP_ATT = (0.8, 1.2)
CLIP_DEF = (0.85, 1.2)
MIN_MINUTES = 270   # abaixo disso o "por 90" é ruído; usamos a mediana do time


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return s.lower().strip()


def read_lineups(round_no: int) -> pd.DataFrame:
    path = LINEUPS / f"rodada_{round_no:02d}.csv"
    if not path.exists():
        return pd.DataFrame(columns=["time", "jogador"])
    df = pd.read_csv(path, comment="#")
    df["time"] = df["time"].map(team)
    return df


def _per90(players: pd.DataFrame) -> pd.DataFrame:
    p = players.copy()
    mins = p["minutes"].clip(lower=1)
    p["xgi90"] = p["expected_goal_involvements"] / mins * 90
    p["xgc90"] = p["expected_goals_conceded"] / mins * 90
    small = p["minutes"] < MIN_MINUTES
    for col in ("xgi90", "xgc90"):
        med = p[~small].groupby(["team", "pos"])[col].median()
        p.loc[small, col] = [med.get((t, pos), np.nan) for t, pos in
                             zip(p.loc[small, "team"], p.loc[small, "pos"])]
    return p


def _match_player(name: str, squad: pd.DataFrame) -> pd.Series | None:
    keys = {}
    for i, r in squad.iterrows():
        for k in (r["web_name"], r["name"], r["name"].split()[-1]):
            keys.setdefault(_norm(k), i)
    hit = difflib.get_close_matches(_norm(name), list(keys), n=1, cutoff=0.75)
    return squad.loc[keys[hit[0]]] if hit else None


def team_multipliers(team_name: str, xi: list[str],
                     players: pd.DataFrame) -> tuple[float, float, list[str]]:
    """(mult_ataque, mult_defesa, jogadores_não_encontrados)."""
    squad = players[players["team"] == team_name]
    usual = squad.nlargest(11, "minutes")
    found, missing = [], []
    for n in xi:
        r = _match_player(n, squad)
        (found.append(r) if r is not None else missing.append(n))
    today = pd.DataFrame(found) if found else squad.iloc[0:0]

    def att(df):
        return df.loc[df["pos"] != "GK", "xgi90"].sum()

    def dfn(df):
        return df.loc[df["pos"].isin(["GK", "DEF"]), "xgc90"].mean()

    # jogador não encontrado conta como um jogador mediano do elenco
    att_today = att(today) + len(missing) * squad["xgi90"].median()
    raw_att = att_today / max(att(usual), 1e-6)
    raw_def = dfn(today) / dfn(usual) if len(today) and dfn(usual) > 0 else 1.0
    m_att = float(np.clip(1 + SHRINK * (raw_att - 1), *CLIP_ATT))
    m_def = float(np.clip(1 + SHRINK * (np.nan_to_num(raw_def, nan=1.0) - 1), *CLIP_DEF))
    return m_att, m_def, missing


def multipliers_for_round(round_no: int, players: pd.DataFrame | None = None) -> dict:
    """{time: {"att": x, "def": y}} para os times com escalação informada."""
    lu = read_lineups(round_no)
    if lu.empty:
        return {}
    if players is not None and "xgi90" not in players:
        players = _per90(players)

    def first(g, col):
        return float(g[col].dropna().iloc[0]) if col in g and g[col].notna().any() else None

    out: dict[str, dict] = {}
    for t, g in lu.groupby("time"):
        a, d = first(g, "ajuste_ataque"), first(g, "ajuste_defesa")
        if a is not None or d is not None:          # ajuste manual tem prioridade
            out[t] = {"att": a or 1.0, "def": d or 1.0, "fonte": "manual"}
            continue
        if players is None:
            try:
                players = _per90(fpl.players())
            except SourceUnavailable as e:
                print(f"[aviso] FPL indisponível ({e}); escalação de {t} ignorada. "
                      "Use as colunas ajuste_ataque/ajuste_defesa.")
                continue
        m_att, m_def, miss = team_multipliers(t, g["jogador"].tolist(), players)
        if miss:
            print(f"[aviso] {t}: não achei no FPL {miss} (contados como medianos)")
        out[t] = {"att": m_att, "def": m_def, "fonte": "FPL"}
    return out
