"""Escalação do dia -> ajuste de força do time (dados do Cartola).

O modelo mede a força do time "típico". Se no dia falta o artilheiro, a
força real é menor. Comparamos o XI informado com o XI "habitual" (os 11
com mais jogos na temporada), usando os scouts do Cartola:

  ataque: soma de (gols + assistências) por jogo dos jogadores de linha.
          XI de hoje / XI habitual = multiplicador do ataque.
  defesa: média de SG por jogo (jogos sem sofrer gol) de goleiro, zagueiros
          e laterais. Hoje MENOR que o habitual = defesa pior (mult > 1).

Diferença para a versão da Premier League: lá havia xG/xA do FPL; aqui o
Cartola só tem gols e assistências reais, que são mais ruidosos. Por isso
o ajuste é ainda mais desconfiável: encolhemos pela metade (SHRINK) e
limitamos. Se quiser, escreva o ajuste à mão em ajuste_ataque/ajuste_defesa.

Formato de data/escalacoes/rodada_XX.csv (uma linha por titular):
    time,jogador[,ajuste_ataque,ajuste_defesa]

Atalho: `python -m brasileirao.lineups 31` cria rodada_31.csv já
preenchido com os "Prováveis" do Cartola, para você só corrigir.
"""
from __future__ import annotations

import difflib
import sys
import unicodedata

import numpy as np
import pandas as pd

from .config import LINEUPS, team
from .sources import SourceUnavailable, cartola

SHRINK = 0.5
CLIP_ATT = (0.8, 1.2)
CLIP_DEF = (0.85, 1.2)
MIN_GAMES = 5        # abaixo disso a média por jogo é ruído: usa a mediana
DEF_POS = ["GOL", "ZAG", "LAT"]


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


def per_game(players: pd.DataFrame) -> pd.DataFrame:
    p = players[players["pos"] != "TEC"].copy()
    g = p["jogos"].clip(lower=1)
    p["ga_jogo"] = (p["G"] + p["A"]) / g
    p["sg_jogo"] = p["SG"] / g
    few = p["jogos"] < MIN_GAMES
    for col in ("ga_jogo", "sg_jogo"):
        med = p[~few].groupby(["team", "pos"])[col].median()
        p.loc[few, col] = [med.get((t, pos), np.nan) for t, pos in
                           zip(p.loc[few, "team"], p.loc[few, "pos"])]
    return p


def _match_player(name: str, squad: pd.DataFrame) -> pd.Series | None:
    keys = {}
    for i, r in squad.iterrows():
        for k in (r["apelido"], r["nome"], str(r["nome"]).split()[-1]):
            keys.setdefault(_norm(k), i)
    hit = difflib.get_close_matches(_norm(name), list(keys), n=1, cutoff=0.75)
    return squad.loc[keys[hit[0]]] if hit else None


def team_multipliers(team_name: str, xi: list[str],
                     players: pd.DataFrame) -> tuple[float, float, list[str]]:
    """(mult_ataque, mult_defesa, jogadores_não_encontrados)."""
    squad = players[players["team"] == team_name]
    usual = squad.nlargest(11, "jogos")
    found, missing = [], []
    for n in xi:
        r = _match_player(n, squad)
        (found.append(r) if r is not None else missing.append(n))
    today = pd.DataFrame(found) if found else squad.iloc[0:0]

    def att(df):
        return df.loc[df["pos"] != "GOL", "ga_jogo"].sum()

    def dfn(df):
        return df.loc[df["pos"].isin(DEF_POS), "sg_jogo"].mean()

    # jogador não encontrado conta como um jogador mediano do elenco
    att_today = att(today) + len(missing) * squad["ga_jogo"].median()
    raw_att = att_today / max(att(usual), 1e-6)
    d_today, d_usual = dfn(today), dfn(usual)
    # SG maior = defesa melhor, então a razão é habitual / hoje
    raw_def = d_usual / d_today if len(today) and d_today > 0 else 1.0
    m_att = float(np.clip(1 + SHRINK * (raw_att - 1), *CLIP_ATT))
    m_def = float(np.clip(1 + SHRINK * (np.nan_to_num(raw_def, nan=1.0) - 1), *CLIP_DEF))
    return m_att, m_def, missing


def multipliers_for_round(round_no: int, players: pd.DataFrame | None = None) -> dict:
    """{time: {"att": x, "def": y}} para os times com escalação informada."""
    lu = read_lineups(round_no)
    if lu.empty:
        return {}
    if players is not None and "ga_jogo" not in players:
        players = per_game(players)

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
                players = per_game(cartola.players())
            except SourceUnavailable as e:
                print(f"[aviso] Cartola indisponível ({e}); escalação de {t} ignorada. "
                      "Use as colunas ajuste_ataque/ajuste_defesa.")
                continue
        m_att, m_def, miss = team_multipliers(t, g["jogador"].dropna().tolist(), players)
        if miss:
            print(f"[aviso] {t}: não achei no Cartola {miss} (contados como medianos)")
        out[t] = {"att": m_att, "def": m_def, "fonte": "Cartola"}
    return out


def desfalques(players: pd.DataFrame, team_name: str, min_games: int = 3) -> pd.DataFrame:
    """Jogadores relevantes (com jogos) fora ou em dúvida, segundo o Cartola."""
    p = players[(players["team"] == team_name) & (players["pos"] != "TEC")
                & players["status"].isin(["Contundido", "Suspenso", "Dúvida"])
                & (players["jogos"] >= min_games)]
    return p.sort_values(["jogos", "G"], ascending=False)[
        ["apelido", "pos", "status", "jogos", "G", "A"]]


def write_probable(round_no: int, teams: list[str] | None = None) -> str:
    """Cria rodada_XX.csv com os 'Prováveis' do Cartola (não sobrescreve)."""
    path = LINEUPS / f"rodada_{round_no:02d}.csv"
    if path.exists():
        return f"{path.name} já existe; não sobrescrevi."
    p = cartola.players()
    p = p[(p["status"] == "Provável") & (p["pos"] != "TEC")]
    if teams:
        p = p[p["team"].isin(teams)]
    p = p.sort_values(["team", "pos"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Pré-preenchido com os 'Prováveis' do Cartola. Corrija no dia do jogo.\n")
        p.rename(columns={"team": "time", "apelido": "jogador"})[["time", "jogador"]].to_csv(
            f, index=False, lineterminator="\n")
    count = p.groupby("team").size()
    short = count[count != 11]
    note = f" Atenção: times sem 11 prováveis: {short.to_dict()}" if len(short) else ""
    return f"Escrevi {path.name} com {len(p)} jogadores.{note}"


if __name__ == "__main__":
    print(write_probable(int(sys.argv[1])))
