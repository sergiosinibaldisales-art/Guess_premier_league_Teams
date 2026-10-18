"""Monta a base de partidas juntando as fontes, com fallback.

Regras que este módulo garante:
  * A RODADA de um jogo vem sempre do calendário congelado (o calendário
    original da temporada, salvo uma única vez em data/). Assim um jogo
    adiado continua na rodada original, mesmo que seja disputado semanas
    depois — exatamente a regra que você pediu.
  * Na Premier League cada confronto mandante x visitante acontece uma vez
    por temporada. Então (home, away) é uma chave única e serve para casar
    as fontes sem depender de data (que muda em adiamentos).
  * Fontes são tentadas em ordem; se uma falha (limite, fora do ar), a
    próxima cobre. O que nenhuma fonte tiver pode vir de
    data/manual/resultados.csv (resultados verificados à mão).
"""
from __future__ import annotations

import warnings

import pandas as pd

from . import config
from .config import FIXTURES_FROZEN, MANUAL, SEASON
from .sources import SourceUnavailable, footballdata_uk, fpl, openfootball, understat

KEY = ["home", "away"]


# --------------------------------------------------------------------------
# Calendário congelado
# --------------------------------------------------------------------------
def freeze_fixtures(season: str = SEASON, force: bool = False) -> pd.DataFrame:
    """Salva o calendário original (rodada, data, mandante, visitante).

    Só roda de verdade na primeira vez: depois o arquivo existe e é lido
    dele. `force=True` sobrescreve — só use se o calendário oficial for
    republicado (raro) e você souber o que está fazendo.
    """
    path = config.DATA / f"calendario_{season}.csv"
    if path.exists() and not force:
        return pd.read_csv(path, parse_dates=["date_original"])
    df = openfootball.load(season)[["round", "date", "home", "away"]]
    df = df.rename(columns={"date": "date_original"})
    assert not df.duplicated(KEY).any(), "confronto repetido no calendário"
    assert len(df) == 380, f"calendário incompleto: {len(df)} jogos"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


# --------------------------------------------------------------------------
# Resultados da temporada atual
# --------------------------------------------------------------------------
def _try(name: str, fn, *args) -> pd.DataFrame | None:
    try:
        df = fn(*args)
        print(f"[ok]    {name}: {len(df)} linhas")
        return df
    except (SourceUnavailable, OSError, ValueError, KeyError) as e:
        print(f"[falha] {name}: {e}")
        return None


def load_manual(season: str = SEASON) -> pd.DataFrame:
    path = MANUAL / f"resultados_{season}.csv"
    if not path.exists():
        return pd.DataFrame(columns=KEY + ["hg", "ag", "date", "fontes", "forcar"])
    df = pd.read_csv(path, comment="#", parse_dates=["date"])
    df["home"] = df["home"].map(config.team)
    df["away"] = df["away"].map(config.team)
    df[["hg", "ag"]] = df[["hg", "ag"]].astype("Int64")
    if "forcar" not in df:
        df["forcar"] = False
    df["forcar"] = df["forcar"].fillna(False).astype(bool)
    return df


def _fill(base: pd.DataFrame, src: pd.DataFrame, label: str) -> pd.DataFrame:
    """Preenche placares que ainda faltam em `base` com os de `src`.

    Também avisa quando duas fontes discordam do placar de um mesmo jogo —
    isso é sinal de erro em alguma delas e merece checagem manual.
    """
    src = src.dropna(subset=["hg", "ag"]).drop_duplicates(KEY)
    src = src[KEY + ["hg", "ag", "date"]].rename(
        columns={"hg": "hg_new", "ag": "ag_new", "date": "date_new"})
    m = base.merge(src, on=KEY, how="left")
    both = m["hg"].notna() & m["hg_new"].notna()
    diff = both & ((m["hg"] != m["hg_new"]) | (m["ag"] != m["ag_new"]))
    for _, r in m[diff].iterrows():
        warnings.warn(f"Conflito {r.home} x {r.away}: {r.source} "
                      f"{r.hg}-{r.ag} vs {label} {r.hg_new}-{r.ag_new}")
    take = m["hg"].isna() & m["hg_new"].notna()
    m.loc[take, ["hg", "ag"]] = m.loc[take, ["hg_new", "ag_new"]].values
    m.loc[take, "date_played"] = m.loc[take, "date_new"]
    m.loc[take, "source"] = label
    return m.drop(columns=["hg_new", "ag_new", "date_new"])


def season_matches(season: str = SEASON, use_network: bool = True) -> pd.DataFrame:
    """Todos os 380 jogos da temporada com placar (se já jogado) e estatísticas."""
    cal = freeze_fixtures(season)
    base = cal.copy()
    base["hg"] = pd.array([pd.NA] * len(base), dtype="Int64")
    base["ag"] = pd.array([pd.NA] * len(base), dtype="Int64")
    base["date_played"] = pd.NaT
    base["source"] = pd.NA

    stats = None
    if use_network:
        # 1º football-data.co.uk (placar + estatísticas + odds)
        fd = _try("football-data.co.uk", footballdata_uk.load, season)
        if fd is not None:
            base = _fill(base, fd, "football-data.co.uk")
            stats = fd.drop(columns=["season", "date", "hg", "ag", "source"])
        # 2º openfootball (só placar) cobre o que faltou
        of = _try("openfootball", openfootball.load, season)
        if of is not None:
            base = _fill(base, of, "openfootball")
        # 3º FPL (placar oficial) cobre o resto
        fx = _try("FPL fixtures", fpl.fixtures)
        if fx is not None:
            fx = fx[fx["finished"].fillna(False).astype(bool)].copy()
            fx["date"] = fx["kickoff"].dt.tz_convert("Europe/London").dt.tz_localize(None).dt.normalize()
            base = _fill(base, fx, "FPL")

    # 4º manual: preenche lacunas; com forcar=True sobrescreve (correção humana)
    man = load_manual(season)
    if len(man):
        forced = man[man["forcar"]]
        base = _fill(base, man, "manual")
        for _, r in forced.iterrows():
            sel = (base["home"] == r.home) & (base["away"] == r.away)
            base.loc[sel, ["hg", "ag", "source"]] = [r.hg, r.ag, "manual (forçado)"]

    if stats is not None:
        base = base.merge(stats.drop_duplicates(KEY), on=KEY, how="left")

    if use_network:
        xg = _try("understat (xG)", understat.load, season)
        if xg is not None:
            base = base.merge(xg.drop(columns="date").drop_duplicates(KEY), on=KEY, how="left")

    base["date_played"] = pd.to_datetime(base["date_played"]).fillna(
        base["date_original"].where(base["hg"].notna()))
    base["status"] = _status(base)
    return base.sort_values(["round", "date_original", "home"]).reset_index(drop=True)


def _status(df: pd.DataFrame) -> pd.Series:
    today = pd.Timestamp.today().normalize()
    played = df["hg"].notna()
    # Dois dias de folga: fontes demoram para lançar o placar.
    late = ~played & (df["date_original"] < today - pd.Timedelta(days=2))
    s = pd.Series("a jogar", index=df.index)
    s[played] = "jogado"
    s[late] = "pendente/adiado"
    return s


# --------------------------------------------------------------------------
# Histórico para treinar o modelo
# --------------------------------------------------------------------------
def history(seasons: list[str], include_championship: bool = False) -> pd.DataFrame:
    """Jogos com placar das temporadas anteriores (openfootball).

    Por que incluir a Championship? Times promovidos (ex.: Coventry, Hull)
    não têm jogos recentes na Premier. Os jogos da 2ª divisão conectam esses
    times aos rebaixados, que já jogaram a Premier — e isso dá ao modelo uma
    estimativa inicial melhor que "time médio". MAS o backtest
    mostrou (18/10/2026) que piora: log-loss 1.043 vs 1.027 em 2025-26 e 0.996 vs 0.992
    em 2024-25. Por isso o padrão é False. A hipótese era razoável, os dados
    disseram que não — fica a opção para testar de novo no futuro.
    """
    frames = []
    for s in seasons:
        leagues = ["en.1", "en.2"] if include_championship else ["en.1"]
        for lg in leagues:
            df = _try(f"openfootball {s} {lg}", openfootball.load, s, lg)
            if df is not None:
                frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    return out.dropna(subset=["hg", "ag"])


def training_set(current: pd.DataFrame, past_seasons: list[str],
                 include_championship: bool = False) -> pd.DataFrame:
    """Histórico + jogos já disputados da temporada atual, no mesmo formato."""
    hist = history(past_seasons, include_championship)
    cur = current[current["hg"].notna()].copy()
    cur["date"] = cur["date_played"]
    cur["league"] = "en.1"
    cols = ["date", "home", "away", "hg", "ag", "league"]
    return pd.concat([hist[cols], cur[cols]], ignore_index=True)
