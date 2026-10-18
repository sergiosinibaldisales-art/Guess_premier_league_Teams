"""Monta a base de partidas juntando as fontes, com fallback.

Regras (as mesmas do projeto da Premier League):
  * A RODADA vem do calendário congelado (data/calendario_2026.csv), salvo
    uma única vez a partir do Cartola, que usa a rodada oficial da CBF.
    Jogo adiado continua na rodada original, mesmo jogado semanas depois.
  * No Brasileirão cada mandante x visitante acontece uma vez por ano
    (turno e returno com mando invertido), então (home, away) é chave única
    e casa as fontes sem depender de data, que muda em adiamentos.
  * Fontes em ordem; se uma falha, a próxima cobre. O que nenhuma tiver
    pode vir de data/manual/resultados_2026.csv.
"""
from __future__ import annotations

import warnings

import pandas as pd

from . import config
from .config import MANUAL, N_ROUNDS, SEASON
from .sources import SourceUnavailable, cartola, espn, footballdata_br, openfootball

KEY = ["home", "away"]


def _try(name: str, fn, *args) -> pd.DataFrame | None:
    try:
        df = fn(*args)
        print(f"[ok]    {name}: {len(df)} linhas")
        return df
    except (SourceUnavailable, OSError, ValueError, KeyError) as e:
        print(f"[falha] {name}: {e}")
        return None


# --------------------------------------------------------------------------
# Calendário congelado
# --------------------------------------------------------------------------
def freeze_fixtures(season: int = SEASON, force: bool = False) -> pd.DataFrame:
    """Salva (rodada, data original, mandante, visitante) uma vez só.

    Fonte: Cartola (rodada oficial); se ele falhar, openfootball. A data
    gravada é a data do Cartola no momento do congelamento; o que importa
    aqui é a RODADA, a data real do jogo vem depois das fontes de placar.
    """
    path = config.DATA / f"calendario_{season}.csv"
    if path.exists() and not force:
        return pd.read_csv(path, parse_dates=["date_original"])
    src = _try("Cartola calendário", cartola.season)
    if src is None:
        src = _try("openfootball calendário", openfootball.load, season)
    if src is None:
        raise SourceUnavailable("nenhuma fonte de calendário respondeu")
    df = src[["round", "date", "home", "away"]].rename(columns={"date": "date_original"})
    assert not df.duplicated(KEY).any(), "confronto repetido no calendário"
    assert len(df) == N_ROUNDS * 10, f"calendário incompleto: {len(df)} jogos"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values(["round", "date_original", "home"]).to_csv(path, index=False)
    return df


# --------------------------------------------------------------------------
# Resultados da temporada atual
# --------------------------------------------------------------------------
def load_manual(season: int = SEASON) -> pd.DataFrame:
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
    """Preenche placares que faltam em `base` com os de `src` e avisa
    quando duas fontes discordam (sinal de erro em uma delas)."""
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


def season_matches(season: int = SEASON, use_network: bool = True) -> pd.DataFrame:
    """Os 380 jogos com placar (se já jogado), estatísticas e odds."""
    cal = freeze_fixtures(season)
    base = cal.copy()
    base["hg"] = pd.array([pd.NA] * len(base), dtype="Int64")
    base["ag"] = pd.array([pd.NA] * len(base), dtype="Int64")
    base["date_played"] = pd.NaT
    base["source"] = pd.NA
    extra = []   # colunas de estatística/odds para juntar no fim

    if use_network:
        # 1º Cartola: placar oficial + data/horário/estádio atuais + forma
        ca = _try("Cartola", cartola.season)
        if ca is not None:
            base = _fill(base, ca, "Cartola")
            extra.append(ca[KEY + ["datetime", "local", "forma_casa", "forma_fora"]])
        # 2º football-data.co.uk: placar + odds de fechamento
        fd = _try("football-data.co.uk", footballdata_br.load, season)
        if fd is not None:
            base = _fill(base, fd, "football-data.co.uk")
            extra.append(fd[KEY + [c for c in fd if c.startswith("odd_")]])
        # 3º openfootball
        of = _try("openfootball", openfootball.load, season)
        if of is not None:
            base = _fill(base, of, "openfootball")

    # manual: preenche lacunas; com forcar=True sobrescreve (correção humana)
    man = load_manual(season)
    if len(man):
        base = _fill(base, man, "manual")
        for _, r in man[man["forcar"]].iterrows():
            sel = (base["home"] == r.home) & (base["away"] == r.away)
            base.loc[sel, ["hg", "ag", "source"]] = [r.hg, r.ag, "manual (forçado)"]

    for e in extra:
        base = base.merge(e.drop_duplicates(KEY), on=KEY, how="left")

    if use_network:
        # ESPN por último: só estatísticas/odds, nos dias em que houve ou haverá jogo
        when = base["datetime"] if "datetime" in base else base["date_original"]
        days = pd.to_datetime(when).dt.normalize().dropna()
        # dias futuros distantes não têm estatística nem odd: não gasta requisição
        days = days[days <= pd.Timestamp.today().normalize() + pd.Timedelta(days=7)].unique()
        es = _try("ESPN (estatísticas/odds)", espn.load_days, days)
        if es is not None and len(es):
            base = _fill(base, es, "ESPN")
            cols = [c for c in es if c[:2] in ("h_", "a_") or c.startswith("odd_mkt")]
            base = base.merge(es[KEY + cols].drop_duplicates(KEY), on=KEY, how="left")

    base["date_played"] = pd.to_datetime(base["date_played"]).fillna(
        base["date_original"].where(base["hg"].notna()))
    base["status"] = _status(base)
    return base.sort_values(["round", "date_original", "home"]).reset_index(drop=True)


def _status(df: pd.DataFrame) -> pd.Series:
    today = pd.Timestamp.today().normalize()
    when = pd.to_datetime(df["datetime"]).dt.normalize() if "datetime" in df else df["date_original"]
    when = when.fillna(df["date_original"])
    played = df["hg"].notna()
    late = ~played & (when < today - pd.Timedelta(days=1))
    s = pd.Series("a jogar", index=df.index)
    s[played] = "jogado"
    s[late] = "pendente/adiado"
    return s


# --------------------------------------------------------------------------
# Histórico para treinar o modelo
# --------------------------------------------------------------------------
def history(seasons: list[int]) -> pd.DataFrame:
    """Jogos com placar das temporadas anteriores (football-data.co.uk)."""
    fd = _try("football-data.co.uk histórico", footballdata_br.load, seasons)
    if fd is None:
        raise SourceUnavailable("sem histórico: football-data.co.uk não respondeu")
    return fd.dropna(subset=["hg", "ag"]).reset_index(drop=True)


def training_set(current: pd.DataFrame, past_seasons: list[int]) -> pd.DataFrame:
    """Histórico + jogos já disputados da temporada atual, no mesmo formato."""
    hist = history(past_seasons)
    cur = current[current["hg"].notna()].copy()
    cur["date"] = cur["date_played"]
    cols = ["date", "home", "away", "hg", "ag"]
    return pd.concat([hist[cols], cur[cols]], ignore_index=True)
