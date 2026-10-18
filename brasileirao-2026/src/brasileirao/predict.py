"""Previsões jogo a jogo e simulação da tabela final."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ZONAS
from .lineups import multipliers_for_round
from .model import DixonColes


def pending(matches: pd.DataFrame, rodada: int | None = None) -> pd.DataFrame:
    m = matches[matches["hg"].isna()]
    return m[m["round"] == rodada] if rodada is not None else m


def predict_fixtures(model: DixonColes, fixtures: pd.DataFrame,
                     use_lineups: bool = True) -> pd.DataFrame:
    """Uma linha por jogo com probabilidades. Usa a escalação se existir."""
    rows, cache = [], {}
    for _, f in fixtures.iterrows():
        mult = {}
        if use_lineups:
            r = int(f["round"])
            if r not in cache:
                cache[r] = multipliers_for_round(r)
            lu = cache[r]
            if f.home in lu:
                mult.update(home_att_mult=lu[f.home]["att"], home_def_mult=lu[f.home]["def"])
            if f.away in lu:
                mult.update(away_att_mult=lu[f.away]["att"], away_def_mult=lu[f.away]["def"])
        p = model.predict(f.home, f.away, **mult)
        when = f["datetime"] if "datetime" in f and pd.notna(f["datetime"]) else f["date_original"]
        p.update(round=int(f["round"]), date=when, escalacao="sim" if mult else "não")
        rows.append(p)
    df = pd.DataFrame(rows)
    first = ["round", "date", "home", "away", "escalacao"]
    return df[first + [c for c in df.columns if c not in first]]


def simulate_season(model: DixonColes, matches: pd.DataFrame, n_sims: int = 10000,
                    seed: int = 42) -> pd.DataFrame:
    """Monte Carlo: sorteia os jogos que faltam n_sims vezes.

    Resultado: chance de título, G4, G6, Sul-Americana (7º-12º) e Z4, além
    de pontos e posição médios. Jogos já disputados ficam fixos.
    Desempate: pontos, vitórias, saldo (como a CBF; gols pró fica de fora
    porque muda pouco e deixaria a simulação mais lenta).
    """
    rng = np.random.default_rng(seed)
    teams = sorted(set(matches.home) | set(matches.away))
    ix = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    pts0, w0, gd0 = np.zeros(n), np.zeros(n), np.zeros(n)
    for _, r in matches[matches["hg"].notna()].iterrows():
        h, a, hg, ag = ix[r.home], ix[r.away], int(r.hg), int(r.ag)
        gd0[h] += hg - ag; gd0[a] += ag - hg
        pts0[h] += 3 if hg > ag else (1 if hg == ag else 0)
        pts0[a] += 3 if ag > hg else (1 if hg == ag else 0)
        w0[h] += hg > ag; w0[a] += ag > hg

    todo = matches[matches["hg"].isna()]
    hi = todo["home"].map(ix).to_numpy(); ai = todo["away"].map(ix).to_numpy()
    lams = np.array([model.expected_goals(h, a) for h, a in zip(todo.home, todo.away)]).reshape(-1, 2)
    hg = rng.poisson(lams[:, 0], size=(n_sims, len(todo)))
    ag = rng.poisson(lams[:, 1], size=(n_sims, len(todo)))
    pts = np.tile(pts0, (n_sims, 1)); gd = np.tile(gd0, (n_sims, 1)); w = np.tile(w0, (n_sims, 1))
    hp = np.where(hg > ag, 3, np.where(hg == ag, 1, 0))
    ap = np.where(ag > hg, 3, np.where(hg == ag, 1, 0))
    for k in range(len(todo)):
        pts[:, hi[k]] += hp[:, k]; pts[:, ai[k]] += ap[:, k]
        w[:, hi[k]] += hp[:, k] == 3; w[:, ai[k]] += ap[:, k] == 3
        gd[:, hi[k]] += hg[:, k] - ag[:, k]; gd[:, ai[k]] += ag[:, k] - hg[:, k]
    score = pts * 1e6 + w * 1e3 + gd + rng.random(pts.shape) * 0.01
    pos = (-score).argsort(axis=1).argsort(axis=1) + 1
    z = ZONAS
    out = pd.DataFrame({
        "pontos_medios": pts.mean(0), "posicao_media": pos.mean(0),
        "p_campeao": (pos == z["titulo"]).mean(0),
        "p_g4": (pos <= z["libertadores_direta"]).mean(0),
        "p_g6": (pos <= z["libertadores"]).mean(0),
        "p_sul_americana": ((pos > z["libertadores"]) & (pos <= z["sul_americana"])).mean(0),
        "p_rebaixado": (pos >= z["rebaixamento"]).mean(0),
    }, index=teams)
    return out.sort_values("pontos_medios", ascending=False).round(3)
