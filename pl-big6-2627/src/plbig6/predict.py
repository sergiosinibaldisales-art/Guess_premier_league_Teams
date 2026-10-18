"""Previsões jogo a jogo dos Big 6 e simulação da tabela final."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import BIG6
from .lineups import multipliers_for_round
from .model import DixonColes


def big6_fixtures(matches: pd.DataFrame, only_pending: bool = True) -> pd.DataFrame:
    m = matches[matches.home.isin(BIG6) | matches.away.isin(BIG6)]
    return m[m["hg"].isna()] if only_pending else m


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
        p.update(round=int(f["round"]), date=f["date_original"],
                 escalacao="sim" if mult else "não")
        rows.append(p)
    df = pd.DataFrame(rows)
    first = ["round", "date", "home", "away", "escalacao"]
    return df[first + [c for c in df.columns if c not in first]]


def simulate_season(model: DixonColes, matches: pd.DataFrame, n_sims: int = 10000,
                    seed: int = 42) -> pd.DataFrame:
    """Monte Carlo: sorteia os jogos que faltam n_sims vezes.

    Resultado: probabilidade de cada time terminar campeão, no top 4 (Champions),
    top 5 etc., e a posição/pontos médios. Jogos já disputados ficam fixos.
    """
    rng = np.random.default_rng(seed)
    teams = sorted(set(matches.home) | set(matches.away))
    ix = {t: i for i, t in enumerate(teams)}
    pts0 = np.zeros(len(teams)); gd0 = np.zeros(len(teams))
    done = matches[matches["hg"].notna()]
    for _, r in done.iterrows():
        h, a, hg, ag = ix[r.home], ix[r.away], int(r.hg), int(r.ag)
        gd0[h] += hg - ag; gd0[a] += ag - hg
        pts0[h] += 3 if hg > ag else (1 if hg == ag else 0)
        pts0[a] += 3 if ag > hg else (1 if hg == ag else 0)

    todo = matches[matches["hg"].isna()]
    hi = todo["home"].map(ix).to_numpy(); ai = todo["away"].map(ix).to_numpy()
    lams = np.array([model.expected_goals(h, a) for h, a in zip(todo.home, todo.away)])
    # Poisson simples na simulação (o ajuste rho muda pouco a tabela final)
    hg = rng.poisson(lams[:, 0], size=(n_sims, len(todo)))
    ag = rng.poisson(lams[:, 1], size=(n_sims, len(todo)))
    pts = np.tile(pts0, (n_sims, 1)); gd = np.tile(gd0, (n_sims, 1))
    hp = np.where(hg > ag, 3, np.where(hg == ag, 1, 0))
    ap = np.where(ag > hg, 3, np.where(hg == ag, 1, 0))
    for k in range(len(todo)):
        pts[:, hi[k]] += hp[:, k]; pts[:, ai[k]] += ap[:, k]
        gd[:, hi[k]] += hg[:, k] - ag[:, k]; gd[:, ai[k]] += ag[:, k] - hg[:, k]
    # desempate por saldo (+ ruído minúsculo para empates totais)
    score = pts * 1000 + gd + rng.random(pts.shape) * 0.01
    pos = (-score).argsort(axis=1).argsort(axis=1) + 1
    out = pd.DataFrame({
        "pontos_medios": pts.mean(0), "posicao_media": pos.mean(0),
        "p_campeao": (pos == 1).mean(0), "p_top4": (pos <= 4).mean(0),
        "p_top5": (pos <= 5).mean(0), "p_rebaixado": (pos >= 18).mean(0),
    }, index=teams)
    return out.sort_values("pontos_medios", ascending=False).round(3)
