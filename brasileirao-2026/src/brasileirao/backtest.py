"""Backtest: o modelo acerta mais que um palpite ingênuo? E perto do mercado?

Para cada rodada, treinamos SÓ com jogos anteriores àquela rodada e
prevemos a rodada. Medimos:
  * log-loss  (menor = melhor; pune muito errar com confiança)
  * Brier     (menor = melhor; erro quadrático das probabilidades 1X2)
  * acerto    (% em que o resultado mais provável aconteceu)
Duas comparações:
  * base "frequência": chutar sempre a % casa/empate/fora da temporada
    (é otimista para a base, porque usa a frequência do próprio ano);
  * mercado: odd de fechamento sem margem (football-data.co.uk; Pinnacle
    quando existe, senão média das casas). É o padrão-ouro gratuito. Se o
    modelo ficar longe, não aposte com ele.

Uso:  python -m brasileirao.backtest 2025
      python -m brasileirao.backtest 2026
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from . import data
from .model import DixonColes
from .sources import cartola, footballdata_br, openfootball


def _target(season: int) -> pd.DataFrame:
    # rodada: Cartola para o ano atual, openfootball para 2025
    t = cartola.season() if season == 2026 else openfootball.load(season)
    t = t.dropna(subset=["hg", "ag"])[["round", "date", "home", "away", "hg", "ag"]]
    odds = footballdata_br.load(season)
    # Pinnacle quando existe; em 2026 o arquivo não traz Pinnacle, então
    # cai para a MÉDIA de fechamento das casas (um pouco menos afiada)
    for c in "hda":
        odds[f"odd_{c}"] = odds[f"odd_ps_{c}"].fillna(odds[f"odd_avg_{c}"])
    odds = odds[["home", "away", "odd_h", "odd_d", "odd_a"]]
    return t.merge(odds.drop_duplicates(["home", "away"]), on=["home", "away"], how="left")


def run(season: int = 2025, n_past: int = 2, start_round: int = 4, **model_kw) -> dict:
    target = _target(season)
    hist = data.history(list(range(season - n_past, season)))
    cols = ["date", "home", "away", "hg", "ag"]
    rows = []
    for rnd in range(start_round, 39):
        test = target[target["round"] == rnd]
        if test.empty:
            continue
        # corte pela 1ª data da rodada: jogo adiado de rodada antiga, se jogado
        # depois, não entra no treino desta rodada (evita "ver o futuro")
        cutoff = test["date"].min() - pd.Timedelta(days=1)
        train = pd.concat([hist[cols], target[target["date"] <= cutoff][cols]])
        m = DixonColes(**model_kw).fit(train, as_of=cutoff)
        for _, g in test.iterrows():
            p = m.predict(g.home, g.away)
            res = 0 if g.hg > g.ag else (1 if g.hg == g.ag else 2)
            q = np.array([1 / g.odd_h, 1 / g.odd_d, 1 / g.odd_a])
            rows.append({"res": res, "p": [p["p_home"], p["p_draw"], p["p_away"]],
                         "mkt": q / q.sum()})
    y = np.array([r["res"] for r in rows]); Y = np.eye(3)[y]
    P = np.array([r["p"] for r in rows]); M = np.array([r["mkt"] for r in rows])
    ok = ~np.isnan(M).any(1)

    def metrics(Q, mask=None):
        mask = np.ones(len(y), bool) if mask is None else mask
        Qm, Ym, ym = Q[mask], Y[mask], y[mask]
        return {"log_loss": float(-np.log(np.clip((Qm * Ym).sum(1), 1e-12, 1)).mean()),
                "brier": float(((Qm - Ym) ** 2).sum(1).mean()),
                "acerto": float((Qm.argmax(1) == ym).mean())}
    return {"jogos": len(y), "jogos_com_odd": int(ok.sum()),
            "modelo": metrics(P), "base_frequencia": metrics(np.tile(Y.mean(0), (len(y), 1))),
            "modelo_mesmos_jogos": metrics(P, ok), "mercado": metrics(M, ok)}


if __name__ == "__main__":
    s = int(sys.argv[1]) if len(sys.argv) > 1 else 2025
    r = run(s)
    print(f"\n{s} | {r['jogos']} jogos ({r['jogos_com_odd']} com odd de fechamento)")
    for k in ("modelo", "base_frequencia", "modelo_mesmos_jogos", "mercado"):
        print(f"  {k:20s}", {m: round(v, 4) for m, v in r[k].items()})
