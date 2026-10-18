"""Backtest: o modelo acerta mais que um palpite ingênuo?

Para cada rodada de uma temporada passada, treinamos SÓ com jogos anteriores
àquela rodada e prevemos a rodada. Medimos:
  * log-loss  (menor = melhor; pune muito errar com confiança)
  * Brier     (menor = melhor; erro quadrático das probabilidades 1X2)
  * acerto    (% em que o resultado mais provável aconteceu)
Comparamos com a base "frequência histórica" (casa ~45%, empate ~25%,
fora ~30%). Se o modelo não bater essa base, ele não serve.

Uso:  python -m plbig6.backtest 2025-26
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from . import data
from .model import DixonColes


def _prev(season: str, k: int) -> list[str]:
    y = int(season[:4])
    return [f"{y - i}-{str(y - i + 1)[-2:]}" for i in range(k, 0, -1)]


def run(season: str = "2025-26", n_past: int = 2, include_championship: bool = False,
        start_round: int = 4, big6_only: bool = False, **model_kw) -> dict:
    from .config import BIG6
    from .sources import openfootball

    target = openfootball.load(season).dropna(subset=["hg", "ag"])
    hist = data.history(_prev(season, n_past), include_championship)
    cols = ["date", "home", "away", "hg", "ag"]
    rows = []
    for rnd in range(start_round, 39):
        test = target[target["round"] == rnd]
        if big6_only:
            test = test[test.home.isin(BIG6) | test.away.isin(BIG6)]
        if test.empty:
            continue
        cutoff = test["date"].min() - pd.Timedelta(days=1)
        train = pd.concat([hist[cols], target[target["date"] <= cutoff][cols]])
        m = DixonColes(**model_kw).fit(train, as_of=cutoff)
        for _, g in test.iterrows():
            p = m.predict(g.home, g.away)
            res = 0 if g.hg > g.ag else (1 if g.hg == g.ag else 2)
            rows.append({"round": rnd, "res": res,
                         "p": [p["p_home"], p["p_draw"], p["p_away"]]})
    P = np.array([r["p"] for r in rows]); y = np.array([r["res"] for r in rows])
    Y = np.eye(3)[y]
    base = Y.mean(0)                         # frequência observada (otimista p/ a base)
    def metrics(Q):
        return {"log_loss": float(-np.log(np.clip((Q * Y).sum(1), 1e-12, 1)).mean()),
                "brier": float(((Q - Y) ** 2).sum(1).mean()),
                "acerto": float((Q.argmax(1) == y).mean())}
    return {"jogos": len(y), "modelo": metrics(P),
            "base_frequencia": metrics(np.tile(base, (len(y), 1)))}


if __name__ == "__main__":
    s = sys.argv[1] if len(sys.argv) > 1 else "2025-26"
    for champ in (True, False):
        r = run(s, include_championship=champ)
        print(f"\n{s} | Championship no treino: {champ} | {r['jogos']} jogos")
        for k in ("modelo", "base_frequencia"):
            print(f"  {k:16s}", {m: round(v, 4) for m, v in r[k].items()})
