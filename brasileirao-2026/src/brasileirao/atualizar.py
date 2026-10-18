"""Atualização semanal (depois da última partida de cada rodada).

Uso:  python -m brasileirao.atualizar
Gera em data/processed/ (CSV em UTF-8 com BOM, abre certo no Excel):
  partidas_2026.csv   os 380 jogos (placar, rodada original, fonte, stats, odds)
  tabela.csv          classificação atual
  previsoes.csv       todos os jogos restantes até a rodada 38
  simulacao.csv       chance de título / G4 / G6 / Z4 por time
  forcas.csv          ataque/defesa estimados de cada time
"""
from __future__ import annotations

import argparse

from . import data, predict, stats
from .config import PAST_SEASONS, PROCESSED, SEASON
from .model import DixonColes


def _save(df, name, index=False):
    num = {c: 4 for c in df.select_dtypes("number")}
    df.round(num).to_csv(PROCESSED / name, index=index, encoding="utf-8-sig")


def main(use_network: bool = True, n_sims: int = 10000) -> dict:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    matches = data.season_matches(SEASON, use_network=use_network)
    train = data.training_set(matches, PAST_SEASONS)
    model = DixonColes().fit(train, as_of=train["date"].max())

    tab = stats.table(matches)
    prev = predict.predict_fixtures(model, predict.pending(matches))
    sim = predict.simulate_season(model, matches, n_sims=n_sims)

    _save(matches, f"partidas_{SEASON}.csv")
    _save(tab, "tabela.csv", index=True)
    _save(prev, "previsoes.csv")
    _save(sim, "simulacao.csv", index=True)
    teams = sorted(set(matches.home))
    _save(model.ratings().loc[teams].sort_values("forca", ascending=False), "forcas.csv", index=True)

    played = matches["hg"].notna().sum()
    pend = matches[matches["status"] == "pendente/adiado"]
    print(f"\n{played} jogos com placar; último: {matches['date_played'].max():%d/%m/%Y}")
    print("Fontes usadas:", matches["source"].value_counts().to_dict())
    if len(pend):
        print("Jogos sem placar com data já passada (adiados ou fonte atrasada):")
        print(pend[["round", "date_original", "home", "away"]].to_string(index=False))
    return {"matches": matches, "model": model, "tabela": tab, "train": train,
            "previsoes": prev, "simulacao": sim}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true",
                    help="não baixa nada; usa só calendário + resultados manuais")
    ap.add_argument("--sims", type=int, default=10000)
    a = ap.parse_args()
    main(use_network=not a.offline, n_sims=a.sims)
