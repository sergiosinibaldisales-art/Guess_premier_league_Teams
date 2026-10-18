"""Atualização semanal (terça-feira, depois da última partida da rodada).

Uso:  python -m plbig6.atualizar
Gera em data/processed/:
  partidas_2026-27.csv   todos os jogos (placar, rodada original, fonte, stats)
  tabela.csv             classificação atual
  previsoes_big6.csv     todos os jogos restantes dos Big 6 até a rodada 38
  simulacao.csv          probabilidade de título/top 4 por time
  forcas.csv             ataque/defesa estimados de cada time
"""
from __future__ import annotations

import argparse

from . import data, predict, stats
from .config import PROCESSED, SEASON
from .model import DixonColes

PAST = ["2023-24", "2024-25", "2025-26"]


def main(use_network: bool = True, n_sims: int = 10000) -> dict:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    matches = data.season_matches(SEASON, use_network=use_network)
    train = data.training_set(matches, PAST)
    model = DixonColes().fit(train, as_of=train["date"].max())

    tab = stats.table(matches)
    prev = predict.predict_fixtures(model, predict.big6_fixtures(matches))
    sim = predict.simulate_season(model, matches, n_sims=n_sims)

    matches.to_csv(PROCESSED / f"partidas_{SEASON}.csv", index=False)
    tab.to_csv(PROCESSED / "tabela.csv")
    prev.round({c: 4 for c in prev.select_dtypes("number")}).to_csv(PROCESSED / "previsoes_big6.csv", index=False)
    sim.to_csv(PROCESSED / "simulacao.csv")
    model.ratings().round(3).to_csv(PROCESSED / "forcas.csv")

    played = matches["hg"].notna().sum()
    pend = matches[matches["status"] == "pendente/adiado"]
    print(f"\n{played} jogos com placar; última data: {matches['date_played'].max():%d/%m/%Y}")
    print("Fontes usadas:", matches["source"].value_counts().to_dict())
    if len(pend):
        print("Jogos sem placar com data já passada (adiados ou fonte atrasada):")
        print(pend[["round", "date_original", "home", "away"]].to_string(index=False))
    return {"matches": matches, "model": model, "tabela": tab,
            "previsoes": prev, "simulacao": sim}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true",
                    help="não baixa nada; usa só calendário + resultados manuais")
    ap.add_argument("--sims", type=int, default=10000)
    a = ap.parse_args()
    main(use_network=not a.offline, n_sims=a.sims)
