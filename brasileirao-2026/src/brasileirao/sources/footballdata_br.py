"""football-data.co.uk — Brasil Série A desde 2012 (arquivo único BRA.csv).

Traz placar e odds de FECHAMENTO (Pinnacle, média e máxima do mercado).
Não traz rodada nem chutes/escanteios. Usamos para:
  * histórico de treino (2024 e 2025 completos);
  * 2ª fonte de placar da temporada atual;
  * comparar o modelo com o mercado no backtest (Pinnacle é a referência
    de odd "afiada": se o modelo não chega perto, desconfie dele).
"""
from __future__ import annotations

import io

import pandas as pd

from ..config import team
from .base import get_text

URL = "https://www.football-data.co.uk/new/BRA.csv"


def load(seasons: int | list[int]) -> pd.DataFrame:
    seasons = [seasons] if isinstance(seasons, int) else list(seasons)
    raw = pd.read_csv(io.StringIO(get_text(URL, max_age_hours=12).lstrip("\ufeff")))
    # filtra ANTES de converter nomes: temporadas antigas têm times (Portuguesa,
    # Avaí...) sem alias e gerariam avisos inúteis
    raw = raw[raw["Season"].isin(seasons)]
    df = pd.DataFrame({
        "season": raw["Season"].astype(int),
        "date": pd.to_datetime(raw["Date"], dayfirst=True),
        "home": raw["Home"].map(team), "away": raw["Away"].map(team),
        "hg": pd.to_numeric(raw["HG"], errors="coerce").astype("Int64"),
        "ag": pd.to_numeric(raw["AG"], errors="coerce").astype("Int64"),
        "odd_ps_h": raw["PSCH"], "odd_ps_d": raw["PSCD"], "odd_ps_a": raw["PSCA"],
        "odd_avg_h": raw["AvgCH"], "odd_avg_d": raw["AvgCD"], "odd_avg_a": raw["AvgCA"],
    })
    df["source"] = "football-data.co.uk"
    return df.reset_index(drop=True)
