"""Testes rápidos e offline (não dependem de rede).

Rodar:  pytest -q
"""
import numpy as np
import pandas as pd
import pytest

from brasileirao import config, data, lineups, stats
from brasileirao.model import DixonColes
from brasileirao.sources import espn


def test_aliases_cover_all_spellings():
    for raw in ["CA Mineiro", "Atletico-MG", "Atlético-MG", "CAM"]:
        assert config.team(raw) == "Atlético-MG"
    assert config.team("CA Paranaense") == "Athletico-PR"
    assert config.team("Red Bull Bragantino") == "Bragantino"
    assert config.team("SC Corinthians Paulista") == "Corinthians"
    assert config.team("Sao Paulo") == config.team("São Paulo FC") == "São Paulo"
    assert config.team("Chapecoense-SC") == "Chapecoense"


def test_american_odds():
    assert espn._american_to_decimal("+275") == pytest.approx(3.75)
    assert espn._american_to_decimal("-155") == pytest.approx(1 + 100 / 155)
    assert espn._american_to_decimal(None) is None


def _toy_matches(n_rounds=30, seed=1):
    """Liga fictícia com forças conhecidas, para ver se o modelo as recupera."""
    rng = np.random.default_rng(seed)
    teams = ["A", "B", "C", "D"]
    att = {"A": 0.5, "B": 0.1, "C": -0.2, "D": -0.4}
    rows, day = [], pd.Timestamp("2025-02-01")
    for _ in range(n_rounds):
        for h in teams:
            for a in teams:
                if h != a:
                    lam = np.exp(0.3 + att[h] - att[a] * 0.5)
                    mu = np.exp(att[a] - att[h] * 0.5)
                    rows.append({"date": day, "home": h, "away": a,
                                 "hg": rng.poisson(lam), "ag": rng.poisson(mu)})
        day += pd.Timedelta(days=7)
    return pd.DataFrame(rows)


def test_model_recovers_strength_order():
    m = DixonColes(xi=0.0).fit(_toy_matches())
    assert m.ratings().index.tolist() == ["A", "B", "C", "D"]
    assert 0.1 < m.home_adv < 0.5


def test_probabilities_sum_to_one():
    p = DixonColes().fit(_toy_matches(10)).predict("A", "D")
    assert p["p_home"] + p["p_draw"] + p["p_away"] == pytest.approx(1, abs=1e-9)
    assert p["p_home"] > p["p_away"]


def test_postponed_match_keeps_original_round(monkeypatch):
    """Jogo da rodada 4 disputado em setembro continua contando como rodada 4."""
    cal = pd.DataFrame({"round": [4, 4, 25], "date_original": pd.to_datetime(
        ["2026-02-25", "2026-02-25", "2026-09-01"]),
        "home": ["A", "C", "A"], "away": ["B", "D", "C"]})
    monkeypatch.setattr(data, "freeze_fixtures", lambda season, force=False: cal)
    man = pd.DataFrame({"home": ["A"], "away": ["B"], "hg": pd.array([2], dtype="Int64"),
                        "ag": pd.array([0], dtype="Int64"),
                        "date": [pd.Timestamp("2026-09-02")], "fontes": ["x"], "forcar": [False]})
    monkeypatch.setattr(data, "load_manual", lambda season: man)
    m = data.season_matches(2026, use_network=False)
    row = m[(m.home == "A") & (m.away == "B")].iloc[0]
    assert row["round"] == 4
    assert row["date_played"] == pd.Timestamp("2026-09-02")
    assert m[(m.home == "C")].iloc[0]["status"] == "pendente/adiado"


def test_table_tiebreak_by_wins():
    """Mesmos pontos: quem tem mais vitórias fica na frente (regra da CBF)."""
    df = pd.DataFrame({"round": [1, 2, 1, 2, 3], "date_played": pd.to_datetime(["2026-02-01"] * 5),
                       "home": ["A", "A", "B", "B", "B"], "away": ["X", "Y", "Z", "W", "V"],
                       "hg": [1, 0, 0, 0, 0], "ag": [0, 5, 0, 0, 0]})
    t = stats.table(df)
    assert t.loc["A", "Pts"] == t.loc["B", "Pts"] == 3
    assert t.loc["A", "Pos"] < t.loc["B", "Pos"]   # A: 1 vitória; B: 0 (saldo melhor não importa)


def test_lineup_without_star_lowers_attack():
    names = [("Goleiro", "GOL", 0, 0, 10), ("Zag1", "ZAG", 1, 0, 8), ("Zag2", "ZAG", 0, 0, 8),
             ("Lat1", "LAT", 0, 2, 8), ("Lat2", "LAT", 0, 1, 8), ("Vol", "MEI", 1, 1, 0),
             ("Meia", "MEI", 3, 5, 0), ("Meia2", "MEI", 2, 2, 0), ("Craque", "ATA", 12, 4, 0),
             ("Ponta", "ATA", 4, 3, 0), ("Ponta2", "ATA", 3, 2, 0), ("Reserva", "ATA", 0, 0, 0)]
    rows = [{"id": i, "team": "Flamengo", "apelido": n, "nome": f"X {n}", "pos": pos,
             "status": "Provável", "jogos": 30 - i, "G": g, "A": a, "SG": sg}
            for i, (n, pos, g, a, sg) in enumerate(names)]
    players = lineups.per_game(pd.DataFrame(rows))
    usual = players.nlargest(11, "jogos")["apelido"].tolist()
    a_full, d_full, _ = lineups.team_multipliers("Flamengo", usual, players)
    no_star = [p for p in usual if p != "Craque"] + ["Reserva"]
    a_weak, _, miss = lineups.team_multipliers("Flamengo", no_star, players)
    assert a_full == pytest.approx(1.0) and d_full == pytest.approx(1.0)
    assert a_weak < 0.9 and not miss
