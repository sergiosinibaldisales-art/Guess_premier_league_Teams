"""Testes rápidos e offline (não dependem de rede).

Rodar:  pytest -q
"""
import numpy as np
import pandas as pd
import pytest

from plbig6 import config, data, stats
from plbig6.model import DixonColes


def test_aliases_cover_all_spellings():
    for raw in ["Manchester City FC", "Man City", "Manchester City"]:
        assert config.team(raw) == "Man City"
    assert config.team("AFC Bournemouth") == "Bournemouth"
    assert config.team("Spurs") == "Tottenham"
    assert config.team("Nottingham Forest FC") == "Nott'm Forest"
    assert config.team("Brighton & Hove Albion FC") == "Brighton"


def _toy_matches(n_rounds=30, seed=1):
    """Liga fictícia com forças conhecidas, para ver se o modelo as recupera."""
    rng = np.random.default_rng(seed)
    teams = ["A", "B", "C", "D"]
    att = {"A": 0.5, "B": 0.1, "C": -0.2, "D": -0.4}
    rows, day = [], pd.Timestamp("2025-08-01")
    for r in range(n_rounds):
        for h in teams:
            for a in teams:
                if h != a:
                    lam = np.exp(0.25 + att[h] - att[a] * 0.5)
                    mu = np.exp(att[a] - att[h] * 0.5)
                    rows.append({"date": day, "home": h, "away": a,
                                 "hg": rng.poisson(lam), "ag": rng.poisson(mu)})
        day += pd.Timedelta(days=7)
    return pd.DataFrame(rows)


def test_model_recovers_strength_order():
    m = DixonColes(xi=0.0).fit(_toy_matches())
    order = m.ratings().index.tolist()
    assert order == ["A", "B", "C", "D"]
    assert 0.1 < m.home_adv < 0.4


def test_probabilities_sum_to_one():
    m = DixonColes().fit(_toy_matches(10))
    p = m.predict("A", "D")
    assert p["p_home"] + p["p_draw"] + p["p_away"] == pytest.approx(1, abs=1e-9)
    assert p["p_home"] > p["p_away"]


def test_lineup_multiplier_changes_expected_goals():
    m = DixonColes().fit(_toy_matches(10))
    base = m.predict("A", "B")["xg_home"]
    weaker = m.predict("A", "B", home_att_mult=0.8)["xg_home"]
    assert weaker == pytest.approx(base * 0.8, abs=0.01)


def test_postponed_match_keeps_original_round(tmp_path, monkeypatch):
    """Jogo da rodada 1 disputado na rodada 5 continua contando como rodada 1."""
    cal = pd.DataFrame({"round": [1, 1, 5], "date_original": pd.to_datetime(
        ["2026-08-20", "2026-08-20", "2026-09-20"]),
        "home": ["A", "C", "A"], "away": ["B", "D", "C"]})
    monkeypatch.setattr(data, "freeze_fixtures", lambda season, force=False: cal)
    man = pd.DataFrame({"home": ["A"], "away": ["B"], "hg": pd.array([2], dtype="Int64"),
                        "ag": pd.array([0], dtype="Int64"),
                        "date": [pd.Timestamp("2026-09-24")], "fontes": ["x"], "forcar": [False]})
    monkeypatch.setattr(data, "load_manual", lambda season: man)
    m = data.season_matches("test", use_network=False)
    row = m[(m.home == "A") & (m.away == "B")].iloc[0]
    assert row["round"] == 1
    assert row["date_played"] == pd.Timestamp("2026-09-24")
    assert m[(m.home == "C")].iloc[0]["status"] == "pendente/adiado"


def test_table_points():
    df = pd.DataFrame({"round": [1, 1], "date_played": pd.to_datetime(["2026-08-20"] * 2),
                       "home": ["A", "C"], "away": ["B", "D"],
                       "hg": [2, 1], "ag": [0, 1]})
    t = stats.table(df)
    assert t.loc["A", "Pts"] == 3 and t.loc["B", "Pts"] == 0 and t.loc["C", "Pts"] == 1


def test_lineup_without_star_lowers_attack():
    from plbig6 import lineups
    rows = [{"id": i, "team": "Arsenal", "pos": pos, "web_name": n, "name": f"X {n}",
             "minutes": 900 - i, "expected_goal_involvements": xgi,
             "expected_goals_conceded": 10.0}
            for i, (n, pos, xgi) in enumerate(
                [("Raya", "GK", 0), ("Saliba", "DEF", 0.5), ("Gabriel", "DEF", 1),
                 ("White", "DEF", 0.5), ("Timber", "DEF", 0.5), ("Rice", "MID", 2),
                 ("Odegaard", "MID", 3), ("Merino", "MID", 1), ("Saka", "MID", 6),
                 ("Martinelli", "MID", 3), ("Gyokeres", "FWD", 6), ("Nwaneri", "MID", 0.3)])]
    players = lineups._per90(pd.DataFrame(rows))
    usual = players.nlargest(11, "minutes")["web_name"].tolist()
    a_full, _, _ = lineups.team_multipliers("Arsenal", usual, players)
    no_saka = [p for p in usual if p != "Saka"] + ["Nwaneri"]
    a_weak, _, miss = lineups.team_multipliers("Arsenal", no_saka, players)
    assert a_full == pytest.approx(1.0)
    assert a_weak < 0.9 and not miss
