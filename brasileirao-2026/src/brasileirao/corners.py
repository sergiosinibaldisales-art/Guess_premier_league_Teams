"""Previsão de escanteios por jogo.

Ideia (a mesma lógica de ataque x defesa do modelo de gols, só que mais simples):
  * Média da liga: o mandante tem em média H escanteios e o visitante A
    (em 2026: ~6,0 x ~4,4; jogar em casa pesa muito).
  * Cada time tem dois índices, medidos como "quanto acima da média":
      pro    = escanteios a favor / média da liga naquele mando
      contra = escanteios cedidos / média da liga naquele mando
    Usamos os jogos em casa e fora juntos (dividindo cada um pela média do
    seu mando), porque separar por mando deixa só ~15 jogos por time.
  * Escanteios esperados do mandante = H x pro(mandante) x contra(visitante),
    e o mesmo para o visitante com A.
  * Encolhimento: com poucos jogos, o índice é puxado para 1 (média):
    peso = n / (n + K). Sem isso, 2 jogos atípicos definiriam o time.

Distribuição: escanteios variam MAIS que um Poisson (na liga, variância do
total ≈ 1,25 x a média). Poisson daria confiança demais nas linhas de
+9,5 etc. Por isso usamos a binomial negativa, com a dispersão medida nos
próprios jogos da temporada (uma para cada lado e outra para o total).

Só usa jogos de 2026: a ESPN só foi consultada para esta temporada.

Backtest (18/10/2026, rodadas 6 a 31, 257 jogos, cada rodada prevista só com
as anteriores) — resultado que manda ler isto com muito ceticismo:
  * TOTAL de escanteios: o modelo NÃO bate a base "frequência da liga até
    ali". +9,5: log-loss 0.699 (modelo) x 0.694 (base); erro médio do total
    2,94 x 2,91. Testamos K = 3 a 50: quanto mais encolhe para a média,
    melhor, ou seja, o "jeito do time" quase não ajuda no total.
  * QUEM TEM MAIS escanteios: o modelo ajuda um pouco. Log-loss 0.881 x
    0.900; acerto 61,5% x 61,1% (a base já acerta muito porque o mandante
    tem mais escanteios na maioria dos jogos).
Por isso o relatório mostra a frequência da liga ao lado das linhas de total.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import nbinom

K = 20                # jogos "fictícios" na média para encolher o índice (ver backtest)
LINES = (8.5, 9.5, 10.5, 11.5)
MAX_C = 40


def _nb_pmf(mean: float, ratio: float) -> np.ndarray:
    """pmf da binomial negativa com média `mean` e variância ratio*mean."""
    k = np.arange(MAX_C + 1)
    if ratio <= 1.0001:                      # sem excesso de variância: Poisson
        from scipy.stats import poisson
        return poisson.pmf(k, mean)
    p = 1 / ratio                            # var = mean / p
    n = mean * p / (1 - p)
    return nbinom.pmf(k, n, p)


class CornersModel:
    def fit(self, matches: pd.DataFrame, as_of: pd.Timestamp | None = None) -> "CornersModel":
        df = matches.dropna(subset=["h_corners", "a_corners"])
        if as_of is not None:
            df = df[pd.to_datetime(df["date_played"]) <= as_of]
        self.H, self.A = df["h_corners"].mean(), df["a_corners"].mean()
        self.disp_h = df["h_corners"].var() / self.H
        self.disp_a = df["a_corners"].var() / self.A
        t = df["h_corners"] + df["a_corners"]
        self.disp_t = t.var() / t.mean()
        rows = pd.concat([
            pd.DataFrame({"team": df.home, "pro": df.h_corners / self.H, "contra": df.a_corners / self.A}),
            pd.DataFrame({"team": df.away, "pro": df.a_corners / self.A, "contra": df.h_corners / self.H}),
        ])
        g = rows.groupby("team").agg(n=("pro", "size"), pro=("pro", "mean"), contra=("contra", "mean"))
        w = g["n"] / (g["n"] + K)
        self.pro = (w * g["pro"] + (1 - w)).to_dict()
        self.contra = (w * g["contra"] + (1 - w)).to_dict()
        self.n_games = len(df)
        self._totals = (df["h_corners"] + df["a_corners"]).to_numpy()
        return self

    def expected(self, home: str, away: str) -> tuple[float, float]:
        eh = self.H * self.pro.get(home, 1) * self.contra.get(away, 1)
        ea = self.A * self.pro.get(away, 1) * self.contra.get(home, 1)
        return eh, ea

    def predict(self, home: str, away: str) -> dict:
        eh, ea = self.expected(home, away)
        ph, pa = _nb_pmf(eh, self.disp_h), _nb_pmf(ea, self.disp_a)
        # O total NÃO é a soma de dois lados independentes: quando um time
        # pressiona, o outro tem menos escanteios (correlação negativa). Somar
        # os lados daria variância ~17 contra ~13 real e linhas achatadas.
        # Então o total tem a própria dispersão, medida nos totais da liga.
        tot = _nb_pmf(eh + ea, self.disp_t)
        k = np.arange(len(tot))
        diff = np.subtract.outer(np.arange(len(ph)), np.arange(len(pa)))
        joint = np.outer(ph, pa)
        out = {"esc_casa": round(float(eh), 1), "esc_fora": round(float(ea), 1),
               "esc_total": round(float(eh + ea), 1),
               "p_mais_esc_casa": joint[diff > 0].sum(), "p_mais_esc_fora": joint[diff < 0].sum()}
        for L in LINES:
            out[f"p_esc_over_{L}"] = float(tot[k > L].sum())
        out["p_mais_esc_casa"] = float(out["p_mais_esc_casa"])
        out["p_mais_esc_fora"] = float(out["p_mais_esc_fora"])
        return out

    def league_rates(self) -> dict:
        """Frequência real de cada linha nos jogos usados no ajuste (a 'base')."""
        return {L: float((self._totals > L).mean()) for L in LINES}


def backtest(matches: pd.DataFrame, start_round: int = 6) -> dict:
    """Prevê cada rodada só com as anteriores. Compara com a base 'frequência
    até ali' (taxa de +9,5 nos jogos anteriores e média do total)."""
    df = matches.dropna(subset=["h_corners", "a_corners"]).copy()
    df["date_played"] = pd.to_datetime(df["date_played"])
    rows = []
    for rnd in range(start_round, int(df["round"].max()) + 1):
        test = df[df["round"] == rnd]
        if test.empty:
            continue
        cutoff = test["date_played"].min() - pd.Timedelta(days=1)
        past = df[df["date_played"] <= cutoff]
        if len(past) < 40:
            continue
        m = CornersModel().fit(past)
        base_rate = ((past.h_corners + past.a_corners) > 9.5).mean()
        base_mean = (past.h_corners + past.a_corners).mean()
        for g in test.itertuples():
            p = m.predict(g.home, g.away)
            tot = g.h_corners + g.a_corners
            rows.append({"y": tot > 9.5, "p": p["p_esc_over_9.5"], "b": base_rate,
                         "err_m": abs(p["esc_total"] - tot), "err_b": abs(base_mean - tot)})
    r = pd.DataFrame(rows)

    def ll(p):
        p = p.clip(1e-6, 1 - 1e-6)
        return float(-(r.y * np.log(p) + (~r.y) * np.log(1 - p)).mean())
    return {"jogos": len(r), "logloss_over95_modelo": ll(r.p), "logloss_over95_base": ll(r.b),
            "erro_medio_total_modelo": float(r.err_m.mean()), "erro_medio_total_base": float(r.err_b.mean())}


if __name__ == "__main__":
    from . import data
    print(backtest(data.season_matches()))
