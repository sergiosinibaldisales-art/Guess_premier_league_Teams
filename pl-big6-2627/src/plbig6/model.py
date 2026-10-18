"""Modelo Dixon-Coles (1997) com decaimento no tempo.

Como funciona, em poucas linhas:
  * Gols de cada time num jogo seguem uma distribuição de Poisson.
  * Gols esperados do mandante  = exp(casa + ataque_mandante + defesa_visitante)
    Gols esperados do visitante = exp(ataque_visitante + defesa_mandante)
    (defesa negativa = boa defesa, porque reduz os gols do adversário).
  * Dixon-Coles adiciona um ajuste `rho` para placares baixos (0-0, 1-0,
    0-1, 1-1), que o Poisson puro erra sistematicamente.
  * Jogos antigos pesam menos: peso = exp(-xi * dias_atrás). Com xi=0.0019
    um jogo de um ano atrás pesa ~50% de um jogo de hoje.
  * Uma penalidade L2 pequena puxa times com poucos jogos para a média,
    em vez de deixar 2 ou 3 resultados definirem a força deles.

Os parâmetros são achados maximizando a verossimilhança (o conjunto de
forças que torna os placares observados os "mais prováveis").
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

MAX_GOALS = 10


def _tau(x, y, lam, mu, rho):
    """Fator de correção de Dixon-Coles para placares baixos (vetorizado)."""
    t = np.ones_like(lam)
    t = np.where((x == 0) & (y == 0), 1 - lam * mu * rho, t)
    t = np.where((x == 0) & (y == 1), 1 + lam * rho, t)
    t = np.where((x == 1) & (y == 0), 1 + mu * rho, t)
    t = np.where((x == 1) & (y == 1), 1 - rho, t)
    return t


@dataclass
class DixonColes:
    xi: float = 0.0019          # decaimento por dia
    l2: float = 0.05            # força da penalidade (regularização)
    teams: list[str] = field(default_factory=list)
    attack: dict[str, float] = field(default_factory=dict)
    defence: dict[str, float] = field(default_factory=dict)
    home_adv: float = 0.0
    rho: float = 0.0
    fitted_at: pd.Timestamp | None = None

    # ------------------------------------------------------------------ fit
    def fit(self, matches: pd.DataFrame, as_of: pd.Timestamp | None = None) -> "DixonColes":
        """Treina com jogos (colunas date, home, away, hg, ag) até `as_of`."""
        as_of = pd.Timestamp(as_of) if as_of is not None else matches["date"].max()
        df = matches[matches["date"] <= as_of].dropna(subset=["hg", "ag"])
        self.teams = sorted(set(df["home"]) | set(df["away"]))
        idx = {t: i for i, t in enumerate(self.teams)}
        n = len(self.teams)
        hi = df["home"].map(idx).to_numpy()
        ai = df["away"].map(idx).to_numpy()
        x = df["hg"].to_numpy(dtype=float)
        y = df["ag"].to_numpy(dtype=float)
        days = (as_of - pd.to_datetime(df["date"])).dt.days.to_numpy()
        w = np.exp(-self.xi * days)

        def nll(p):
            att, dfn = p[:n], p[n:2 * n]
            home, rho = p[2 * n], p[2 * n + 1]
            lam = np.exp(home + att[hi] + dfn[ai])
            mu = np.exp(att[ai] + dfn[hi])
            tau = np.clip(_tau(x, y, lam, mu, rho), 1e-10, None)
            ll = (np.log(tau) + poisson.logpmf(x, lam) + poisson.logpmf(y, mu))
            # sum(att)=0 fixa a escala (senão ataque+k e defesa-k dão o mesmo)
            pen = self.l2 * (att @ att + dfn @ dfn) + 10 * att.sum() ** 2
            return -(w * ll).sum() + pen

        p0 = np.concatenate([np.zeros(2 * n), [0.25, -0.05]])
        bounds = [(-3, 3)] * (2 * n) + [(-1, 1), (-0.3, 0.3)]
        res = minimize(nll, p0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 2000})
        if not res.success:
            print(f"[aviso] otimização não convergiu: {res.message}")
        p = res.x
        self.attack = dict(zip(self.teams, p[:n]))
        self.defence = dict(zip(self.teams, p[n:2 * n]))
        self.home_adv, self.rho = float(p[2 * n]), float(p[2 * n + 1])
        self.fitted_at = as_of
        return self

    # -------------------------------------------------------------- predict
    def expected_goals(self, home: str, away: str,
                       home_att_mult: float = 1.0, away_att_mult: float = 1.0,
                       home_def_mult: float = 1.0, away_def_mult: float = 1.0):
        """Gols esperados (lambda, mu). Os multiplicadores vêm da escalação.

        def_mult > 1 significa defesa PIOR (deixa o adversário marcar mais).
        Time desconhecido (sem jogos) entra como média (0), e avisamos.
        """
        for t in (home, away):
            if t not in self.attack:
                print(f"[aviso] {t} sem histórico no modelo: tratado como time médio")
        lam = np.exp(self.home_adv + self.attack.get(home, 0) + self.defence.get(away, 0))
        mu = np.exp(self.attack.get(away, 0) + self.defence.get(home, 0))
        return lam * home_att_mult * away_def_mult, mu * away_att_mult * home_def_mult

    def score_matrix(self, lam: float, mu: float) -> np.ndarray:
        g = np.arange(MAX_GOALS + 1)
        m = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu))
        X, Y = np.meshgrid(g, g, indexing="ij")
        m *= _tau(X, Y, np.full_like(m, lam), np.full_like(m, mu), self.rho)
        return m / m.sum()

    def predict(self, home: str, away: str, **mults) -> dict:
        lam, mu = self.expected_goals(home, away, **mults)
        m = self.score_matrix(lam, mu)
        ph, pd_, pa = np.tril(m, -1).sum(), np.trace(m), np.triu(m, 1).sum()
        i, j = np.unravel_index(m.argmax(), m.shape)
        tot = np.add.outer(np.arange(MAX_GOALS + 1), np.arange(MAX_GOALS + 1))
        return {
            "home": home, "away": away,
            "xg_home": round(lam, 2), "xg_away": round(mu, 2),
            "p_home": ph, "p_draw": pd_, "p_away": pa,
            "placar_mais_provavel": f"{i}-{j}", "p_placar": m[i, j],
            "p_over_2_5": m[tot > 2.5].sum(),
            "p_btts": m[1:, 1:].sum(),
            "p_clean_sheet_home": m[:, 0].sum(),
            "p_clean_sheet_away": m[0, :].sum(),
            "odd_justa_home": 1 / ph, "odd_justa_draw": 1 / pd_, "odd_justa_away": 1 / pa,
        }

    def ratings(self) -> pd.DataFrame:
        df = pd.DataFrame({"ataque": self.attack, "defesa": self.defence})
        # Força líquida: quanto o time marca a mais do que sofre contra um médio
        df["forca"] = df["ataque"] - df["defesa"]
        return df.sort_values("forca", ascending=False)
