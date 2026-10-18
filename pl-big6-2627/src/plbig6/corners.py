"""Modelo de escanteios (antes o projeto só previa gols).

Mesma ideia do modelo de gols, adaptada:
  * Escanteios esperados do mandante  = exp(base + casa + gera[mandante] + cede[visitante])
    Escanteios esperados do visitante = exp(base + gera[visitante] + cede[mandante])
    "gera" = quanto o time força escanteios; "cede" = quanto deixa o rival forçar.
  * Jogos antigos pesam menos (mesmo decaimento do modelo de gols) e uma
    penalidade L2 puxa times com poucos jogos para a média.
  * Escanteio varia MUITO mais que gol (o Man United teve 3 e 13 em jogos
    seguidos). Poisson subestima isso, então as probabilidades usam a binomial
    negativa, com a dispersão medida nos próprios dados.

Treino: football-data.co.uk (colunas HC/AC) das temporadas passadas + a atual.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import nbinom, poisson

from .sources import SourceUnavailable, footballdata_uk

MAX_CORNERS = 30
LINES_TOTAL = (8.5, 9.5, 10.5, 11.5)
LINES_TEAM = (3.5, 4.5, 5.5)


def history(seasons: list[str]) -> pd.DataFrame:
    """Escanteios das temporadas passadas (football-data.co.uk)."""
    frames = []
    for s in seasons:
        try:
            df = footballdata_uk.load(s)
            print(f"[ok]    football-data.co.uk {s} (escanteios): {len(df)} linhas")
            frames.append(df[["date", "home", "away", "h_corners", "a_corners"]])
        except (SourceUnavailable, OSError, ValueError, KeyError) as e:
            print(f"[falha] football-data.co.uk {s} (escanteios): {e}")
    if not frames:
        return pd.DataFrame(columns=["date", "home", "away", "h_corners", "a_corners"])
    return pd.concat(frames, ignore_index=True)


def training_set(current: pd.DataFrame, past_seasons: list[str]) -> pd.DataFrame:
    hist = history(past_seasons)
    cur = current[current["hg"].notna()].copy()
    cur["date"] = cur["date_played"]
    cols = ["date", "home", "away", "h_corners", "a_corners"]
    if "h_corners" not in cur:
        return hist.dropna(subset=cols[3:])
    return pd.concat([hist[cols], cur[cols]], ignore_index=True).dropna(subset=cols[3:])


@dataclass
class CornersModel:
    xi: float = 0.0019
    l2: float = 8.0            # backtest 24-25/25-26: 8 > 2 > 0.5
    teams: list[str] = field(default_factory=list)
    gen: dict[str, float] = field(default_factory=dict)
    conc: dict[str, float] = field(default_factory=dict)
    base: float = 1.6
    home_adv: float = 0.0
    k: float = np.inf            # dispersão da binomial negativa (inf = Poisson)

    def fit(self, df: pd.DataFrame, as_of: pd.Timestamp | None = None) -> "CornersModel":
        as_of = pd.Timestamp(as_of) if as_of is not None else df["date"].max()
        df = df[pd.to_datetime(df["date"]) <= as_of].dropna(subset=["h_corners", "a_corners"])
        self.teams = sorted(set(df["home"]) | set(df["away"]))
        idx = {t: i for i, t in enumerate(self.teams)}
        n = len(self.teams)
        hi, ai = df["home"].map(idx).to_numpy(), df["away"].map(idx).to_numpy()
        x, y = df["h_corners"].to_numpy(float), df["a_corners"].to_numpy(float)
        w = np.exp(-self.xi * (as_of - pd.to_datetime(df["date"])).dt.days.to_numpy())

        def nll(p):
            g, c, b, h = p[:n], p[n:2 * n], p[2 * n], p[2 * n + 1]
            lam = np.exp(b + h + g[hi] + c[ai])
            mu = np.exp(b + g[ai] + c[hi])
            ll = poisson.logpmf(x, lam) + poisson.logpmf(y, mu)
            pen = self.l2 * (g @ g + c @ c) + 10 * (g.sum() ** 2 + c.sum() ** 2)
            return -(w * ll).sum() + pen

        p0 = np.concatenate([np.zeros(2 * n), [np.log(max(x.mean(), y.mean(), 1)), 0.1]])
        res = minimize(nll, p0, method="L-BFGS-B",
                       bounds=[(-2, 2)] * (2 * n) + [(0, 3), (-1, 1)])
        p = res.x
        self.gen, self.conc = dict(zip(self.teams, p[:n])), dict(zip(self.teams, p[n:2 * n]))
        self.base, self.home_adv = float(p[2 * n]), float(p[2 * n + 1])

        # Dispersão: var = mu + mu^2/k (método dos momentos nos resíduos)
        lam = np.exp(self.base + self.home_adv + p[:n][hi] + p[n:2 * n][ai])
        mu = np.exp(self.base + p[:n][ai] + p[n:2 * n][hi])
        m = np.concatenate([lam, mu]); o = np.concatenate([x, y])
        excess = ((o - m) ** 2 - m).sum() / (m ** 2).sum()
        self.k = 1 / excess if excess > 1e-6 else np.inf
        return self

    def expected(self, home: str, away: str) -> tuple[float, float]:
        lam = np.exp(self.base + self.home_adv + self.gen.get(home, 0) + self.conc.get(away, 0))
        mu = np.exp(self.base + self.gen.get(away, 0) + self.conc.get(home, 0))
        return float(lam), float(mu)

    def _pmf(self, m: float) -> np.ndarray:
        g = np.arange(MAX_CORNERS + 1)
        if np.isinf(self.k):
            return poisson.pmf(g, m)
        return nbinom.pmf(g, self.k, self.k / (self.k + m))

    def predict(self, home: str, away: str) -> dict:
        lam, mu = self.expected(home, away)
        ph, pa = self._pmf(lam), self._pmf(mu)
        tot = np.convolve(ph, pa)[: 2 * MAX_CORNERS + 1]
        g = np.arange(len(tot))
        joint = np.outer(ph, pa)
        out = {"esc_home": round(lam, 2), "esc_away": round(mu, 2),
               "esc_total": round(lam + mu, 2),
               "esc_faixa_80": f"{int(g[np.searchsorted(tot.cumsum(), 0.1)])}-"
                               f"{int(g[np.searchsorted(tot.cumsum(), 0.9)])}"}
        for line in LINES_TOTAL:
            out[f"p_esc_over_{line}".replace(".", "_")] = float(tot[g > line].sum())
        for side, pmf in (("home", ph), ("away", pa)):
            for line in LINES_TEAM:
                out[f"p_esc_{side}_over_{line}".replace(".", "_")] = float(
                    pmf[np.arange(len(pmf)) > line].sum())
        out["p_mais_esc_home"] = float(np.tril(joint, -1).sum())
        out["p_mais_esc_away"] = float(np.triu(joint, 1).sum())
        return out

    def ratings(self) -> pd.DataFrame:
        df = pd.DataFrame({"gera": self.gen, "cede": self.conc})
        df["esc_pro_vs_medio"] = np.exp(self.base + df["gera"]).round(2)
        df["esc_contra_vs_medio"] = np.exp(self.base + df["cede"]).round(2)
        return df.sort_values("gera", ascending=False)


def backtest(season: str = "2025-26", past: list[str] | None = None,
             start_round: int = 4) -> dict:
    """Walk-forward por mês: treina com o que havia antes e prevê o mês seguinte.

    Compara o erro (MAE) do total de escanteios com o palpite ingênuo
    "média da liga".
    """
    y = int(season[:4])
    past = past or [f"{y - i}-{str(y - i + 1)[-2:]}" for i in (2, 1)]
    hist = history(past)
    target = footballdata_uk.load(season).dropna(subset=["h_corners", "a_corners"])
    target = target.sort_values("date")
    start = target["date"].iloc[start_round * 10]
    rows = []
    for month_start in pd.date_range(start.normalize(), target["date"].max(), freq="MS"):
        test = target[(target["date"] >= month_start) &
                      (target["date"] < month_start + pd.offsets.MonthBegin(1))]
        if test.empty:
            continue
        train = pd.concat([hist, target[target["date"] < month_start]])
        cutoff = month_start - pd.Timedelta(days=1)
        m = CornersModel().fit(train, as_of=cutoff)
        avg = (train["h_corners"] + train["a_corners"]).mean()
        for _, g in test.iterrows():
            p = m.predict(g.home, g.away)
            real = g.h_corners + g.a_corners
            rows.append({"real": real, "prev": p["esc_total"], "media": avg,
                         "h": g.h_corners, "a": g.a_corners, "ph": p["esc_home"],
                         "pa": p["esc_away"], "hm": train["h_corners"].mean(),
                         "am": train["a_corners"].mean(),
                         "p_over_9_5": p["p_esc_over_9_5"]})
    r = pd.DataFrame(rows)
    over = (r["real"] > 9.5).astype(float)
    team_err = ((r.h - r.ph).abs().mean() + (r.a - r.pa).abs().mean()) / 2
    team_base = ((r.h - r.hm).abs().mean() + (r.a - r.am).abs().mean()) / 2
    dec = r[r.h != r.a]
    return {"jogos": len(r),
            "mae_por_time_modelo": round(float(team_err), 3),
            "mae_por_time_media_liga": round(float(team_base), 3),
            "acerto_quem_tem_mais": round(float(((dec.ph > dec.pa) == (dec.h > dec.a)).mean()), 3),
            "mae_modelo": round(float((r.real - r.prev).abs().mean()), 3),
            "mae_media_liga": round(float((r.real - r.media).abs().mean()), 3),
            "brier_over_9_5_modelo": round(float(((r.p_over_9_5 - over) ** 2).mean()), 4),
            "brier_over_9_5_base": round(float(((over.mean() - over) ** 2).mean()), 4)}


if __name__ == "__main__":
    import sys
    print(backtest(sys.argv[1] if len(sys.argv) > 1 else "2025-26"))
