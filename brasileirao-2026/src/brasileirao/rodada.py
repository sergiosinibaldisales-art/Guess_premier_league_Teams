"""Relatório dos jogos que faltam de uma rodada (Markdown + CSV).

Uso:  python -m brasileirao.rodada          # rodada atual do Cartola
      python -m brasileirao.rodada 32
Gera relatorios/rodada_XX.md e data/processed/rodada_XX.csv.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import atualizar, lineups, stats
from .config import PROCESSED, REPORTS
from .sources import SourceUnavailable, cartola

DIAS = ["seg", "ter", "qua", "qui", "sex", "sáb", "dom"]


def _pct(x) -> str:
    return "-" if pd.isna(x) else f"{100 * x:.0f}%"


def _when(ts) -> str:
    ts = pd.Timestamp(ts)
    hm = "" if ts.hour == 0 and ts.minute == 0 else f" {ts:%H:%M}"
    return f"{DIAS[ts.dayofweek]} {ts:%d/%m}{hm}"


def _no_vig(h, d, a):
    """Odds decimais -> probabilidades sem a margem da casa (normalização simples)."""
    if any(pd.isna(v) or v is None for v in (h, d, a)):
        return None
    inv = np.array([1 / h, 1 / d, 1 / a])
    return inv / inv.sum()


def _venue_line(lf: pd.DataFrame, team: str, venue: str) -> str:
    x = lf[(lf.team == team) & (lf.venue == venue)]
    if x.empty:
        return "sem jogos"
    v, e, d = (x.res == "V").sum(), (x.res == "E").sum(), (x.res == "D").sum()
    return (f"{len(x)}J {v}V {e}E {d}D, {x.gf.sum()}-{x.ga.sum()} em gols, "
            f"aproveitamento {100 * x.pts.sum() / (3 * len(x)):.0f}%")


def _avg(lf: pd.DataFrame, team: str, venue: str, col: str) -> str:
    x = lf[(lf.team == team) & (lf.venue == venue)]
    if col not in x or x[col].notna().sum() == 0:
        return "-"
    return f"{x[col].mean():.1f}"


def _top_scores(model, lam, mu, k=3) -> str:
    m = model.score_matrix(lam, mu)
    flat = np.argsort(m, axis=None)[::-1][:k]
    return ", ".join(f"{i}-{j} ({100 * m[i, j]:.0f}%)"
                     for i, j in (np.unravel_index(f, m.shape) for f in flat))


def build(res: dict, rodada: int) -> tuple[str, pd.DataFrame]:
    matches, model, tab, train = res["matches"], res["model"], res["tabela"], res["train"]
    games = matches[(matches["round"] == rodada)].copy()
    todo = games[games["hg"].isna()]
    done = games[games["hg"].notna()]
    lf = stats.long_format(matches)
    try:
        players = cartola.players()
    except SourceUnavailable as e:
        print(f"[aviso] sem dados de jogadores do Cartola: {e}")
        players = None
    lu = lineups.multipliers_for_round(rodada, lineups.per_game(players) if players is not None else None)

    out = [f"# Brasileirão 2026 · Rodada {rodada}",
           f"_Gerado em {pd.Timestamp.now(tz='America/Sao_Paulo'):%d/%m/%Y %H:%M} (Brasília). "
           f"Base: {int(matches['hg'].notna().sum())} jogos com placar._", ""]
    if len(done):
        out += ["## Já disputados nesta rodada", ""]
        out += [f"- {r.home} {int(r.hg)} x {int(r.ag)} {r.away}" for r in done.itertuples()]
        out.append("")
    out += [f"## Faltam {len(todo)} jogos", "",
            "| Jogo | Quando | Casa | Empate | Fora | Placar provável | +2,5 gols | Ambos marcam |",
            "|---|---|---|---|---|---|---|---|"]
    rows, blocks = [], []
    for g in todo.itertuples():
        mult = {}
        if g.home in lu:
            mult.update(home_att_mult=lu[g.home]["att"], home_def_mult=lu[g.home]["def"])
        if g.away in lu:
            mult.update(away_att_mult=lu[g.away]["att"], away_def_mult=lu[g.away]["def"])
        p = model.predict(g.home, g.away, **mult)
        lam, mu = model.expected_goals(g.home, g.away, **mult)
        when = g.datetime if hasattr(g, "datetime") and pd.notna(g.datetime) else g.date_original
        out.append(f"| {g.home} x {g.away} | {_when(when)} | {_pct(p['p_home'])} | "
                   f"{_pct(p['p_draw'])} | {_pct(p['p_away'])} | {p['placar_mais_provavel']} | "
                   f"{_pct(p['p_over_2_5'])} | {_pct(p['p_btts'])} |")
        mk = _no_vig(getattr(g, "odd_mkt_h", None), getattr(g, "odd_mkt_d", None),
                     getattr(g, "odd_mkt_a", None))
        th, ta = tab.loc[g.home], tab.loc[g.away]
        b = [f"### {g.home} x {g.away}",
             f"{_when(when)} · {getattr(g, 'local', '') or 'local não informado'}"
             + (" · **com escalação informada**" if mult else ""), "",
             f"- **Tabela:** {g.home} {th.Pos}º ({th.Pts} pts, {th.V}V {th.E}E {th.D}D, "
             f"saldo {th.SG:+d}, últimos 5: {th.Forma}) · {g.away} {ta.Pos}º ({ta.Pts} pts, "
             f"{ta.V}V {ta.E}E {ta.D}D, saldo {ta.SG:+d}, últimos 5: {ta.Forma})",
             f"- **{g.home} em casa:** {_venue_line(lf, g.home, 'casa')}",
             f"- **{g.away} fora:** {_venue_line(lf, g.away, 'fora')}",
             "", "| Média por jogo (mando do jogo) | " + g.home + " (casa) | " + g.away + " (fora) |",
             "|---|---|---|"]
        for col, lab in (("gf", "Gols marcados"), ("ga", "Gols sofridos"), ("shots", "Finalizações"),
                         ("sot", "Finalizações no alvo"), ("shots_contra", "Finalizações sofridas"),
                         ("corners", "Escanteios"), ("possession", "Posse (%)"), ("fouls", "Faltas")):
            b.append(f"| {lab} | {_avg(lf, g.home, 'casa', col)} | {_avg(lf, g.away, 'fora', col)} |")
        h2h = stats.head_to_head(train, g.home, g.away).tail(5)
        b += ["", "- **Confronto direto (desde 2024):** " + (
            "; ".join(f"{d:%d/%m/%y} {h} {int(x)}-{int(y)} {a}"
                      for d, h, x, y, a in h2h.itertuples(index=False)) or "sem jogos na base")]
        b.append(f"- **Modelo:** gols esperados {lam:.2f} x {mu:.2f} · {g.home} {_pct(p['p_home'])}, "
                 f"empate {_pct(p['p_draw'])}, {g.away} {_pct(p['p_away'])} · placares mais "
                 f"prováveis: {_top_scores(model, lam, mu)} · odd justa "
                 f"{p['odd_justa_home']:.2f} / {p['odd_justa_draw']:.2f} / {p['odd_justa_away']:.2f}")
        if mk is not None:
            b.append(f"- **Mercado (DraftKings via ESPN, sem margem):** {_pct(mk[0])} / "
                     f"{_pct(mk[1])} / {_pct(mk[2])} · diferença modelo − mercado no mandante: "
                     f"{100 * (p['p_home'] - mk[0]):+.0f} p.p.")
        else:
            b.append("- **Mercado:** odds não publicadas pela ESPN para este jogo.")
        if players is not None:
            for t in (g.home, g.away):
                df = lineups.desfalques(players, t)
                if len(df):
                    b.append(f"- **Desfalques/dúvidas {t} (Cartola):** " + "; ".join(
                        f"{r.apelido} ({r.pos}, {r.status}, {r.jogos}J {r.G}G {r.A}A)"
                        for r in df.head(6).itertuples()))
        blocks += b + [""]
        rows.append({**p, "round": rodada, "quando": when,
                     "p_mercado_home": mk[0] if mk is not None else np.nan,
                     "p_mercado_draw": mk[1] if mk is not None else np.nan,
                     "p_mercado_away": mk[2] if mk is not None else np.nan})
    out += ["", "Probabilidades do modelo Dixon-Coles. Leia com ceticismo: veja o backtest no README.", ""]
    out += blocks

    late = matches[matches["status"] == "pendente/adiado"]
    if len(late):
        out += ["## Jogos atrasados (contam na rodada original)", ""]
        out += [f"- Rodada {int(r['round'])}: {r.home} x {r.away}" for _, r in late.iterrows()]
        out.append("")
    sim = res["simulacao"]
    out += ["## Classificação atual e simulação do fim do campeonato", "",
            "| Pos | Time | Pts | J | V | SG | Últimos 5 | Título | G4 | G6 | Z4 |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for t, r in tab.iterrows():
        s = sim.loc[t]
        out.append(f"| {r.Pos} | {t} | {r.Pts} | {r.J} | {r.V} | {r.SG:+d} | {r.Forma} | "
                   f"{_pct(s.p_campeao)} | {_pct(s.p_g4)} | {_pct(s.p_g6)} | {_pct(s.p_rebaixado)} |")
    out += ["", "G6 é aproximado: as vagas extras na Libertadores dependem da Copa do Brasil."]
    return "\n".join(out) + "\n", pd.DataFrame(rows)


def main(rodada: int | None = None, use_network: bool = True) -> str:
    res = atualizar.main(use_network=use_network)
    if rodada is None:
        try:
            rodada = cartola.current_round()
        except SourceUnavailable:
            rodada = int(res["matches"].loc[res["matches"]["hg"].isna(), "round"].min())
    md, df = build(res, rodada)
    REPORTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / f"rodada_{rodada:02d}.md"
    path.write_text(md, encoding="utf-8")
    df.round({c: 4 for c in df.select_dtypes("number")}).to_csv(
        PROCESSED / f"rodada_{rodada:02d}.csv", index=False, encoding="utf-8-sig")
    print(f"\nRelatório: {path}")
    return md


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("rodada", nargs="?", type=int)
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args()
    main(a.rodada, use_network=not a.offline)
