"""API pública do Cartola FC (Globo) — fonte principal da temporada atual.

Por que ela é a 1ª fonte:
  * traz as 38 rodadas com o número OFICIAL da rodada da CBF. Jogo adiado
    continua na rodada original (ex.: Flamengo x Mirassol da 4ª rodada,
    jogado em setembro, segue como rodada 4) — é a regra que você pediu;
  * placar oficial atualizado logo depois do jogo;
  * dados de jogador (scouts: gols, assistências, finalizações, SG...) e
    status (Provável, Dúvida, Contundido, Suspenso), usados nas escalações.
Contras: é uma API não documentada; pode mudar de formato sem aviso. Por
isso o resto do código trata qualquer erro aqui como "fonte indisponível".
"""
from __future__ import annotations

import pandas as pd

from ..config import CARTOLA_IDS, N_ROUNDS
from .base import SourceUnavailable, get_json

BASE = "https://api.cartola.globo.com"
STATUS_ATLETA = {2: "Dúvida", 3: "Suspenso", 5: "Contundido", 6: "Nulo", 7: "Provável"}
POSICAO = {1: "GOL", 2: "LAT", 3: "ZAG", 4: "MEI", 5: "ATA", 6: "TEC"}


def _club(cid) -> str:
    try:
        return CARTOLA_IDS[int(cid)]
    except KeyError:
        raise SourceUnavailable(f"clube id {cid} sem nome em CARTOLA_IDS") from None


def current_round() -> int:
    return int(get_json(f"{BASE}/mercado/status", max_age_hours=1)["rodada_atual"])


def round_matches(rodada: int, max_age_hours: float = 1) -> pd.DataFrame:
    d = get_json(f"{BASE}/partidas/{rodada}", max_age_hours=max_age_hours)
    rows = []
    for p in d["partidas"]:
        hg, ag = p.get("placar_oficial_mandante"), p.get("placar_oficial_visitante")
        rows.append({
            "round": rodada,
            "datetime": pd.Timestamp(p["partida_data"]),   # horário de Brasília
            "home": _club(p["clube_casa_id"]), "away": _club(p["clube_visitante_id"]),
            "hg": hg if hg is not None else pd.NA,
            "ag": ag if ag is not None else pd.NA,
            "local": p.get("local"),
            "status_tr": p.get("status_transmissao_tr"),
            "forma_casa": "".join(p.get("aproveitamento_mandante") or []).upper(),
            "forma_fora": "".join(p.get("aproveitamento_visitante") or []).upper(),
        })
    df = pd.DataFrame(rows)
    df[["hg", "ag"]] = df[["hg", "ag"]].astype("Int64")
    # Placar de jogo em andamento NÃO conta como resultado final. Rodadas
    # antigas vêm com status vazio ("") e placar oficial: essas valem.
    live = ~df["status_tr"].fillna("").isin(["ENCERRADA", ""])
    df.loc[live, ["hg", "ag"]] = pd.NA
    df["date"] = df["datetime"].dt.normalize()
    df["source"] = "Cartola"
    return df


def season(max_age_hours: float = 6) -> pd.DataFrame:
    """Todas as rodadas (1 a 38). Rodadas antigas mudam pouco: cache mais longo."""
    return pd.concat([round_matches(r, max_age_hours) for r in range(1, N_ROUNDS + 1)],
                     ignore_index=True)


def players() -> pd.DataFrame:
    """Atletas do mercado com scouts acumulados na temporada e status."""
    d = get_json(f"{BASE}/atletas/mercado", max_age_hours=3)
    rows = []
    for a in d["atletas"]:
        cid = int(a["clube_id"])
        if cid not in CARTOLA_IDS:
            continue
        sc = a.get("scout") or {}
        rows.append({
            "id": a["atleta_id"], "team": CARTOLA_IDS[cid], "apelido": a["apelido"],
            "nome": a["nome"], "pos": POSICAO.get(a["posicao_id"], "?"),
            "status": STATUS_ATLETA.get(a["status_id"], str(a["status_id"])),
            "jogos": a.get("jogos_num") or 0, "media_cartola": a.get("media_num"),
            "G": sc.get("G", 0), "A": sc.get("A", 0),
            "finalizacoes": sc.get("FD", 0) + sc.get("FF", 0) + sc.get("FT", 0),
            "SG": sc.get("SG", 0), "GS": sc.get("GS", 0), "DE": sc.get("DE", 0),
            "CA": sc.get("CA", 0), "CV": sc.get("CV", 0),
        })
    return pd.DataFrame(rows)
