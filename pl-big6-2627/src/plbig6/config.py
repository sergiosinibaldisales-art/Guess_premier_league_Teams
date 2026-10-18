"""Configurações centrais: temporada, Big 6, caminhos e nomes de times.

Por que um arquivo só para isso? Cada fonte de dados escreve o nome dos times
de um jeito ("Man City", "Manchester City FC", "Man City"...). Se cada módulo
tratar isso sozinho, um dia um deles erra e um jogo some da base sem aviso.
Centralizar aqui garante um único nome "canônico" por time.
"""
from __future__ import annotations

import re
import unicodedata
import warnings
from pathlib import Path

SEASON = "2026-27"          # temporada alvo (formato openfootball)
SEASON_START = "2026-08-01"

BIG6 = ["Arsenal", "Chelsea", "Liverpool", "Man City", "Man United", "Tottenham"]

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
CACHE = DATA / "cache"            # respostas brutas das fontes (não versionado)
MANUAL = DATA / "manual"          # resultados digitados/verificados à mão
LINEUPS = DATA / "escalacoes"     # escalações que você preenche no dia do jogo
PROCESSED = DATA / "processed"    # saídas prontas (tabela, previsões...)
FIXTURES_FROZEN = DATA / f"calendario_{SEASON}.csv"  # rodadas ORIGINAIS congeladas

# Nome canônico <- variações vistas nas fontes. Chave sempre em minúsculas,
# sem acento e sem sufixo FC/AFC (ver _clean).
_ALIASES = {
    "arsenal": "Arsenal",
    "aston villa": "Aston Villa",
    "bournemouth": "Bournemouth",
    "brentford": "Brentford",
    "brighton": "Brighton", "brighton & hove albion": "Brighton",
    "brighton and hove albion": "Brighton",
    "burnley": "Burnley",
    "chelsea": "Chelsea",
    "coventry": "Coventry", "coventry city": "Coventry",
    "crystal palace": "Crystal Palace",
    "everton": "Everton",
    "fulham": "Fulham",
    "hull": "Hull", "hull city": "Hull",
    "ipswich": "Ipswich", "ipswich town": "Ipswich",
    "leeds": "Leeds", "leeds united": "Leeds",
    "leicester": "Leicester", "leicester city": "Leicester",
    "liverpool": "Liverpool",
    "luton": "Luton", "luton town": "Luton",
    "man city": "Man City", "manchester city": "Man City",
    "man united": "Man United", "man utd": "Man United",
    "manchester united": "Man United", "manchester utd": "Man United",
    "newcastle": "Newcastle", "newcastle united": "Newcastle", "newcastle utd": "Newcastle",
    "nott'm forest": "Nott'm Forest", "nottm forest": "Nott'm Forest",
    "nottingham forest": "Nott'm Forest", "nottingham": "Nott'm Forest",
    "sheffield united": "Sheffield United", "sheffield utd": "Sheffield United",
    "southampton": "Southampton",
    "sunderland": "Sunderland",
    "tottenham": "Tottenham", "tottenham hotspur": "Tottenham", "spurs": "Tottenham",
    "west ham": "West Ham", "west ham united": "West Ham",
    "wolves": "Wolves", "wolverhampton": "Wolves", "wolverhampton wanderers": "Wolves",
}

_unknown_seen: set[str] = set()


def _clean(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    s = s.lower().strip()
    s = re.sub(r"^afc\s+|\s+(a?fc)$", "", s)   # "AFC Bournemouth", "Arsenal FC"
    return re.sub(r"\s+", " ", s)


def team(name: str) -> str:
    """Converte qualquer grafia para o nome canônico.

    Times fora da lista (ex.: da Championship) passam com o nome limpo.
    Se o nome NÃO veio no formato "X FC" do openfootball, avisamos uma vez:
    pode ser uma grafia nova de um time da Premier que precisa de alias,
    e sem o alias o jogo viraria um "time fantasma" na base.
    """
    key = _clean(name)
    if key in _ALIASES:
        return _ALIASES[key]
    canon = key.title()
    if canon not in _unknown_seen:
        _unknown_seen.add(canon)
        if not re.search(r"\ba?fc$", name.strip(), re.IGNORECASE):
            warnings.warn(f"Time sem alias conhecido: {name!r} -> {canon!r}", stacklevel=2)
    return canon
