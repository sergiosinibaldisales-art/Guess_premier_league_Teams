"""Configurações centrais: temporada, caminhos e nomes de times.

Mesmo motivo do projeto da Premier League: cada fonte escreve o time de um
jeito ("CA Mineiro", "Atletico-MG", "Atlético-MG", "CAM"...). Se o nome não
for unificado, o merge entre fontes falha em silêncio e o jogo some.
Aqui todo nome vira um único nome "canônico".
"""
from __future__ import annotations

import re
import unicodedata
import warnings
from pathlib import Path

SEASON = 2026               # o Brasileirão é ano-calendário (jan a dez)
PAST_SEASONS = [2024, 2025]  # histórico usado no treino do modelo
N_ROUNDS = 38

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
CACHE = DATA / "cache"            # respostas brutas das fontes (não versionado)
MANUAL = DATA / "manual"          # resultados digitados/verificados à mão
LINEUPS = DATA / "escalacoes"     # escalações que você preenche no dia do jogo
PROCESSED = DATA / "processed"    # saídas prontas (tabela, previsões...)
REPORTS = ROOT / "relatorios"     # relatório de cada rodada em Markdown
FIXTURES_FROZEN = DATA / f"calendario_{SEASON}.csv"  # rodadas ORIGINAIS congeladas

# Zonas da tabela (2026: 4 vagas diretas na Libertadores + 2 na pré, Z4 cai).
# As vagas extras dependem de quem ganha a Copa do Brasil/Libertadores,
# então "top 6" é aproximação.
ZONAS = {"titulo": 1, "libertadores_direta": 4, "libertadores": 6,
         "sul_americana": 12, "rebaixamento": 17}

# Cartola identifica os clubes por id; o nome lá vem só como sigla.
CARTOLA_IDS = {
    262: "Flamengo", 263: "Botafogo", 264: "Corinthians", 265: "Bahia",
    266: "Fluminense", 267: "Vasco", 275: "Palmeiras", 276: "São Paulo",
    277: "Santos", 280: "Bragantino", 282: "Atlético-MG", 283: "Cruzeiro",
    284: "Grêmio", 285: "Internacional", 287: "Vitória", 293: "Athletico-PR",
    294: "Coritiba", 315: "Chapecoense", 364: "Remo", 2305: "Mirassol",
}

# Chave: nome limpo (minúsculo, sem acento) -> canônico.
_ALIASES = {
    "athletico-pr": "Athletico-PR", "athletico pr": "Athletico-PR",
    "athletico paranaense": "Athletico-PR", "ca paranaense": "Athletico-PR",
    "atletico-pr": "Athletico-PR", "cap": "Athletico-PR",
    "atletico-mg": "Atlético-MG", "atletico mg": "Atlético-MG", "ca mineiro": "Atlético-MG",
    "atletico mineiro": "Atlético-MG", "cam": "Atlético-MG",
    "atletico-go": "Atlético-GO", "atletico go": "Atlético-GO",
    "atletico goianiense": "Atlético-GO", "atl. goianiense": "Atlético-GO",
    "ac goianiense": "Atlético-GO",
    "bahia": "Bahia", "ec bahia": "Bahia", "bah": "Bahia",
    "botafogo": "Botafogo", "botafogo rj": "Botafogo", "botafogo fr": "Botafogo", "bot": "Botafogo",
    "bragantino": "Bragantino", "rb bragantino": "Bragantino",
    "red bull bragantino": "Bragantino", "rbb": "Bragantino",
    "ceara": "Ceará", "ceara sc": "Ceará",
    "chapecoense": "Chapecoense", "chapecoense-sc": "Chapecoense",
    "chapecoense af": "Chapecoense", "cha": "Chapecoense",
    "corinthians": "Corinthians", "sc corinthians paulista": "Corinthians", "cor": "Corinthians",
    "coritiba": "Coritiba", "coritiba fbc": "Coritiba", "cfc": "Coritiba",
    "criciuma": "Criciúma", "criciuma ec": "Criciúma",
    "cruzeiro": "Cruzeiro", "cruzeiro ec": "Cruzeiro", "cru": "Cruzeiro",
    "cuiaba": "Cuiabá", "cuiaba ec": "Cuiabá",
    "flamengo": "Flamengo", "flamengo rj": "Flamengo", "cr flamengo": "Flamengo", "fla": "Flamengo",
    "fluminense": "Fluminense", "fluminense fc": "Fluminense", "flu": "Fluminense",
    "fortaleza": "Fortaleza", "fortaleza ec": "Fortaleza",
    "goias": "Goiás",
    "gremio": "Grêmio", "gremio fbpa": "Grêmio", "gre": "Grêmio",
    "internacional": "Internacional", "sc internacional": "Internacional", "int": "Internacional",
    "juventude": "Juventude", "ec juventude": "Juventude",
    "mirassol": "Mirassol", "mirassol fc": "Mirassol", "mir": "Mirassol",
    "palmeiras": "Palmeiras", "se palmeiras": "Palmeiras", "pal": "Palmeiras",
    "remo": "Remo", "clube do remo": "Remo", "rem": "Remo",
    "santos": "Santos", "santos fc": "Santos", "san": "Santos",
    "sao paulo": "São Paulo", "sao paulo fc": "São Paulo", "sao": "São Paulo",
    "sport": "Sport", "sport recife": "Sport", "sport club do recife": "Sport",
    "vasco": "Vasco", "vasco da gama": "Vasco", "cr vasco da gama": "Vasco", "vas": "Vasco",
    "vitoria": "Vitória", "ec vitoria": "Vitória", "vit": "Vitória",
}

_unknown_seen: set[str] = set()


def _clean(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", s.lower().strip())


def team(name: str) -> str:
    """Converte qualquer grafia para o nome canônico.

    Nome desconhecido passa como veio e gera UM aviso: é sinal de que uma
    fonte mudou a grafia e falta um alias aqui (senão vira "time fantasma").
    """
    key = _clean(name)
    if key in _ALIASES:
        return _ALIASES[key]
    if name not in _unknown_seen:
        _unknown_seen.add(name)
        warnings.warn(f"Time sem alias conhecido: {name!r}", stacklevel=2)
    return str(name).strip()
