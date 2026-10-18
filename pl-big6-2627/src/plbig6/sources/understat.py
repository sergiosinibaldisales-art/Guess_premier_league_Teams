"""Understat — xG (gols esperados) por jogo, gratuito.

xG mede a qualidade das chances criadas. É um sinal melhor de força que
o placar em amostras pequenas (6-8 rodadas), porque placar tem muito acaso.
O site mudou de formato algumas vezes; tentamos o endpoint JSON atual e,
se falhar, o formato antigo (JSON embutido no HTML). Se os dois falharem,
levantamos SourceUnavailable e o modelo segue só com gols.
"""
from __future__ import annotations

import codecs
import json
import re

import pandas as pd
import requests

from ..config import team
from .base import HEADERS, SourceUnavailable, get_text

API = "https://understat.com/getLeagueData/EPL/{year}"
PAGE = "https://understat.com/league/EPL/{year}"


def _from_api(year: int) -> list[dict]:
    r = requests.get(API.format(year=year), timeout=20,
                     headers={**HEADERS, "X-Requested-With": "XMLHttpRequest"})
    if r.status_code != 200:
        raise SourceUnavailable(f"understat API HTTP {r.status_code}")
    return r.json()["dates"]


def _from_page(year: int) -> list[dict]:
    html = get_text(PAGE.format(year=year))
    m = re.search(r"datesData\s*=\s*JSON\.parse\('(.+?)'\)", html)
    if not m:
        raise SourceUnavailable("understat: formato da página mudou")
    return json.loads(codecs.decode(m.group(1), "unicode_escape"))


def load(season: str) -> pd.DataFrame:
    year = int(season[:4])
    try:
        matches = _from_api(year)
    except (requests.RequestException, SourceUnavailable, KeyError, ValueError):
        matches = _from_page(year)
    rows = [{
        "date": pd.Timestamp(m["datetime"]).normalize(),
        "home": team(m["h"]["title"]),
        "away": team(m["a"]["title"]),
        "h_xg": float(m["xG"]["h"]),
        "a_xg": float(m["xG"]["a"]),
    } for m in matches if m.get("isResult")]
    return pd.DataFrame(rows)
