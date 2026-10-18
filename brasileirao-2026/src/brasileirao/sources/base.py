"""Download com cache e tratamento de limite de requisições.

Ideia central: nenhuma fonte gratuita é confiável 100% do tempo (limite de
requisições, site fora do ar, bloqueio de rede). Então toda fonte:
  1. tenta baixar;
  2. se der erro/limite (HTTP 429, 403, timeout), levanta SourceUnavailable;
  3. quem chama (data.py) captura e tenta a PRÓXIMA fonte da lista.
O cache evita bater na mesma URL várias vezes no mesmo dia, o que é o jeito
mais simples de não estourar limite.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import requests

from ..config import CACHE

HEADERS = {"User-Agent": "Mozilla/5.0 (brasileirao-2026 research script)"}


class SourceUnavailable(RuntimeError):
    """A fonte não respondeu, bloqueou ou estourou o limite."""


def _cache_path(url: str) -> Path:
    return CACHE / (hashlib.sha1(url.encode()).hexdigest()[:16] + ".cache")


def get_text(url: str, max_age_hours: float = 6, timeout: float = 20,
             retries: int = 2) -> str:
    """Baixa `url` como texto, reaproveitando o cache se for recente."""
    path = _cache_path(url)
    if path.exists() and (time.time() - path.stat().st_mtime) < max_age_hours * 3600:
        return path.read_text(encoding="utf-8")

    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout)
        except requests.RequestException as e:      # rede bloqueada, DNS, timeout
            last_err = e
        else:
            if r.status_code == 200:
                CACHE.mkdir(parents=True, exist_ok=True)
                path.write_text(r.text, encoding="utf-8")
                return r.text
            last_err = SourceUnavailable(f"HTTP {r.status_code} em {url}")
            if r.status_code not in (429, 500, 502, 503, 504):
                break                                 # 403/404: insistir não ajuda
        time.sleep(2 ** attempt)                      # espera 1s, 2s, 4s...

    # Se tem cache antigo, melhor usar dado velho (e avisar) do que nada.
    if path.exists():
        print(f"[aviso] {url} indisponível ({last_err}); usando cache antigo.")
        return path.read_text(encoding="utf-8")
    raise SourceUnavailable(str(last_err))


def get_json(url: str, **kw) -> dict:
    return json.loads(get_text(url, **kw))
