#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_enrichers.py — utilitario para probar enriquecedores externos (inicio: Lusha)

Uso básico (Windows PowerShell):
  python test_enrichers.py lusha --domain example.com

Variables de entorno (en .env o entorno):
  LUSHA_API_KEY  -> API key de Lusha
  LUSHA_API_BASE -> Opcional. Base URL (por defecto intenta https://api.lusha.com)

Acciones:
  - Consulta company/enrich por dominio
  - Muestra status, créditos y campos principales
  - Cachea respuestas por 24h en cache/lusha_<domain>.json
"""

import os
import json
import time
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import requests
from dotenv import load_dotenv


CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)


def _cache_path(name: str) -> Path:
    h = hashlib.md5(name.encode("utf-8")).hexdigest()
    return CACHE_DIR / f"{h}.json"


def cache_get(name: str, max_age: timedelta = timedelta(hours=24)) -> Optional[Dict[str, Any]]:
    p = _cache_path(name)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text("utf-8"))
    except Exception:
        return None
    ts = data.get("_ts")
    try:
        if ts and datetime.fromisoformat(ts) + max_age > datetime.now():
            return data.get("content")
    except Exception:
        pass
    return None


def cache_set(name: str, content: Dict[str, Any]) -> None:
    p = _cache_path(name)
    p.write_text(json.dumps({"_ts": datetime.now().isoformat(), "content": content}, ensure_ascii=False, indent=2), "utf-8")


def lusha_company_by_domain(domain: str, *, force: bool = False, base_override: Optional[str] = None, verbose: bool = False) -> Dict[str, Any]:
    """Prueba la API de Lusha para enriquecer compañía por dominio.

    Retorna un dict con { ok: bool, status_code, data, error }
    """
    load_dotenv()
    api_key = os.getenv("LUSHA_API_KEY")
    base = base_override or os.getenv("LUSHA_API_BASE") or "https://api.lusha.com"
    if not api_key:
        return {"ok": False, "error": "Falta LUSHA_API_KEY en entorno/.env"}

    cache_key = f"lusha:company:{domain}"
    if not force:
        cached = cache_get(cache_key)
        if cached is not None:
            return {"ok": True, "status_code": 200, "data": cached, "cached": True}

    # Nota: Lusha documenta endpoints mediante MCP y docs; la ruta exacta puede variar según plan.
    # Intentamos endpoints comunes; si falla, reportamos respuesta.
    # Bases posibles (algunas integraciones usan .com y otras .co)
    bases = []
    for cand in [base, "https://api.lusha.com", "https://api.lusha.co"]:
        if cand and cand not in bases:
            bases.append(cand)

    # Auth styles: Bearer o X-API-Key
    auth_headers_variants = [
        {"Authorization": f"Bearer {api_key}"},
        {"X-API-Key": api_key},
        {"x-api-key": api_key},
        {"X-Lusha-Api-Key": api_key},
        {"X-Lusha-API-Key": api_key},
        {"apikey": api_key}
    ]

    # Endpoints candidatos (POST JSON preferido, GET fallback con params)
    paths = [
        "/v1/companies/enrich",
        "/companies/enrich",
        "/v1/companies/search",
        "/companies/search",
        "/v1/companies/lookup",
        "/companies/lookup",
    ]

    last_err = None
    for b in bases:
        for headers_auth in auth_headers_variants:
            session = requests.Session()
            session.headers.update({
                **headers_auth,
                "Accept": "application/json",
                "User-Agent": "enrich-test/1.0"
            })
            for path in paths:
                url = f"{b.rstrip('/')}{path}"
                try:
                    if verbose:
                        print(f"→ Trying {url} with headers {list(headers_auth.keys())}")
                    payload = {"domain": domain}
                    r = session.post(url, json=payload, timeout=(10, 20))
                    if (r.status_code == 404 or r.status_code == 405) and ("/search" in path or "/enrich" in path):
                        # intentar GET con params como fallback
                        r = session.get(url, params={"domain": domain}, timeout=(10, 20))
                    if verbose:
                        print(f"  ← HTTP {r.status_code}")
                    if r.status_code == 401:
                        last_err = "No autorizado (API key inválida o plan sin acceso)"
                        continue
                    if r.status_code == 429:
                        return {"ok": False, "status_code": 429, "error": "Rate limited (429)"}
                    if r.status_code >= 400:
                        last_err = f"HTTP {r.status_code}: {r.text[:400]}"
                        continue
                    data = r.json()
                    cache_set(cache_key, data)
                    return {"ok": True, "status_code": r.status_code, "data": data}
                except requests.RequestException as e:
                    last_err = str(e)
                except Exception as e:
                    last_err = f"{type(e).__name__}: {e}"

    return {"ok": False, "error": last_err or "No se pudo resolver endpoint Lusha"}


def contactout_company_by_domain(domain: str, *, force: bool = False, base_override: Optional[str] = None, verbose: bool = False) -> Dict[str, Any]:
    """Prueba ContactOut: company by domain (si el plan lo permite).
    Variables: CONTACTOUT_API_KEY, CONTACTOUT_API_BASE (opcional)
    """
    load_dotenv()
    api_key = os.getenv("CONTACTOUT_API_KEY")
    base = base_override or os.getenv("CONTACTOUT_API_BASE") or "https://api.contactout.com"
    if not api_key:
        return {"ok": False, "error": "Falta CONTACTOUT_API_KEY en entorno/.env"}

    cache_key = f"contactout:company:{domain}"
    if not force:
        cached = cache_get(cache_key)
        if cached is not None:
            return {"ok": True, "status_code": 200, "data": cached, "cached": True}

    bases = []
    for cand in [base, "https://api.contactout.com"]:
        if cand and cand not in bases:
            bases.append(cand)
    auth_headers_variants = [
        {"token": api_key},  # official docs
        {"Authorization": f"Bearer {api_key}"},
        {"X-API-Key": api_key},
        {"x-api-key": api_key},
        {"apikey": api_key}
    ]
    paths = [
        "/v1/domain/enrich",  # official
        "/v1/companies/enrich",
        "/companies/enrich",
        "/v1/companies/search",
        "/companies/search",
    ]
    last_err = None
    for b in bases:
        for headers_auth in auth_headers_variants:
            session = requests.Session()
            session.headers.update({**headers_auth, "Accept": "application/json", "User-Agent": "enrich-test/1.0"})
            for path in paths:
                url = f"{b.rstrip('/')}{path}"
                try:
                    if verbose:
                        print(f"→ [ContactOut] {url} with {list(headers_auth.keys())}")
                    payload = {"domain": domain}
                    r = session.post(url, json=payload, timeout=(10, 20))
                    if (r.status_code in (404, 405)):
                        r = session.get(url, params={"domain": domain}, timeout=(10, 20))
                    if verbose:
                        print(f"  ← HTTP {r.status_code}")
                    if r.status_code == 401:
                        last_err = "No autorizado (API key inválida o plan sin acceso)"
                        continue
                    if r.status_code == 429:
                        return {"ok": False, "status_code": 429, "error": "Rate limited (429)"}
                    if r.status_code >= 400:
                        last_err = f"HTTP {r.status_code}: {r.text[:400]}"
                        continue
                    data = r.json()
                    cache_set(cache_key, data)
                    return {"ok": True, "status_code": r.status_code, "data": data}
                except requests.RequestException as e:
                    last_err = str(e)
                except Exception as e:
                    last_err = f"{type(e).__name__}: {e}"
    return {"ok": False, "error": last_err or "No se pudo resolver endpoint ContactOut"}


def rocketreach_company_by_domain(domain: str, *, force: bool = False, base_override: Optional[str] = None, verbose: bool = False) -> Dict[str, Any]:
    """Prueba RocketReach: company by domain.
    Variables: ROCKETREACH_API_KEY, ROCKETREACH_API_BASE (opcional)
    """
    load_dotenv()
    api_key = os.getenv("ROCKETREACH_API_KEY")
    base = base_override or os.getenv("ROCKETREACH_API_BASE") or "https://api.rocketreach.co"
    if not api_key:
        return {"ok": False, "error": "Falta ROCKETREACH_API_KEY en entorno/.env"}

    cache_key = f"rocketreach:company:{domain}"
    if not force:
        cached = cache_get(cache_key)
        if cached is not None:
            return {"ok": True, "status_code": 200, "data": cached, "cached": True}

    bases = []
    for cand in [base, "https://api.rocketreach.co", "https://api.rocketreach.com"]:
        if cand and cand not in bases:
            bases.append(cand)
    auth_headers_variants = [
        {"Authorization": f"Bearer {api_key}"},
        {"X-API-Key": api_key},
        {"x-api-key": api_key},
        {"apikey": api_key}
    ]
    # Endpoints típicos de RocketReach para company lookup
    paths = [
        "/v2/api/search/companies/enrich",
        "/v1/api/search/companies/enrich",
        "/v2/api/search/companies",
        "/v1/api/search/companies",
        "/v2/api/lookup/company",
        "/v1/api/lookup/company",
        "/v2/api/companies/lookup",
        "/v1/api/companies/lookup",
    ]
    last_err = None
    for b in bases:
        for headers_auth in auth_headers_variants:
            session = requests.Session()
            session.headers.update({**headers_auth, "Accept": "application/json", "User-Agent": "enrich-test/1.0"})
            for path in paths:
                url = f"{b.rstrip('/')}{path}"
                try:
                    if verbose:
                        print(f"→ [RocketReach] {url} with {list(headers_auth.keys())}")
                    payload = {"domain": domain}
                    r = session.post(url, json=payload, timeout=(10, 20))
                    if (r.status_code in (404, 405)):
                        r = session.get(url, params={"domain": domain}, timeout=(10, 20))
                    if verbose:
                        print(f"  ← HTTP {r.status_code}")
                    if r.status_code == 401:
                        last_err = "No autorizado (API key inválida o plan sin acceso)"
                        continue
                    if r.status_code == 429:
                        return {"ok": False, "status_code": 429, "error": "Rate limited (429)"}
                    if r.status_code >= 400:
                        last_err = f"HTTP {r.status_code}: {r.text[:400]}"
                        continue
                    data = r.json()
                    cache_set(cache_key, data)
                    return {"ok": True, "status_code": r.status_code, "data": data}
                except requests.RequestException as e:
                    last_err = str(e)
                except Exception as e:
                    last_err = f"{type(e).__name__}: {e}"
    return {"ok": False, "error": last_err or "No se pudo resolver endpoint RocketReach"}


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Probar enriquecedores externos (Lusha/ContactOut/RocketReach)")
    ap.add_argument("provider", choices=["lusha", "contactout", "rocketreach"], help="Proveedor a probar")
    ap.add_argument("--domain", required=True, help="Dominio de la empresa a enriquecer (ej: example.com)")
    ap.add_argument("--force", action="store_true", help="Ignorar cache y forzar llamada")
    ap.add_argument("--base", dest="base", help="Override base URL de la API (ej: https://api.lusha.com)")
    ap.add_argument("--verbose", action="store_true", help="Logs verbosos de intentos")
    args = ap.parse_args()

    if args.provider == "lusha":
        res = lusha_company_by_domain(args.domain, force=bool(args.force), base_override=args.base, verbose=bool(args.verbose))
    elif args.provider == "contactout":
        res = contactout_company_by_domain(args.domain, force=bool(args.force), base_override=args.base, verbose=bool(args.verbose))
    elif args.provider == "rocketreach":
        res = rocketreach_company_by_domain(args.domain, force=bool(args.force), base_override=args.base, verbose=bool(args.verbose))
    # Salida resumida y amigable
    print(f"\n=== {args.provider.capitalize()}: Company by Domain ===")
    print(f"Domain: {args.domain}")
    if res.get("ok"):
        if res.get("cached"):
            print("(from cache)")
        print(f"Status: {res.get('status_code')}")
        data = res.get("data") or {}
        # Mostrar algunos campos comunes si existen
        name = data.get("name") or data.get("company_name") or data.get("company")
        website = data.get("website") or data.get("domain")
        size = data.get("employee_count") or data.get("size")
        industry = data.get("industry") or data.get("industries")
        print(f"Company: {name}")
        print(f"Website: {website}")
        print(f"Size: {size}")
        print(f"Industry: {industry}")
        # Persistir raw para inspección
        out_dir = Path("out/enrich_tests"); out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{args.provider}_company_{args.domain.replace('.', '_')}.json"
        out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")
        print(f"Raw guardado en: {out_path}")
    else:
        print("ERROR en llamada:")
        print(f"  {res.get('error')}")


if __name__ == "__main__":
    main()
