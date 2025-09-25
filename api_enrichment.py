import logging
import os
import re
from datetime import datetime
from pathlib import Path
import pickle
import time
from typing import Dict, Any, Optional, Tuple

import requests
from dotenv import load_dotenv

def setup_api_logging():
    """Configura los loggers para cada API"""
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True)
    
    # Configurar loggers
    loggers = {}
    for api in ["lusha", "contactout", "rocketreach"]:
        logger = logging.getLogger(api)
        logger.setLevel(logging.DEBUG)
        
        # Verificar si ya tiene handlers para evitar duplicados
        if not logger.handlers:
            # File handler
            fh = logging.FileHandler(logs_dir / f"{api}_log.log", encoding='utf-8')
            fh.setLevel(logging.DEBUG)
            
            # Formatter
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            
            # Add handler
            logger.addHandler(fh)
        
        loggers[api] = logger
    
    return loggers


def _now_iso() -> str:
    return datetime.now().isoformat()


def _normalize_domain(value: str) -> str:
    """Normaliza un dominio: quita esquema, www., paths y lower-case.
    Acepta entradas como 'https://www.example.com/page' o 'example.com/'.
    """
    if not value:
        return value
    v = value.strip().lower()
    # remove scheme
    v = re.sub(r"^https?://", "", v)
    # split by slash and take first
    v = v.split("/")[0]
    # strip port if any
    v = v.split(":")[0]
    # drop common www
    if v.startswith("www."):
        v = v[4:]
    return v


def _http_json(
    method: str,
    url: str,
    logger: logging.Logger,
    *,
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Any]] = None,
    json_body: Optional[Dict[str, Any]] = None,
    timeout: int = 20,
    retries: int = 1,
    backoff: float = 0.5,
) -> Tuple[int, Optional[Dict[str, Any]], Optional[str]]:
    """Realiza una solicitud HTTP y devuelve (status_code, json, text) con reintentos simples."""
    for attempt in range(retries + 1):
        try:
            resp = requests.request(
                method.upper(),
                url,
                headers=headers or {},
                params=params,
                json=json_body,
                timeout=timeout,
            )
            ct = resp.headers.get("Content-Type", "")
            text = resp.text
            data = None
            if "application/json" in ct:
                try:
                    data = resp.json()
                except Exception:
                    data = None
            logger.debug(
                f"HTTP {method} {url} -> {resp.status_code}; params={params} body={json_body}; headers={headers}; resp_ct={ct}"
            )
            return resp.status_code, data, text
        except requests.RequestException as e:
            logger.warning(f"HTTP error {method} {url} attempt {attempt+1}/{retries+1}: {e}")
            if attempt < retries:
                time.sleep(backoff * (2 ** attempt))
            else:
                return 0, None, str(e)


def save_api_cache(api_name: str, domain: str, data: Dict):
    """Guarda respuesta de API en cache"""
    cache_file = Path("cache") / f"{api_name}_{domain}.pkl"
    with open(cache_file, "wb") as f:
        pickle.dump({"data": data, "timestamp": datetime.now().isoformat()}, f)

def load_api_cache(api_name: str, domain: str) -> Optional[Dict]:
    """Carga respuesta de API desde cache"""
    cache_file = Path("cache") / f"{api_name}_{domain}.pkl"
    if cache_file.exists():
        try:
            with open(cache_file, "rb") as f:
                cached = pickle.load(f)
                # Cache válido por 24 horas
                cache_time = datetime.fromisoformat(cached["timestamp"])
                if (datetime.now() - cache_time).total_seconds() < 86400:
                    return cached["data"]
        except Exception:
            pass
    return None

def query_lusha(domain: str, logger: logging.Logger) -> Dict[str, Any]:
    """Consulta la API de Lusha"""
    logger.info(f"Consultando Lusha para dominio: {domain}")
    
    # Intentar cargar desde cache
    norm_domain = _normalize_domain(domain)
    cached_data = load_api_cache("lusha", norm_domain)
    if cached_data:
        logger.info(f"Usando datos en cache para {norm_domain}")
        return cached_data
    
    try:
        load_dotenv()
        api_key = os.getenv("LUSHA_API_KEY") or os.getenv("LUSHA_TOKEN")
        if not api_key:
            logger.warning("LUSHA_API_KEY no configurado en .env; omitiendo llamada real")
            response = {"status": "skipped", "reason": "missing_api_key", "last_check": _now_iso()}
            save_api_cache("lusha", norm_domain, response)
            return response

        # Intento mínimo basado en docs: GET https://api.lusha.com/v2/company?domain=example.com
        headers = {
            "api_key": api_key,
            "Accept": "application/json",
        }
        status, data, text = _http_json(
            "GET",
            "https://api.lusha.com/v2/company",
            logger,
            headers=headers,
            params={"domain": norm_domain},
            timeout=25,
            retries=1,
        )

        if status == 200 and isinstance(data, dict) and data:
            # Tratar de mapear algunos campos comunes si existen
            company = {
                "name": data.get("name"),
                "domain": data.get("domain") or norm_domain,
                "website": data.get("website"),
                "linkedin_url": data.get("url") or data.get("linkedin_url"),
                "industry": data.get("industry"),
                "size": data.get("size"),
                "country": data.get("country"),
                "headquarter": data.get("headquarter"),
                "founded_at": data.get("founded_at"),
                "revenue": data.get("revenue"),
                "logo_url": data.get("logo_url"),
                "locations": data.get("locations"),
                "type": data.get("type"),
                "specialties": data.get("specialties"),
            }
            response = {
                "status": "ok",
                "last_check": _now_iso(),
                "company": company,
                "raw": data,
            }
        elif status in (401, 403):
            response = {
                "status": "unauthorized",
                "last_check": _now_iso(),
                "http_status": status,
                "message": "Clave inválida o plan sin acceso",
            }
        elif status == 404:
            response = {"status": "not_found", "last_check": _now_iso(), "http_status": status}
        else:
            response = {
                "status": "not_found",
                "last_check": _now_iso(),
                "http_status": status,
                "raw_text": text[:1000] if text else None,
            }
        
        logger.debug(f"Respuesta de Lusha: {response}")
        save_api_cache("lusha", norm_domain, response)
        return response
        
    except Exception as e:
        logger.error(f"Error consultando Lusha: {str(e)}")
        return {"status": "error", "last_check": _now_iso(), "error": str(e)}

def query_contactout(domain: str, logger: logging.Logger) -> Dict[str, Any]:
    """Consulta la API de ContactOut"""
    logger.info(f"Consultando ContactOut para dominio: {domain}")
    
    # Intentar cargar desde cache
    norm_domain = _normalize_domain(domain)
    cached_data = load_api_cache("contactout", norm_domain)
    if cached_data:
        logger.info(f"Usando datos en cache para {norm_domain}")
        return cached_data
    
    try:
        load_dotenv()
        token = os.getenv("CONTACTOUT_API_KEY") or os.getenv("CONTACTOUT_TOKEN")
        if not token:
            logger.warning("CONTACTOUT_API_KEY no configurado en .env; omitiendo llamada real")
            response = {"status": "skipped", "reason": "missing_api_key", "last_check": _now_iso()}
            save_api_cache("contactout", norm_domain, response)
            return response

        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "token": token,
        }
        payload = {"domains": [norm_domain]}
        status, data, text = _http_json(
            "POST",
            "https://api.contactout.com/v1/domain/enrich",
            logger,
            headers=headers,
            json_body=payload,
            timeout=25,
            retries=1,
        )

        def _first_company(d: Any) -> Optional[Dict[str, Any]]:
            comps = None
            if isinstance(d, dict):
                comps = d.get("companies")
            if isinstance(comps, list) and comps:
                entry = comps[0]
                if isinstance(entry, dict) and entry:
                    # The API returns [{"domain": {...company...}}]
                    # Grab first inner dict
                    inner_vals = list(entry.values())
                    if inner_vals and isinstance(inner_vals[0], dict):
                        return inner_vals[0]
            return None

        if status == 200 and isinstance(data, dict):
            comp = _first_company(data)
            if comp:
                company = {
                    "name": comp.get("name"),
                    "domain": comp.get("domain") or norm_domain,
                    "website": comp.get("website"),
                    "linkedin_url": comp.get("li_vanity") or comp.get("url"),
                    "industry": comp.get("industry"),
                    "size": comp.get("size"),
                    "country": comp.get("country"),
                    "headquarter": comp.get("headquarter"),
                    "founded_at": comp.get("founded_at"),
                    "revenue": comp.get("revenue"),
                    "logo_url": comp.get("logo_url"),
                    "locations": comp.get("locations"),
                    "type": comp.get("type"),
                    "specialties": comp.get("specialties"),
                }
                response = {
                    "status": "ok",
                    "last_check": _now_iso(),
                    "company": company,
                    "raw": data,
                }
            else:
                response = {"status": "not_found", "last_check": _now_iso(), "http_status": status}
        elif status in (401, 403):
            response = {
                "status": "unauthorized",
                "last_check": _now_iso(),
                "http_status": status,
                "message": "Clave inválida, sin créditos o sin acceso",
            }
        else:
            response = {
                "status": "not_found",
                "last_check": _now_iso(),
                "http_status": status,
                "raw_text": text[:1000] if text else None,
            }
        
        logger.debug(f"Respuesta de ContactOut: {response}")
        save_api_cache("contactout", norm_domain, response)
        return response
        
    except Exception as e:
        logger.error(f"Error consultando ContactOut: {str(e)}")
        return {"status": "error", "last_check": _now_iso(), "error": str(e)}

def query_rocketreach(domain: str, logger: logging.Logger) -> Dict[str, Any]:
    """Consulta la API de RocketReach"""
    logger.info(f"Consultando RocketReach para dominio: {domain}")
    
    # Intentar cargar desde cache
    norm_domain = _normalize_domain(domain)
    cached_data = load_api_cache("rocketreach", norm_domain)
    if cached_data:
        logger.info(f"Usando datos en cache para {norm_domain}")
        return cached_data
    
    try:
        load_dotenv()
        api_key = os.getenv("ROCKETREACH_API_KEY") or os.getenv("ROCKETREACH_TOKEN")
        if not api_key:
            logger.warning("ROCKETREACH_API_KEY no configurado en .env; omitiendo llamada real")
            response = {"status": "skipped", "reason": "missing_api_key", "last_check": _now_iso()}
            save_api_cache("rocketreach", norm_domain, response)
            return response

        # Preferir endpoints oficiales v2
        headers = {
            "Accept": "application/json",
            "X-Api-Key": api_key,
        }

        # 1) Intentar Company Lookup (requiere acceso de exports segun docs)
        status, data, text = _http_json(
            "GET",
            "https://api.rocketreach.co/api/v2/company/lookup",
            logger,
            headers=headers,
            params={"domain": norm_domain},
            timeout=25,
            retries=1,
        )

        def _map_rr_company(d: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "name": d.get("name") or d.get("company_name"),
                "domain": d.get("domain") or norm_domain,
                "website": d.get("website") or d.get("url"),
                "linkedin_url": d.get("linkedin_url") or d.get("linkedin"),
                "industry": d.get("industry"),
                "size": d.get("employee_count") or d.get("size"),
                "country": d.get("country"),
                "headquarter": d.get("headquarters") or d.get("hq"),
                "founded_at": d.get("founded") or d.get("year_founded"),
                "revenue": d.get("revenue"),
                "logo_url": d.get("logo"),
                "locations": d.get("locations"),
                "type": d.get("type"),
                "specialties": d.get("specialties"),
            }

        if status == 200 and isinstance(data, dict) and data:
            company = _map_rr_company(data)
            response = {
                "status": "ok",
                "last_check": _now_iso(),
                "company": company,
                "raw": data,
            }
        else:
            # 2) Fallback: Company Search por dominio
            payload = {"domain": norm_domain}
            status2, data2, text2 = _http_json(
                "POST",
                "https://api.rocketreach.co/api/v2/searchCompany",
                logger,
                headers={**headers, "Content-Type": "application/json"},
                json_body=payload,
                timeout=25,
                retries=1,
            )

            def _first_rr_company(d: Any) -> Optional[Dict[str, Any]]:
                if isinstance(d, dict):
                    # readme style often returns list under 'results' or 'companies'
                    results = d.get("results") or d.get("companies") or d.get("data")
                    if isinstance(results, list) and results:
                        return results[0]
                return None

            if status2 == 200 and isinstance(data2, dict):
                comp = _first_rr_company(data2)
                if comp:
                    company = _map_rr_company(comp)
                    response = {
                        "status": "ok",
                        "last_check": _now_iso(),
                        "company": company,
                        "raw": data2,
                    }
                else:
                    response = {"status": "not_found", "last_check": _now_iso(), "http_status": status2}
            elif status in (401, 403) or status2 in (401, 403):
                response = {
                    "status": "unauthorized",
                    "last_check": _now_iso(),
                    "http_status": status if status in (401, 403) else status2,
                    "message": "Clave inválida o plan sin acceso",
                }
            else:
                response = {
                    "status": "not_found",
                    "last_check": _now_iso(),
                    "http_status": status2,
                    "raw_text": (text or text2)[:1000] if (text or text2) else None,
                }
        
        logger.debug(f"Respuesta de RocketReach: {response}")
        save_api_cache("rocketreach", norm_domain, response)
        return response
        
    except Exception as e:
        logger.error(f"Error consultando RocketReach: {str(e)}")
        return {"status": "error", "last_check": _now_iso(), "error": str(e)}

def get_enrichment_sources(domain: str) -> Dict[str, Any]:
    """Obtiene información de los servicios de enriquecimiento para roles específicos"""
    enrichment_data = {
        "lusha": {"status": "pending", "last_check": datetime.now().isoformat()},
        "contactout": {"status": "pending", "last_check": datetime.now().isoformat()},
        "rocketreach": {"status": "pending", "last_check": datetime.now().isoformat()}
    }
    
    # Configurar logging
    loggers = setup_api_logging()
    
    # Consultar cada API
    enrichment_data["lusha"] = query_lusha(domain, loggers["lusha"])
    enrichment_data["contactout"] = query_contactout(domain, loggers["contactout"])
    enrichment_data["rocketreach"] = query_rocketreach(domain, loggers["rocketreach"])
    
    return enrichment_data