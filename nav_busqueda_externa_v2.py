# nav_busqueda_externa_v2.py — Etapa 2 y 3: Búsqueda web externa con enriquecimiento
# Objetivo: Buscar señales externas (emails, roles, nombres) y enriquecer con información contextual

import os
import re
import json
import time
import random
import logging
import hashlib
from typing import List, Dict, Any, Optional, Tuple, Set
from urllib.parse import quote_plus, urlparse, urljoin
from datetime import datetime, timedelta
from pathlib import Path
import subprocess

import requests
from bs4 import BeautifulSoup
from io import BytesIO

from api_enrichment import get_enrichment_sources

# Cargar variables de entorno si existe .env
try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover - opcional
    def load_dotenv(*args, **kwargs):
        return False
load_dotenv()

# Configuración de directorios
OUT_DIR = "out"
CACHE_DIR = "cache"  # Directorio para caché
for dir_path in [OUT_DIR, CACHE_DIR]:
    Path(dir_path).mkdir(exist_ok=True)
OUT_JSON = os.path.join(OUT_DIR, "etapa1_2_V2_V3.json")

# Configuración de búsqueda
USER_AGENTS = [
    ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
     "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
    ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
     "(KHTML, like Gecko) Edge/124.0 Safari/537.36"),
    ("Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) "
     "Gecko/20100101 Firefox/124.0"),
    ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
     "(KHTML, like Gecko) Version/17.0 Safari/605.1.15")
]
# Motores de búsqueda con configuraciones
SERP_RESULTS_LIMIT = int(os.getenv("SERP_RESULTS_LIMIT", "100") or "100")
SERP_FOLLOW_LIMIT = int(os.getenv("SERP_FOLLOW_LIMIT", "100") or "100")

SEARCH_ENGINES = {
    # Priorizar Google primero para mayor estabilidad; Yahoo después
    "google": {
        "url": "https://www.google.com/search",
        "param": "q",
        "rate_limit": 4.0,
        "max_retries": 3,
        # Google soporta num=100
        "extra_params": {"num": str(SERP_RESULTS_LIMIT)}
    },
    "yahoo": {
        "url": "https://search.yahoo.com/search",
        "param": "p",
        "rate_limit": 3.0,  # segundos entre solicitudes
        "max_retries": 3,
        # Pedir más resultados si el motor lo soporta
        "extra_params": {"n": str(SERP_RESULTS_LIMIT)}
    }
}

# Directorios locales de Florida
FLORIDA_DIRECTORIES = [
    "https://www.floridachamber.com/",
    "https://www.myflorida.com/",
    "https://www.sunbiz.org/"
]

# Configuración de tiempos
PAUSE_MIN, PAUSE_MAX = 3.0, 6.0  # Aumentados para evitar bloqueos
TIMEOUT = (15, 25)  # Aumentados los timeouts
CACHE_DURATION = timedelta(hours=24)  # Duración del caché

# Patrones de búsqueda
# Patrones extendidos de búsqueda
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
NAME_TOKEN = r"(?:[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+\.?|[A-Z]{2,3})"
NAME_RE = re.compile(rf"\b{NAME_TOKEN}(?:\s+{NAME_TOKEN}){{1,3}}\b")
SOCIAL_HOSTS = {
    "facebook.com/": "Facebook",
    "instagram.com/": "Instagram",
    "linkedin.com/": "LinkedIn",
    "youtube.com/": "YouTube",
    "twitter.com/": "Twitter",
    "yelp.com/": "Yelp"
}

# Proveedores externos de email permitidos además del dominio
ALLOWED_EXTERNAL_DOMAINS = {
    'comcast.net', 'gmail.com', 'me.com', 'icloud.com', 'outlook.com',
    'hotmail.com', 'yahoo.com', 'aol.com', 'proton.me'
}

# Obfuscation handling (e.g., "name [at] domain [dot] com")
OBFUSCATION_PATTERNS = [
    (re.compile(r"\s*\[\s*at\s*\]\s*", re.I), "@"),
    (re.compile(r"\s*\(\s*at\s*\)\s*", re.I), "@"),
    (re.compile(r"\s+at\s+", re.I), "@"),
    (re.compile(r"\s*\[\s*dot\s*\]\s*", re.I), "."),
    (re.compile(r"\s*\(\s*dot\s*\)\s*", re.I), "."),
    (re.compile(r"\s+dot\s+", re.I), "."),
    (re.compile(r"\s*\[\s*arroba\s*\]\s*", re.I), "@"),
    (re.compile(r"\s*\[\s*punto\s*\]\s*", re.I), "."),
    (re.compile(r"\s+arroba\s+", re.I), "@"),
    (re.compile(r"\s+punto\s+", re.I), ".")
]

def deobfuscate_email_text(t: str) -> str:
    if not t:
        return t
    for pat, rep in OBFUSCATION_PATTERNS:
        t = pat.sub(rep, t)
    t = re.sub(r"\s*@\s*", "@", t)
    t = re.sub(r"\s*\.\s*", ".", t)
    return t

def decode_cloudflare_cfemail(hex_string: str) -> Optional[str]:
    """Decode Cloudflare-protected emails stored in data-cfemail attributes."""
    try:
        r = bytes.fromhex(hex_string)
        key = r[0]
        decoded = bytes([b ^ key for b in r[1:]]).decode('utf-8')
        # quick sanity: must contain '@'
        return decoded if '@' in decoded else None
    except Exception:
        return None

class RequestCache:
    """Sistema de caché para evitar solicitudes repetidas"""
    
    def __init__(self, cache_dir: str = CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def _sanitize_filename(self, name: str, max_len: int = 120) -> str:
        """Sanitiza un nombre para usarlo como archivo en Windows/Linux"""
        # Reemplazar caracteres no válidos y normalizar espacios
        name = re.sub(r"[\\/:*?\"<>|]+", "_", name)
        name = name.replace(" ", "_")
        # Colapsar múltiples guiones bajos
        name = re.sub(r"_+", "_", name)
        # Limitar longitud para evitar rutas demasiado largas
        return name[:max_len].strip("._") or "cache"

    def _extract_domain_from_query(self, query: str) -> Optional[str]:
        """Intenta extraer el dominio a partir de un prompt de búsqueda (ej: 'site:example.com ...')."""
        m = re.search(r"site:([A-Za-z0-9.-]+)", query)
        if m:
            return m.group(1)
        # Fallback: tomar el primer token con un punto (posible dominio)
        for token in re.split(r"\s+", query):
            if "." in token and not token.startswith("@"):  # evitar emails
                # limpiar puntuación simple
                token = token.strip(",;:()[]{}\"'<>“”’`“””)")
                if re.match(r"^[A-Za-z0-9.-]+$", token):
                    return token
        return None

    def _friendly_cache_filename(self, key: str) -> Optional[str]:
        """Crea un nombre de archivo legible basado en 'dominio + prompt' cuando es posible."""
        try:
            if key.startswith("search:"):
                # Formato esperado: search:{engine}:{query}
                parts = key.split(":", 2)
                engine = parts[1] if len(parts) > 1 else "engine"
                query = parts[2] if len(parts) > 2 else "query"
                domain = self._extract_domain_from_query(query) or "unknown"
                prompt = self._sanitize_filename(query)
                dom = self._sanitize_filename(domain, max_len=80)
                eng = self._sanitize_filename(engine, max_len=20)
                return f"{dom}__{prompt}__{eng}.json"
            if key.startswith("enrichment:"):
                # Formato esperado: enrichment:{domain}
                domain = key.split(":", 1)[1] if ":" in key else "unknown"
                dom = self._sanitize_filename(domain, max_len=80)
                return f"{dom}__enrichment.json"
        except Exception:
            return None
        return None

    def _get_cache_path(self, key: str) -> Path:
        """Genera la ruta del archivo de caché usando 'dominio + prompt' si es posible; si no, usa hash."""
        friendly = self._friendly_cache_filename(key)
        if friendly:
            return self.cache_dir / friendly
        # Fallback al esquema anterior basado en hash
        hash_key = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{hash_key}.json"
    
    def get(self, key: str) -> Optional[Dict]:
        """Obtiene un valor del caché si existe y no ha expirado"""
        cache_path = self._get_cache_path(key)
        logger.debug(f"[CACHE] Intentando leer cache: {cache_path} (key: {key})")
        # Intentar con nombre amigable; si no existe, intentar con hash antiguo para retrocompatibilidad
        fallback_hashed = None
        if not cache_path.exists():
            hash_key = hashlib.md5(key.encode()).hexdigest()
            fallback_hashed = self.cache_dir / f"{hash_key}.json"
            if fallback_hashed.exists():
                logger.debug(f"[CACHE] No existe cache amigable. Probando fallback hashed: {fallback_hashed}")
                cache_path = fallback_hashed
        if cache_path.exists():
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    logger.debug(f"[CACHE] Cache encontrado para key: {key}")
                    if datetime.fromisoformat(data['timestamp']) + CACHE_DURATION > datetime.now():
                        logger.debug(f"[CACHE] Cache válido para key: {key}")
                        return data['content']
                    else:
                        logger.debug(f"[CACHE] Cache expirado para key: {key}")
            except Exception as e:
                logger.error(f"Error reading cache: {str(e)}")
        else:
            logger.debug(f"[CACHE] No existe cache para key: {key}")
        return None
    
    def set(self, key: str, value: Any):
        """Guarda un valor en el caché"""
        cache_path = self._get_cache_path(key)
        logger.debug(f"[CACHE] Guardando cache: {cache_path} (key: {key})")
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'content': value
                }, f, ensure_ascii=False, indent=2)
            logger.debug(f"[CACHE] Cache guardado para key: {key}")
        except Exception as e:
            logger.error(f"Error writing cache: {str(e)}")

# Instancia global del caché
cache = RequestCache()

class APIKeyRotator:
    """Sistema de rotación de API keys"""
    
    def __init__(self, keys: List[str]):
        self.keys = keys
        self.current_index = 0
        self.last_used = datetime.now()
    
    def get_next_key(self) -> str:
        """Obtiene la siguiente API key disponible"""
        now = datetime.now()
        if (now - self.last_used).total_seconds() < 1.0:
            time.sleep(1.0)
        
        self.current_index = (self.current_index + 1) % len(self.keys)
        self.last_used = datetime.now()
        return self.keys[self.current_index]

_engine_circuit: Dict[str, Dict[str, Any]] = {}

def _engine_allowed(engine: str) -> bool:
    st = _engine_circuit.get(engine)
    if not st:
        return True
    until = st.get("suspended_until")
    if until and datetime.now() < until:
        return False
    return True

def _note_engine_error(engine: str, status: Optional[int] = None):
    st = _engine_circuit.setdefault(engine, {"errors": 0, "suspended_until": None})
    st["errors"] = int(st.get("errors", 0)) + 1
    # Si recibe 5xx repetidos, suspender por 10 minutos
    if status and 500 <= status < 600:
        # incrementar errores 5xx y suspender si supera 2
        if st["errors"] >= 2:
            st["suspended_until"] = datetime.now() + timedelta(minutes=10)
            logger.warning(f"[ENGINE] Suspendiendo {engine} por 10 min debido a errores 5xx repetidos")

def _note_engine_success(engine: str):
    if engine in _engine_circuit:
        _engine_circuit[engine] = {"errors": 0, "suspended_until": None}

def search_with_retry(engine: str, query: str, max_retries: int = 3) -> Optional[str]:
    """Realiza una búsqueda con reintentos y manejo de errores"""
    engine_config = SEARCH_ENGINES[engine]
    retry_count = 0
    logger.debug(f"[FUNC] search_with_retry llamada con engine={engine}, query={query}")
    # Verificar caché primero
    cache_key = f"search:{engine}:{query}"
    cached_result = cache.get(cache_key)
    if cached_result:
        logger.debug(f"[CACHE] search_with_retry cache hit para key: {cache_key}")
        # Manejar tanto string como dict en caché
        if isinstance(cached_result, dict) and 'html' in cached_result:
            return cached_result['html']
        return cached_result
    # Respetar suspensión temporal del motor
    if not _engine_allowed(engine):
        logger.info(f"[ENGINE] Saltando motor suspendido: {engine}")
        return None
    # Intentar primero con Puppeteer si está disponible
    try:
        if _puppeteer_available():
            pup = _puppeteer_search(engine, query, SERP_RESULTS_LIMIT, timeout_ms=int(TIMEOUT[1]*1000))
            if pup and isinstance(pup.get("html"), str):
                cache.set(cache_key, {
                    'engine': engine,
                    'query': query,
                    'html': pup["html"],
                    'links': pup.get('links', [])
                })
                _note_engine_success(engine)
                return pup["html"]
    except Exception:
        pass
    while retry_count < max_retries:
        try:
            logger.debug(f"[FUNC] search llamada desde search_with_retry (engine={engine}, query={query}, intento={retry_count+1})")
            result = search(engine, query)
            if result:
                # Guardar en caché
                cache.set(cache_key, {
                    'engine': engine,
                    'query': query,
                    'html': result
                })
                logger.debug(f"[CACHE] search_with_retry cache set para key: {cache_key}")
                _note_engine_success(engine)
                return result
        except Exception as e:
            logger.warning(f"Intento {retry_count + 1} fallido para {engine}: {str(e)}")
        retry_count += 1
        time.sleep(engine_config['rate_limit'] * (retry_count + 1))
    logger.debug(f"[FUNC] search_with_retry no obtuvo resultado para engine={engine}, query={query}")
    return None

def search(engine: str, query: str) -> Optional[str]:
    """Realiza una búsqueda en el motor especificado con configuraciones mejoradas"""
    engine_config = SEARCH_ENGINES[engine]
    q = quote_plus(query)
    url = f"{engine_config['url']}?{engine_config['param']}={q}"
    # Adjuntar parámetros extra (p.ej., num=100)
    extra = engine_config.get("extra_params") or {}
    if extra:
        for k, v in extra.items():
            url += f"&{k}={quote_plus(str(v))}"
    logger.debug(f"[FUNC] search llamada con engine={engine}, query={query}, url={url}")
    # Rotar User-Agent
    ua = random.choice(USER_AGENTS)
    headers = {
        "User-Agent": ua,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1"
    }
    try:
        logger.debug(f"Buscando en {engine}: {query}")
        r = requests.get(url, timeout=TIMEOUT, headers=headers)
        logger.debug(f"[FUNC] search respuesta status={r.status_code} para url={url}")
        if r.status_code >= 400:
            if engine == "yahoo" and r.status_code == 502:
                logger.error("Error 502 en yahoo — se intentará más tarde o se continuará con otros motores")
            _note_engine_error(engine, r.status_code)
            return None
        logger.debug(f"Búsqueda exitosa en {engine}")
        _note_engine_success(engine)
        return r.text
    except Exception as e:
        logger.error(f"Error en búsqueda {engine}: {str(e)}")
        _note_engine_error(engine, None)
        return None

def get_base_url(domain: str) -> str:
    """Determina la URL base que funciona para el dominio"""
    base_urls = [
        f"https://www.{domain}",
        f"https://{domain}",
        f"http://www.{domain}",
        f"http://{domain}"
    ]
    
    for url in base_urls:
        try:
            ua = random.choice(USER_AGENTS)
            headers = {
                "User-Agent": ua,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5"
            }
            r = requests.get(url, timeout=TIMEOUT, headers=headers, allow_redirects=True)
            if r.status_code == 200:
                return r.url  # Retorna la URL final después de redirecciones
        except:
            continue
    return f"https://{domain}"  # URL por defecto si ninguna funciona

def _same_site(host: str, base_host: str) -> bool:
    host = (host or "").lower().lstrip("www.")
    base = (base_host or "").lower().lstrip("www.")
    return host == base or host.endswith("." + base)

def get_internal_links(html: str, base_url: str) -> List[str]:
    """Extrae enlaces internos de una página (tolera www/subdominios)."""
    soup = BeautifulSoup(html, 'html.parser')
    links = set()
    base_host = urlparse(base_url).netloc
    for a in soup.find_all('a', href=True):
        href = a['href']
        if href.startswith('#') or href.startswith('mailto:') or href.startswith('javascript:'):
            continue
        if href.startswith('/'):
            links.add(urljoin(base_url, href))
        elif href.startswith(('http://', 'https://')):
            parsed = urlparse(href)
            if _same_site(parsed.netloc, base_host):
                links.add(href)
    return list(links)

def scrape_website(url: str, visited: set) -> Tuple[Optional[str], List[str]]:
    """Extrae el contenido directamente del sitio web y retorna los enlaces internos"""
    if url in visited:
        return None, []
        
    visited.add(url)
    try:
        ua = random.choice(USER_AGENTS)
        headers = {
            "User-Agent": ua,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5"
        }
        r = requests.get(url, timeout=TIMEOUT, headers=headers)
        if r.status_code == 200:
            return r.text, get_internal_links(r.text, url)
    except Exception as e:
        logger.error(f"Error scraping {url}: {str(e)}")
    return None, []

# Cache sencillo para fetch de URLs (SERP targets)
def fetch_page_cached(url: str) -> Optional[str]:
    key = f"fetch:{url}"
    cached = cache.get(key)
    if cached:
        if isinstance(cached, dict) and 'html' in cached:
            return cached['html']
        if isinstance(cached, str):
            return cached
    # Validar URL básica antes de solicitar
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            logger.debug(f"[FETCH] URL con esquema no soportado, se omite: {url}")
            return None
        if not parsed.netloc or "." not in parsed.netloc:
            logger.debug(f"[FETCH] URL con host inválido, se omite: {url}")
            return None
    except Exception:
        logger.debug(f"[FETCH] URL malformada, se omite: {url}")
        return None
    try:
        ua = random.choice(USER_AGENTS)
        headers = {
            "User-Agent": ua,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5"
        }
        r = requests.get(url, timeout=TIMEOUT, headers=headers)
        if r.status_code == 200:
            ctype = (r.headers.get('Content-Type') or '').lower()
            # Detectar PDF por content-type o extensión
            if 'application/pdf' in ctype or url.lower().endswith('.pdf'):
                text = None
                try:
                    from pdfminer.high_level import extract_text as pdf_extract_text  # type: ignore
                    text = pdf_extract_text(BytesIO(r.content))
                except Exception as e:
                    logger.warning(f"[PDF] No se pudo extraer texto de PDF {url}: {e}")
                if text:
                    cache.set(key, {"url": url, "html": text, "type": "pdf"})
                    return text
                else:
                    # fallback: devolver vacío para no romper flujo
                    cache.set(key, {"url": url, "html": "", "type": "pdf"})
                    return ""
            else:
                html = r.text or ''
                # Fallback: si es muy corto o .aspx, intentar headless
                is_aspx = url.lower().endswith('.aspx') or '.aspx?' in url.lower()
                if (len(html) < 600 or is_aspx) and _puppeteer_available():
                    pup = _puppeteer_fetch(url, timeout_ms=int(TIMEOUT[1]*1000))
                    if isinstance(pup, dict):
                        # Priorizar HTML si es más largo; pero guardamos también señales para extracción posterior
                        ph = pup.get('html') or ''
                        if isinstance(ph, str) and len(ph) > len(html):
                            html = ph
                        # Guardar señales adicionales en caché para consumo aguas arriba
                        extra = {
                            'url': url,
                            'html': html,
                            'text': pup.get('text') or '',
                            'mailtos': pup.get('mailtos') or [],
                            'cfemails': pup.get('cfemails') or [],
                            'nodeTexts': pup.get('nodeTexts') or []
                        }
                        cache.set(key, extra)
                        return html
                cache.set(key, {"url": url, "html": html})
                return html
    except Exception as e:
        logger.error(f"Error fetch_page_cached {url}: {e}")
    return None

def parse_search_result_links(engine: str, html: str, domain: str) -> List[str]:
    """Extrae URLs de resultados en el SERP y filtra por el dominio objetivo."""
    soup = BeautifulSoup(html or "", 'html.parser')
    out = []
    base_host = domain.lower().lstrip('www.')
    for a in soup.find_all('a', href=True):
        href = a['href']
        # manejar google /url?q=
        if href.startswith('/url?q='):
            try:
                # extraer destino real
                q = href.split('/url?q=', 1)[1].split('&', 1)[0]
                href = requests.utils.unquote(q)
            except Exception:
                pass
        if not href.startswith(('http://', 'https://')):
            continue
        host = urlparse(href).netloc.lower().lstrip('www.')
        if host == base_host or host.endswith('.' + base_host):
            out.append(href)
    # dedup conservando orden
    seen, dedup = set(), []
    for u in out:
        if u not in seen:
            seen.add(u); dedup.append(u)
    # Limitar por configuración (default 100)
    return dedup[:SERP_RESULTS_LIMIT]

def extract_emails_from_text(text: str, domain: str, url: str, emails_found: Set[str]) -> List[Dict[str, Any]]:
    """Extrae emails de un texto y genera datos de contacto"""
    # Determinar si el texto proviene de una página interna del dominio objetivo
    try:
        page_host = urlparse(url).netloc
        is_internal_page = _same_site(page_host, domain)
    except Exception:
        is_internal_page = False
    def extract_name_from_context(ctx: str, email_value: str) -> Optional[str]:
        if not ctx:
            return None
        try:
            low = ctx.lower()
            idx = low.find(email_value.lower())
            # ventana alrededor del email
            start = max(0, idx - 220) if idx >= 0 else 0
            end = min(len(ctx), (idx + len(email_value) + 220)) if idx >= 0 else len(ctx)
            window = ctx[start:end]
            # buscar el nombre más cercano al email
            best_name = None
            best_dist = 10**9
            for m in NAME_RE.finditer(window):
                nm = m.group(0).strip()
                # Heurística simple: descartar tokens que lucen como secciones genéricas
                if len(nm.split()) < 2:
                    continue
                # Excluir palabras comunes indeseadas
                BAD = {"contact", "about", "team", "office", "general", "support", "sales", "info"}
                tl = nm.lower()
                if any(b in tl for b in BAD):
                    continue
                # distancia al centro del email
                mid = (idx - start) if idx >= 0 else 0
                dist = abs(((m.start() + m.end()) // 2) - mid)
                if dist < best_dist:
                    best_dist = dist; best_name = nm
            return best_name
        except Exception:
            return None
    def fallback_name_from_email(email_value: str) -> Optional[str]:
        try:
            local = (email_value or "").split("@", 1)[0]
            local = re.sub(r"\d+", " ", local)  # quitar dígitos
            local = local.replace("_", " ").replace("-", " ").replace(".", " ")
            parts = [p for p in local.split() if p]
            GENERIC = {"info","contact","support","sales","events","office","admin","hello","hi","team"}
            if not parts or parts[0].lower() in GENERIC:
                return None
            # Capitalizar 2-3 partes
            parts = parts[:3]
            titled = " ".join(w[:1].upper() + w[1:].lower() for w in parts if len(w) >= 2)
            if len(titled.split()) >= 2:
                return titled
            return None
        except Exception:
            return None
    def is_plausible_domain(d: str) -> bool:
        try:
            d = (d or '').lower().strip().strip('.')
            if '.' not in d:
                return False
            labels = d.split('.')
            # TLD alfabético de 2 a 10 chars
            tld = labels[-1]
            if not (2 <= len(tld) <= 10 and tld.isalpha()):
                return False
            # Al menos una letra en todo el dominio
            if not any(any(ch.isalpha() for ch in lab) for lab in labels):
                return False
            # Evitar patrones de teléfono en cualquier label (e.g., 352-637-4424)
            phone_like = re.compile(r"^\d{3,}(-\d{2,})+\d*$")
            for lab in labels:
                if phone_like.match(lab):
                    return False
            return True
        except Exception:
            return False
    contacts = []
    for email in EMAIL_RE.finditer(text):
        email_addr = email.group().lower()
        email_domain = email_addr.split('@')[-1]
        # Regla de aceptación:
        # - Si la página es interna del sitio, aceptar solo dominios plausibles (evita falsos positivos tipo teléfonos);
        #   permite dominios externos reales como .gov/.us
        # - Si es externa (SERP u otros dominios), mantener filtro estricto: mismo dominio o proveedores libres permitidos
        accept = False
        if is_internal_page:
            accept = is_plausible_domain(email_domain)
        else:
            accept = email_addr.endswith(domain) or (email_domain in ALLOWED_EXTERNAL_DOMAINS)
        if accept and email_addr not in emails_found:
            
            # Buscar contexto alrededor del email
            idx = text.lower().find(email_addr)
            start = max(0, idx - 100)
            end = min(len(text), idx + len(email_addr) + 100)
            context = text[start:end].strip()
            
            # Determinar rol basado en el contexto
            context_lower = context.lower()
            role_title = "Staff Member"
            department = "Unknown"
            confidence = 0.9  # Alta confianza por estar en el sitio
            
            if any(term in context_lower for term in ['events', 'catering', 'banquet']):
                role_title = "Events & Catering Staff"
                department = "Events & Catering"
            elif any(term in context_lower for term in ['marketing', 'communications']):
                role_title = "Marketing Staff"
                department = "Marketing"
            elif any(term in context_lower for term in ['golf', 'pro shop']):
                role_title = "Golf Staff"
                department = "Golf Operations"
            
            # Intentar extraer un nombre cercano
            possible_name = extract_name_from_context(context, email_addr) or fallback_name_from_email(email_addr)
            emails_found.add(email_addr)
            contact_data = {
                "email": email_addr,
                "name": possible_name or "",
                "roles": [{
                    "title": role_title,
                    "confidence": confidence,
                    "sources": [{
                        "type": "website",
                        "url": url,
                        "context": context,
                        "timestamp": datetime.now().isoformat()
                    }]
                }],
                "department": department,
                "description": "Found on website"
            }
            contacts.append(contact_data)
    
    return contacts

def search_emails(domain: str, site_name: Optional[str] = None) -> List[Dict[str, Any]]:
    """Busca emails relacionados con el dominio"""
    logger.info(f"Buscando emails para {domain}")
    emails_found = set()
    contacts = []
    visited = set()

    # --- Nuevo: Scraping de redes sociales (Facebook) si el sitio lo tiene ---
    def load_socials_from_etapa1(dom: str) -> List[str]:
        """Carga las URLs de redes sociales desde out/etapa1_v1.json si existe."""
        etapa1_path = Path(OUT_DIR) / "etapa1_v1.json"
        urls: List[str] = []
        if etapa1_path.exists():
            try:
                with open(etapa1_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                sites = data.get('sites') if isinstance(data, dict) else None
                if isinstance(sites, list):
                    for site in sites:
                        if isinstance(site, dict) and site.get('domain') == dom:
                            socials = site.get('socials') or []
                            for s in socials:
                                url = (s or {}).get('url')
                                if isinstance(url, str):
                                    urls.append(url)
                            break
            except Exception as e:
                logger.debug(f"[SOCIAL] No se pudo leer etapa1_v1.json: {e}")
        return urls

    def sanitize_fb_url(raw: str) -> Optional[str]:
        """Intenta normalizar una URL de Facebook y descartar basura.
        - Extrae el primer http(s)://...facebook.com...
        - O agrega https:// si aparece facebook.com/... sin esquema
        """
        if not isinstance(raw, str) or not raw:
            return None
        s = raw.strip()
        # Buscar http(s)://...facebook.com...
        m = re.search(r"https?://[^\s\"'>]*facebook\.com[^\s\"'>]*", s, flags=re.IGNORECASE)
        if m:
            return m.group(0)
        # Buscar dominio sin esquema
        m2 = re.search(r"(?:www\.)?facebook\.com/[^\s\"'>]*", s, flags=re.IGNORECASE)
        if m2:
            return "https://" + m2.group(0)
        return None

    def collect_social_links(dom: str, base_url: str) -> List[str]:
        """Retorna enlaces sociales, priorizando Facebook. Usa etapa1 y fallback al homepage."""
        links: List[str] = []
        # 1) Etapa1
        for u in load_socials_from_etapa1(dom):
            fb = sanitize_fb_url(u)
            if fb:
                links.append(fb)
        # 2) Fallback: buscar enlaces externos en el homepage
        try:
            html, _ = scrape_website(base_url, set())
            if html:
                soup = BeautifulSoup(html, 'html.parser')
                for a in soup.find_all('a', href=True):
                    href = a['href']
                    if 'facebook.com' in (href or ''):
                        fb = sanitize_fb_url(href)
                        if fb:
                            links.append(fb)
        except Exception:
            pass
        # dedup manteniendo orden
        seen, out_links = set(), []
        for u in links:
            if u and u not in seen:
                seen.add(u); out_links.append(u)
        return out_links

    def normalize_fb_candidates(url: str) -> List[str]:
        """Genera variantes tipo m.facebook.com y about para facilitar scraping."""
        try:
            # Sanitizar primero
            su = sanitize_fb_url(url) or url
            p = urlparse(su)
            host = p.netloc.lower()
            path_q = p.path or '/'
            if p.query:
                path_q += ('?' + p.query)
            candidates = [url]
            if 'facebook.com' in host and not host.startswith('m.'):
                murl = f"https://m.facebook.com{path_q}"
                candidates.append(murl)
                # variantes de about
                if not path_q.rstrip('/').endswith('/about'):
                    candidates.append(murl.rstrip('/') + '/about')
                    candidates.append(murl.split('?', 1)[0].rstrip('/') + '?sk=about')
            else:
                # ya es m.facebook.com → agregar about
                if not path_q.rstrip('/').endswith('/about'):
                    candidates.append(su.rstrip('/') + '/about')
                    base_no_q = su.split('?', 1)[0].rstrip('/')
                    candidates.append(base_no_q + '?sk=about')
            # dedup
            seen, outc = set(), []
            for c in candidates:
                sc = sanitize_fb_url(c)
                if sc and sc not in seen:
                    seen.add(sc); outc.append(sc)
            return outc
        except Exception:
            sc = sanitize_fb_url(url)
            return [sc] if sc else []

    def scrape_facebook_emails(fb_url: str, dom: str) -> List[Dict[str, Any]]:
        found: List[Dict[str, Any]] = []
        for cand in normalize_fb_candidates(fb_url):
            page_html = fetch_page_cached(cand)
            if not page_html:
                continue
            try:
                soup = BeautifulSoup(page_html, 'html.parser')
                text = ' '.join(list(soup.stripped_strings))
            except Exception:
                text = page_html if isinstance(page_html, str) else ''
            for m in EMAIL_RE.finditer(text):
                email_addr = m.group().lower()
                email_domain = email_addr.split('@')[-1]
                if ((email_addr.endswith(dom) or email_domain in ALLOWED_EXTERNAL_DOMAINS)
                    and email_addr not in emails_found):
                    emails_found.add(email_addr)
                    found.append({
                        "email": email_addr,
                        "roles": [{
                            "title": "Staff Member",
                            "confidence": 0.75,
                            "sources": [{
                                "type": "social",
                                "url": cand,
                                "context": "Facebook page",
                                "timestamp": datetime.now().isoformat()
                            }]
                        }],
                        "department": "Unknown",
                        "description": "Found on Facebook"
                    })
        return found

    # Modo especial jaxgcc con Bing deshabilitado
    if domain.lower().strip().lstrip("www.") == "jaxgcc.com":
        logger.info("[NAVEGADO] Modo especial jaxgcc deshabilitado (Bing no permitido). Se continúa con Yahoo/Google estándar.")
    
    # Obtener URL base que funciona
    base_url = get_base_url(domain)
    urls_to_check = [base_url]

    # Buscar y scrapear Facebook si existe
    try:
        social_links = collect_social_links(domain, base_url)
        fb_links = [u for u in social_links if 'facebook.com' in (u or '')]
        total_fb = 0
        for fb in fb_links[:3]:  # limitar a 3 para no exceder tiempo
            new_from_fb = scrape_facebook_emails(fb, domain)
            if new_from_fb:
                contacts.extend(new_from_fb)
                total_fb += len(new_from_fb)
        if total_fb:
            logger.debug(f"[SOCIAL] Facebook → {total_fb} emails capturados")
    except Exception as e:
        logger.debug(f"[SOCIAL] Error scraping Facebook: {e}")
    
    # Agregar URLs comunes de contacto
    for path in [
        '/contact', '/about', '/staff', '/team', '/directory',
        '/contact-us', '/about-us', '/booking', '/book', '/private-events',
        '/events', '/press', '/media', '/join-our-team', '/employment'
    ]:
        urls_to_check.append(urljoin(base_url, path))

    # Extra: patrones municipales típicos (.aspx)
    try:
        dom_l = (domain or '').lower().strip()
        extra_paths = set()
        # Sitios .gov frecuentemente usan ASP.NET con rutas Directory/Departments/ContactUs
        if dom_l.endswith('.gov'):
            extra_paths.update({
                '/Directory.aspx', '/directory.aspx',
                '/Departments.aspx', '/departments.aspx',
                '/ContactUs.aspx', '/contactus.aspx',
            })
        # Pista específica para Inverness (directorio conocido)
        if dom_l == 'inverness-fl.gov':
            extra_paths.update({'/Directory.aspx?did=5'})
        for p in extra_paths:
            urls_to_check.append(urljoin(base_url, p))
    except Exception:
        pass
    
    # Explorar el sitio web recursivamente
    while urls_to_check and len(visited) < 30:  # Limitar a 30 páginas para mayor cobertura
        url = urls_to_check.pop(0)
        logger.debug(f"Explorando {url}")
        
        html, internal_links = scrape_website(url, visited)
        if html:
            soup = BeautifulSoup(html, 'html.parser')
            # Buscar en el texto visible
            # 1) Cloudflare email protection
            try:
                for sp in soup.select('span.__cf_email__'):
                    data = sp.get('data-cfemail')
                    if data:
                        dec = decode_cloudflare_cfemail(data)
                        if dec:
                            ctx = sp.parent.get_text(" ", strip=True) if sp.parent else dec
                            new_contacts = extract_emails_from_text(ctx, domain, url, emails_found)
                            contacts.extend(new_contacts)
            except Exception:
                pass
            # 1.5) Si esta URL fue obtenida con fallback headless y hay señales extra en cache, úsalas
            try:
                extra_cached = cache.get(f"fetch:{url}")
                if isinstance(extra_cached, dict):
                    # cfemails (Cloudflare)
                    for hx in (extra_cached.get('cfemails') or []):
                        dec = decode_cloudflare_cfemail(str(hx))
                        if dec:
                            new_contacts = extract_emails_from_text(dec, domain, url, emails_found)
                            contacts.extend(new_contacts)
                    # mailtos directos
                    for mt in (extra_cached.get('mailtos') or []):
                        new_contacts = extract_emails_from_text(mt, domain, url, emails_found)
                        contacts.extend(new_contacts)
                    # textos de nodos/innerText
                    for tx in (extra_cached.get('nodeTexts') or []):
                        new_contacts = extract_emails_from_text(deobfuscate_email_text(str(tx)), domain, url, emails_found)
                        contacts.extend(new_contacts)
                    # texto plano de body
                    body_text = extra_cached.get('text') or ''
                    if isinstance(body_text, str) and body_text:
                        new_contacts = extract_emails_from_text(deobfuscate_email_text(body_text), domain, url, emails_found)
                        contacts.extend(new_contacts)
            except Exception:
                pass

            # 2) Onclick/mail() patterns or plain text with obfuscations
            for text in soup.stripped_strings:
                new_contacts = extract_emails_from_text(deobfuscate_email_text(text), domain, url, emails_found)
                contacts.extend(new_contacts)
            
            # Buscar en atributos href="mailto:"
            for a in soup.find_all('a', href=True):
                href = a['href']
                if href.startswith('mailto:'):
                    email = deobfuscate_email_text(href[7:])  # Remover 'mailto:'
                    # Enriquecer contexto con el texto del enlace y del padre
                    link_txt = a.get_text(" ", strip=True)
                    parent_txt = a.parent.get_text(" ", strip=True) if a.parent else ""
                    ctx = " ".join([t for t in [link_txt, parent_txt, email] if t])
                    new_contacts = extract_emails_from_text(ctx, domain, url, emails_found)
                    contacts.extend(new_contacts)
                else:
                    # onclick based obfuscations like onclick="location.href='mailto:' + 'name' + '@' + 'dom.com'"
                    onclick = a.get('onclick') or ''
                    if 'mailto' in onclick.lower():
                        try:
                            parts = re.findall(r"'([^']+)'|\"([^\"]+)\"", onclick)
                            joined = ''.join([p[0] or p[1] for p in parts])
                            if 'mailto:' in joined.lower():
                                m = re.search(r"mailto:([^\s]+)", joined, re.I)
                                if m:
                                    ctx = deobfuscate_email_text(m.group(1))
                                    new_contacts = extract_emails_from_text(ctx, domain, url, emails_found)
                                    contacts.extend(new_contacts)
                        except Exception:
                            pass
            # Segunda pasada con texto completo de la página para mejorar contexto
            try:
                page_text_full = deobfuscate_email_text(' '.join(list(soup.stripped_strings)))
                new_contacts = extract_emails_from_text(page_text_full, domain, url, emails_found)
                contacts.extend(new_contacts)
            except Exception:
                pass
            
            # Agregar nuevos enlaces a explorar, priorizando los que parecen relevantes
            for link in internal_links:
                if link not in visited and len(visited) < 20:
                    if any(term in link.lower() for term in ['contact', 'about', 'staff', 'team', 'directory']):
                        urls_to_check.insert(0, link)  # Priorizar estos enlaces
                    else:
                        urls_to_check.append(link)
        
        time.sleep(random.uniform(PAUSE_MIN, PAUSE_MAX))
    
    # Búsqueda externa (UNA PASADA) usando BUSQUEDA_QUERIES_JSON del entorno
    raw_env = os.getenv("BUSQUEDA_QUERIES_JSON") or "[]"
    try:
        env_queries = json.loads(raw_env)
        if not isinstance(env_queries, list):
            env_queries = []
    except Exception:
        env_queries = []

    nombre = (site_name or domain.split('.')[:1][0]).strip()
    dom = domain.strip()

    def _expand(q: Any) -> str:
        if not isinstance(q, str):
            return ""
        return q.replace("${NOMBRE_SITIO}", nombre).replace("${DOMINIO}", dom)

    def _normalize_query(q: str) -> str:
        # Colapsar espacios y asegurar prefijo site:DOMINIO
        t = " ".join((q or "").split())
        if not t:
            return ""
        if "site:" not in t:
            t = f"site:{dom} " + t
        return t.strip()

    expanded_raw = [_expand(q) for q in env_queries if _expand(q)]
    expanded_queries = []
    seen_q = set()
    for q in expanded_raw:
        nq = _normalize_query(q)
        if nq and nq not in seen_q:
            seen_q.add(nq); expanded_queries.append(nq)
    if not expanded_queries:
        for q in ["email", "contact", "staff"]:
            nq = _normalize_query(q)
            if nq and nq not in seen_q:
                seen_q.add(nq); expanded_queries.append(nq)

    logger.info(f"[SEARCH] Pasada única de búsquedas externas con {len(expanded_queries)} queries desde ENV")

    BASE_CONF = 0.65

    for query in expanded_queries:
        confidence = BASE_CONF
        logger.debug(f"[SEARCH] Query: '{query}' (conf: {confidence})")
        for engine in SEARCH_ENGINES.keys():
            html = search_with_retry(engine, query, max_retries=SEARCH_ENGINES[engine].get('max_retries', 3))
            if html:
                # 1) Intentar extraer emails directamente del SERP (por si el buscador muestra el texto)
                soup = BeautifulSoup(html, 'html.parser')
                texts = [text for text in soup.stripped_strings]
                all_text = ' '.join(texts)
                extracted = 0
                for email in EMAIL_RE.finditer(all_text):
                    email_addr = email.group().lower()
                    email_domain = email_addr.split('@')[-1]
                    if ((email_addr.endswith(domain) or email_domain in ALLOWED_EXTERNAL_DOMAINS)
                        and email_addr not in emails_found):
                        # ventana de contexto alrededor del email en el SERP
                        idx = all_text.lower().find(email_addr)
                        s = max(0, idx - 100) if idx >= 0 else 0
                        e = min(len(all_text), idx + len(email_addr) + 100) if idx >= 0 else len(all_text)
                        context = all_text[s:e].strip()
                        # intentar nombre cercano
                        possible_name = None
                        try:
                            for m in NAME_RE.finditer(context):
                                cand = m.group(0).strip()
                                if len(cand.split()) >= 2:
                                    possible_name = cand; break
                        except Exception:
                            possible_name = None
                        emails_found.add(email_addr)
                        contacts.append({
                            "email": email_addr,
                            "name": possible_name or "",
                            "roles": [{
                                "title": "Staff Member",
                                "confidence": confidence,
                                "sources": [{
                                    "type": "web_search",
                                    "url": f"{engine} search",
                                    "context": context,
                                    "timestamp": datetime.now().isoformat()
                                }]
                            }],
                            "department": "Unknown",
                            "description": "Found via web search"
                        })
                        extracted += 1
                # 2) Seguir enlaces del SERP y extraer de las páginas destino
                target_links = parse_search_result_links(engine, html, domain)
                fetched = 0
                for link in target_links:
                    page_html = fetch_page_cached(link)
                    if not page_html:
                        continue
                    fetched += 1
                    try:
                        soup2 = BeautifulSoup(page_html, 'html.parser')
                        page_text = ' '.join(list(soup2.stripped_strings))
                        new_contacts = extract_emails_from_text(page_text, domain, link, emails_found)
                        contacts.extend(new_contacts)
                    except Exception:
                        pass
                    if fetched >= SERP_FOLLOW_LIMIT:
                        break
                logger.debug(f"[SEARCH] {engine} -> {extracted} emails en SERP y {len(contacts)} total acumulado para query '{query}'")
            time.sleep(random.uniform(PAUSE_MIN, PAUSE_MAX))
    
    logger.info(f"Encontrados {len(contacts)} emails para {domain}")
    return contacts

# Configuración de logging
def setup_logging():
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logger = logging.getLogger("busqueda_externa")
    logger.setLevel(logging.DEBUG)
    
    fh = logging.FileHandler(log_dir / "busqueda_externa.log", encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    
    if not logger.handlers:
        logger.addHandler(fh)
        # Agregar salida a consola para ver progreso en tiempo real
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
        logger.addHandler(ch)
    logger.propagate = False
    return logger

logger = setup_logging()

# ---- Puppeteer integration (optional) ----
def _tools_dir() -> Path:
    return Path(__file__).parent / "tools"

def _puppeteer_available() -> bool:
    try:
        if (os.getenv("USE_PUPPETEER") or "0").strip().lower() not in ("1","true","yes","on"):
            return False
        tools = _tools_dir()
        if not (tools / "puppeteer_search.js").exists():
            return False
        # verify Node is available
        subprocess.run(["node", "-v"], capture_output=True, text=True, check=True)
        # verify dependencies installed: require node_modules presence
        if not (tools / "node_modules").exists():
            return False
        return True
    except Exception:
        return False

def _puppeteer_search(engine: str, query: str, results: int, timeout_ms: int) -> Optional[Dict[str, Any]]:
    try:
        tools = _tools_dir()
        cmd = [
            "node",
            str(tools / "puppeteer_search.js"),
            f"--engine={engine}",
            f"--query={query}",
            f"--results={int(results)}",
            f"--timeout={int(timeout_ms)}"
        ]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(tools),
            timeout=(timeout_ms/1000.0 + 10)
        )
        out = (proc.stdout or "").strip()
        if not out:
            return None
        data = json.loads(out)
        if isinstance(data, dict) and data.get("ok"):
            return data
        return None
    except Exception as e:
        logger.debug(f"[PUPPETEER] Error ejecutando puppeteer: {e}")
        return None

def _puppeteer_fetch(url: str, timeout_ms: int) -> Optional[Dict[str, Any]]:
    try:
        tools = _tools_dir()
        cmd = [
            "node",
            str(tools / "puppeteer_fetch.js"),
            f"--url={url}",
            f"--timeout={int(timeout_ms)}"
        ]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(tools),
            timeout=(timeout_ms/1000.0 + 10)
        )
        out = (proc.stdout or "").strip()
        if not out:
            return None
        data = json.loads(out)
        if isinstance(data, dict) and data.get("ok"):
            return data
        return None
    except Exception as e:
        logger.debug(f"[PUPPETEER] Error en fetch: {e}")
        return None

def process_domain(domain: str, site_name: str = None, city: str = None) -> Dict[str, Any]:
    """Procesa un dominio específico y retorna los datos estructurados"""
    logger.info(f"Procesando dominio: {domain}")
    
    logger.debug(f"[FUNC] process_domain llamada con domain={domain}, site_name={site_name}, city={city}")
    # Guardar en cache el llamado y respuesta de cada API de enriquecimiento
    enrichment_cache_key = f"enrichment:{domain}"
    enrichment_cached = cache.get(enrichment_cache_key)
    if enrichment_cached:
        enrichment_sources = enrichment_cached
        logger.debug(f"[CACHE] enrichment cache hit para key: {enrichment_cache_key}")
    else:
        logger.debug(f"[FUNC] get_enrichment_sources llamada desde process_domain para domain={domain}")
        enrichment_sources = get_enrichment_sources(domain)
        cache.set(enrichment_cache_key, enrichment_sources)
        logger.debug(f"[CACHE] enrichment cache set para key: {enrichment_cache_key}")
    site_data = {
        "site_info": {
            "name": site_name or domain.split('.')[0].capitalize(),
            "location": city or "Unknown",
            "type": "Business"
        },
    "contacts": search_emails(domain, site_name),  # Buscar y agregar emails
        "enrichment_sources": enrichment_sources,
        "metadata": {
            "last_updated": datetime.now().isoformat(),
            "confidence_scoring": {
                "description": (
                    "0.0-1.0 scale where:\n"
                    "0.9+ Direct website listing with role\n"
                    "0.7-0.8 Indirect website mention with context\n"
                    "0.5-0.6 External source or unclear context\n"
                    "<0.5 Uncertain association"
                )
            }
        }
    }
    logger.debug(f"[FUNC] process_domain completado para domain={domain}")
    return site_data

def main():
    """Función principal del script"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Procesa dominios web para extraer y enriquecer información de contacto")
    parser.add_argument("domain", nargs="?", help="Dominio a procesar (ej: example.com)")
    parser.add_argument("--site-name", help="Nombre del sitio")
    parser.add_argument("--city", help="Ciudad del negocio")
    parser.add_argument("--output", default="etapa1_2.json", help="Archivo de salida JSON")
    # Modo batch desde etapa1_v1.json
    parser.add_argument("--batch-from-etapa1", dest="batch_etapa1", help="Ruta a out/etapa1_v1.json para procesar en lote")
    parser.add_argument("--only-missing-emails", action="store_true", help="Solo sitios con emails vacíos en etapa1_v1.json")
    parser.add_argument("--max-band-score", type=int, default=None, help="Descarta sitios con band.score > valor indicado")
    parser.add_argument("--min-band-score", type=int, default=None, help="Descarta sitios con band.score < valor indicado")
    parser.add_argument("--require-florida-ok", action="store_true", help="Solo incluir sitios con florida_ok = true")
    parser.add_argument("--start", type=int, default=0, help="Índice de inicio para el lote")
    parser.add_argument("--limit", type=int, default=0, help="Cantidad máxima a procesar (0 = todos)")
    args = parser.parse_args()
    
    def ensure_parent_dir(path: str):
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)

    # Modo batch
    if args.batch_etapa1:
        etapa1_path = args.batch_etapa1
        try:
            with open(etapa1_path, 'r', encoding='utf-8') as f:
                etapa1 = json.load(f)
        except Exception as e:
            logger.error(f"No se pudo leer {etapa1_path}: {e}")
            return
        sites = etapa1.get('sites') if isinstance(etapa1, dict) else None
        if not isinstance(sites, list):
            logger.error("Formato inesperado en etapa1_v1.json (no se encontró 'sites')")
            return

        # Filtrar dominios
        items = []
        for s in sites:
            if not isinstance(s, dict):
                continue
            dom = s.get('domain')
            if not dom:
                continue
            # Filtrar por florida_ok si se requiere
            if args.require_florida_ok and not bool(s.get('florida_ok')):
                logger.info(f"[FILTER] Skip {dom} por florida_ok = false")
                continue
            # Filtrar por band.score
            band_score = None
            band = s.get('band')
            if isinstance(band, dict):
                band_score = band.get('score')
            elif isinstance(band, (int, float)):
                band_score = band
            if args.max_band_score is not None and isinstance(band_score, (int, float)) and band_score > args.max_band_score:
                logger.info(f"[FILTER] Skip {dom} por band.score {band_score} > {args.max_band_score}")
                continue
            if args.min_band_score is not None and isinstance(band_score, (int, float)) and band_score < args.min_band_score:
                logger.info(f"[FILTER] Skip {dom} por band.score {band_score} < {args.min_band_score}")
                continue
            if args.only_missing_emails:
                emails = s.get('emails', [])
                if isinstance(emails, list) and len(emails) > 0:
                    continue
            items.append({
                'domain': dom,
                'site_name': s.get('site_name'),
                'city': s.get('city')
            })

        # Slicing por start/limit
        start = max(0, int(args.start or 0))
        limit = int(args.limit or 0)
        if limit > 0:
            batch = items[start:start+limit]
        else:
            batch = items[start:]

        logger.info(f"Procesando lote desde etapa1: total={len(items)}, a procesar={len(batch)}, start={start}, limit={limit}")

        # Cargar salida existente o crearla vacía
        out_data = {}
        ensure_parent_dir(args.output)
        if os.path.exists(args.output):
            try:
                with open(args.output, 'r', encoding='utf-8') as f:
                    out_data = json.load(f)
            except Exception:
                out_data = {}
        else:
            # Crear archivo vacío
            try:
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump({}, f, indent=2, ensure_ascii=False)
                logger.info(f"Salida no existía, creada: {args.output}")
            except Exception as e:
                logger.error(f"No se pudo crear archivo de salida {args.output}: {e}")
                return

        processed = 0
        for it in batch:
            dom = it['domain']
            sname = it.get('site_name')
            cty = it.get('city')
            logger.info(f"[BATCH] Procesando {dom} ({processed+1}/{len(batch)})")
            try:
                site_data = process_domain(dom, sname, cty)
                out_data[dom] = site_data
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(out_data, f, indent=2, ensure_ascii=False)
            except KeyboardInterrupt:
                logger.warning("Batch interrumpido por el usuario.")
                break
            except Exception as e:
                logger.error(f"[BATCH] Error procesando {dom}: {e}")
            processed += 1
            # pequeña pausa entre dominios para ser amables
            time.sleep(random.uniform(1.0, 2.0))

        logger.info(f"Batch completado: procesados={processed}")
        return

    # Modo single
    if not args.domain:
        logger.error("Debe especificar un dominio o usar --batch-from-etapa1")
        return

    logger.info(f"Iniciando procesamiento para {args.domain}")

    site_data = process_domain(args.domain, args.site_name, args.city)

    data = {}
    ensure_parent_dir(args.output)
    if os.path.exists(args.output):
        try:
            with open(args.output, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            logger.error(f"Error al leer archivo {args.output}")
            pass
    else:
        # Crear archivo vacío
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump({}, f, indent=2, ensure_ascii=False)
            logger.info(f"Salida no existía, creada: {args.output}")
        except Exception as e:
            logger.error(f"No se pudo crear archivo de salida {args.output}: {e}")
            return

    data[args.domain] = site_data
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info(f"Procesamiento completado para {args.domain}")

if __name__ == "__main__":
    main()
