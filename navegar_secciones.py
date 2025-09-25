import os
import re
import csv
import json
import time
import random
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
from urllib.parse import urljoin, urlparse
from urllib.parse import quote_plus

from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException, NoSuchElementException, StaleElementReferenceException, WebDriverException
)

# ====== CONFIG ======
USE_BRAVE = False  # True para Brave, False para Chrome
BRAVE_PATH = r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe"  # ajustar
CHROMEDRIVER_PATH = os.path.join(os.path.dirname(__file__), "chromedriver_win32", "chromedriver.exe")  # Ruta local al ChromeDriver
HEADLESS = False
MAX_LINKS = 30
PAUSE_MIN, PAUSE_MAX = 0.35, 0.9

OUT_DIR = "out"
OUT_JSON = os.path.join(OUT_DIR, "extract.json")
OUT_EMAILS_CSV = os.path.join(OUT_DIR, "emails.csv")

NAV_SELECTORS = [
    "header nav a",
    "nav a",
    "[role='navigation'] a",
    ".menu a, .navbar a, .nav a, .site-nav a, .main-nav a, .navigation a",
    "a[href^='#']"
]

COOKIE_SELECTORS = [
    "button[aria-label*='Aceptar']",
    "button[aria-label*='Accept']",
    "button.cookie, button#onetrust-accept-btn-handler",
    "[id*='accept'] button, [class*='accept'] button"
]

# Diccionario de roles (ES/EN) ampliable
ROLE_KEYWORDS = [
    # EN
    "owner","co-owner","founder","co-founder","ceo","cfo","coo","cto","cmo","manager","general manager",
    "director","executive director","artistic director","music director","operations manager","project manager",
    "producer","promoter","talent buyer","booking manager","booking agent","event manager","event coordinator",
    "marketing manager","social media manager","communications manager","front of house","foh",
    "technical director","stage manager","sound engineer","audio engineer","lighting designer","ld",
    "house manager","box office manager","venue manager","bar manager","chef","head chef","sommelier",
    "human resources","hr manager","accounts payable","accounting","finance manager","facilities manager",
    # ES
    "dueño","co-dueño","propietario","fundador","cofundador","director ejecutivo","gerente general","gerente",
    "jefe de operaciones","jefe de marketing","encargado","coordinador de eventos","productor","promotor",
    "responsable de prensa","comunicaciones","rrhh","recursos humanos","encargado de sala","sonidista","iluminador",
    "jefe de barra","chef","caja","boletería","taquilla","programador","programación","booking","contratación"
]
ROLE_RE = re.compile(r"\b(" + "|".join(re.escape(k) for k in ROLE_KEYWORDS) + r")\b", re.IGNORECASE)

# Emails: admite puntos, guiones, +, subdominios, TLDs comunes
EMAIL_RE = re.compile(r"[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,24}", re.IGNORECASE)

# Nombre simple (heurística): 2–4 palabras capitalizadas (John M. Smith | Ana María Pérez)
NAME_TOKEN = r"(?:[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+\.?)"
NAME_RE = re.compile(rf"\b{NAME_TOKEN}(?:\s+{NAME_TOKEN}){{1,3}}\b")

# Patrones “Nombre – Rol” en variantes comunes
PAIR_PATTERNS = [
    re.compile(rf"(?P<name>{NAME_TOKEN}(?:\s+{NAME_TOKEN}){{1,3}})\s*[,–\-|:]\s*(?P<role>[^|/\n\r]{{3,70}})", re.IGNORECASE),
    re.compile(rf"(?P<role>[^|/\n\r]{{3,70}})\s*[,–\-|:]\s*(?P<name>{NAME_TOKEN}(?:\s+{NAME_TOKEN}){{1,3}})", re.IGNORECASE),
]

@dataclass
class PersonHit:
    name: str
    role: str = ""
    email: str = ""
    context: str = ""   # breve fragmento
    source: str = ""    # e.g., 'text', 'mailto', 'schema'

@dataclass
class PageExtract:
    page_url: str
    title: str
    emails: List[str]
    persons: List[Dict[str, Any]]


# ====== Helpers ======
def human_pause(a=PAUSE_MIN, b=PAUSE_MAX):
    time.sleep(random.uniform(a, b))

def build_driver():
    opts = Options()
    if USE_BRAVE:
        opts.binary_location = BRAVE_PATH
    if HEADLESS:
        opts.add_argument("--headless=new")
    opts.add_argument("--start-maximized")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    from selenium.webdriver.chrome.service import Service
    service = Service(CHROMEDRIVER_PATH)
    driver = webdriver.Chrome(service=service, options=opts)
    driver.set_page_load_timeout(45)
    return driver

def is_same_domain(base_url, candidate_url):
    try:
        b = urlparse(base_url)
        c = urlparse(candidate_url)
        return (c.netloc == "" or c.netloc == b.netloc)
    except:
        return False

def is_visible_link(el):
    try:
        if not el.is_displayed():
            return False
        href = el.get_attribute("href") or ""
        if not href or href.startswith("javascript:"):
            return False
        rect = el.rect
        if rect and (rect.get("width", 0) < 5 or rect.get("height", 0) < 5):
            return False
        return True
    except StaleElementReferenceException:
        return False
    except Exception:
        return False

def accept_cookies_if_any(driver, wait):
    for css in COOKIE_SELECTORS:
        try:
            btns = driver.find_elements(By.CSS_SELECTOR, css)
            if btns:
                for b in btns:
                    try:
                        if b.is_displayed() and b.is_enabled():
                            b.click()
                            human_pause()
                            return
                    except:
                        continue
        except:
            continue

def collect_nav_links(driver, base_url):
    links = []
    for sel in NAV_SELECTORS:
        try:
            found = driver.find_elements(By.CSS_SELECTOR, sel)
            for el in found:
                if is_visible_link(el):
                    href = el.get_attribute("href") or ""
                    if href.startswith("#"):
                        full = base_url.rstrip("/") + "/" + href
                    else:
                        full = urljoin(base_url, href)
                    if is_same_domain(base_url, full):
                        links.append(full)
        except Exception:
            continue
    # Quitar duplicados
    seen = set(); ordered = []
    for u in links:
        if u not in seen:
            seen.add(u); ordered.append(u)
    return ordered

def scroll_page(driver):
    try:
        driver.execute_script("window.scrollTo(0, 0);")
        human_pause()
        h = driver.execute_script("return document.body.scrollHeight || document.documentElement.scrollHeight;") or 2000
        steps = 3
        for i in range(1, steps + 1):
            y = int(h * i / steps)
            driver.execute_script(f"window.scrollTo(0, {y});")
            human_pause(0.25, 0.6)
        driver.execute_script("window.scrollTo(0, 0);")
    except:
        pass

def text_chunks(text: str, size: int = 260):
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return [text[i:i+size] for i in range(0, min(len(text), 2000), size)]

def find_emails(soup: BeautifulSoup) -> List[str]:
    emails = set()
    # mailto:
    for a in soup.select("a[href^='mailto:']"):
        href = a.get("href") or ""
        m = EMAIL_RE.search(href)
        if m:
            emails.add(m.group(0))
    # texto plano
    for txt in soup.stripped_strings:
        for m in EMAIL_RE.finditer(txt):
            emails.add(m.group(0))
    return sorted(emails)

def find_persons_and_roles(soup: BeautifulSoup) -> List[PersonHit]:
    persons: List[PersonHit] = []

    # 1) Schema.org Person
    for p in soup.select('[itemtype*="schema.org/Person"], [typeof*="schema:Person"]'):
        name = ""; role = ""; email = ""
        # name
        cand = p.select_one('[itemprop="name"], [property="schema:name"], .name, .person-name, h2, h3')
        if cand and cand.get_text(strip=True):
            name = cand.get_text(strip=True)
        # role
        cand = p.select_one('[itemprop="jobTitle"], [property="schema:jobTitle"], .role, .title, .position')
        if cand and cand.get_text(strip=True):
            role = cand.get_text(strip=True)
        # email
        cand = p.select_one('a[href^="mailto:"]')
        if cand:
            m = EMAIL_RE.search(cand.get("href") or "")
            if m: email = m.group(0)
        if name:
            persons.append(PersonHit(name=name, role=role, email=email, source="schema"))

    # 2) Bloques con cards/staff típicos
    candidate_blocks = soup.select(".staff, .team, .people, .person, .profile, [class*='team'], [class*='staff']")
    for blk in candidate_blocks:
        txt = blk.get_text(separator=" ", strip=True)
        if not txt: continue
        # Pairs Nombre – Rol
        for pat in PAIR_PATTERNS:
            for m in pat.finditer(txt):
                nm = (m.groupdict().get("name") or "").strip()
                rl = (m.groupdict().get("role") or "").strip()
                if nm and (rl and ROLE_RE.search(rl)):
                    persons.append(PersonHit(name=nm, role=rl, context=txt[:200], source="text"))
        # Nombre suelto + rol suelto en el mismo bloque
        for m in NAME_RE.finditer(txt):
            nm = m.group(0).strip()
            # Buscar rol cercano
            win = txt[max(0, m.start()-80): m.end()+80]
            r = ROLE_RE.search(win)
            role = r.group(0) if r else ""
            if nm:
                persons.append(PersonHit(name=nm, role=role, context=win, source="text"))

    # 3) Barrido general del documento (fallback)
    full = soup.get_text(separator=" ", strip=True)
    for pat in PAIR_PATTERNS:
        for m in pat.finditer(full):
            nm = (m.groupdict().get("name") or "").strip()
            rl = (m.groupdict().get("role") or "").strip()
            if nm and rl and ROLE_RE.search(rl):
                ctx = full[max(0, m.start()-100): m.end()+100]
                persons.append(PersonHit(name=nm, role=rl, context=ctx, source="text"))

    # 4) Deduplicar por (name, role, email)
    seen = set(); deduped = []
    for p in persons:
        key = (p.name.lower(), (p.role or "").lower(), (p.email or "").lower())
        if key in seen: continue
        seen.add(key); deduped.append(p)

    return deduped

def extract_from_current_page(driver, page_url) -> PageExtract:
    title = ""
    try:
        title = driver.title or ""
    except:
        pass
    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")

    emails = find_emails(soup)

    # Si encontramos emails con patrones de persona (p.ej. "john@..."), tratamos de mapear nombre a partir de alrededor
    persons = find_persons_and_roles(soup)

    # También mapear mailto asociados a nombres cercanos en la UI (etiquetas vecinas)
    for a in soup.select("a[href^='mailto:']"):
        email = None
        m = EMAIL_RE.search(a.get("href") or "")
        if m: email = m.group(0)
        if not email: continue
        label = a.get_text(" ", strip=True)
        # Mirar elementos hermanos/padre
        ctx_txt = " ".join([label] + [s.get_text(" ", strip=True) for s in a.parents if hasattr(s, "get_text")][:1])
        # ¿Hay nombre cerca?
        nm = None
        mm = NAME_RE.search(ctx_txt)
        if mm:
            nm = mm.group(0)
        # ¿Hay rol cerca?
        rl = ""
        rr = ROLE_RE.search(ctx_txt)
        if rr:
            rl = rr.group(0)
        persons.append(PersonHit(name=nm or "", role=rl, email=email, context=ctx_txt[:200], source="mailto"))

    # Deduplicar personas final
    seen = set(); final_persons = []
    for p in persons:
        key = ( (p.name or "").lower(), (p.role or "").lower(), (p.email or "").lower() )
        if key in seen: continue
        seen.add(key); final_persons.append(p)

    return PageExtract(
        page_url=page_url,
        title=title,
        emails=sorted(set(emails)),
        persons=[asdict(p) for p in final_persons]
    )

def safe_screenshot(driver, outdir, label):
    os.makedirs(outdir, exist_ok=True)
    fname = os.path.join(outdir, f"{label}.png")
    try:
        driver.save_screenshot(fname)
    except:
        pass

# ====== Core de navegación + extracción ======
def navegar_y_extraer(url: str, output_dir: str = None):
    out_dir = output_dir or OUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    driver = build_driver()
    wait = WebDriverWait(driver, 20)

    parsed = urlparse(url)
    base_origin = f"{parsed.scheme}://{parsed.netloc}"
    results: List[PageExtract] = []
    visited = set()

    try:
        print(f"[INFO] Abriendo: {url}")
        driver.get(url)
        human_pause(0.8, 1.4)
        accept_cookies_if_any(driver, wait)
        scroll_page(driver)
        safe_screenshot(driver, out_dir, "home")

        # Extraer de la home
        pe = extract_from_current_page(driver, driver.current_url)
        results.append(pe)
        visited.add(driver.current_url.split("#")[0])

        # Colectar enlaces de navegación
        nav_links = collect_nav_links(driver, base_origin)[:MAX_LINKS]
        print(f"[INFO] Secciones detectadas: {len(nav_links)}")

        idx = 1
        for link in nav_links:
            base = link.split("#")[0]
            if base in visited:
                continue
            visited.add(base)

            print(f"  → [{idx}/{len(nav_links)}] {link}")
            idx += 1

            # Mismo path con #ancla
            if "#" in link and urlparse(link).path == urlparse(driver.current_url).path:
                try:
                    anchor = link.split("#", 1)[1]
                    if anchor:
                        target = driver.find_elements(By.CSS_SELECTOR, f"#{anchor}")
                        if target:
                            driver.execute_script("arguments[0].scrollIntoView({behavior:'smooth',block:'center'});", target[0])
                            human_pause(0.6, 1.2)
                            safe_screenshot(driver, out_dir, f"anchor_{anchor}")
                            # Extraer igual (estado actual de la página)
                            pe = extract_from_current_page(driver, driver.current_url + "#" + anchor)
                            results.append(pe)
                            continue
                except Exception:
                    pass

            # Navegar
            try:
                driver.execute_script("window.open('','_self');")
                driver.get(link)
            except WebDriverException:
                try:
                    driver.switch_to.active_element.send_keys(Keys.CONTROL, "l")
                    driver.switch_to.active_element.send_keys(link + "\n")
                except:
                    continue

            try:
                accept_cookies_if_any(driver, wait)
            except:
                pass

            try:
                wait.until(lambda d: d.execute_script("return document.readyState") == "complete")
            except TimeoutException:
                pass

            scroll_page(driver)
            label = urlparse(link).path.strip("/").replace("/", "_") or "root"
            label = ("sec_" + label)[:120]
            safe_screenshot(driver, out_dir, label)

            pe = extract_from_current_page(driver, driver.current_url)
            results.append(pe)
            human_pause(0.8, 1.4)

        # Guardar resultados
        out_json = os.path.join(out_dir, "extract.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in results], f, ensure_ascii=False, indent=2)

        # Export CSV de emails (dedupe por email)
        seen_emails = set()
        rows = []
        for r in results:
            for em in r.emails:
                if em in seen_emails: continue
                seen_emails.add(em)
                rows.append({"page_url": r.page_url, "email": em})

        out_emails_csv = os.path.join(out_dir, "emails.csv")
        with open(out_emails_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["page_url", "email"])
            writer.writeheader()
            writer.writerows(rows)

        print(f"[OK] Guardado JSON: {out_json}")
        print(f"[OK] Guardado CSV emails: {out_emails_csv}")

    finally:
        driver.quit()


# ====== SERP (Bing) -> Click en resultados -> extracción de emails ======
def navegar_serp_bing_y_extraer_emails(domain: str, query: str = None, max_results: int = 10) -> List[str]:
    """
    Abre Bing, busca site:domain con términos de contacto y recorre hasta max_results resultados
    del mismo dominio, extrayendo emails en cada página. Devuelve una lista de emails únicos.

    Guarda capturas en out/navegado_{domain} para visibilidad del recorrido.
    """
    NAV_OUT_DIR = os.path.join(OUT_DIR, f"navegado_{domain.replace('.', '_')}")
    os.makedirs(NAV_OUT_DIR, exist_ok=True)

    if not query:
        query = f"site:{domain} email OR contact OR staff OR directory OR team @{domain}"

    driver = build_driver()
    wait = WebDriverWait(driver, 20)
    emails: List[str] = []
    seen_links = set()

    def same_site(href: str) -> bool:
        try:
            h = urlparse(href).netloc.lower().lstrip("www.")
            base = domain.lower().lstrip("www.")
            return h == base or h.endswith("." + base)
        except:
            return False

    try:
        q = quote_plus(query)
        serp_url = f"https://www.bing.com/search?q={q}"
        print(f"[SERP] Abriendo: {serp_url}")
        driver.get(serp_url)
        accept_cookies_if_any(driver, wait)
        human_pause(0.8, 1.4)
        safe_screenshot(driver, NAV_OUT_DIR, "bing_serp")

        # recolectar enlaces de resultados
        links = []
        try:
            cards = driver.find_elements(By.CSS_SELECTOR, "li.b_algo h2 a, li.b_algo a[href]")
            for a in cards:
                try:
                    href = a.get_attribute("href") or ""
                    if not href: continue
                    if not same_site(href):
                        continue
                    if href in seen_links:
                        continue
                    seen_links.add(href)
                    links.append(href)
                    if len(links) >= max_results:
                        break
                except Exception:
                    continue
        except Exception:
            pass

        print(f"[SERP] Enlaces filtrados para {domain}: {len(links)}")
        # visitar cada enlace
        idx = 1
        for href in links:
            label = f"result_{idx}"
            idx += 1
            try:
                driver.execute_script("window.open('', '_self');")
                driver.get(href)
            except WebDriverException:
                try:
                    driver.switch_to.active_element.send_keys(Keys.CONTROL, "l")
                    driver.switch_to.active_element.send_keys(href + "\n")
                except:
                    continue

            try:
                accept_cookies_if_any(driver, wait)
            except:
                pass
            try:
                wait.until(lambda d: d.execute_script("return document.readyState") == "complete")
            except TimeoutException:
                pass
            scroll_page(driver)
            safe_screenshot(driver, NAV_OUT_DIR, label)

            html = driver.page_source
            soup = BeautifulSoup(html, "html.parser")
            page_emails = find_emails(soup)
            for e in page_emails:
                # filtrar por dominio objetivo o externos típicos
                edom = e.split("@")[-1].lower()
                if edom == domain.lower() or edom.endswith("." + domain.lower()) or edom in {
                    "gmail.com","outlook.com","hotmail.com","yahoo.com","aol.com","icloud.com","me.com","proton.me","comcast.net"
                }:
                    emails.append(e)
            human_pause()

        # dedupe
        emails = sorted(set(emails))
        # evidencia final
        with open(os.path.join(NAV_OUT_DIR, "emails.json"), "w", encoding="utf-8") as f:
            json.dump(emails, f, ensure_ascii=False, indent=2)

        print(f"[SERP] Emails encontrados: {len(emails)}")
        return emails
    finally:
        try:
            driver.quit()
        except Exception:
            pass


# ====== CLI ======
if __name__ == "__main__":
    import sys
    import argparse
    parser = argparse.ArgumentParser(description="Navega secciones de un sitio y extrae emails/personas. También soporta lote desde etapa1_v1.json.")
    parser.add_argument("url", nargs="?", help="URL completa del sitio (ej: https://ejemplo.com)")
    parser.add_argument("--out-dir", dest="out_dir", help="Directorio de salida (por sitio)")
    # Batch desde etapa1_v1.json
    parser.add_argument("--batch-from-etapa1", dest="batch_etapa1", help="Ruta a out/etapa1_v1.json para procesar en lote")
    parser.add_argument("--only-missing-emails", action="store_true", help="Solo sitios con emails vacíos en etapa1_v1.json")
    parser.add_argument("--require-florida-ok", action="store_true", help="Solo incluir sitios con florida_ok = true")
    parser.add_argument("--max-band-score", type=int, default=None, help="Descarta sitios con band.score > valor indicado")
    parser.add_argument("--min-band-score", type=int, default=None, help="Descarta sitios con band.score < valor indicado")
    parser.add_argument("--start", type=int, default=0, help="Índice de inicio para el lote")
    parser.add_argument("--limit", type=int, default=0, help="Cantidad máxima a procesar (0 = todos)")
    parser.add_argument("--out-base", dest="out_base", default=OUT_DIR, help="Base de salida para lotes (default: out)")
    args = parser.parse_args()

    def select_base_url(domain: str) -> str:
        # Elegir un candidato razonable sin hacer requests previas
        return f"https://{domain}"

    if args.batch_etapa1:
        path = args.batch_etapa1
        try:
            with open(path, "r", encoding="utf-8") as f:
                etapa1 = json.load(f)
        except Exception as e:
            print(f"[ERR] No se pudo leer {path}: {e}")
            sys.exit(2)
        sites = etapa1.get("sites") if isinstance(etapa1, dict) else None
        if not isinstance(sites, list):
            print("[ERR] Formato inesperado en etapa1_v1.json (no 'sites')")
            sys.exit(2)

        # Filtrado similar a v2
        items = []
        for s in sites:
            if not isinstance(s, dict):
                continue
            dom = s.get("domain")
            if not dom:
                continue
            if args.require_florida_ok and not bool(s.get("florida_ok")):
                print(f"[FILTER] Skip {dom} por florida_ok = false")
                continue
            band_score = None
            band = s.get("band")
            if isinstance(band, dict):
                band_score = band.get("score")
            elif isinstance(band, (int, float)):
                band_score = band
            if args.max_band_score is not None and isinstance(band_score, (int, float)) and band_score > args.max_band_score:
                print(f"[FILTER] Skip {dom} por band.score {band_score} > {args.max_band_score}")
                continue
            if args.min_band_score is not None and isinstance(band_score, (int, float)) and band_score < args.min_band_score:
                print(f"[FILTER] Skip {dom} por band.score {band_score} < {args.min_band_score}")
                continue
            if args.only_missing_emails:
                emails = s.get("emails", [])
                if isinstance(emails, list) and len(emails) > 0:
                    continue
            items.append({"domain": dom})

        start = max(0, int(args.start or 0))
        limit = int(args.limit or 0)
        if limit > 0:
            batch = items[start:start+limit]
        else:
            batch = items[start:]

        print(f"[INFO] Lote navegar_secciones: total={len(items)}, a procesar={len(batch)}, start={start}, limit={limit}")
        processed = 0
        for it in batch:
            dom = it["domain"]
            url = select_base_url(dom)
            out_dir = os.path.join(args.out_base, f"navegado_{dom.replace('.', '_')}")
            print(f"[BATCH] Navegando {dom} → {url}")
            try:
                navegar_y_extraer(url, output_dir=out_dir)
            except KeyboardInterrupt:
                print("[WARN] Batch interrumpido por el usuario")
                break
            except Exception as e:
                print(f"[ERR] Error navegando {dom}: {e}")
            processed += 1
            time.sleep(random.uniform(0.6, 1.2))
        print(f"[INFO] Batch finalizar: procesados={processed}")
        sys.exit(0)

    # Modo URL única
    if not args.url:
        print("Uso: python navegar_secciones.py https://ejemplo.com [--out-dir OUT]")
        sys.exit(1)
    navegar_y_extraer(args.url, output_dir=args.out_dir)
