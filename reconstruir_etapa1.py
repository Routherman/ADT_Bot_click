import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).parent
OUT_DIR = ROOT / "out"
STATUS_LOG = OUT_DIR / "status_log.json"
CACHE_DIR = OUT_DIR / "etapa1_cache"
ETAPA1_JSON = OUT_DIR / "etapa1_v1.json"
RECON_JSON = OUT_DIR / "etapa1_recon.json"


EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+\-]+@[a-zA-Z0-9\-]+\.[a-zA-Z0-9\-.]+")


def clean_email(email: str) -> str:
    if not email:
        return ""
    e = email.strip().lower()
    # strip common wrappers
    e = re.sub(r"^mailto:", "", e)
    e = e.replace("(at)", "@").replace("[at]", "@").replace("{at}", "@").replace("\u0040", "@")
    # remove trailing punctuation
    e = e.strip(" ,.;:|/\\<>\"'()[]{}")
    # basic sanity
    return e if EMAIL_RE.fullmatch(e or "") else ""


def uniq_keep_order(seq):
    seen = set()
    out = []
    for x in seq:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def load_json(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


# Ponderación por zonas para el score de "band_hits"
ZONE_WEIGHTS = {
    "title": 5.0,
    "menu": 2.0,
    "body": 1.0,
    "footer": 0.5,
}


def score_page_band_hits(band_hits) -> float:
    """
    Calcula un puntaje para una página a partir de su objeto band_hits.
    band_hits tiene forma:
      {
        "keyword": {
          "total": int,
          "zones": { "title": int, "body": int, "menu": int, "footer": int },
          "pages": [ ... ]
        },
        ...
      }
    """
    if not isinstance(band_hits, dict):
        return 0.0

    page_score = 0.0
    for _, info in band_hits.items():
        if not isinstance(info, dict):
            continue
        zones = info.get("zones")
        total = info.get("total") or 0
        if isinstance(zones, dict) and zones:
            zsum = 0.0
            for zname, cnt in zones.items():
                if cnt is None:
                    continue
                w = ZONE_WEIGHTS.get(str(zname).lower(), 1.0)
                try:
                    zsum += w * float(cnt)
                except Exception:
                    pass
            page_score += zsum
        else:
            # Si no hay detalle por zonas, usar el total plano
            try:
                page_score += float(total)
            except Exception:
                pass
    return page_score


def gather_from_cache(cache_path: Path):
    data = load_json(cache_path)
    if not isinstance(data, dict):
        return {}

    # Expected keys in cache file
    site_url = data.get("site_url") or data.get("url")
    domain = data.get("domain") or (site_url or "").split("//")[-1].split("/")[0]
    per_page = data.get("per_page", {})

    emails = []
    phones = []
    socials = []
    band_score_total = 0.0
    band_pages = 0
    for pinfo in per_page.values():
        emails.extend(pinfo.get("emails", []) if isinstance(pinfo, dict) else [])
        phones.extend(pinfo.get("phones", []) if isinstance(pinfo, dict) else [])
        socials.extend(pinfo.get("socials", []) if isinstance(pinfo, dict) else [])
        # Calcular score por página desde band_hits
        if isinstance(pinfo, dict) and "band_hits" in pinfo:
            ps = score_page_band_hits(pinfo.get("band_hits"))
            if ps > 0:
                band_pages += 1
                band_score_total += ps

    # Clean emails
    emails = [clean_email(e) for e in emails]
    emails = [e for e in emails if e]
    emails = uniq_keep_order(emails)

    socials = uniq_keep_order(socials)
    phones = uniq_keep_order(phones)

    people = data.get("people") or []
    if isinstance(people, dict):
        people = [people]

    # Si el cache ya trae un band_score y el nuestro es 0, conservar el existente,
    # pero preferimos el calculado desde band_hits cuando esté disponible.
    cache_band_score = data.get("band_score")
    final_band_score = int(round(band_score_total)) if band_score_total > 0 else (cache_band_score if cache_band_score is not None else None)

    pages_scanned = data.get("pages_scanned")
    if pages_scanned is None and isinstance(per_page, dict):
        pages_scanned = len(per_page)

    return {
        "site_url": site_url,
        "domain": domain,
        "emails": emails,
        "phones": phones,
        "socials": socials,
        "people": people,
        "band_score": final_band_score,
        "band_pages": band_pages,
        "pages_scanned": pages_scanned,
        "last_updated": data.get("last_updated"),
        "source": "cache",
    }


def reconstruct(min_band_score: int = None, include_non_florida: bool = False, limit: int = None, restrict_to_etapa1: bool = False):
    status = load_json(STATUS_LOG)
    if not status or not isinstance(status, dict):
        raise SystemExit(f"No válido o no encontrado: {STATUS_LOG}")

    domains = status.get("domains") or {}
    if not isinstance(domains, dict):
        raise SystemExit("Estructura inesperada en status_log.json (falta 'domains')")

    # Si se requiere restringir a los dominios actuales de etapa1_v1.json
    allowed_domains = None  # mantener orden
    if restrict_to_etapa1 and ETAPA1_JSON.exists():
        etapa1 = load_json(ETAPA1_JSON) or {}
        sites = etapa1.get("sites") or []
        seen = set()
        ordered = []
        for s in sites:
            if not isinstance(s, dict):
                continue
            d = (s.get("domain") or "").lower()
            if not d or d in seen:
                continue
            seen.add(d)
            ordered.append(d)
        allowed_domains = ordered

    items = []
    # Dominios a procesar
    def candidate_domains():
        if allowed_domains is not None:
            # Mantener orden idéntico al etapa1_v1.json
            for d in allowed_domains:
                yield d
        else:
            for d in domains.keys():
                yield d

    for domain in candidate_domains():
        meta = domains.get(domain, {})
        if not isinstance(meta, dict):
            meta = {}
        florida_ok = meta.get("florida_ok")
        # Si estamos restringiendo a etapa1, no filtrar por Florida para mantener el mismo universo
        if not restrict_to_etapa1:
            if not include_non_florida and florida_ok is False:
                continue

        cache_path = meta.get("cache_path")
        cache_file = None
        if cache_path:
            cache_file = ROOT / cache_path if not os.path.isabs(cache_path) else Path(cache_path)
        # Fallback: cache por nombre de dominio
        if not cache_file or not cache_file.exists():
            guess = CACHE_DIR / f"{domain}.json"
            if guess.exists():
                cache_file = guess
        if not cache_file or not cache_file.exists():
            continue

        item = gather_from_cache(cache_file)
        if not item:
            continue

        # augment with meta
        item["domain"] = (item.get("domain") or domain).lower()
        item["site_category"] = meta.get("site_category")
        item["florida_ok"] = florida_ok
        item["cache_path"] = str(cache_file.relative_to(ROOT)) if str(cache_file).startswith(str(ROOT)) else str(cache_file)
        item["status_last_updated"] = meta.get("last_updated")

        # Filtrar por band_score luego de calcularlo desde el cache
        # Si estamos restringiendo a etapa1, no filtrar por band_score para preservar el recuento
        if not restrict_to_etapa1:
            if min_band_score is not None:
                bs = item.get("band_score")
                if bs is None or bs < min_band_score:
                    continue

        items.append(item)

        if limit and len(items) >= limit:
            break

    # Deduplicate by domain, keep latest updated
    items_by_domain = {}
    def parse_dt(s):
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00")) if s else datetime.min
        except Exception:
            return datetime.min

    for it in items:
        d = (it.get("domain") or "").lower()
        prev = items_by_domain.get(d)
        if not prev:
            items_by_domain[d] = it
        else:
            if parse_dt(it.get("last_updated")) >= parse_dt(prev.get("last_updated")):
                items_by_domain[d] = it

    sites = list(items_by_domain.values())
    sites.sort(key=lambda x: (x.get("band_score") or 0, x.get("domain") or ""), reverse=True)

    return {
        "version": 3,
        "source": "reconstructed_from_status_log_and_cache",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "filters": {
            "min_band_score": min_band_score,
            "include_non_florida": include_non_florida,
            "limit": limit,
        },
        "total": len(sites),
        "sites": sites,
    }


def main():
    parser = argparse.ArgumentParser(description="Reconstruye etapa1 desde status_log y etapa1_cache")
    parser.add_argument("--min-band-score", type=int, default=None, help="Filtrar por band_score mínimo")
    parser.add_argument("--include-non-florida", action="store_true", help="Incluir dominios con florida_ok == False")
    parser.add_argument("--limit", type=int, default=None, help="Limitar cantidad de sitios procesados")
    parser.add_argument("--dry-run", action="store_true", help="No escribir archivos, solo reportar")
    parser.add_argument("--replace-etapa1", action="store_true", help="Respaldar etapa1_v1.json y reemplazarlo con la reconstrucción")
    parser.add_argument("--restrict-to-etapa1", action="store_true", help="Solo incluir dominios presentes en etapa1_v1.json actual")

    args = parser.parse_args()

    recon = reconstruct(
        min_band_score=args.min_band_score,
        include_non_florida=args.include_non_florida,
        limit=args.limit,
        restrict_to_etapa1=args.restrict_to_etapa1,
    )

    print(f"Candidatos reconstruidos: {recon['total']}")
    sample = recon["sites"][:5]
    for i, s in enumerate(sample, 1):
        print(f" {i}. {s.get('domain')} emails={len(s.get('emails', []))} band={s.get('band_score')} florida={s.get('florida_ok')}")

    if args.dry_run:
        return

    # Write recon file
    RECON_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(RECON_JSON, "w", encoding="utf-8") as f:
        json.dump(recon, f, ensure_ascii=False, indent=2)
    print(f"Escrito: {RECON_JSON}")

    if args.replace_etapa1:
        # Backup existing etapa1
        if ETAPA1_JSON.exists():
            backup_path = ETAPA1_JSON.with_suffix(".bak.json")
            os.replace(ETAPA1_JSON, backup_path)
            print(f"Respaldo creado: {backup_path}")
        # Create etapa1 structure similar to original if needed
        etapa1_payload = {"version": 3, "sites": recon["sites"]}
        with open(ETAPA1_JSON, "w", encoding="utf-8") as f:
            json.dump(etapa1_payload, f, ensure_ascii=False, indent=2)
        print(f"Reemplazado: {ETAPA1_JSON}")


if __name__ == "__main__":
    main()
