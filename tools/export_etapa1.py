import json
import pathlib
import re
from typing import Any, Dict, List, Optional, Set, Tuple
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "out"
INPUT = OUT_DIR / "etapa1_v1.json"
BUSQ_EXT = OUT_DIR / "busquedas_externas.json"
ENRIQ = OUT_DIR / "enriquecidos.json"
ENRIQ_V3 = OUT_DIR / "enriquecidov3.json"
V2_JSON = OUT_DIR / "etapa1_2_V2_V3.json"
EXPORTS_DIR = OUT_DIR / "exports"
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_XLSX = EXPORTS_DIR / "etapa1_export.xlsx"
OUTPUT_ROWS_JSON = EXPORTS_DIR / "etapa1_export_rows.json"


def load_json(path: pathlib.Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


EMAIL_RE = re.compile(r"^[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,24}$", re.I)


def _norm_email(e: Optional[str]) -> Optional[str]:
    if not isinstance(e, str):
        return None
    t = e.strip().strip(',;|')
    if not t:
        return None
    return t if EMAIL_RE.match(t) else None


def _extract_domain(value: Optional[str]) -> str:
    if not isinstance(value, str):
        return ""
    v = value.strip().lower()
    if not v:
        return ""
    # if it already looks like a bare domain
    if '://' not in v and '/' not in v and '@' not in v:
        return v.lstrip('www.')
    # strip scheme and path
    if v.startswith('http://'):
        v = v[7:]
    elif v.startswith('https://'):
        v = v[8:]
    host = v.split('/')[0]
    return host.lstrip('www.')


def _emails_from_site(site: Dict[str, Any]) -> Tuple[Set[str], Set[str]]:
    """Return (emails_web, emails_people) from etapa1 site object."""
    # emails at site level
    emails_web: Set[str] = set()
    for e in (site.get("emails") or []):
        val: Optional[str] = None
        if isinstance(e, dict):
            val = e.get("value") or e.get("email") or e.get("mail")
        elif isinstance(e, str):
            val = e
        ne = _norm_email(val)
        if ne:
            emails_web.add(ne)

    # emails from people
    emails_people: Set[str] = set()
    people = site.get("people") or []
    if isinstance(people, list):
        for p in people:
            if not isinstance(p, dict):
                continue
            ne = _norm_email(p.get("email"))
            if ne:
                emails_people.add(ne)
            # sometimes people has 'emails' list
            for pe in (p.get("emails") or []):
                ne2 = _norm_email(pe)
                if ne2:
                    emails_people.add(ne2)
    return emails_web, emails_people


def _emails_from_busquedas_externas(bext: Dict[str, Any], domain: str) -> Set[str]:
    rec = bext.get(domain) if isinstance(bext, dict) else None
    out: Set[str] = set()
    if isinstance(rec, dict):
        for e in (rec.get("emails") or []):
            ne = _norm_email(e)
            if ne:
                out.add(ne)
    return out


def _emails_from_enriquecidos(enriq: Dict[str, Any], domain: str) -> Set[str]:
    node = enriq.get(domain) if isinstance(enriq, dict) else None
    out: Set[str] = set()
    if not isinstance(node, dict):
        return out
    # contacts_enriched: preferred direct source
    ce = node.get("contacts_enriched")
    if isinstance(ce, list):
        for item in ce:
            if not isinstance(item, dict):
                continue
            sp = item.get("source_person")
            if isinstance(sp, dict):
                ne = _norm_email(sp.get("email"))
                if ne:
                    out.add(ne)
            # Recursively scan 'enrich' provider blocks (profile.email, personal_email, etc.)
            enr = item.get("enrich")
            if isinstance(enr, dict):
                out.update(_collect_emails_from_enrich(enr))
    # Optional: allow a flat list 'emails' at root if present
    root_emails = node.get("emails")
    if isinstance(root_emails, list):
        for x in root_emails:
            ne = _norm_email(x)
            if ne:
                out.add(ne)
    return out


def _collect_emails_from_enrich(obj: Any) -> Set[str]:
    """Recursively collect emails from any nested provider block.
    Looks for keys containing 'email' and gathers string/list values.
    """
    acc: Set[str] = set()
    try:
        if isinstance(obj, dict):
            for k, v in obj.items():
                lk = str(k).lower()
                if 'email' in lk:
                    if isinstance(v, str):
                        for token in re.findall(r"[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,24}", v, re.I):
                            ne = _norm_email(token)
                            if ne:
                                acc.add(ne)
                    elif isinstance(v, list):
                        for it in v:
                            if isinstance(it, str):
                                ne = _norm_email(it)
                                if ne:
                                    acc.add(ne)
                            elif isinstance(it, dict):
                                acc.update(_collect_emails_from_enrich(it))
                    elif isinstance(v, dict):
                        acc.update(_collect_emails_from_enrich(v))
                else:
                    if isinstance(v, (dict, list)):
                        acc.update(_collect_emails_from_enrich(v))
        elif isinstance(obj, list):
            for it in obj:
                acc.update(_collect_emails_from_enrich(it))
    except Exception:
        pass
    return acc


def _map_emails_from_enriq_v3(enrv3: Dict[str, Any]) -> Dict[str, Set[str]]:
    """Build domain->emails set from enriquecidov3.json (sites list)."""
    out: Dict[str, Set[str]] = {}
    if not isinstance(enrv3, dict):
        return out
    sites = enrv3.get("sites")
    if not isinstance(sites, list):
        return out
    for s in sites:
        if not isinstance(s, dict):
            continue
        domain = _extract_domain((s.get("domain") or "").strip().lower())
        if not domain:
            domain = _extract_domain((s.get("site_url") or "").strip())
        if not domain:
            continue
        emails_web, emails_people = _emails_from_site(s)
        if domain not in out:
            out[domain] = set()
        out[domain].update(emails_web)
        out[domain].update(emails_people)
    return out


def _map_emails_from_v2(v2: Dict[str, Any]) -> Dict[str, Set[str]]:
    """Build domain->emails set from etapa1_2_V2_V3.json."""
    out: Dict[str, Set[str]] = {}
    if not isinstance(v2, dict):
        return out
    for domain, node in v2.items():
        d = (domain or "").strip().lower()
        if not d:
            continue
        emails: Set[str] = set()
        if isinstance(node, dict):
            for c in (node.get("contacts") or []):
                if isinstance(c, dict):
                    ne = _norm_email(c.get("email"))
                    if ne:
                        emails.add(ne)
        if emails:
            out[d] = emails
    return out


def flatten_site(site: Dict[str, Any], bext: Dict[str, Any], enriq: Dict[str, Any], enrv3_map: Dict[str, Set[str]], v2_map: Dict[str, Set[str]]) -> Dict[str, Any]:
    domain = _extract_domain((site.get("domain") or "").strip())
    site_url = (site.get("site_url") or "").strip()
    if not domain:
        domain = _extract_domain(site_url)
    name = (
        (site.get("source_csv") or {}).get("row", {}).get("Nombre")
        if isinstance(site.get("source_csv"), dict)
        else None
    ) or site.get("site_name") or domain

    # score (band.score) if present
    band = site.get("band") or {}
    score = band.get("score") if isinstance(band, dict) else None
    # emails grouped by source
    emails_web, emails_people = _emails_from_site(site)
    emails_ext = _emails_from_busquedas_externas(bext, domain)
    emails_enrq = _emails_from_enriquecidos(enriq, domain)
    emails_v3  = enrv3_map.get(domain, set())
    emails_v2  = v2_map.get(domain, set())
    emails_c2  = _emails_from_etapa2_cache(domain)

    emails_all = sorted(set().union(emails_web, emails_people, emails_ext, emails_enrq, emails_v3, emails_v2, emails_c2))

    people = site.get("people") or []
    if not isinstance(people, list):
        people = []

    return {
        "domain": domain,
        "site_url": site_url,
        "nombre": name,
        "score": score,
        "emails_web": ", ".join(sorted(emails_web)),
        "emails_people": ", ".join(sorted(emails_people)),
        "emails_externos": ", ".join(sorted(emails_ext)),
        "emails_enriquecidos": ", ".join(sorted(emails_enrq)),
        "emails_v2": ", ".join(sorted(emails_v2)),
        "emails_enriquecidov3": ", ".join(sorted(emails_v3)),
        "emails_etapa2_cache": ", ".join(sorted(emails_c2)),
        "emails_todos": ", ".join(emails_all),
        "count_web": len(emails_web),
        "count_people": len(emails_people),
        "count_externos": len(emails_ext),
        "count_enriquecidos": len(emails_enrq),
        "count_v2": len(emails_v2),
        "count_enriquecidov3": len(emails_v3),
        "count_etapa2_cache": len(emails_c2),
        "count_total": len(emails_all),
        "people_count": len(people),
    }


def _emails_from_etapa2_cache(domain: str, max_files: int = 200) -> Set[str]:
    emails: Set[str] = set()
    if not domain:
        return emails
    cache_dir = OUT_DIR / "etapa2_cache"
    if not cache_dir.exists():
        return emails
    key = domain.lower()
    count = 0
    pattern = re.compile(r"[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,24}", re.I)
    for fp in cache_dir.rglob('*'):
        if not fp.is_file():
            continue
        name = fp.name.lower()
        if key not in name:
            continue
        try:
            if count >= max_files:
                break
            try:
                text = fp.read_text(encoding='utf-8', errors='ignore')
            except Exception:
                text = fp.read_bytes().decode('utf-8', errors='ignore')
            for m in pattern.findall(text or ''):
                ne = _norm_email(m)
                if ne:
                    emails.add(ne)
            count += 1
        except Exception:
            continue
    return emails


def build_frames(data: Dict[str, Any], bext: Dict[str, Any], enriq: Dict[str, Any], enrv3: Dict[str, Any], v2: Dict[str, Any]):
    sites = data.get("sites") or []
    enrv3_map = _map_emails_from_enriq_v3(enrv3)
    v2_map = _map_emails_from_v2(v2)
    rows = [flatten_site(s, bext, enriq, enrv3_map, v2_map) for s in sites if isinstance(s, dict)]
    df_all = pd.DataFrame(rows)
    if not df_all.empty:
        # sort by score desc then domain
        if "score" in df_all.columns:
            df_all = df_all.sort_values(by=["score", "domain"], ascending=[False, True])
        else:
            df_all = df_all.sort_values(by=["domain"]) 
    df_with_emails = df_all[df_all["count_total"] > 0].copy() if not df_all.empty else df_all.copy()
    return df_all, df_with_emails, rows


def main():
    if not INPUT.exists():
        raise SystemExit(f"No se encuentra {INPUT}")
    data = load_json(INPUT)
    bext = load_json(BUSQ_EXT) if BUSQ_EXT.exists() else {}
    enriq = load_json(ENRIQ) if ENRIQ.exists() else {}
    enrv3 = load_json(ENRIQ_V3) if ENRIQ_V3.exists() else {}
    v2 = load_json(V2_JSON) if V2_JSON.exists() else {}
    if not isinstance(data, dict) or not isinstance(data.get("sites"), list):
        raise SystemExit("Formato inesperado de etapa1_v1.json")

    df_all, df_with_emails, rows = build_frames(data, bext, enriq, enrv3, v2)

    with pd.ExcelWriter(OUTPUT_XLSX, engine="xlsxwriter") as writer:
        df_all.to_excel(writer, index=False, sheet_name="Todos")
        df_with_emails.to_excel(writer, index=False, sheet_name="conMails")

    # dump rows as json for later comparisons
    with OUTPUT_ROWS_JSON.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print(f"Exportado a {OUTPUT_XLSX}")


if __name__ == "__main__":
    main()
