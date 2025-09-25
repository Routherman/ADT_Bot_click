import os
import json
import re
from datetime import datetime
from typing import Any, Dict, List, Tuple

try:
    import pandas as pd
except Exception:
    pd = None  # fallback to CSV only

ROOT = os.path.dirname(__file__)
OUT_DIR = os.path.join(ROOT, "out")
EXPORTS_DIR = os.path.join(OUT_DIR, "exports")
V3_PATH = os.path.join(OUT_DIR, "enriquecidov3.json")
RAW_PATH = os.path.join(OUT_DIR, "enriquecidos.json")
EXT_A = os.path.join(OUT_DIR, "busquedas_externas.json")
EXT_B = os.path.join(OUT_DIR, "busqueda_externa.json")


def _load_json(path: str) -> Any:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_domain_from_str(s: str) -> str:
    if not s:
        return ""
    v = str(s).strip()
    v = re.sub(r"^mailto:", "", v, flags=re.IGNORECASE)
    v = v.replace("https://", "").replace("http://", "").replace("ftp://", "")
    v = v.split("/")[0].split("?")[0].split("#")[0]
    return v.strip().lower()


def _domain_variants(d: str) -> List[str]:
    n = (d or "").strip().lower()
    if not n:
        return []
    if n.startswith("www."):
        bare = n[4:]
        return [n, bare]
    return [n, f"www.{n}"]


def _collect_external_maps() -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    emails_map: Dict[str, set] = {}
    names_map: Dict[str, set] = {}
    for path in (EXT_A, EXT_B):
        try:
            data = _load_json(path)
            if not isinstance(data, dict):
                continue
            for dom, payload in data.items():
                if not isinstance(payload, dict):
                    continue
                emails_set = set()
                for key in ["emails", "emails_found", "found_emails", "site_emails", "emails_web"]:
                    vals = payload.get(key)
                    if isinstance(vals, list):
                        for item in vals:
                            if isinstance(item, str):
                                v = item.strip()
                                if v:
                                    emails_set.add(v)
                            elif isinstance(item, dict):
                                v = (item.get("email") or item.get("value") or item.get("addr") or "").strip()
                                if v:
                                    emails_set.add(v)
                names_set = set()
                vals_n = payload.get("names")
                if isinstance(vals_n, list):
                    for item in vals_n:
                        if isinstance(item, str):
                            v = item.strip()
                            if v:
                                names_set.add(v)
                        elif isinstance(item, dict):
                            v = (item.get("name") or item.get("value") or "").strip()
                            if v:
                                names_set.add(v)
                if emails_set:
                    for dv in _domain_variants(dom):
                        emails_map.setdefault(dv, set()).update(emails_set)
                if names_set:
                    for dv in _domain_variants(dom):
                        names_map.setdefault(dv, set()).update(names_set)
        except Exception:
            continue
    return (
        {d: sorted(list(es)) for d, es in emails_map.items()},
        {d: sorted(list(ns)) for d, ns in names_map.items()},
    )


def _collect_external_values(map_by_domain: Dict[str, List[str]], record_domain: str) -> List[str]:
    out: List[str] = []
    seen = set()
    # exact variants
    for dv in _domain_variants(record_domain):
        for val in map_by_domain.get(dv, []):
            lv = (val or "").strip()
            if not lv:
                continue
            lvn = lv.lower()
            if lvn in seen:
                continue
            seen.add(lvn)
            out.append(lv)
    # subdomains
    suf = f".{(record_domain or '').strip().lower()}"
    for dom_key, vals in map_by_domain.items():
        dk = (dom_key or "").strip().lower()
        if dk.endswith(suf):
            for v in vals:
                lv = (v or "").strip()
                if not lv:
                    continue
                lvn = lv.lower()
                if lvn in seen:
                    continue
                seen.add(lvn)
                out.append(lv)
    return out


def _get_enriched_emails_from_raw(raw_data: Dict[str, Any], domain: str) -> List[str]:
    site = (raw_data or {}).get(domain) or {}
    out: List[str] = []
    seen = set()
    for it in site.get("contacts_enriched") or []:
        if not isinstance(it, dict):
            continue
        enrich = it.get("enrich") or {}
        # ContactOut
        for key in ("contactout_people", "contactout_linkedin"):
            co = enrich.get(key)
            if isinstance(co, dict):
                status = co.get("status_code")
                profile = co.get("profile") if status == 200 else None
                if isinstance(profile, dict):
                    emails_raw: List[str] = []
                    for fld in ("email", "personal_email", "work_email"):
                        vals = profile.get(fld)
                        if isinstance(vals, list):
                            emails_raw.extend([v for v in vals if isinstance(v, str)])
                        elif isinstance(vals, str):
                            emails_raw.append(vals)
                    for e in emails_raw:
                        ee = (e or "").strip()
                        if ee and "@" in ee and ee.lower() not in seen:
                            seen.add(ee.lower()); out.append(ee)
        # Lusha person / RocketReach (simplificado)
        lp = enrich.get("lusha_person")
        if isinstance(lp, dict):
            candidates = []
            if isinstance(lp.get("data"), list):
                candidates = lp["data"]
            elif isinstance(lp.get("data"), dict):
                candidates = [lp["data"]]
            elif isinstance(lp, list):
                candidates = lp
            else:
                candidates = [lp]
            for c in candidates:
                if not isinstance(c, dict):
                    continue
                for k in ("email","workEmail","emailAddress"):
                    v = c.get(k)
                    if isinstance(v, str) and "@" in v:
                        ee = v.strip()
                        if ee.lower() not in seen:
                            seen.add(ee.lower()); out.append(ee)
        rr = enrich.get("rocketreach")
        if isinstance(rr, dict):
            items = rr.get("results") or rr.get("data") or rr.get("profiles") or []
            for it2 in (items or []):
                if not isinstance(it2, dict):
                    continue
                for k in ("email","work_email","current_work_email"):
                    v = it2.get(k)
                    if isinstance(v, str) and "@" in v:
                        ee = v.strip()
                        if ee.lower() not in seen:
                            seen.add(ee.lower()); out.append(ee)
    return out


def _get_social(rec: Dict[str, Any], plat: str) -> str:
    socials = (rec.get("socials") or [])
    for s in socials:
        if isinstance(s, dict):
            p = (s.get("platform") or "").lower()
            url = (s.get("url") or s.get("link") or "").strip()
            if plat in p or plat in url.lower():
                return url
    return ""


def _get_mails_todos(rec: Dict[str, Any], ext_emails_map: Dict[str, List[str]], raw_data: Dict[str, Any]) -> str:
    domain = (rec.get("domain") or _extract_domain_from_str(rec.get("site_url")) or "").strip()
    emails = []
    # base emails
    for e in (rec.get("emails") or []):
        if isinstance(e, str):
            emails.append(e)
        elif isinstance(e, dict):
            v = e.get("value") or e.get("email")
            if v:
                emails.append(v)
    # from people
    for p in (rec.get("people") or []):
        v = (p.get("email") or "").strip()
        if v:
            emails.append(v)
    # emails_web saved in v3
    for e in (rec.get("emails_web") or []):
        emails.append(e)
    # external
    if domain:
        emails += _collect_external_values(ext_emails_map, domain)
        # enriched raw
        emails += _get_enriched_emails_from_raw(raw_data, domain)
    # dedupe
    out = []
    seen = set()
    for e in emails:
        ee = (e or "").strip()
        if not ee or "@" not in ee:
            continue
        eel = ee.lower()
        if eel in seen:
            continue
        seen.add(eel)
        out.append(ee)
    return ", ".join(out)


def build_rows(v3_sites: List[Dict[str, Any]], ext_emails_map: Dict[str, List[str]], raw_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for r in v3_sites:
        try:
            band = (r.get("band") or {}).get("score")
            score = int(band) if band is not None else 0
        except Exception:
            score = 0
        nombre = r.get("site_name") or r.get("nombre") or r.get("domain") or (r.get("site_url") or "").replace("https://"," ").replace("http://"," ")
        web = r.get("domain") or _extract_domain_from_str(r.get("site_url"))
        mails_todos = _get_mails_todos(r, ext_emails_map, raw_data)
        facebook = _get_social(r, "facebook")
        linkedin = _get_social(r, "linkedin")
        # redes extra
        redes = []
        for plat in ["instagram","youtube","tiktok","x","twitter","bandcamp","soundcloud"]:
            u = _get_social(r, plat)
            if u:
                redes.append(f"{plat}: {u}")
        direcciones = "; ".join([(a.get("value") or "").strip() for a in (r.get("addresses") or []) if isinstance(a, dict) and a.get("value")])
        telefonos = ", ".join([(p.get("value") or "").strip() for p in (r.get("phones") or []) if isinstance(p, dict) and p.get("value")])
        rows.append({
            "Score": score,
            "Nombre": nombre or "",
            "web": web or "",
            "mails_todos": mails_todos,
            "red_social_facebook": facebook,
            "red_social_linkedin": linkedin,
            "redes": "; ".join(redes),
            "direcciones": direcciones,
            "telefonos": telefonos,
        })
    return rows


def main():
    os.makedirs(EXPORTS_DIR, exist_ok=True)
    v3 = _load_json(V3_PATH)
    if not (isinstance(v3, dict) and isinstance(v3.get("sites"), list)):
        print(f"[ERROR] No se pudo leer sitios desde {V3_PATH}")
        return 2
    v3_sites = v3["sites"]
    raw = _load_json(RAW_PATH) or {}
    ext_emails_map, _names_map = _collect_external_maps()
    rows = build_rows(v3_sites, ext_emails_map, raw)
    # Ordenar por score desc
    rows.sort(key=lambda x: x.get("Score", 0), reverse=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"baseFL_Todos_{ts}"
    xlsx_todos = os.path.join(EXPORTS_DIR, base_name + ".xlsx")
    xlsx_conmails = os.path.join(EXPORTS_DIR, base_name.replace("Todos","conMails") + ".xlsx")
    csv_todos = os.path.join(EXPORTS_DIR, base_name + ".csv")
    csv_conmails = os.path.join(EXPORTS_DIR, base_name.replace("Todos","conMails") + ".csv")

    df = None
    try:
        df = pd.DataFrame(rows) if pd is not None else None
    except Exception:
        df = None

    if df is not None:
        try:
            df_conmails = df[df["mails_todos"].str.strip() != ""]
            df.to_excel(xlsx_todos, index=False)
            df_conmails.to_excel(xlsx_conmails, index=False)
            print(f"[OK] Excel exportado: {xlsx_todos} ({len(df)} registros)")
            print(f"[OK] Excel exportado: {xlsx_conmails} ({len(df_conmails)} con mails)")
            return 0
        except Exception as ex:
            print(f"[WARN] No se pudo exportar Excel ({ex}). Exportando CSV...")
    # CSV fallback
    import csv
    with open(csv_todos, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    rows_mail = [r for r in rows if (r.get("mails_todos") or "").strip()]
    with open(csv_conmails, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        for r in rows_mail:
            writer.writerow(r)
    print(f"[OK] CSV exportado: {csv_todos} ({len(rows)} registros)")
    print(f"[OK] CSV exportado: {csv_conmails} ({len(rows_mail)} con mails)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
