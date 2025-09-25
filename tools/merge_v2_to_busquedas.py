import os
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "out"
V2_PATH = OUT / "etapa1_2_V2_V3.json"
BUSQ_PATH = OUT / "busquedas_externas.json"

def load_json(p: Path):
    if not p.exists():
        return {}
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_json(p: Path, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def extract_from_v2(domain: str, v2: dict):
    entry = v2.get(domain) or {}
    contacts = entry.get("contacts") or []
    emails = []
    names = []
    socials = []
    for c in contacts:
        em = (c or {}).get("email")
        if em:
            emails.append(em.strip().lower())
        nm = (c or {}).get("name")
        if nm:
            names.append(nm.strip())
        # collect social URLs from sources if present
        for r in (c or {}).get("roles") or []:
            for src in (r or {}).get("sources") or []:
                u = (src or {}).get("url")
                if isinstance(u, str) and ("instagram.com" in u or "facebook.com" in u or "linkedin.com" in u or "x.com" in u or "twitter.com" in u or "youtube.com" in u):
                    socials.append(u)
    return {
        "emails": sorted({e for e in emails if e}),
        "names": sorted({n for n in names if n}),
        "socials": sorted({s for s in socials if s}),
        "search_text": {}
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: merge_v2_to_busquedas.py <domain>")
        sys.exit(1)
    domain = sys.argv[1].strip().lower()

    v2 = load_json(V2_PATH)
    if domain not in v2:
        print(f"Domain {domain} not found in {V2_PATH}")
        sys.exit(2)
    payload = extract_from_v2(domain, v2)

    data = load_json(BUSQ_PATH)
    before = data.get(domain)
    data[domain] = {
        "emails": sorted({*(before or {}).get("emails", []), *payload["emails"]}),
        "socials": sorted({*(before or {}).get("socials", []), *payload["socials"]}),
        "names": sorted({*(before or {}).get("names", []), *payload["names"]}),
        "search_text": (before or {}).get("search_text") or {}
    }
    # backup
    bak = BUSQ_PATH.with_suffix(".bak.json")
    if BUSQ_PATH.exists():
        try:
            bak.write_text(BUSQ_PATH.read_text(encoding="utf-8"), encoding="utf-8")
        except Exception:
            pass
    save_json(BUSQ_PATH, data)
    print(f"Merged {domain} into {BUSQ_PATH}")

if __name__ == "__main__":
    main()
