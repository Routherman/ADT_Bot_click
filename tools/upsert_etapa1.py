import json
import sys
import os
from pathlib import Path
from datetime import datetime

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "out"
ETAPA1 = OUT / "etapa1_v1.json"

def load_json_array(path: Path):
    if not path.exists():
        return {"version": 3, "sites": []}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def upsert(domain: str, url: str, email: str | None = None, city: str | None = None):
    data = load_json_array(ETAPA1)
    sites = data.get("sites") or []
    # find existing
    idx = None
    for i, s in enumerate(sites):
        if isinstance(s, dict) and s.get("domain") == domain:
            idx = i
            break
    now = datetime.now().isoformat(timespec="seconds")
    rec = {
        "site_url": url,
        "domain": domain,
        "florida_ok": True,
        "band": {"score": 10, "unique_keywords": 0, "weighted_hits": 0, "per_keyword": {}},
        "emails": ([{"value": email, "pages": []}] if email else []),
        "phones": [],
        "addresses": ([{"value": "", "city": city, "state": "FL", "zip": "", "pages": []}] if city else []),
        "socials": [],
        "people": [],
        "last_updated": now,
        "pages_scanned": 0
    }
    if idx is None:
        sites.append(rec)
    else:
        # merge shallow
        cur = sites[idx]
        cur.update({k: v for k, v in rec.items() if v})
        sites[idx] = cur
    data["sites"] = sites
    save_json(ETAPA1, data)
    print(f"Upserted {domain} into {ETAPA1}")

def main():
    if len(sys.argv) < 3:
        print("Usage: upsert_etapa1.py <domain> <url> [email] [city]")
        sys.exit(1)
    domain = sys.argv[1].strip().lower()
    url = sys.argv[2].strip()
    email = sys.argv[3].strip() if len(sys.argv) >= 4 else None
    city = sys.argv[4].strip() if len(sys.argv) >= 5 else None
    upsert(domain, url, email, city)

if __name__ == "__main__":
    main()
