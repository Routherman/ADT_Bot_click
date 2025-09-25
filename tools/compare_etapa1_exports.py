import json
import pathlib
from typing import Dict, Any, List, Tuple

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "out" / "exports"
NEW_JSON = OUT_DIR / "exportacion_etapa1_*.json"  # we will glob


def load_latest_json() -> pathlib.Path:
    files = sorted(OUT_DIR.glob("exportacion_etapa1_*.json"))
    if not files:
        raise SystemExit("No exportacion_etapa1_*.json found in out/exports")
    return files[-1]


def index_by_domain(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    idx = {}
    for r in rows:
        # try to infer domain from WEB or socials if domain not present
        web = (r.get("WEB") or "").strip()
        dom = web.replace("https://", "").replace("http://", "").split("/")[0].lower() if web else ""
        idx[dom] = r
    return idx


def main():
    latest = load_latest_json()
    with latest.open("r", encoding="utf-8") as f:
        old = json.load(f)
    old_rows = old.get("data") or []

    rows_path = OUT_DIR / ".." / ".." / "out" / "exports" / "etapa1_export_rows.json"
    # adjust: rows JSON lives at out/exports/etapa1_export_rows.json
    rows_path = ROOT / "out" / "exports" / "etapa1_export_rows.json"
    if not rows_path.exists():
        raise SystemExit("etapa1_export_rows.json not found. Run tools/export_etapa1.py first.")
    with rows_path.open("r", encoding="utf-8") as f:
        new_rows = json.load(f)

    # Compare coverage by domain (approx via WEB url hostname)
    old_idx = index_by_domain(old_rows)

    completed = []
    for rec in new_rows:
        dom = (rec.get("domain") or "").lower()
        if not dom:
            continue
        # consider "completo" if emails_todos present and non-empty OR emails_Web in old was empty
        count = rec.get("count_total") or 0
        old_dom = old_idx.get(dom)
        completed.append({
            "domain": dom,
            "new_emails": rec.get("emails_todos"),
            "count_total": count,
            "was_present_in_old": bool(old_dom)
        })

    # Print a small summary
    have_new = sum(1 for c in completed if (c.get("count_total") or 0) > 0)
    print(f"Domains with emails now: {have_new} / {len(completed)}")
    # Save a diff snapshot
    diff_path = OUT_DIR / "compare_diff.json"
    with diff_path.open("w", encoding="utf-8") as f:
        json.dump({"completed": completed}, f, ensure_ascii=False, indent=2)
    print(f"Diff saved to {diff_path}")


if __name__ == "__main__":
    main()
