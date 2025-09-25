import json
import os
from pathlib import Path
import csv

OUT_DIR = Path("out")
INPUT_JSON = OUT_DIR / "busquedas_externas.json"
EXPORT_DIR = OUT_DIR / "exports"
EXPORT_DIR.mkdir(exist_ok=True, parents=True)

def load_data():
    if not INPUT_JSON.exists():
        return {}
    try:
        with open(INPUT_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def build_rows(data: dict):
    rows = []
    for domain, obj in (data or {}).items():
        emails = obj.get("emails") or []
        names = obj.get("names") or []
        rows.append({
            "domain": domain,
            "emails_count": len(emails),
            "names_count": len(names),
            "emails": "; ".join(sorted(set(emails))),
            "names": "; ".join(sorted(set(names))),
        })
    return rows

def save_csv(rows):
    p = EXPORT_DIR / "busquedas_externas_resumen.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["domain", "emails_count", "names_count", "emails", "names"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return p

def save_json(rows):
    p = EXPORT_DIR / "busquedas_externas_resumen.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    return p

def save_md(rows):
    p = EXPORT_DIR / "busquedas_externas_resumen.md"
    lines = ["# Resumen busquedas_externas\n", "\n"]
    total = len(rows)
    avg_emails = (sum(r["emails_count"] for r in rows) / total) if total else 0.0
    avg_names = (sum(r["names_count"] for r in rows) / total) if total else 0.0
    lines += [
        f"- dominios: {total}\n",
        f"- promedio emails/dom: {avg_emails:.2f}\n",
        f"- promedio names/dom: {avg_names:.2f}\n",
        "\n",
        "| domain | emails_count | names_count |\n",
        "|---|---:|---:|\n",
    ]
    for r in rows[:100]:
        lines.append(f"| {r['domain']} | {r['emails_count']} | {r['names_count']} |\n")
    with open(p, "w", encoding="utf-8") as f:
        f.writelines(lines)
    return p

def main():
    data = load_data()
    rows = build_rows(data)
    csv_p = save_csv(rows)
    json_p = save_json(rows)
    md_p = save_md(rows)
    print("Reporte generado:")
    print(" -", csv_p)
    print(" -", json_p)
    print(" -", md_p)

if __name__ == "__main__":
    main()
