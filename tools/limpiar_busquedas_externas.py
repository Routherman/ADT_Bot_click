#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Limpia correos inválidos de out/busquedas_externas.json usando reglas simples:
- TLD debe ser alfabético de 2 a 10 chars
- Debe haber al menos una letra en el dominio
- Se descartan dominios que parecen teléfonos (e.g., 352-637-4424)
Uso:
  python tools/limpiar_busquedas_externas.py --input out/busquedas_externas.json --only cccourthouse.org
"""
import argparse, json, re
from pathlib import Path

def is_plausible_domain(d: str) -> bool:
    try:
        d = (d or '').lower().strip().strip('.')
        if '.' not in d:
            return False
        labels = d.split('.')
        tld = labels[-1]
        if not (2 <= len(tld) <= 10 and tld.isalpha()):
            return False
        if not any(any(ch.isalpha() for ch in lab) for lab in labels):
            return False
        phone_like = re.compile(r"^\d{3,}(-\d{2,})+\d*$")
        for lab in labels:
            if phone_like.match(lab):
                return False
        return True
    except Exception:
        return False

def clean_emails(emails):
    seen = set(); out = []
    for e in emails or []:
        e = (e or '').strip().lower()
        if not e or '@' not in e: continue
        dom = e.split('@')[-1]
        if not is_plausible_domain(dom):
            continue
        if e not in seen:
            seen.add(e); out.append(e)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='out/busquedas_externas.json')
    ap.add_argument('--only', default='')
    args = ap.parse_args()

    p = Path(args.input)
    data = json.loads(p.read_text('utf-8'))

    targets = []
    if args.only:
        targets = [d.strip() for d in args.only.split(',') if d.strip()]
    else:
        targets = list(data.keys())

    updated = 0
    for dom in targets:
        node = data.get(dom)
        if not isinstance(node, dict):
            continue
        before = list(node.get('emails') or [])
        after = clean_emails(before)
        if after != before:
            node['emails'] = after
            data[dom] = node
            updated += 1
            print(f"  - {dom}: {len(before)} -> {len(after)} emails")

    if updated:
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), 'utf-8')
        print(f"Guardado. Entradas actualizadas: {updated}")
    else:
        print("Nada para actualizar.")

if __name__ == '__main__':
    main()
