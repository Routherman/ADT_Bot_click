#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Re-run external search for domains that are empty in out/busquedas_externas.json
(empty = emails=[], socials=[], names=[], search_text={}).
Forces USE_PUPPETEER=1 to try headless SERP path first.

Usage (PowerShell):
  python tools/rerun_empty_busquedas_externas.py --input out/busquedas_externas.json --limit 50

Options:
  --input     Path to busquedas_externas.json (default out/busquedas_externas.json)
  --start     Start index within the empty list (default 0)
  --limit     Max domains to process (0 = all)
  --only      Comma-separated domain list to process (overrides empty-scan)
  --dry-run   Do not write file, just print what would be updated
  --no-puppeteer  Do not force USE_PUPPETEER
"""

from __future__ import annotations
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from nav_busqueda_externa_v2 import process_domain  # type: ignore


def is_empty_entry(entry: Dict[str, Any]) -> bool:
    if not isinstance(entry, dict):
        return False
    emails = entry.get('emails') or []
    names = entry.get('names') or []
    socials = entry.get('socials') or []
    stx = entry.get('search_text') or {}
    return len(emails) == 0 and len(names) == 0 and len(socials) == 0 and (not stx)


def upsert_domain(obj: Dict[str, Any], domain: str, site_data: Dict[str, Any]) -> Dict[str, Any]:
    # Extract from site_data['contacts']
    contacts = site_data.get('contacts') or []
    emails = sorted({(c or {}).get('email', '').strip().lower() for c in contacts if (c or {}).get('email')})
    names = sorted({(c or {}).get('name', '').strip() for c in contacts if (c or {}).get('name')})
    payload = {
        'emails': emails,
        'socials': [],  # reserved for future
        'names': names,
        'search_text': {}
    }
    cur = obj.get(domain) or {'emails': [], 'socials': [], 'names': [], 'search_text': {}}
    cur['emails'] = sorted(set((cur.get('emails') or []) + payload['emails']))
    cur['socials'] = sorted(set((cur.get('socials') or []) + payload['socials']))
    cur['names'] = sorted(set((cur.get('names') or []) + payload['names']))
    stx = cur.get('search_text') or {}
    stx.update(payload.get('search_text') or {})
    cur['search_text'] = stx
    obj[domain] = cur
    return obj


def main():
    ap = argparse.ArgumentParser(description='Re-run external search for empty entries in busquedas_externas.json')
    ap.add_argument('--input', dest='input_path', default=str(PROJECT_ROOT / 'out' / 'busquedas_externas.json'))
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--only', type=str, default='')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--no-puppeteer', action='store_true')
    args = ap.parse_args()

    # Force Puppeteer unless explicitly disabled
    if not args.no_puppeteer:
        os.environ['USE_PUPPETEER'] = os.getenv('USE_PUPPETEER', '1') or '1'
        # keep default PUPPETEER_HEADLESS=1

    in_path = Path(args.input_path)
    if not in_path.exists():
        print(f"ERROR: No existe el archivo {in_path}")
        sys.exit(1)

    try:
        data = json.loads(in_path.read_text('utf-8'))
    except Exception as e:
        print(f"ERROR: No se pudo leer JSON {in_path}: {e}")
        sys.exit(1)

    # Build target list
    domains: List[str] = []
    if args.only.strip():
        domains = [d.strip() for d in args.only.split(',') if d.strip()]
    else:
        for dom, entry in (data.items() if isinstance(data, dict) else []):
            if is_empty_entry(entry):
                domains.append(dom)

    if not domains:
        print('No hay dominios vacíos para reprocesar.')
        return

    start = max(0, int(args.start or 0))
    limit = int(args.limit or 0)
    if limit > 0:
        work = domains[start:start+limit]
    else:
        work = domains[start:]

    print(f"A procesar: {len(work)} (de {len(domains)} vacíos encontrados)\n")

    processed = 0
    updated = 0
    errors: List[str] = []

    for i, dom in enumerate(work, 1):
        print('-'*72)
        print(f"[{i}/{len(work)}] {dom}")
        try:
            site_name = dom.split('.')[0].title()
            site_data = process_domain(dom, site_name=site_name, city=None)
            # Tomar snapshot profundo de la entrada antes de modificar (evita comparar la misma referencia)
            before = data.get(dom)
            try:
                before_snapshot = json.loads(json.dumps(before)) if before is not None else None
            except Exception:
                before_snapshot = None
            data = upsert_domain(data, dom, site_data)
            after = data.get(dom)
            changed = (before_snapshot != after)
            if changed:
                updated += 1
                if not args.dry_run:
                    in_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), 'utf-8')
                    print('  -> Actualizado en busquedas_externas.json')
                else:
                    print('  -> (dry-run) Se actualizaría entrada')
            else:
                print('  -> Sin cambios')
        except KeyboardInterrupt:
            print('Interrumpido por el usuario.')
            break
        except Exception as e:
            emsg = f"ERROR {dom}: {e}"
            print('  ' + emsg)
            errors.append(emsg)
        processed += 1
        time.sleep(random.uniform(0.8, 1.6))

    print('\nResumen:')
    print(f"  Procesados: {processed}")
    print(f"  Actualizados: {updated}")
    if errors:
        print(f"  Errores: {len(errors)}")
        for e in errors[:5]:
            print('   - ' + e)


if __name__ == '__main__':
    main()
