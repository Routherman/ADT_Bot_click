#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Batch runner to test enrichment providers (using test_enrichers functions)
for the first N domains with band.score == 100 from etapa1 JSON.

Usage (PowerShell):
  python tools/run_enrich_batch.py --from out/etapa1_v1.json --count 5 --providers contactout rocketreach
"""

import argparse
import json
import re
from pathlib import Path
from typing import List

from dotenv import load_dotenv

import sys
sys.path.insert(0, str(Path('.').resolve()))

try:
    import test_enrichers as te
except Exception as e:
    print("ERROR: No se pudo importar test_enrichers.py:", e)
    sys.exit(1)


def extract_top_domains(etapa_path: Path, count: int) -> List[str]:
    obj = json.loads(etapa_path.read_text('utf-8'))
    res: List[str] = []
    for s in obj.get('sites', []):
        band = s.get('band') or {}
        if band.get('score') == 100:
            dom = s.get('domain')
            if not dom:
                url = s.get('site_url', '')
                dom = re.sub(r'^https?://', '', url).split('/')[0].lower()
                if dom.startswith('www.'):
                    dom = dom[4:]
            if dom:
                res.append(dom)
        if len(res) >= count:
            break
    return res


def main():
    ap = argparse.ArgumentParser(description='Batch enrichment tests from etapa1 JSON (score==100)')
    ap.add_argument('--from', dest='from_path', required=True, help='Path to etapa1 JSON (e.g., out/etapa1_v1.json)')
    ap.add_argument('--count', type=int, default=5, help='How many domains to take (default 5)')
    ap.add_argument('--providers', nargs='+', default=['contactout', 'rocketreach'], choices=['contactout', 'rocketreach', 'lusha'], help='Providers to test')
    ap.add_argument('--force', action='store_true', help='Force ignore cache')
    args = ap.parse_args()

    etapa_path = Path(args.from_path)
    if not etapa_path.exists():
        print(f"ERROR: No existe el archivo {etapa_path}")
        sys.exit(1)

    load_dotenv()
    domains = extract_top_domains(etapa_path, args.count)
    if not domains:
        print("No se encontraron dominios con score 100.")
        sys.exit(0)

    print(f"Dominios (score=100) seleccionados ({len(domains)}): {', '.join(domains)}\n")

    out_dir = Path('out/enrich_tests')
    out_dir.mkdir(parents=True, exist_ok=True)

    for dom in domains:
        print(f"=== {dom} ===")
        if 'contactout' in args.providers:
            r = te.contactout_company_by_domain(dom, force=args.force, verbose=False)
            if r.get('ok'):
                data = r.get('data') or {}
                (out_dir / f"contactout_company_{dom.replace('.', '_')}.json").write_text(
                    json.dumps(data, ensure_ascii=False, indent=2), 'utf-8')
                print("  ContactOut: OK (raw guardado)")
            else:
                print("  ContactOut: ERROR ->", r.get('error'))
        if 'rocketreach' in args.providers:
            r = te.rocketreach_company_by_domain(dom, force=args.force, verbose=False)
            if r.get('ok'):
                data = r.get('data') or {}
                (out_dir / f"rocketreach_company_{dom.replace('.', '_')}.json").write_text(
                    json.dumps(data, ensure_ascii=False, indent=2), 'utf-8')
                print("  RocketReach: OK (raw guardado)")
            else:
                print("  RocketReach: ERROR ->", r.get('error'))
        if 'lusha' in args.providers:
            r = te.lusha_company_by_domain(dom, force=args.force, verbose=False)
            if r.get('ok'):
                data = r.get('data') or {}
                (out_dir / f"lusha_company_{dom.replace('.', '_')}.json").write_text(
                    json.dumps(data, ensure_ascii=False, indent=2), 'utf-8')
                print("  Lusha: OK (raw guardado)")
            else:
                print("  Lusha: ERROR ->", r.get('error'))
        print()


if __name__ == '__main__':
    main()
