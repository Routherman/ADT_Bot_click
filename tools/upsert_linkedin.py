import json
import math
import sys
from pathlib import Path

INPUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('out/etapa1_v1.json')
DOMAIN = sys.argv[2] if len(sys.argv) > 2 else 'plazaliveorlando.org'
URL = sys.argv[3] if len(sys.argv) > 3 else 'https://www.linkedin.com/company/the-plaza-live/'


def replace_nans(obj):
    """Recursively replace float('nan') and 'NaN' strings with None so JSON is valid."""
    if isinstance(obj, dict):
        return {k: replace_nans(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [replace_nans(x) for x in obj]
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, str) and obj.strip().lower() == 'nan':
        return None
    return obj


def main():
    # Load even if file contains NaN literals (Python json supports them by default)
    data = json.loads(INPUT.read_text(encoding='utf-8'))

    sites = data.get('sites', [])
    target = next((s for s in sites if s.get('domain') == DOMAIN), None)
    if not target:
        print(f'domain {DOMAIN} not found')
        return 1

    socials = target.setdefault('socials', [])
    # check existing
    if any(s.get('url') == URL for s in socials):
        print('already present')
    else:
        socials.insert(0, {
            'platform': 'linkedin',
            'url': URL,
            'pages': [target.get('site_url') or f'https://{DOMAIN}/']
        })
        print('inserted')

    # Sanitize NaNs and write back with strict JSON
    data = replace_nans(data)
    OUTPUT = INPUT
    OUTPUT.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
