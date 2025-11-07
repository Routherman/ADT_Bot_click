#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import shlex
import sqlite3
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timezone
import re
import json
import os
import sqlite3

# Cargar variables de entorno desde .env si está disponible
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# Asegurar que podamos importar desde el directorio actual
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

# Archivo donde persistimos rubros marcados como "completados" a efectos del resumen
COMPLETED_RUBROS_PATH = Path('out/rubros_completados.json')
# Archivo donde persistimos las anotaciones locales de validación/comentario por rubro
COMENTARIOS_JSON_PATH = Path('comentarios.json')

def _load_completed_rubros() -> set:
    try:
        if COMPLETED_RUBROS_PATH.exists():
            data = json.load(COMPLETED_RUBROS_PATH.open('r', encoding='utf-8'))
            if isinstance(data, list):
                return {str(x).strip() for x in data}
    except Exception:
        pass
    return set()

def _save_completed_rubros(s: set) -> None:
    try:
        COMPLETED_RUBROS_PATH.parent.mkdir(parents=True, exist_ok=True)
        json.dump(sorted(list(s)), COMPLETED_RUBROS_PATH.open('w', encoding='utf-8'), ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[WARN] No se pudo guardar rubros completados: {e}")

def _atomic_write_json(path: Path, obj: Any) -> None:
    """Escribe JSON de forma atómica (tmp + replace) para minimizar corrupción en Windows."""
    try:
        tmp = path.with_suffix(path.suffix + '.tmp')
        # Asegurar carpeta si aplica
        if path.parent and str(path.parent) not in ('', '.'):
            path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        # Reemplazo atómico best-effort en Windows
        try:
            os.replace(tmp, path)
        except Exception:
            # Fallback: eliminar destino y renombrar
            try:
                if path.exists():
                    path.unlink()
            except Exception:
                pass
            os.rename(tmp, path)
    except Exception as e:
        print(f"[WARN] No se pudo escribir {path.name}: {e}")

def _load_comentarios() -> List[Dict[str, Any]]:
    try:
        if COMENTARIOS_JSON_PATH.exists():
            data = json.load(COMENTARIOS_JSON_PATH.open('r', encoding='utf-8'))
            if isinstance(data, list):
                return data
    except Exception:
        pass
    return []

def _merge_comentarios(existing: List[Dict[str, Any]], nuevos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Fusiona entradas manteniendo las anotaciones existentes del usuario.
    - Clave de coincidencia: (rubro, id) si id no vacío, de lo contrario (rubro, dominio).
    - Si ya existe una entrada y tiene comentario no vacío o item_valido distinto de 'SI', NO se sobreescribe.
    - En caso contrario, se actualiza/añade la entrada nueva (por lo general con valores por defecto 'SI' y '').
    """
    # Construir índice existente
    index: Dict[Tuple[str, str, str], int] = {}
    def _key(ent: Dict[str, Any]) -> Tuple[str, str, str]:
        rub = str(ent.get('rubro') or '').strip().lower()
        _id = str(ent.get('id') or '').strip()
        dom = str(ent.get('dominio') or '').strip().lower()
        if _id:
            return (rub, 'id', _id)
        return (rub, 'dominio', dom)
    for i, ent in enumerate(existing):
        index[_key(ent)] = i
    # Aplicar merges
    for ent in nuevos:
        k = _key(ent)
        if k in index:
            i = index[k]
            cur = existing[i]
            # Conservar anotaciones del usuario si existen
            cur_item = str(cur.get('item_valido') or '').strip().upper()
            cur_comm = str(cur.get('comentario') or '').strip()
            if (cur_item and cur_item != 'SI') or cur_comm:
                # No sobreescribir
                continue
            # Reemplazar por defaults más recientes (no debería cambiar mucho)
            existing[i] = ent
        else:
            existing.append(ent)
            index[k] = len(existing) - 1
    return existing

def _coment_key_for(rubro: str, _id: str, dominio: str) -> Tuple[str, str, str]:
    rub = str(rubro or '').strip().lower()
    _id = str(_id or '').strip()
    dom = str(dominio or '').strip().lower()
    if _id:
        return (rub, 'id', _id)
    return (rub, 'dominio', dom)

def _build_comentarios_index(entries: List[Dict[str, Any]]) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    idx: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for ent in entries:
        rub = str(ent.get('rubro') or '').strip().lower()
        _id = str(ent.get('id') or '').strip()
        dom = str(ent.get('dominio') or '').strip().lower()
        key = _coment_key_for(rub, _id, dom)
        idx[key] = ent
    return idx

def _apply_comentarios_to_etapa1(output_path: Optional[Path] = None) -> Optional[Path]:
    """Anota los sites en out/etapa1_v1.json con item_valido/comentario si hay coincidencia.
    Escribe a out/etapa1_v1_annotated.json por defecto (para evitar sobrescribir el original).
    """
    try:
        src_path = Path('out/etapa1_v1.json')
        if not src_path.exists():
            return None
        data = json.load(src_path.open('r', encoding='utf-8'))
        sites = data.get('sites', []) if isinstance(data, dict) else []
        entries = _load_comentarios()
        idx = _build_comentarios_index(entries)
        changed = 0
        for s in sites:
            try:
                src = s.get('source_csv') or {}
                row = src.get('row') or {}
                rub = str(src.get('rubro') or '').strip().lower()
                _id = str(row.get('ID') or '')
                dom = (s.get('domain') or '').strip().lower()
                if not dom:
                    try:
                        from exportar_etapa1 import extract_domain as _extract
                        dom = (_extract(s.get('site_url') or '') or '').lstrip('www.').lower()
                    except Exception:
                        dom = ''
                k = _coment_key_for(rub, _id, dom)
                ent = idx.get(k)
                if ent:
                    s['item_valido'] = ent.get('item_valido', 'SI')
                    s['comentario'] = ent.get('comentario', '')
                    changed += 1
            except Exception:
                continue
        out_p = output_path or Path('out/etapa1_v1_annotated.json')
        out_p.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(out_p, data)
        print(f"[ETAPA1] Anotados {changed} sitios con comentarios → {out_p}")
        return out_p
    except Exception as e:
        print(f"[ETAPA1] Aviso: no se pudo aplicar comentarios a etapa1: {e}")
        return None

# =========================
# Master "mapa_sheet_ADT-ots" sync (Sheets + Keywords_map)
# =========================

MAPA_SHEET_ID_DEFAULT = "1OgyiER-zPgG-iqGBWBFzXy3LJqKJ4tOKxQXZX-t-ucU"

def _load_json_safe(path: Path) -> Any:
    try:
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        return None
    return None

def _save_json_safe(path: Path, data: Any) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def _gspread_client():
    try:
        import gspread  # type: ignore
        cred_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS','').strip() or None
        if cred_path and os.path.isfile(cred_path):
            return gspread.service_account(cred_path)
        # fallback default
        return gspread.service_account()
    except Exception as e:
        raise RuntimeError(f"gspread not available or credentials invalid: {e}")

def _ensure_worksheet(sh, title: str):
    try:
        try:
            ws = sh.worksheet(title)
            return ws
        except Exception:
            return sh.add_worksheet(title=title, rows=1000, cols=26)
    except Exception as e:
        raise RuntimeError(f"Cannot ensure worksheet {title}: {e}")

def _count_valid_usa_for_csv(etapa1_path: Path, csv_path: Path) -> int:
    try:
        if not etapa1_path.exists():
            return 0
        data = json.load(etapa1_path.open('r', encoding='utf-8'))
        sites = data.get('sites') or []
        target = str(csv_path.resolve())
        cnt = 0
        for s in sites:
            sc = s.get('source_csv') or {}
            f = (sc.get('file') or '')
            # normalize windows paths
            f_norm = str(Path(str(f)).resolve()) if f else ''
            if f_norm == target:
                if bool(s.get('florida_ok')):
                    cnt += 1
        return cnt
    except Exception:
        return 0

def _get_rubro_sheet_url(rubro_title: str) -> Optional[str]:
    mapping_path = Path('out/sheets_rubros_map.json')
    mp = _load_json_safe(mapping_path) or {}
    rec = mp.get(rubro_title)
    if isinstance(rec, dict):
        sid = rec.get('spreadsheet_id') or rec.get('id')
        if sid:
            return f"https://docs.google.com/spreadsheets/d/{sid}/edit"
    return None

def _get_rubro_sheet_row_count(rubro_title: str) -> int:
    try:
        mapping_path = Path('out/sheets_rubros_map.json')
        mp = _load_json_safe(mapping_path) or {}
        rec = mp.get(rubro_title)
        sid = None
        if isinstance(rec, dict):
            sid = rec.get('spreadsheet_id') or rec.get('id')
            ws_name = rec.get('worksheet') or rubro_title
        else:
            ws_name = rubro_title
        if not sid:
            return 0
        gc = _gspread_client()
        sh = gc.open_by_key(sid)
        # Assume single worksheet named as rubro title
        try:
            ws = sh.worksheet(ws_name)
        except Exception:
            ws = sh.sheet1
        values = ws.get_all_values()
        return max(0, len(values)-1) if values else 0
    except Exception:
        return 0

def sync_master_mapa_sheet() -> None:
    """Update the master spreadsheet: Sheet1 'Sheets' with rubro links and counts, and 'Keywords_map' with config map.
    Also reads back 'Keywords_map' to update configuraciones/config_map.json.
    """
    try:
        gc = _gspread_client()
        mapa_id = os.getenv('MAPA_SHEET_ID','').strip() or MAPA_SHEET_ID_DEFAULT
        sh = gc.open_by_key(mapa_id)
        # Registrar mapeo de la hoja maestra
        try:
            mapping_path = Path('out/sheets_rubros_map.json')
            mp = _load_json_safe(mapping_path) or {}
            mp['__master__'] = {
                'spreadsheet_id': mapa_id,
                'url': getattr(sh, 'url', None),
                'worksheet': 'Sheets'
            }
            _save_json_safe(mapping_path, mp)
        except Exception:
            pass
        # 1) Sheets worksheet
        ws_sheets = _ensure_worksheet(sh, 'Sheets')
        header = [
            'fecha_actualizacion','rubro','cantidad de registros procesados','cantidad de registros Validos USA',
            'cantidad de registros en Maps sin procesar','cantidad de registros sin revision',
            '% cobertura correo interno','% cobertura correo externo','ultima_actualizacion_rubro'
        ]
        rows: List[List[str]] = [header]
        csv_dir = Path('csv')
        etapa1_path = Path('out/etapa1_v1.json')
        # Cargar búsquedas externas consolidando variantes (singular/plural)
        try:
            from exportar_etapa1 import load_busquedas_externas  # type: ignore
            busq_ext_data = load_busquedas_externas('out/busquedas_externas.json')
        except Exception:
            busq_ext_path = Path('out/busquedas_externas.json')
            busq_ext_data = _load_json_safe(busq_ext_path) or {}
        nowts = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00','Z')
        if csv_dir.exists():
            for csv_file in sorted(csv_dir.glob('*.csv')):
                try:
                    title = csv_file.stem
                    # progress counts
                    total_targets, processed_count, pending_count = _compute_nav_pro_pending(str(csv_file))
                    # prefer DB unique count if available
                    def _db_count_for(csv_fp: Path) -> Optional[int]:
                        try:
                            db_path = Path('out/adt_procesado.db')
                            if not db_path.exists():
                                return None
                            conn = sqlite3.connect(str(db_path))
                            cur = conn.cursor()
                            q = "SELECT COUNT(DISTINCT web) FROM procesado WHERE csv_file = ?"
                            row = cur.execute(q, (str(csv_fp),)).fetchone()
                            conn.close()
                            if row and row[0] is not None:
                                return int(row[0])
                        except Exception:
                            return None
                        return None
                    db_cnt = _db_count_for(csv_file)
                    processed_count = db_cnt if (db_cnt is not None) else processed_count
                    valid_usa = _count_valid_usa_for_csv(etapa1_path, csv_file)
                    # link to rubro sheet
                    link = _get_rubro_sheet_url(title) or title
                    # count rows in rubro sheet (excluding header)
                    sin_revision = _get_rubro_sheet_row_count(title)
                    # cobertura interna/externa + última actualización
                    cov_int = 0.0
                    cov_ext = 0.0
                    last_up = ''
                    try:
                        if etapa1_path.exists():
                            data = json.load(etapa1_path.open('r', encoding='utf-8'))
                            sites = data.get('sites') or []
                            # Usar helper robusto para mapear CSV -> subset de etapa1 con fallback por dominio
                            subset = _build_subset_for_csv_from_etapa1(sites, str(csv_file))
                            n = len(subset)
                            if n > 0:
                                # interna: presencia de emails on-site
                                with_int = 0
                                for s in subset:
                                    try:
                                        ems = s.get('emails') or []
                                        if isinstance(ems, list) and len(ems) > 0:
                                            with_int += 1
                                    except Exception:
                                        continue
                                cov_int = round(with_int / n * 100.0, 2)
                                # externa: presencia de emails en búsquedas externas por dominio
                                with_ext = 0
                                for s in subset:
                                    try:
                                        d = _extract_domain_local(s.get('domain') or s.get('site_url') or '')
                                        node = busq_ext_data.get(d) if (isinstance(busq_ext_data, dict) and d) else None
                                        emails_ext = []
                                        if isinstance(node, dict):
                                            emails_ext = [e for e in (node.get('emails') or []) if str(e).strip()]
                                        if emails_ext:
                                            with_ext += 1
                                    except Exception:
                                        continue
                                cov_ext = round(with_ext / n * 100.0, 2)
                                # última actualización
                                try:
                                    last_up = max((s.get('last_updated') or '') for s in subset if s.get('last_updated')) or ''
                                except Exception:
                                    last_up = ''
                    except Exception:
                        pass
                    # Build hyperlink formula if we have URL
                    if link.startswith('http'):
                        rubro_cell = f"=HYPERLINK(\"{link}\", \"{title}\")"
                    else:
                        rubro_cell = title
                    rows.append([
                        nowts,
                        rubro_cell,
                        str(processed_count),
                        str(valid_usa),
                        str(pending_count),
                        str(sin_revision),
                        f"{cov_int}",
                        f"{cov_ext}",
                        last_up
                    ])
                except Exception:
                    continue
        # Write all rows (clear first)
        ws_sheets.clear()
        ws_sheets.update('A1', rows)

        # 2) Keywords_map worksheet: read, merge into config_map.json, then write canonical view
        ws_kw = _ensure_worksheet(sh, 'Keywords_map')
        cfg_path = Path('configuraciones/config_map.json')
        # Try reading
        try:
            values = ws_kw.get_all_values()
        except Exception:
            values = []

        cfg = _load_json_safe(cfg_path) or {}
        parsed_ok = False
        try:
            if values and values[0] and 'rubro' in [h.strip().lower() for h in values[0]]:
                # Expect layout: special rows for defaults_dom_name, defaults, scoring; then rows of rubros
                headers = [h.strip().lower() for h in values[0]]
                colmap = {h:i for i,h in enumerate(headers)}
                def _get(row, key, default=""):
                    idx = colmap.get(key)
                    return row[idx] if (idx is not None and idx < len(row)) else default
                # Initialize structure
                new_cfg = {"$schema": cfg.get("$schema"), "version": cfg.get("version", 2)}
                new_cfg["defaults_Dom_Name"] = {"boost_keywords": [], "penalty_keywords": [], "discard_keywords": []}
                new_cfg["defaults"] = {"boost_keywords": [], "penalty_keywords": [], "discard_keywords": []}
                new_cfg["rubros"] = {}
                new_cfg["scoring"] = {"boost_delta": 10, "penalty_delta": 15, "discard_min_hits": 1}
                for row in values[1:]:
                    if not row or not any(c.strip() for c in row):
                        continue
                    kind = _get(row, 'rubro', '').strip()
                    if not kind:
                        continue
                    def _splitlist(s: str) -> List[str]:
                        s = (s or '').strip()
                        if not s:
                            return []
                        # split by | or ;
                        parts = [p.strip() for p in re.split(r"[|;]", s) if p.strip()]
                        return parts
                    if kind.lower() == 'defaults_dom_name':
                        new_cfg['defaults_Dom_Name'] = {
                            'boost_keywords': _splitlist(_get(row,'dom_boost','')),
                            'penalty_keywords': _splitlist(_get(row,'dom_penalty','')),
                            'discard_keywords': _splitlist(_get(row,'dom_discard','')),
                        }
                    elif kind.lower() == 'defaults':
                        new_cfg['defaults'] = {
                            'boost_keywords': _splitlist(_get(row,'body_boost','')),
                            'penalty_keywords': _splitlist(_get(row,'body_penalty','')),
                            'discard_keywords': _splitlist(_get(row,'body_discard','')),
                        }
                    elif kind.lower() == 'scoring':
                        try:
                            new_cfg['scoring'] = {
                                'boost_delta': int(_get(row,'boost_delta','10') or 10),
                                'penalty_delta': int(_get(row,'penalty_delta','15') or 15),
                                'discard_min_hits': int(_get(row,'discard_min_hits','1') or 1),
                            }
                        except Exception:
                            pass
                    else:
                        rk = kind.strip()
                        new_cfg['rubros'].setdefault(rk, {})
                        dn = {
                            'boost_keywords': _splitlist(_get(row,'dom_boost','')),
                            'penalty_keywords': _splitlist(_get(row,'dom_penalty','')),
                            'discard_keywords': _splitlist(_get(row,'dom_discard','')),
                        }
                        body = {
                            'boost_keywords': _splitlist(_get(row,'body_boost','')),
                            'penalty_keywords': _splitlist(_get(row,'body_penalty','')),
                            'discard_keywords': _splitlist(_get(row,'body_discard','')),
                        }
                        if any(v for v in dn.values()):
                            new_cfg['rubros'][rk]['dom_name'] = dn
                        # merge with possible existing rubro keys
                        rb = new_cfg['rubros'][rk]
                        rb['boost_keywords'] = body['boost_keywords']
                        rb['penalty_keywords'] = body['penalty_keywords']
                        rb['discard_keywords'] = body['discard_keywords']
                # keep schema/version if present
                if not new_cfg.get('$schema'):
                    new_cfg['$schema'] = cfg.get('$schema')
                _save_json_safe(cfg_path, new_cfg)
                cfg = new_cfg
                parsed_ok = True
        except Exception:
            parsed_ok = False

        # Write canonical view to Keywords_map (header + rows)
        try:
            headers = ['rubro','dom_boost','dom_penalty','dom_discard','body_boost','body_penalty','body_discard','boost_delta','penalty_delta','discard_min_hits']
            out_rows = [headers]
            # defaults rows
            def _join(lst: List[str]) -> str:
                return " | ".join(lst or [])
            ddom = (cfg.get('defaults_Dom_Name') or {})
            out_rows.append([
                'defaults_dom_name', _join(ddom.get('boost_keywords') or []), _join(ddom.get('penalty_keywords') or []), _join(ddom.get('discard_keywords') or []), '', '', '', '', '', ''
            ])
            dbody = (cfg.get('defaults') or {})
            out_rows.append([
                'defaults', '', '', '', _join(dbody.get('boost_keywords') or []), _join(dbody.get('penalty_keywords') or []), _join(dbody.get('discard_keywords') or []), '', '', ''
            ])
            scoring = (cfg.get('scoring') or {})
            out_rows.append([
                'scoring', '', '', '', '', '', '', str(scoring.get('boost_delta','')), str(scoring.get('penalty_delta','')), str(scoring.get('discard_min_hits',''))
            ])
            # rubros
            rubs = (cfg.get('rubros') or {})
            for rk in sorted(rubs.keys()):
                r = rubs.get(rk) or {}
                dn = (r.get('dom_name') or {})
                out_rows.append([
                    rk,
                    _join(dn.get('boost_keywords') or []),
                    _join(dn.get('penalty_keywords') or []),
                    _join(dn.get('discard_keywords') or []),
                    _join(r.get('boost_keywords') or []),
                    _join(r.get('penalty_keywords') or []),
                    _join(r.get('discard_keywords') or []),
                    '', '', ''
                ])
            ws_kw.clear()
            ws_kw.update('A1', out_rows)
        except Exception:
            pass
    except Exception as e:
        print(f"[MAPA] Sync error: {e}")
def _ensure_local_venv_python():
    """If a local .venv exists, re-exec this script with its python to ensure deps (gspread) are present."""
    try:
        from pathlib import Path as _P
        import os as _os, sys as _sys
        here = _P(__file__).resolve().parent
        if _os.name == 'nt':
            venv_py = here / '.venv' / 'Scripts' / 'python.exe'
        else:
            venv_py = here / '.venv' / 'bin' / 'python'
        if venv_py.exists():
            cur = _P(_sys.executable).resolve()
            if cur != venv_py.resolve():
                print(f"[ENV] Reinvocando con venv: {venv_py}")
                _os.execv(str(venv_py), [str(venv_py), __file__, *_sys.argv[1:]])
    except Exception:
        pass



def load_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def normalize_url(url: str) -> str:
    url = str(url or '').strip()
    if not url:
        return url
    url = url.lower()
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    return url


def _compute_nav_pro_pending(csv_path: str) -> Tuple[int, int, int]:
    """Calcula (total_targets, processed_count, pending_count) basado en status_log.json."""
    try:
        p = Path(csv_path)
        if not p.exists():
            return (0, 0, 0)
        import csv as _csv
        rows: List[Dict[str, Any]] = []
        with p.open('r', encoding='utf-8', errors='ignore') as f:
            reader = _csv.DictReader(f)
            for r in reader:
                rows.append(r)
        urls: List[str] = []
        for r in rows:
            for key in ("Sitio Web", "sitio web", "website", "Website", "url", "URL"):
                if key in r and r[key]:
                    urls.append(str(r[key]).strip())
                    break
        seen = set()
        targets: List[str] = []
        from urllib.parse import urlparse as _urlparse
        for raw in urls:
            raw_s = str(raw or '').strip()
            if not raw_s:
                continue
            if not raw_s.startswith(('http://','https://')):
                raw_s = 'https://' + raw_s
            try:
                pu = _urlparse(raw_s)
                dom = pu.netloc.replace('www.', '')
                if dom and dom not in seen:
                    seen.add(dom)
                    targets.append(dom)
            except Exception:
                continue
        total_targets = len(targets)
        # Cargar status_log
        status_path = Path('out/status_log.json')
        processed = set()
        if status_path.exists():
            try:
                sdata = json.load(status_path.open('r', encoding='utf-8'))
                domains_map = sdata.get('domains') or {}
                processed = set(domains_map.keys())
            except Exception:
                processed = set()
        processed_count = len([d for d in targets if d in processed])
        pending_count = total_targets - processed_count
        return (total_targets, processed_count, pending_count)
    except Exception:
        return (0, 0, 0)

def get_csv_progress_stats(csv_path: Path) -> dict:
    """Obtiene estadísticas de progreso para un CSV específico.
    Ahora el porcentaje se calcula por dominios realmente procesados (status_log/etapa1),
    no por presencia en la última exportación (que puede excluir por filtros).
    También se informa cobertura de la última export (si existe).
    """
    try:
        # Leer CSV (URLs objetivo)
        try:
            df = pd.read_csv(csv_path, encoding='utf-8')
        except Exception:
            df = pd.read_csv(csv_path, encoding='latin1')

        url_cols = ['Sitio Web', 'Website', 'URL', 'url', 'web', 'Web']
        url_col = next((col for col in url_cols if col in df.columns), None)
        if not url_col:
            return {
                'total': 0,
                'procesados': 0,
                'pendientes': 0,
                'validos_zona': 0,
                'error': 'No se encontró columna URL'
            }

        # Objetivo por dominio (evitar desajustes http/https, www, trailing slash)
        try:
            from exportar_etapa1 import extract_domain as _extract_domain  # type: ignore
        except Exception:
            def _extract_domain(u: str) -> str:
                from urllib.parse import urlparse
                u = str(u or '').strip()
                try:
                    p = urlparse(u if '://' in u else 'http://' + u)
                    host = (p.netloc or p.path).lower()
                    return host.lstrip('www.')
                except Exception:
                    return u.lower().lstrip('www/')

        url_values = [str(u) for u in df[url_col].dropna()]
        target_domains = {_extract_domain(u) for u in url_values if str(u).strip()}
        target_domains = {d for d in target_domains if d}
        total_targets = len(target_domains)

        # Identidad del rubro (para marcar "completado" a efectos del resumen)
        title = csv_path.stem  # rubro_xxx
        simple_key = title.replace('rubro_', '')
        completed_set = _load_completed_rubros()
        is_completed_override = (title in completed_set) or (simple_key in completed_set)

        # Procesados reales (status_log) por dominio
        status_path = Path('out/status_log.json')
        processed_domains = set()
        if status_path.exists():
            try:
                sdata = json.load(status_path.open('r', encoding='utf-8'))
                domains_map = sdata.get('domains') or {}
                processed_domains = {str(k).lower() for k in domains_map.keys()}
            except Exception:
                processed_domains = set()
        # Fallback a etapa1 si no hay status_log
        if not processed_domains and Path('out/etapa1_v1.json').exists():
            try:
                e1 = json.load(open('out/etapa1_v1.json', 'r', encoding='utf-8'))
                e_sites = e1.get('sites', []) if isinstance(e1, dict) else []
                for s in e_sites:
                    dom = (s.get('domain') or '').strip().lower()
                    if not dom:
                        dom = _extract_domain(s.get('site_url') or '')
                    dom = (dom or '').lstrip('www.')
                    if dom:
                        processed_domains.add(dom)
            except Exception:
                pass

        processed_in_csv = target_domains.intersection(processed_domains)
        not_completed = sorted(list(target_domains - processed_in_csv))

        # Válidos en zona: usar etapa1 (florida_ok) para los dominios procesados del CSV
        valid_in_zone = 0
        try:
            e1 = json.load(open('out/etapa1_v1.json', 'r', encoding='utf-8'))
            e_sites = e1.get('sites', []) if isinstance(e1, dict) else []
            ok_domains = set()
            for s in e_sites:
                dom = (s.get('domain') or '').strip().lower().lstrip('www.')
                if not dom:
                    dom = _extract_domain(s.get('site_url') or '')
                dom = (dom or '').lstrip('www.')
                if dom and dom in target_domains and dom in processed_in_csv and (s.get('florida_ok') is True):
                    ok_domains.add(dom)
            valid_in_zone = len(ok_domains)
        except Exception:
            valid_in_zone = 0

        # Cobertura de la última export (opcional, informativa)
        exports_dir = Path('out/exports')
        exported_in_csv = set()
        if exports_dir.exists():
            export_files = list(exports_dir.glob('exportacion_etapa1_*.json'))
            if export_files:
                latest_export = max(export_files, key=lambda x: x.stat().st_mtime)
                processed_data = load_json(str(latest_export))
                # Derivar dominio desde el campo WEB en la export
                for site in processed_data.get('data', []) or []:
                    dom = _extract_domain(site.get('WEB', ''))
                    if dom and dom in target_domains:
                        exported_in_csv.add(dom)

        # Si el rubro está marcado como completado, forzar 100% pero informar no completados
        if is_completed_override and total_targets > 0:
            return {
                'total': total_targets,
                'procesados': total_targets,
                'pendientes': 0,
                'validos_zona': valid_in_zone,
                'exportados': len(exported_in_csv),
                'pct_exportados': (len(exported_in_csv) / total_targets * 100.0) if total_targets else 0.0,
                'dominios_pendientes': not_completed,
                'urls_pendientes': not_completed,
                'no_completados': len(not_completed),
                'override_100': True
            }
        else:
            return {
                'total': total_targets,
                'procesados': len(processed_in_csv),
                'pendientes': total_targets - len(processed_in_csv),
                'validos_zona': valid_in_zone,
                'exportados': len(exported_in_csv),
                'pct_exportados': (len(exported_in_csv) / total_targets * 100.0) if total_targets else 0.0,
                'dominios_pendientes': not_completed,
                'urls_pendientes': not_completed
            }
    except Exception as e:
        return {
            'total': 0,
            'procesados': 0,
            'pendientes': 0,
            'validos_zona': 0,
            'error': str(e)
        }

def show_progress_menu() -> Optional[Tuple[str, int, List[str]]]:
    """[DEPRECATED] Usar choose_csv()."""
    return choose_csv()

def choose_csv() -> Optional[Tuple[str, int, List[str]]]:
    """Muestra el listado de CSVs (2 columnas) y permite seleccionar uno."""
    csv_dir = Path("csv")
    if not csv_dir.exists():
        print("Error: no existe directorio csv/")
        return None
    csvs = list(csv_dir.glob("*.csv"))
    if not csvs:
        print("Error: no hay archivos CSV en csv/")
        return None
    print("\n=== CSVs disponibles (2 columnas) ===")
    entries = []
    for i, csv_file in enumerate(csvs, 1):
        stats = get_csv_progress_stats(csv_file)
        rubro = csv_file.stem.replace('rubro_', '')
        pct = (stats['procesados']/stats['total']*100) if stats['total'] else 0.0
        entries.append(f"{i}. {rubro} ({pct:.1f}%)")
    max_len = min(50, max((len(e) for e in entries), default=0))
    cols = 2
    for row_idx in range(0, len(entries), cols):
        row_items = entries[row_idx:row_idx+cols]
        padded = [s.ljust(max_len) for s in row_items[:-1]] + [row_items[-1]]
        print("  " + "  ".join(padded))
    while True:
        try:
            choice = int(input("\nSeleccione un CSV por número (0 para cancelar): "))
            if choice == 0:
                return None
            if 1 <= choice <= len(csvs):
                csv_file = csvs[choice-1]
                stats = get_csv_progress_stats(csv_file)
                return (
                    str(csv_file),
                    stats['procesados'],
                    sorted(list(stats.get('urls_pendientes', [])))
                )
            print("Opción inválida")
        except ValueError:
            print("Por favor ingrese un número válido")
    return None

def print_progress_summary() -> None:
    """Imprime un resumen no interactivo de progreso para todos los CSV en csv/."""
    csv_dir = Path("csv")
    if not csv_dir.exists():
        print("Error: no existe directorio csv/")
        return

    csvs = list(csv_dir.glob("*.csv"))
    if not csvs:
        print("Error: no hay archivos CSV en csv/")
        return

    print("\n=== RESUMEN DE PROGRESO POR CSV (no interactivo) ===\n")
    # Pre-cargar comentarios para conteos por rubro
    comentarios_entries = _load_comentarios()
    comentarios_idx = _build_comentarios_index(comentarios_entries)
    for i, csv_file in enumerate(csvs, 1):
        stats = get_csv_progress_stats(csv_file)
        rubro = csv_file.stem.replace('rubro_', '')

        print(f"{i}. {rubro}")
        print(f"   Total URLs: {stats['total']}")
        pct = (stats['procesados']/stats['total']*100) if stats['total'] else 0
        print(f"   Procesados: {stats['procesados']} ({pct:.1f}%)")
        print(f"   Pendientes: {stats['pendientes']}")
        print(f"   Válidos en zona: {stats['validos_zona']}")
        # Conteos de validación/comentarios
        # Claves usan rubro en minúsculas
        rub_key = rubro.strip().lower()
        total_no = 0
        total_com = 0
        # contamos por entradas con esta clave (ignoramos ID/DOM concretos, sumamos todas)
        for k, ent in comentarios_idx.items():
            if k[0] == rub_key:
                if str(ent.get('item_valido') or '').strip().upper() == 'NO':
                    total_no += 1
                if str(ent.get('comentario') or '').strip():
                    total_com += 1
        if total_no or total_com:
            print(f"   Item_Valido=NO: {total_no}  |  Con comentario: {total_com}")
        if 'override_100' in stats:
            print(f"   No completados (forzado 100%): {stats.get('no_completados', 0)}")
        if 'error' in stats:
            print(f"   ⚠️ {stats['error']}")
        print()

def _extract_domain_local(u: str) -> str:
    """Pequeño helper robusto para extraer dominio de una URL o texto.
    Intenta usar exportar_etapa1.extract_domain si está disponible; si no, usa urlparse.
    """
    try:
        from exportar_etapa1 import extract_domain as _extract  # type: ignore
        return (_extract(u) or '').lstrip('www.').lower()
    except Exception:
        from urllib.parse import urlparse
        try:
            s = str(u or '').strip()
            if not s:
                return ''
            if '://' not in s:
                s = 'http://' + s
            p = urlparse(s)
            host = (p.netloc or p.path).lower()
            return host.lstrip('www.')
        except Exception:
            return ''

def _build_subset_for_csv_from_etapa1(etapa1_sites: List[Dict[str, Any]], csv_path: str) -> List[Dict[str, Any]]:
    """Devuelve la lista de sites de etapa1 que corresponden a un CSV dado.
    Primero intenta por coincidencia directa de source_csv.file. Si no hay/está vacío,
    hace un fallback por dominio: cruza dominios del CSV con dominios ya procesados
    en etapa1 y monta un source_csv sintético con la fila del CSV para exportación.
    """
    norm_csv_local = csv_path.replace('/', os.sep).replace('\\', os.sep)
    # 1) Coincidencias directas por source_csv.file
    subset: List[Dict[str, Any]] = []
    seen_dom: set = set()
    for s in etapa1_sites:
        try:
            src = s.get('source_csv') or {}
            if not isinstance(src, dict):
                continue
            f = (src.get('file') or '').replace('/', os.sep).replace('\\', os.sep)
            if f and (f.endswith(norm_csv_local) or f == norm_csv_local):
                subset.append(s)
                d = (s.get('domain') or '').strip().lower()
                if d:
                    seen_dom.add(d)
        except Exception:
            continue
    if subset:
        return subset
    # 2) Fallback por dominio: mapear dominios del CSV
    def _looks_like_url(val: Any) -> bool:
        try:
            s = str(val or '').strip()
            if not s:
                return False
            if '://' in s:
                return True
            # domain-like: contains a dot and no spaces, not a pure email
            if (' ' not in s) and ('.' in s) and ('@' not in s):
                return True
            return False
        except Exception:
            return False

    try:
        try:
            df = pd.read_csv(csv_path, encoding='utf-8')
        except Exception:
            df = pd.read_csv(csv_path, encoding='latin1')
    except Exception:
        # último intento: sin encabezado
        try:
            df = pd.read_csv(csv_path, header=None, encoding='utf-8', engine='python')
        except Exception:
            try:
                df = pd.read_csv(csv_path, header=None, encoding='latin1', engine='python')
            except Exception:
                return []
    url_cols = ['Sitio Web', 'Website', 'URL', 'url', 'Web']
    url_col = next((c for c in url_cols if c in df.columns), None)
    # Heurística: si no hay columna URL estándar, detectar la columna con más valores tipo URL
    if not url_col:
        try:
            best_col = None
            best_hits = -1
            for c in df.columns:
                try:
                    hits = 0
                    sample = df[c].head(1000)  # limitar para rendimiento
                    for v in sample:
                        if _looks_like_url(v):
                            hits += 1
                    if hits > best_hits:
                        best_hits = hits
                        best_col = c
                except Exception:
                    continue
            # Si encontramos una columna con al menos algunos hits, usarla
            if best_col is not None and best_hits > 0:
                url_col = best_col
        except Exception:
            pass
    if url_col is None:
        return []
    # construir índice dominio -> row(dict)
    domain_map: Dict[str, Dict[str, Any]] = {}
    for _, row in df.iterrows():
        web_val = str(row.get(url_col, '') or '').strip()
        dom = _extract_domain_local(web_val)
        if not dom:
            continue
        k = dom.lstrip('www.').lower()
        if k and k not in domain_map:
            # convertir a dict llano para evitar tipos pandas
            domain_map[k] = {str(col): (row.get(col) if not pd.isna(row.get(col)) else '') for col in df.columns}
    # 3) Tomar sites de etapa1 cuyo dominio esté en el CSV y crear source_csv sintético
    rubro_val = Path(csv_path).stem  # ej: rubro_resorts_florida
    out: List[Dict[str, Any]] = []
    seen_out: set = set()
    for s in etapa1_sites:
        d = (s.get('domain') or '').strip().lower()
        k = d.lstrip('www.') if d else ''
        # Buscar fila del CSV para este dominio
        row_dict = domain_map.get(k) if k else None
        if not row_dict:
            # Fallback: intentar por site_url incluso si k existe, por si el dominio en etapa1 difiere
            k2 = _extract_domain_local(s.get('site_url') or '')
            if k2 and not row_dict:
                row_dict = domain_map.get(k2)
        # Si no hay coincidencia en el CSV, omitir este site
        if not row_dict:
            continue
        # Key de dominio para deduplicar salida
        key_dom = k or _extract_domain_local(s.get('site_url') or '')
        if not key_dom:
            continue
        if key_dom in seen_out:
            continue
        seen_out.add(key_dom)
        s2 = dict(s)  # copia superficial para no mutar etapa1
        s2['source_csv'] = {
            'file': csv_path,
            'rubro': rubro_val,
            'row': row_dict
        }
        out.append(s2)
    return out

def _export_rubro_google_sheet(csv_path: str, subset_sites: List[Dict[str, Any]]) -> None:
    """Crea/actualiza una Google Sheet por rubro con TODO el CSV y columnas extra para procesados.
    - Archivo/título de la spreadsheet: nombre del CSV (por ej. rubro_resorts_with_live_entertainment_florida)
    - Hoja única con el mismo nombre que el archivo
    - Columnas base (del CSV): ID, Nombre, Dirección, Teléfono, Sitio Web, Ciudad
    - Columnas extra para procesados: procesado_Manual (Si/No), Venue(=rubro), puntaje_web, emails_Web, CantidadEmails, social1..6, FechaExport
    - Excluir filas cuyo dominio o emails coincidan con la hoja NIC/Exclusions (ENRICH_EXCLUDE_SHEET_ID)
    - Ubicar el archivo dentro de la carpeta de Drive SHEETS_RUBROS_FOLDER_ID (fallback al ID compartido provisto)
    Requiere tener gspread configurado con credenciales de servicio (GOOGLE_APPLICATION_CREDENTIALS).
    """
    try:
        import gspread  # type: ignore
        from exportar_etapa1 import (
            extract_domain, clean_email, is_valid_email,
            load_busquedas_externas, load_json_or_empty, collect_all_web_emails,
            _load_exclusions_cached, load_usa_exclusion_sets
        )
        from social_utils import normalize_social_url  # type: ignore
    except Exception as e:
        print(f"[GSHEET] No se pudieron importar dependencias para Google Sheets: {e}")
        return

    # Cargar CSV original
    import pandas as pd  # local alias
    try:
        try:
            df = pd.read_csv(csv_path, encoding='utf-8')
        except Exception:
            df = pd.read_csv(csv_path, encoding='latin1')
    except Exception as e:
        print(f"[GSHEET] Error leyendo CSV {csv_path}: {e}")
        return

    # Detectar columnas del CSV
    def _pick(colnames: List[str]) -> Optional[str]:
        for c in colnames:
            if c in df.columns:
                return c
        return None

    col_id = _pick(['ID','Id','id'])
    col_nombre = _pick(['Nombre','Name','NOMBRE'])
    col_direccion = _pick(['Dirección','Direccion','Address','ADDRESS'])
    col_telefono = _pick(['Teléfono','Telefono','Phone','TEL'])
    col_web = _pick(['Sitio Web','Website','URL','url','Web'])
    col_ciudad = _pick(['Ciudad','City','CIUDAD'])
    if not col_web:
        print('[GSHEET] No se encontró columna de Sitio Web/URL en el CSV; se cancela exportación a Google Sheet.')
        return

    # Mapear dominio -> site procesado (subset_sites)
    subset_map: Dict[str, Dict[str, Any]] = {}
    for s in subset_sites:
        dom = (s.get('domain') or '').strip().lower()
        if not dom:
            # fallback al site_url si hiciera falta
            dom = extract_domain(s.get('site_url') or '')
        dom = extract_domain(dom)
        if dom:
            subset_map[dom] = s

    # Cargar fuentes auxiliares para emails combinados
    busq_ext = load_busquedas_externas('out/busquedas_externas.json')
    v2 = load_json_or_empty('out/etapa1_2_V2_V3.json')
    enriq = load_json_or_empty('out/enriquecidos.json')
    enrv3 = load_json_or_empty('out/enriquecidov3.json')

    # Cargar exclusiones (NIC/USA) para descartar filas
    excl_sheet_id = os.getenv('ENRICH_EXCLUDE_SHEET_ID','').strip()
    if not excl_sheet_id:
        print('[GSHEET] Falta ENRICH_EXCLUDE_SHEET_ID en .env; no se puede aplicar filtro USA/NIC para este rubro. Aborta export.')
        return
    excl_domains: set = set()
    excl_emails: set = set()
    excl_names: set = set()
    excl_web_domains: set = set()
    try:
        # Forzar refresh del cache de exclusiones para evitar usar datos viejos
        prev_refresh = os.environ.get('EXCLUSIONS_CACHE_REFRESH')
        os.environ['EXCLUSIONS_CACHE_REFRESH'] = '1'
        cred_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS','').strip() or None
        excl_domains, excl_emails = _load_exclusions_cached(excl_sheet_id, cred_path)
        full_ex = load_usa_exclusion_sets(excl_sheet_id, cred_path)
        excl_names = full_ex.get('names', set()) or set()
        excl_web_domains = full_ex.get('web_domains', set()) or set()
        print(f"[GSHEET] Exclusiones cargadas: dominios={len(excl_domains)} emails={len(excl_emails)} nombres={len(excl_names)} webs={len(excl_web_domains)}")
        # Restaurar estado anterior de la variable si existía
        if prev_refresh is None:
            os.environ.pop('EXCLUSIONS_CACHE_REFRESH', None)
        else:
            os.environ['EXCLUSIONS_CACHE_REFRESH'] = prev_refresh
    except Exception as e:
        print(f"[GSHEET] Aviso: no se pudo cargar hoja de exclusiones: {e}")

    # Preparar filas para la hoja
    rubro_title = Path(csv_path).stem  # ej: rubro_resorts_with_live_entertainment_florida
    venue_value = rubro_title.replace('rubro_','')  # ej: resorts_with_live_entertainment_florida
    # Requisito: agregar 2 columnas al principio antes de ID
    #  - Item_Valido (por defecto "NO_Revisado"); si se coloca "NO", completar "comentario"
    #  - comentario (lista de opciones)
    header = [
        'Item_Valido','comentario',
        'ID','Nombre','Dirección','Teléfono','Sitio Web','Ciudad',
        'procesado_Manual','Venue','puntaje_web','emails_Web','emails_externos','CantidadEmails',
        'social1','social2','social3','social4','social5','social6','FechaExport'
    ]
    rows: List[List[str]] = [header]
    # Colección para comentarios.json (uno por fila exportada)
    comentarios_batch: List[Dict[str, Any]] = []
    ts = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00','Z')
    # Forzar inclusión de dominios (ej. webookem.com)
    _force_env = os.getenv('FORCE_INCLUDE_DOMAINS','').strip()
    if _force_env:
        force_include = {d.strip().lower() for d in _force_env.split(',') if d.strip()}
    else:
        force_include = {'webookem.com'}

    for _, row in df.iterrows():
        web_val = str(row.get(col_web, '') or '').strip()
        dom = extract_domain(web_val)
        if not dom:
            # fila sin dominio utilizable => incluir básica si no está excluida por dominio vacío
            dom = ''
        dom_key = dom.lstrip('www.') if dom else ''
        # Excluir por NIC si corresponde
        if dom_key:
            dk = dom_key.lower()
            if dk in excl_domains and dk not in force_include:
                continue

        site = subset_map.get(dom_key)
        procesado = site is not None
        # Solo exportar filas procesadas (Si)
        if not procesado:
            continue
        # Valores base del CSV
        val_id = str(row.get(col_id,'') if col_id else '')
        val_nombre = str(row.get(col_nombre,'') if col_nombre else '')
        val_dir = str(row.get(col_direccion,'') if col_direccion else '')
        val_tel = str(row.get(col_telefono,'') if col_telefono else '')
        val_web = web_val
        val_ciudad = str(row.get(col_ciudad,'') if col_ciudad else '')

        # Excluir por coincidencia con USA sheet por nombre/website/email
        nm_low = val_nombre.strip().lower() if val_nombre else ''
        if nm_low and nm_low in excl_names:
            continue
        if dom_key and dom_key in excl_web_domains:
            continue
        # Campos procesados
        proc_flag = 'Si' if procesado else 'No'
        venue = venue_value if procesado else ''
        puntaje = ''
        emails_comb = ''
        emails_ext = ''
        cant_emails = 0
        socials_vals = ['']*6
        if procesado and isinstance(site, dict):
            # score
            band = site.get('band') or {}
            puntaje = str(band.get('score') or site.get('band_score') or '')
            # emails combinados (web + externos)
            try:
                emw, emx = collect_all_web_emails(site, busq_ext, v2, enriq, enrv3)
            except Exception:
                # retrocompatibilidad si la función antigua devolvía string
                emw = collect_all_web_emails(site, busq_ext, v2, enriq, enrv3)
                emx = ''
            emails_comb = emw or ''
            emails_ext = emx or ''
            # Fallback si dominio forzado y sin emails: inyectar de busquedas_externas y v2
            if (not emails_comb and not emails_ext) and dom_key and dom_key.lower() in force_include:
                extras = set()
                try:
                    node = busq_ext.get(dom_key.lower()) if isinstance(busq_ext, dict) else None
                    if isinstance(node, dict):
                        for em in (node.get('emails') or []):
                            from exportar_etapa1 import clean_email, is_valid_email  # type: ignore
                            em2 = clean_email(str(em))
                            if em2 and is_valid_email(em2):
                                extras.add(em2)
                except Exception:
                    pass
                try:
                    from exportar_etapa1 import emails_from_v2_for_domain  # type: ignore
                    for em in emails_from_v2_for_domain(v2, dom_key.lower()):
                        extras.add(em)
                except Exception:
                    pass
                if extras:
                    emails_ext = ', '.join(sorted(extras))
            # Excluir por emails si alguno está en exclusiones
            if (emails_comb or emails_ext) and excl_emails and (not (dom_key and dom_key.lower() in force_include)):
                any_excl = False
                to_check = []
                to_check.extend([e.strip().lower() for e in (emails_comb.split(',') if emails_comb else []) if e.strip()])
                to_check.extend([e.strip().lower() for e in (emails_ext.split(',') if emails_ext else []) if e.strip()])
                for e in to_check:
                    if e and e in excl_emails:
                        any_excl = True
                        break
                if any_excl:
                    continue
            # Contar emails totales (internos + externos)
            cnt_in = len([e for e in (emails_comb.split(',') if emails_comb else []) if e.strip()])
            cnt_ex = len([e for e in (emails_ext.split(',') if emails_ext else []) if e.strip()])
            cant_emails = cnt_in + cnt_ex
            # socials normalizados
            raw_soc = []
            for s in (site.get('socials') or []):
                url = s.get('url') if isinstance(s, dict) else s
                if url:
                    raw_soc.append(normalize_social_url(str(url)))
            # dedup preservando orden
            seen_s = set(); ordered_s = []
            for u in raw_soc:
                if u and u not in seen_s:
                    seen_s.add(u); ordered_s.append(u)
            for i in range(min(6, len(ordered_s))):
                socials_vals[i] = ordered_s[i]

        # Si NO está procesado, aún así intentamos poblar emails_externos desde fuentes externas
        if (not procesado) and dom_key:
            try:
                node = busq_ext.get(dom_key.lower()) if isinstance(busq_ext, dict) else None
                extras: set = set()
                if isinstance(node, dict):
                    for em in (node.get('emails') or []):
                        em2 = clean_email(str(em))
                        if em2 and is_valid_email(em2):
                            extras.add(em2)
                # V2 as fallback
                try:
                    from exportar_etapa1 import emails_from_v2_for_domain  # type: ignore
                    for em in emails_from_v2_for_domain(v2, dom_key.lower()):
                        extras.add(em)
                except Exception:
                    pass
                if extras and not emails_ext:
                    emails_ext = ', '.join(sorted(extras))
                # Excluir por emails si alguno está en exclusiones (salvo forzados)
                if excl_emails and (not (dom_key and dom_key.lower() in force_include)):
                    to_check = [e.strip().lower() for e in (emails_ext.split(',') if emails_ext else []) if e.strip()]
                    if any(e in excl_emails for e in to_check):
                        continue
                # Ajustar contador si estaba en 0
                if not cant_emails:
                    cant_emails = len([e for e in (emails_ext.split(',') if emails_ext else []) if e.strip()])
            except Exception:
                pass

        # Nuevas columnas: Item_Valido (default "NO_Revisado"), comentario (vacío)
        item_valido = 'NO_Revisado'
        comentario = ''
        rows.append([
            item_valido, comentario,
            val_id, val_nombre, val_dir, val_tel, val_web, val_ciudad,
            proc_flag, venue, puntaje, emails_comb, emails_ext, str(cant_emails),
            *socials_vals, ts
        ])
        # Acumular entrada para comentarios.json
        comentarios_batch.append({
            'rubro': venue_value,
            'id': val_id,
            'dominio': dom_key,
            'item_valido': item_valido,
            'comentario': comentario
        })

    if len(rows) == 1:
        print('[GSHEET] No hay filas para subir (todas excluidas o CSV vacío).')
        return

    # Crear/abrir spreadsheet y subir
    folder_id = os.getenv('SHEETS_RUBROS_FOLDER_ID','').strip() or '15htoK4GqCW-7OqG7rgt6W946twVY9wyv'
    cred_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS','').strip() or None
    try:
        if cred_path and os.path.isfile(cred_path):
            gc = gspread.service_account(filename=cred_path)
        else:
            gc = gspread.service_account()
    except Exception as e:
        print(f"[GSHEET] Error inicializando credenciales Google: {e}")
        return

    title = rubro_title
    sh = None
    # 1) Override por ID (evita creaciones y problemas de cuota)
    sheet_id_override = None
    try:
        # 1) Archivo de mapeo
        map_file = os.getenv('SHEETS_RUBROS_MAP_FILE','').strip()
        if map_file and os.path.isfile(map_file):
            try:
                mapping = json.load(open(map_file,'r',encoding='utf-8'))
                if isinstance(mapping, dict):
                    sheet_id_override = mapping.get(title) or mapping.get(venue_value)
            except Exception:
                pass
        # 2) JSON embebido
        if not sheet_id_override:
            map_json = os.getenv('SHEETS_RUBROS_MAP_JSON','').strip()
            if map_json:
                try:
                    mapping = json.loads(map_json)
                    if isinstance(mapping, dict):
                        sheet_id_override = mapping.get(title) or mapping.get(venue_value)
                except Exception:
                    pass
        # 3) Variables simples
        sheet_id_override = sheet_id_override or os.getenv('SHEETS_RUBRO_ID') or os.getenv('RUBRO_SHEET_ID')
    except Exception:
        sheet_id_override = None

    if sheet_id_override:
        # sheet_id_override puede venir como string (ID) o dict con metadatos {spreadsheet_id, worksheet, url}
        target_ws_name = title
        sheet_key = sheet_id_override
        if isinstance(sheet_id_override, dict):
            try:
                target_ws_name = sheet_id_override.get('worksheet') or title
                sheet_key = sheet_id_override.get('spreadsheet_id') \
                            or sheet_id_override.get('id') \
                            or sheet_id_override.get('sheetId') \
                            or sheet_id_override.get('sheet_id')
            except Exception:
                sheet_key = None
        try:
            if sheet_key:
                sh = gc.open_by_key(str(sheet_key))
                print(f"[GSHEET] Usando hoja preexistente por ID: {sheet_key}")
            else:
                raise ValueError("sheet_id_override sin ID válido")
        except Exception as oe:
            print(f"[GSHEET] No se pudo abrir por ID {sheet_id_override}: {oe}")
            sh = None

    # 2) Abrir por título si no hubo override
    if sh is None:
        try:
            sh = gc.open(title)
            print(f"[GSHEET] Actualizando spreadsheet existente por título: {title}")
        except Exception:
            # 3) Intentar crear (puede fallar por cuota)
            try:
                sh = gc.create(title, folder_id=folder_id)
                print(f"[GSHEET] Creada nueva spreadsheet en carpeta destino: {title}")
            except Exception as ce:
                msg = str(ce)
                print(f"[GSHEET] Error creando spreadsheet en carpeta destino: {msg}")
                # Fallback: crear en el 'My Drive' del Service Account (sin folder_id)
                try:
                    sh = gc.create(title)
                    print(f"[GSHEET] Fallback: creada spreadsheet en el Drive del Service Account: {title}")
                    try:
                        print(f"[GSHEET] URL: {sh.url}")
                    except Exception:
                        pass
                    print("[GSHEET] Sugerencia: comparte la carpeta destino con el Service Account y/o mueve este archivo a esa carpeta.")
                except Exception as ce2:
                    print(f"[GSHEET] Fallback también falló al crear sin carpeta: {ce2}")
                    return
    try:
        # Intentar abrir la worksheet específica si fue provista en el mapeo; sino fallback a la primera
        try:
            ws = sh.worksheet(target_ws_name)
        except Exception:
            ws = sh.sheet1
        # Renombrar hoja a rubro si es posible
        try:
            ws.update_title(title)
        except Exception:
            pass
        # 1) Leer anotaciones actuales (ANTES de limpiar)
        existing_annotations: Dict[Tuple[str, str, str], Dict[str, str]] = {}
        try:
            values = ws.get_all_values()  # lista de filas
            if values and len(values) > 1:
                hdr = [h.strip() for h in (values[0] or [])]
                def _col_ix(name: str) -> Optional[int]:
                    try:
                        return hdr.index(name)
                    except Exception:
                        return None
                def _col_ix_multi(options: List[str]) -> Optional[int]:
                    for n in options:
                        i = _col_ix(n)
                        if i is not None:
                            return i
                    return None
                ix_item = _col_ix('Item_Valido')
                ix_com = _col_ix('comentario')
                ix_id = _col_ix_multi(['ID','Id','id'])
                ix_web = _col_ix_multi(['Sitio Web','Website','URL','url','Web'])
                rub_key = title.replace('rubro_','')
                ann_count = 0
                for r in values[1:]:
                    try:
                        v_id = (r[ix_id] if (ix_id is not None and ix_id < len(r)) else '').strip() if r else ''
                        v_web = (r[ix_web] if (ix_web is not None and ix_web < len(r)) else '').strip() if r else ''
                        from exportar_etapa1 import extract_domain as _extract
                        dom = _extract(v_web)
                        dom_key = (dom or '').lstrip('www.').lower()
                        k = _coment_key_for(rub_key, v_id, dom_key)
                        val_item = (r[ix_item] if (ix_item is not None and ix_item < len(r)) else '').strip().upper() if r else ''
                        val_com = (r[ix_com] if (ix_com is not None and ix_com < len(r)) else '').strip() if r else ''
                        if val_item or val_com:
                            existing_annotations[k] = {
                                'item_valido': val_item or 'SI',
                                'comentario': val_com or ''
                            }
                            ann_count += 1
                    except Exception:
                        continue
                if ann_count:
                    print(f"[GSHEET] Anotaciones previas detectadas: {ann_count}")
        except Exception:
            existing_annotations = {}
        # 2) Limpiar y preparar hoja de destino
        try:
            ws.clear()
        except Exception:
            pass
        try:
            ws.resize(rows=len(rows), cols=len(header))
        except Exception:
            pass
        # Antes de subir, aplicamos anotaciones existentes sobre las filas a enviar y pre-llenamos comentarios_batch
        try:
            rub_key = title.replace('rubro_','')
            # Rehacer filas (rows[0] es header)
            new_rows = [rows[0]]
            for r in rows[1:]:
                # r estructura:
                # [Item_Valido, comentario, ID, Nombre, Dirección, Teléfono, Sitio Web, Ciudad, ...]
                v_id = str(r[2] or '')
                v_web = str(r[6] or '')
                from exportar_etapa1 import extract_domain as _extract
                dom = _extract(v_web)
                dom_key = (dom or '').lstrip('www.').lower()
                k = _coment_key_for(rub_key, v_id, dom_key)
                ann = existing_annotations.get(k)
                if ann:
                    # Sobrescribir columnas A y B con lo que hay en la sheet
                    r[0] = (ann.get('item_valido') or 'SI').upper()
                    r[1] = ann.get('comentario') or ''
                # Actualizar batch a partir de la fila final
                comentarios_batch.append({
                    'rubro': rub_key,
                    'id': v_id,
                    'dominio': dom_key,
                    'item_valido': str(r[0] or '').upper() or 'SI',
                    'comentario': str(r[1] or '')
                })
                new_rows.append(r)
            rows = new_rows
        except Exception:
            # Si falla, seguimos con rows tal cual y dejamos comentarios_batch para el post-subida
            pass
        # Subir contenido
        ws.update(rows, value_input_option='RAW')
        # Data validation: columna A (Item_Valido) con opciones SI/NO y
        # columna B (comentario) con el listado solicitado
        try:
            sheet_id = ws.id
            total_rows = len(rows)
            dv_requests = {
                "requests": [
                    {
                        "setDataValidation": {
                            "range": {
                                "sheetId": sheet_id,
                                "startRowIndex": 1,  # después del header
                                "endRowIndex": total_rows,
                                "startColumnIndex": 0,  # Columna A
                                "endColumnIndex": 1
                            },
                            "rule": {
                                "condition": {
                                    "type": "ONE_OF_LIST",
                                    "values": [
                                        {"userEnteredValue": "SI"},
                                        {"userEnteredValue": "NO"},
                                        {"userEnteredValue": "HECHO"}
                                    ]
                                },
                                "showCustomUi": True,
                                "strict": True
                            }
                        }
                    },
                    {
                        "setDataValidation": {
                            "range": {
                                "sheetId": sheet_id,
                                "startRowIndex": 1,
                                "endRowIndex": total_rows,
                                "startColumnIndex": 1,  # Columna B (comentario)
                                "endColumnIndex": 2
                            },
                            "rule": {
                                "condition": {
                                    "type": "ONE_OF_LIST",
                                    "values": [
                                        {"userEnteredValue": "NO hace eventos de musica"},
                                        {"userEnteredValue": "Espacio reducido"},
                                        {"userEnteredValue": "No contratria banda"},
                                        {"userEnteredValue": "Listador de eventos"},
                                        {"userEnteredValue": "Venta de Tickects o Agencia"},
                                        {"userEnteredValue": "Otro o duplicado (especificar)"}
                                    ]
                                },
                                "showCustomUi": True,
                                "strict": False
                            }
                        }
                    }
                ]
            }
            sh.batch_update(dv_requests)
        except Exception as e:
            print(f"[GSHEET] Aviso: no se pudo aplicar data validation: {e}")
        print(f"[GSHEET] Subidas {len(rows)-1} filas a la hoja '{title}'.")
        # Persistir comentarios.json sincronizado: preferimos los valores que subimos (o los que leímos de la sheet)
        try:
            existing_local = _load_comentarios()
            merged = _merge_comentarios(existing_local, comentarios_batch)
            _atomic_write_json(COMENTARIOS_JSON_PATH, merged)
            print(f"[LOCAL] comentarios.json actualizado ({len(comentarios_batch)} filas, total {len(merged)} entradas)")
        except Exception as pe:
            print(f"[LOCAL] Aviso: no se pudo actualizar comentarios.json: {pe}")
        # Actualizar mapeo local de hojas por rubro para usos posteriores (pestaña 'Sheets' del master)
        try:
            mapping_path = Path('out/sheets_rubros_map.json')
            mp = _load_json_safe(mapping_path) or {}
            mp[title] = {
                'spreadsheet_id': getattr(sh, 'id', None) or getattr(sh, 'spreadsheet_id', None),
                'url': getattr(sh, 'url', None),
                'worksheet': title
            }
            _save_json_safe(mapping_path, mp)
        except Exception:
            pass
        # Sincronizar hoja maestra (Sheets + Keywords_map)
        try:
            sync_master_mapa_sheet()
        except Exception as _merr:
            print(f"[MAPA] Aviso: no se pudo sincronizar hoja maestra: {_merr}")
    except Exception as ue:
        print(f"[GSHEET] Error actualizando worksheet: {ue}")

if __name__ == "__main__":
    _ensure_local_venv_python()
    import argparse
    parser = argparse.ArgumentParser(description="Menú de progreso por CSV: interactivo o resumen no interactivo")
    parser.add_argument("--summary", action="store_true", help="Mostrar resumen no interactivo y salir")
    parser.add_argument("--no-run", action="store_true", help="No ejecutar nav_pro automáticamente tras seleccionar un CSV")
    parser.add_argument("--extra-args", type=str, default="", help="Argumentos extra a pasar a nav_pro (en una sola cadena)")
    parser.add_argument("--bulk-export-all", action="store_true", help="Procesa todos los registros existentes en etapa1_v1.json y los vuelca a SQLite + export global")
    parser.add_argument("--socials-max-len", type=int, default=1500, help="Máximo de caracteres para campo socials en exportaciones (default 1500)")
    parser.add_argument("--report-rubros", action="store_true", help="Genera un reporte agregado por rubro/csv desde SQLite")
    parser.add_argument("--emit-sheets-map", action="store_true", help="Genera un JSON plantilla (rubro -> SHEET_ID) para todas las entradas en ./csv y sale")
    parser.add_argument("--sheets-map-out", type=str, default="out/sheets_rubros_map.json", help="Ruta donde guardar el JSON plantilla (default: out/sheets_rubros_map.json)")
    parser.add_argument("--mark-rubro-done", type=str, default=None, help="Marca un rubro como completado a efectos del resumen (acepta 'rubro_xxx', 'xxx' o nombre de archivo .csv)")
    parser.add_argument("--unmark-rubro-done", type=str, default=None, help="Desmarca un rubro como completado")
    parser.add_argument("--list-rubros-done", action="store_true", help="Lista los rubros marcados como completados")
    # Todos_ADT_OTS_Expo manual sync options
    parser.add_argument("--todos-rebuild-unprocessed", action="store_true", help="Reconstruye la hoja 'sin_procesar_bot' (tab 2) en Todos_ADT_OTS_Expo")
    parser.add_argument("--todos-sync-procesados", action="store_true", help="Sincroniza procesados/discards (tabs 1 y 3) desde etapa1_v1.json")
    parser.add_argument("--filter-csv-dom-name", action="store_true", help="Filtrar con DOM_Name todos los CSV y subir 'Filtrados_sin_procesar_bot'")
    parser.add_argument("--todos-backfill-short-descrip", action="store_true", help="Rellenar Short_descrip (Tab1) para filas vacías con IA usando el cache")
    parser.add_argument("--todos-apply-domname-procesados", action="store_true", help="Mover de procesados_bot a procesados_paratanda2 si DOM_Name hit (motivo 'Filtro: <kw>')")
    parser.add_argument("--todos-force-domname-keywords", type=str, default=None, help="Forzar mover por keywords (coma-separadas) de Tab1 a Tab3")
    # Short_descrip desde cache (export/backfill)
    parser.add_argument("--shortdesc-export", type=str, default=None, help="Exportar Short_descrip desde cache a archivo (.jsonl o .csv)")
    parser.add_argument("--shortdesc-limit", type=int, default=None, help="Límite de dominios a procesar para Short_descrip")
    parser.add_argument("--shortdesc-backfill", action="store_true", help="Backfill en TAB1 de Short_descrip (solo vacíos) usando cache (~150 chars)")
    args = parser.parse_args()
    # Gestión rápida de rubros marcados como completados
    if args.list_rubros_done:
        s = _load_completed_rubros()
        if not s:
            print("[DONE] No hay rubros marcados como completados.")
        else:
            print("[DONE] Rubros completados:")
            for k in sorted(s):
                print(" -", k)
        sys.exit(0)

    # Acciones directas para sync de hoja general (no interactivo)
    if args.todos_rebuild_unprocessed or args.todos_sync_procesados or args.filter_csv_dom_name or args.todos_backfill_short_descrip or args.todos_apply_domname_procesados or args.todos_force_domname_keywords or args.shortdesc_export or args.shortdesc_backfill:
        try:
            from todos_expo_push import rebuild_unprocessed_from_csv, sync_processed_from_etapa1, rebuild_filtered_unprocessed_from_csv, backfill_short_descrip_tab1, apply_dom_name_filter_on_tab1_move_to_tab3, force_move_by_keyword_on_tab1  # type: ignore
        except Exception as e:
            print(f"[TODOS-EXPO] No se pudo importar helper: {e}")
            sys.exit(1)
        # Siempre refrescar filtros (Keywords_map → configuraciones/config_map.json) antes de cualquier sync/export
        try:
            sync_master_mapa_sheet()
            print("[MAPA] Keywords_map refrescado → config_map.json actualizado.")
        except Exception as e:
            print(f"[MAPA] Aviso: no se pudo refrescar Keywords_map antes del sync: {e}")
        # Short_descrip export/backfill vía generador offline (cache)
        if args.shortdesc_export or args.shortdesc_backfill:
            try:
                cmd_parts = [sys.executable, 'gen_short_descrip_from_cache.py']
                if args.shortdesc_export:
                    cmd_parts += ['--out', args.shortdesc_export]
                if args.shortdesc_limit is not None:
                    cmd_parts += ['--limit', str(args.shortdesc_limit)]
                if args.shortdesc_backfill:
                    cmd_parts += ['--update-tab1']
                print("[SHORTDESC] Ejecutando:", ' '.join(shlex.quote(p) for p in cmd_parts))
                res = subprocess.run(cmd_parts, check=False)
                if res.returncode != 0:
                    print(f"[SHORTDESC] Terminó con código {res.returncode}")
            except Exception as e:
                print(f"[SHORTDESC] Error ejecutando generador: {e}")
        if args.todos_rebuild_unprocessed:
            try:
                n = rebuild_unprocessed_from_csv('csv')
                print(f"[TODOS-EXPO] 'sin_procesar_bot' reconstruida: {n} filas")
            except Exception as e:
                print(f"[TODOS-EXPO] Error reconstruyendo tab 2: {e}")
        if args.todos_sync_procesados:
            try:
                a1, a3 = sync_processed_from_etapa1('out/etapa1_v1.json')
                print(f"[TODOS-EXPO] Sincronizado: procesados_bot +{a1}, procesados_paratanda2 +{a3}")
                # Actualizar pestaña 'stats' tras sincronizar tabs 1/3
                try:
                    from todos_expo_push import update_stats_tab  # type: ignore
                    res = update_stats_tab()
                    if isinstance(res, dict) and res.get('ok'):
                        print(f"[STATS] Actualizada pestaña 'stats' (rubros={res.get('rubros',0)}, total_items={res.get('total_items',0)}).")
                    else:
                        print(f"[STATS] Aviso: no se pudo actualizar stats: {res}")
                except Exception as _se:
                    print(f"[STATS] Error actualizando 'stats': {_se}")
            except Exception as e:
                print(f"[TODOS-EXPO] Error sincronizando tabs 1/3: {e}")
        if args.filter_csv_dom_name:
            try:
                n4 = rebuild_filtered_unprocessed_from_csv('csv')
                print(f"[TODOS-EXPO] 'Filtrados_sin_procesar_bot' reconstruida: {n4} filas")
            except Exception as e:
                print(f"[TODOS-EXPO] Error reconstruyendo tab 4: {e}")
        if args.todos_backfill_short_descrip:
            try:
                nbf = backfill_short_descrip_tab1()
                print(f"[TODOS-EXPO] Short_descrip backfilled: {nbf} filas")
            except Exception as e:
                print(f"[TODOS-EXPO] Error en backfill de Short_descrip: {e}")
        if args.todos_apply_domname_procesados:
            try:
                mv = apply_dom_name_filter_on_tab1_move_to_tab3()
                print(f"[TODOS-EXPO] DOM_Name aplicado a procesados: movidos {mv}")
            except Exception as e:
                print(f"[TODOS-EXPO] Error aplicando DOM_Name en procesados: {e}")
        if args.todos_force_domname_keywords:
            try:
                kws = [x.strip() for x in (args.todos_force_domname_keywords or '').split(',') if x.strip()]
                mv2 = force_move_by_keyword_on_tab1(kws)
                print(f"[TODOS-EXPO] DOM_Name force keywords: movidos {mv2}")
            except Exception as e:
                print(f"[TODOS-EXPO] Error en force keywords: {e}")
        sys.exit(0)
    if args.mark_rubro_done:
        key = args.mark_rubro_done.strip()
        # Normalizar: aceptar con o sin 'rubro_' y con o sin '.csv'
        key = key.replace('.csv','')
        s = _load_completed_rubros()
        s.add(key)
        if key.startswith('rubro_'):
            s.add(key.replace('rubro_',''))
        else:
            s.add('rubro_' + key)
        _save_completed_rubros(s)
        print(f"[DONE] Marcado como completado: {key}")
        sys.exit(0)
    if args.unmark_rubro_done:
        key = args.unmark_rubro_done.strip().replace('.csv','')
        s = _load_completed_rubros()
        removed = False
        for cand in [key, key.replace('rubro_',''), ('rubro_' + key if not key.startswith('rubro_') else key)]:
            if cand in s:
                s.discard(cand)
                removed = True
        _save_completed_rubros(s)
        print(f"[DONE] Desmarcado: {key}" + (" (no estaba)" if not removed else ""))
        sys.exit(0)

    def format_socials(socials: List[str], max_len: int) -> str:
        """Deduplica, ordena y concatena socials; trunca si excede max_len conservando URLs completas tanto como sea posible."""
        cleaned = []
        for s in socials:
            if not s:
                continue
            cs = str(s).strip()
            if not cs:
                continue
            cleaned.append(cs)
        # dedup preservando orden
        seen = set()
        ordered = []
        for u in cleaned:
            if u not in seen:
                seen.add(u)
                ordered.append(u)
        # unir por coma
        out = ",".join(ordered)
        if len(out) <= max_len:
            return out
        # truncar manteniendo inicio y agregando marcador
        trunc_marker = " ...[truncado]"
        keep = max_len - len(trunc_marker)
        if keep < 0:
            return out[:max_len]
        return out[:keep] + trunc_marker

    if args.emit_sheets_map:
        try:
            csv_dir = Path("csv")
            csvs = sorted(csv_dir.glob("*.csv")) if csv_dir.exists() else []
            mapping: Dict[str, str] = {}
            for p in csvs:
                title = p.stem  # ej: rubro_resorts_with_live_entertainment_florida
                simple = title.replace('rubro_','')
                # placeholders vacíos para que el usuario pegue los IDs
                if title not in mapping:
                    mapping[title] = ""
                if simple not in mapping:
                    mapping[simple] = ""
            out_path = Path(args.sheets_map_out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(mapping, f, ensure_ascii=False, indent=2)
            print(f"[SHEETS] Plantilla de mapeo creada: {out_path}")
            print("Configura uno de estos métodos para que el export use esas hojas por ID:")
            print(f"  1) SHEETS_RUBROS_MAP_FILE={out_path}")
            print("  2) SHEETS_RUBROS_MAP_JSON='{" + "..." + "}' (pegando el contenido del JSON)")
        except Exception as e:
            print(f"[SHEETS] Error generando plantilla de mapeo: {e}")
        sys.exit(0)

    if args.report_rubros:
        db_path = Path('out/adt_procesado.db')
        if not db_path.exists():
            print('No existe la base out/adt_procesado.db. Ejecuta primero una exportación.')
            sys.exit(1)
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        # Crear índice si no existe (best effort)
        try:
            cur.execute("CREATE INDEX IF NOT EXISTS idx_procesado_updated_at ON procesado(updated_at)")
        except Exception:
            pass
        # Asegurar columnas nuevas por compatibilidad (item_valido, comentario)
        try:
            cur.execute("SELECT item_valido, comentario FROM procesado LIMIT 1")
        except Exception:
            try:
                cur.execute("ALTER TABLE procesado ADD COLUMN item_valido TEXT")
            except Exception:
                pass
            try:
                cur.execute("ALTER TABLE procesado ADD COLUMN comentario TEXT")
            except Exception:
                pass
        # Consulta agregada
        sql = """
        SELECT 
            venue,
            csv_file,
            COUNT(*) AS total,
            AVG(puntaje) AS avg_puntaje,
            SUM(CASE WHEN (emails_web IS NOT NULL AND TRIM(emails_web) <> '') THEN 1 ELSE 0 END) AS con_emails,
            SUM(CASE WHEN (socials IS NOT NULL AND TRIM(socials) <> '') THEN 1 ELSE 0 END) AS con_socials,
            SUM(CASE WHEN (UPPER(item_valido) = 'NO') THEN 1 ELSE 0 END) AS items_no,
            SUM(CASE WHEN (comentario IS NOT NULL AND TRIM(comentario) <> '') THEN 1 ELSE 0 END) AS con_comentario,
            MAX(updated_at) AS last_update
        FROM procesado
        GROUP BY venue, csv_file
        ORDER BY venue COLLATE NOCASE
        """
        rows = cur.execute(sql).fetchall()
        conn.close()
        if not rows:
            print('No hay datos en la tabla procesado.')
            sys.exit(0)
        # Preparar datos
        report = []
        for venue, csv_file, total, avg_p, con_em, con_soc, items_no, con_com, last_up in rows:
            pct_em = (con_em / total * 100.0) if total else 0
            pct_soc = (con_soc / total * 100.0) if total else 0
            pct_no = (items_no / total * 100.0) if total else 0
            pct_com = (con_com / total * 100.0) if total else 0
            report.append({
                'venue': venue,
                'csv_file': csv_file,
                'total_registros': total,
                'avg_puntaje': avg_p or 0,
                'con_emails': con_em,
                'pct_con_emails': round(pct_em,2),
                'con_socials': con_soc,
                'pct_con_socials': round(pct_soc,2),
                'item_valido_no': items_no,
                'pct_item_valido_no': round(pct_no,2),
                'con_comentario': con_com,
                'pct_con_comentario': round(pct_com,2),
                'last_update': last_up
            })
        # Exportar
        out_dir = Path('out/reports')
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        json_path = out_dir / f'resumen_rubros_{ts}.json'
        csv_path = out_dir / f'resumen_rubros_{ts}.csv'
        try:
            json.dump({'generated_at': ts, 'total_rubros': len(report), 'data': report}, open(json_path,'w',encoding='utf-8'), ensure_ascii=False, indent=2)
            import csv as _csv
            with open(csv_path,'w',encoding='utf-8',newline='') as fcsv:
                writer = _csv.DictWriter(fcsv, fieldnames=list(report[0].keys()))
                writer.writeheader()
                writer.writerows(report)
            print(f'Reporte generado:\n  JSON: {json_path}\n  CSV:  {csv_path}\n  Rubros: {len(report)}')
        except Exception as e:
            print(f'Error escribiendo reporte: {e}')
        sys.exit(0)

    if args.bulk_export_all:
        etapa1_path = 'out/etapa1_v1.json'
        if not Path(etapa1_path).exists():
            print('No existe out/etapa1_v1.json')
            sys.exit(1)
        try:
            data = json.load(open(etapa1_path, 'r', encoding='utf-8')).get('sites', [])
        except Exception as e:
            print(f'Error leyendo etapa1: {e}')
            sys.exit(1)
        # Intentar importar helpers avanzados
        try:
            from exportar_etapa1 import collect_all_web_emails, load_busquedas_externas, load_json_or_empty
            adv = True
        except Exception:
            adv = False
        if adv:
            busq_ext = load_busquedas_externas('out/busquedas_externas.json')
            v2 = load_json_or_empty('out/etapa1_2_V2_V3.json')
            enriq = load_json_or_empty('out/enriquecidos.json')
            enrv3 = load_json_or_empty('out/enriquecidov3.json')
        else:
            busq_ext = v2 = enriq = enrv3 = {}
        # Índice de comentarios para enriquecer outputs
        comentarios_idx_all = _build_comentarios_index(_load_comentarios())
        # Filtrar sólo los que tienen source_csv
        records = []
        for item in data:
            src = item.get('source_csv') or {}
            if not isinstance(src, dict):
                continue
            row = src.get('row') or {}
            if not isinstance(row, dict):
                row = {}
            if adv:
                try:
                    emails_web = collect_all_web_emails(item, busq_ext, v2, enriq, enrv3)  # type: ignore
                except Exception:
                    emails_web = ', '.join(sorted(set(e if isinstance(e, str) else (e.get('value','') if isinstance(e, dict) else '') for e in (item.get('emails') or []))))
            else:
                emails_raw = []
                for e in (item.get('emails') or []):
                    if isinstance(e, dict):
                        emails_raw.append(e.get('value',''))
                    else:
                        emails_raw.append(str(e))
                emails_web = ', '.join(sorted(set(emails_raw)))
            socials_list = []
            for s in (item.get('socials') or []):
                url = s.get('url') if isinstance(s, dict) else s
                if url:
                    socials_list.append(str(url))
            socials_joined = format_socials(socials_list, args.socials_max_len)
            # Comentarios lookup
            rub_key = str(src.get('rubro','')).strip().lower()
            row_id = str(row.get('ID') or '')
            try:
                from exportar_etapa1 import extract_domain as _extract
                dom_key = (_extract(record.get('WEB')) or '').lstrip('www.').lower()
            except Exception:
                dom_key = ''
            ent_c = comentarios_idx_all.get(_coment_key_for(rub_key, row_id, dom_key), {})
            item_val = str(ent_c.get('item_valido') or 'SI').upper()
            comm = str(ent_c.get('comentario') or '')
            record = {
                'csv_file': src.get('file',''),
                'Nombre': row.get('Nombre', row.get('Name', '')),
                'Venue': src.get('rubro', ''),
                'WEB': row.get('Sitio Web', row.get('Website', row.get('URL', ''))),
                'puntaje_web': item.get('band_score') or (item.get('band') or {}).get('score') or 0,
                'emails_Web': emails_web,
                'socials': socials_joined,
                'Direccion_Completa': (item.get('addresses') or [{}])[0].get('value','') if item.get('addresses') else row.get('Dirección', row.get('Address','')),
                'Item_Valido': item_val,
                'comentario': comm
            }
            if record['csv_file']:
                records.append(record)
        print(f'Registros a volcar: {len(records)}')
        # Volcar a SQLite
        try:
            db_path = Path('out/adt_procesado.db')
            db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute("""
            CREATE TABLE IF NOT EXISTS procesado (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                csv_file TEXT,
                nombre TEXT,
                venue TEXT,
                web TEXT,
                puntaje REAL,
                emails_web TEXT,
                socials TEXT,
                direccion TEXT,
                item_valido TEXT,
                comentario TEXT,
                created_at TEXT,
                updated_at TEXT,
                UNIQUE(csv_file, web)
            )
            """)
            # Migración ligera: si falta columna updated_at intentar añadir
            try:
                cur.execute("SELECT updated_at FROM procesado LIMIT 1")
            except Exception:
                try:
                    cur.execute("ALTER TABLE procesado ADD COLUMN updated_at TEXT")
                except Exception:
                    pass
            now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00','Z')
            rows = [(
                r['csv_file'], r['Nombre'], r['Venue'], r['WEB'], r['puntaje_web'], r['emails_Web'], r['socials'], r['Direccion_Completa'], now, now
            ) for r in records]
            cur.executemany("""
                INSERT INTO procesado (csv_file, nombre, venue, web, puntaje, emails_web, socials, direccion, item_valido, comentario, created_at, updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(csv_file, web) DO UPDATE SET
                    nombre=excluded.nombre,
                    venue=excluded.venue,
                    puntaje=excluded.puntaje,
                    emails_web=excluded.emails_web,
                    socials=excluded.socials,
                    direccion=excluded.direccion,
                    item_valido=excluded.item_valido,
                    comentario=excluded.comentario,
                    updated_at=excluded.updated_at
            """, rows)
            conn.commit()
            conn.close()
            print(f'Upsert en SQLite completado: {len(rows)} filas (out/adt_procesado.db)')
        except Exception as e:
            print(f'Error SQLite: {e}')
        # Export global
        try:
            out_dir = Path('out/exports_sqlite')
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            excel_path = out_dir / f'export_bulk_{ts}.xlsx'
            json_path = out_dir / f'export_bulk_{ts}.json'
            df = pd.DataFrame(records)
            with pd.ExcelWriter(excel_path) as writer:
                df.to_excel(writer, sheet_name='Datos', index=False)
            json.dump({'total': len(records), 'data': df.to_dict(orient='records')}, open(json_path,'w',encoding='utf-8'), ensure_ascii=False, indent=2)
            print(f'Export global creada:\n  Excel: {excel_path}\n  JSON:  {json_path}')
            # Sincronizar hoja maestra luego de exportación global
            try:
                sync_master_mapa_sheet()
            except Exception as _e:
                print(f"[MAPA] Aviso: no se pudo sincronizar hoja maestra post export global: {_e}")
        except Exception as e:
            print(f'Error export global: {e}')
        sys.exit(0)

    if args.summary:
        print_progress_summary()
    else:
        while True:
            print("\n=== Opciones ===")
            print("1. Continuar procesamiento (nav_pro)")
            print("2. Exportar lo procesado de este CSV")
            print("3. Volver al listado de CSVs")
            print("4. Exportar desde SQLite por rango de fechas")
            print("5. Exportar Etapa1 Solo rubro (seleccionado) + hoja diferencial (Google Sheet)")
            print("6. Exportar Etapa1 GLOBAL con etiqueta USA_NIC para items ya presentes")
            print("7. Exportar Todos: Individualmente cada rubro")
            print("8. Extraer nuevos rubros desde Google Maps (Node) y sincronizar ./csv")
            print("9. Sincronizar 'Todos_ADT_OTS_Expo' (tabs 1/2/3)")
            print("10. Short_descrip desde cache (export/backfill)")
            print("0. Salir")
            act = input("Seleccione acción: ").strip()
            if act == '0':
                sys.exit(0)
            if act == '3':
                # Solo mostrar listado informativo de CSVs
                choose_csv()
                continue
            if act == '1':
                # Continuar procesamiento: primero elegir CSV
                sel = choose_csv()
                if not sel:
                    continue
                csv_path, start_idx, pending_urls = sel
                # Continuar procesamiento
                print(f"\nContinuar procesando {csv_path}")
                print(f"Inicio sugerido (desde índice): {start_idx} — Pendientes: {len(pending_urls)}")
                user_limit = None
                try:
                    raw = input("¿Cuántos procesar ahora? [Enter = todos los pendientes]: ").strip()
                    if raw:
                        val = int(raw)
                        if val > 0:
                            user_limit = val
                except Exception:
                    user_limit = None
                if args.no_run:
                    print("\nOmisión solicitada (--no-run): no se ejecutará nav_pro automáticamente.")
                    print("Sugerencia manual:")
                    cmd_preview = f"{sys.executable} nav_pro.py --csv \"{csv_path}\" --start {start_idx}"
                    if args.extra_args:
                        cmd_preview += f" {args.extra_args}"
                    if user_limit is not None:
                        cmd_preview += f" --limit {user_limit}"
                    print(cmd_preview)
                else:
                    cmd_parts = [sys.executable, "nav_pro.py", "--csv", csv_path, "--start", str(start_idx)]
                    if user_limit is not None:
                        cmd_parts += ["--limit", str(user_limit)]
                    if args.extra_args:
                        cmd_parts += shlex.split(args.extra_args)
                    # Recalcular pendientes reales según status_log
                    total_t, real_processed, real_pending = _compute_nav_pro_pending(csv_path)
                    adjusted_start = 0
                    reason_msg = None
                    # Si el start_idx>=real_pending o real_pending==0 -> dejar 0
                    if real_pending <= 0:
                        reason_msg = f"No hay pendientes reales (status_log). Procesados={real_processed} Total={total_t}."
                    else:
                        # El start que nav_pro usa es sobre la lista de pendientes, no el total original.
                        # Si el usuario ingresó un start_idx basado en 'procesados' del menú (puede estar desfasado), lo ajustamos a 0.
                        if start_idx >= real_pending or real_processed != start_idx:
                            reason_msg = (f"Ajuste start → 0 (menu procesados={start_idx}, status_log procesados={real_processed}, pendientes reales={real_pending}).")
                        else:
                            adjusted_start = start_idx
                    if reason_msg:
                        print(f"[MENU->NAV] {reason_msg}")
                    # Reemplazar parámetro --start en cmd_parts
                    for i, val in enumerate(cmd_parts):
                        if val == '--start' and i+1 < len(cmd_parts):
                            cmd_parts[i+1] = str(adjusted_start)
                            break
                    # Activar realtime append si variable global sheet está presente
                    realtime_enabled = False
                    if os.getenv('ENRICH_GLOBAL_EXPORT_SHEET_ID'):
                        os.environ['REALTIME_GLOBAL_APPEND'] = '1'
                        realtime_enabled = True
                        print('[REALTIME] REALTIME_GLOBAL_APPEND=1 activado (append incremental habilitado).')
                    print("\nLanzando procesamiento con:")
                    print(" ".join(shlex.quote(p) for p in cmd_parts))
                    try:
                        result = subprocess.run(cmd_parts, check=False)
                        if result.returncode == 0:
                            print("\nnav_pro finalizó sin errores (exit=0)")
                        else:
                            print(f"\nAdvertencia: nav_pro terminó con código {result.returncode}")
                    except Exception as e:
                        print(f"\nError al ejecutar nav_pro: {e}")
                    finally:
                        if realtime_enabled:
                            os.environ.pop('REALTIME_GLOBAL_APPEND', None)
                # luego de procesar, actualizar stats
                # fin acción 1
                continue
            if act == '2':
                # Exportar lo procesado: submenú para 1) solo este CSV o 2) todos los CSVs
                print("\nSubmenú exportación:")
                print("1. Exportar SOLO el CSV seleccionado")
                print("2. Exportar TODOS los CSVs (batch)")
                print("0. Cancelar")
                sub = input("Seleccione: ").strip()
                if sub == '0':
                    continue
                export_all = (sub == '2')
                target_csvs: List[str]
                if export_all:
                    # tomar todos los CSVs de ./csv
                    target_csvs = [str(p) for p in Path('csv').glob('*.csv')]
                    if not target_csvs:
                        print('No hay CSVs en ./csv para exportar.')
                        continue
                    print(f"Export batch de {len(target_csvs)} CSVs...")
                else:
                    sel = choose_csv()
                    if not sel:
                        continue
                    csv_path, _, _ = sel
                    target_csvs = [csv_path]
                # Cargar etapa1
                try:
                    from exportar_etapa1 import collect_all_web_emails, load_busquedas_externas, load_json_or_empty, extract_domain, is_valid_email, clean_email, normalize_social_url
                except Exception:
                    print("No se pudo importar utilidades de exportación, se exportará solo campos básicos.")
                    collect_all_web_emails = None  # type: ignore
                etapa1_path = 'out/etapa1_v1.json'
                if not Path(etapa1_path).exists():
                    print("No existe etapa1_v1.json, no se puede exportar.")
                    continue
                try:
                    etapa1_data = json.load(open(etapa1_path, 'r', encoding='utf-8')).get('sites', [])
                except Exception as e:
                    print(f"Error leyendo etapa1: {e}")
                    continue
                # Helper para exportar un solo CSV (reutiliza lógica existente)
                def export_one_csv(one_csv_path: str):
                    # Subset directo por source_csv o fallback por dominio si falta source
                    subset_local = _build_subset_for_csv_from_etapa1(etapa1_data, one_csv_path)
                    if not subset_local:
                        print(f"[EXPORT] No hay registros procesados aún para {Path(one_csv_path).name}.")
                        return
                    print(f"[EXPORT] {Path(one_csv_path).name}: {len(subset_local)} registros procesados. Generando export...")
                    out_dir_local = Path('out/exports_parciales')
                    out_dir_local.mkdir(parents=True, exist_ok=True)
                    ts_local = datetime.now().strftime('%Y%m%d_%H%M%S')
                    base_name_local = Path(one_csv_path).stem
                    excel_path_local = out_dir_local / f'export_parcial_{base_name_local}_{ts_local}.xlsx'
                    json_path_local = out_dir_local / f'export_parcial_{base_name_local}_{ts_local}.json'
                    # cargar externos si tenemos helpers
                    if collect_all_web_emails:
                        try:
                            busq_ext_local = load_busquedas_externas('out/busquedas_externas.json')  # type: ignore
                            v2_local = load_json_or_empty('out/etapa1_2_V2_V3.json')  # type: ignore
                            enriq_local = load_json_or_empty('out/enriquecidos.json')  # type: ignore
                            enrv3_local = load_json_or_empty('out/enriquecidov3.json')  # type: ignore
                        except Exception:
                            busq_ext_local = {}
                            v2_local = {}
                            enriq_local = {}
                            enrv3_local = {}
                    records_local = []
                    comentarios_idx = _build_comentarios_index(_load_comentarios())
                    for item in subset_local:
                        src = item.get('source_csv') or {}
                        if not isinstance(src, dict):
                            src = {}
                        row = src.get('row') or {}
                        if not isinstance(row, dict):
                            row = {}
                        if collect_all_web_emails:
                            try:
                                emw, emx = collect_all_web_emails(item, busq_ext_local, v2_local, enriq_local, enrv3_local)  # type: ignore
                            except Exception:
                                # retrocompatibilidad por si devuelve string
                                emw = collect_all_web_emails(item, busq_ext_local, v2_local, enriq_local, enrv3_local)  # type: ignore
                                emx = ''
                            emails_web_local = emw
                            emails_ext_local = emx
                        else:
                            emails_raw_local = []
                            for e in (item.get('emails') or []):
                                if isinstance(e, dict):
                                    emails_raw_local.append(e.get('value',''))
                                else:
                                    emails_raw_local.append(str(e))
                            emails_web_local = ', '.join(sorted(set(emails_raw_local)))
                            emails_ext_local = ''
                        socials_list_local = []
                        for s in (item.get('socials') or []):
                            url = s.get('url') if isinstance(s, dict) else s
                            if url:
                                socials_list_local.append(str(url))
                        socials_joined_local = format_socials(socials_list_local, args.socials_max_len)
                        # Comentarios lookup
                        row_id = str(row.get('ID') or '')
                        web_val_local = row.get('Sitio Web', row.get('Website', row.get('URL', '')))
                        try:
                            dom_key_local = extract_domain(web_val_local)
                        except Exception:
                            dom_key_local = ''
                        dom_key_local = (dom_key_local or '').lstrip('www.').lower()
                        rub_key_local = str(src.get('rubro', '')).strip().lower()
                        ent_c_local = comentarios_idx.get(_coment_key_for(rub_key_local, row_id, dom_key_local), {})
                        item_val_local = str(ent_c_local.get('item_valido') or 'SI').upper()
                        comm_local = str(ent_c_local.get('comentario') or '')
                        record_local = {
                            'Nombre': row.get('Nombre', row.get('Name', '')),
                            'Venue': src.get('rubro', ''),
                            'WEB': row.get('Sitio Web', row.get('Website', row.get('URL', ''))),
                            'puntaje_web': item.get('band_score') or (item.get('band') or {}).get('score') or 0,
                            'emails_Web': emails_web_local,
                            'emails_externos': emails_ext_local,
                            'socials': socials_joined_local,
                            'Direccion_Completa': (item.get('addresses') or [{}])[0].get('value','') if item.get('addresses') else row.get('Dirección', row.get('Address','')),
                            'Item_Valido': item_val_local,
                            'comentario': comm_local
                        }
                        records_local.append(record_local)
                    # Guardar Excel + JSON
                    try:
                        with pd.ExcelWriter(excel_path_local) as writer:
                            pd.DataFrame(records_local).to_excel(writer, sheet_name='Datos', index=False)
                        json.dump({'data': records_local, 'total': len(records_local)}, open(json_path_local, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
                        print(f"[EXPORT] Parcial creado:\n  Excel: {excel_path_local}\n  JSON:  {json_path_local}")
                    except Exception as e:
                        print(f"Error generando export: {e}")
                    # Persistir en SQLite
                    try:
                        db_path_local = Path('out/adt_procesado.db')
                        conn_local = sqlite3.connect(db_path_local)
                        cur_local = conn_local.cursor()
                        cur_local.execute("""
                        CREATE TABLE IF NOT EXISTS procesado (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            csv_file TEXT,
                            nombre TEXT,
                            venue TEXT,
                            web TEXT,
                            puntaje REAL,
                            emails_web TEXT,
                            socials TEXT,
                            direccion TEXT,
                            item_valido TEXT,
                            comentario TEXT,
                            created_at TEXT,
                            updated_at TEXT,
                            UNIQUE(csv_file, web)
                        )
                        """)
                        # Migración: añadir columnas que puedan faltar en tablas antiguas
                        try:
                            cur_local.execute("SELECT item_valido FROM procesado LIMIT 1")
                        except Exception:
                            try:
                                cur_local.execute("ALTER TABLE procesado ADD COLUMN item_valido TEXT")
                            except Exception:
                                pass
                        try:
                            cur_local.execute("SELECT comentario FROM procesado LIMIT 1")
                        except Exception:
                            try:
                                cur_local.execute("ALTER TABLE procesado ADD COLUMN comentario TEXT")
                            except Exception:
                                pass
                        try:
                            cur_local.execute("SELECT updated_at FROM procesado LIMIT 1")
                        except Exception:
                            try:
                                cur_local.execute("ALTER TABLE procesado ADD COLUMN updated_at TEXT")
                            except Exception:
                                pass
                        now_local = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00','Z')
                        rows_local = [(
                            one_csv_path,
                            r['Nombre'],
                            r['Venue'],
                            r['WEB'],
                            r['puntaje_web'],
                            r['emails_Web'],
                            r['socials'],
                            r['Direccion_Completa'],
                            r.get('Item_Valido','SI'),
                            r.get('comentario',''),
                            now_local,
                            now_local
                        ) for r in records_local]
                        cur_local.executemany("""
                            INSERT INTO procesado (csv_file, nombre, venue, web, puntaje, emails_web, socials, direccion, item_valido, comentario, created_at, updated_at)
                            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                            ON CONFLICT(csv_file, web) DO UPDATE SET
                                nombre=excluded.nombre,
                                venue=excluded.venue,
                                puntaje=excluded.puntaje,
                                emails_web=excluded.emails_web,
                                socials=excluded.socials,
                                direccion=excluded.direccion,
                                item_valido=excluded.item_valido,
                                comentario=excluded.comentario,
                                updated_at=excluded.updated_at
                        """, rows_local)
                        conn_local.commit()
                        conn_local.close()
                        print(f"[EXPORT] SQLite actualizado: {len(rows_local)} filas (out/adt_procesado.db)")
                    except Exception as e:
                        print(f"Error guardando en SQLite: {e}")
                    # Export adicional a Google Sheet por rubro (con TODO el CSV + campos procesados y exclusiones NIC)
                    try:
                        _export_rubro_google_sheet(one_csv_path, subset_local)
                    except Exception as e:
                        print(f"[GSHEET] Aviso: exportación Google Sheet por rubro falló ({Path(one_csv_path).stem}): {e}")
                    # Aplicar comentarios a etapa1 (archivo anotado)
                    try:
                        _apply_comentarios_to_etapa1()
                    except Exception as e:
                        print(f"[ETAPA1] Aviso: anotación post-export falló: {e}")

                # Ejecutar export para cada CSV de destino
                for one in target_csvs:
                    export_one_csv(one)
                # Anotación adicional por si hubo múltiples rubros
                try:
                    _apply_comentarios_to_etapa1()
                except Exception as e:
                    print(f"[ETAPA1] Aviso: anotación post-export (batch) falló: {e}")
                continue
            if act == '4':
                # Exportar desde SQLite por rango
                from_date = input("Fecha desde (YYYY-MM-DD) [Enter=sin filtro]: ").strip() or None
                to_date = input("Fecha hasta (YYYY-MM-DD) [Enter=sin filtro]: ").strip() or None
                csv_filter = input("Filtro que contenga en nombre de CSV (substring) [Enter=sin filtro]: ").strip() or None
                try:
                    def export_sqlite_rango(from_d: Optional[str], to_d: Optional[str], csv_sub: Optional[str]):
                        db_path = Path('out/adt_procesado.db')
                        if not db_path.exists():
                            print('No existe base SQLite todavía.')
                            return
                        conn = sqlite3.connect(db_path)
                        cur = conn.cursor()
                        where = []
                        params: List[str] = []
                        if from_d:
                            where.append('created_at >= ?')
                            params.append(from_d + 'T00:00:00')
                        if to_d:
                            where.append('created_at <= ?')
                            params.append(to_d + 'T23:59:59')
                        if csv_sub:
                            where.append('csv_file LIKE ?')
                            params.append(f'%{csv_sub}%')
                        sql = 'SELECT csv_file,nombre,venue,web,puntaje,emails_web,socials,direccion,created_at FROM procesado'
                        if where:
                            sql += ' WHERE ' + ' AND '.join(where)
                        rows = cur.execute(sql, params).fetchall()
                        conn.close()
                        if not rows:
                            print('No hay resultados para ese filtro.')
                            return
                        out_dir = Path('out/exports_sqlite')
                        out_dir.mkdir(parents=True, exist_ok=True)
                        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                        suffix = f"{from_d or 'ALL'}_{to_d or 'ALL'}"
                        excel_path = out_dir / f'export_sqlite_{suffix}_{ts}.xlsx'
                        json_path = out_dir / f'export_sqlite_{suffix}_{ts}.json'
                        cols = ['csv_file','nombre','venue','web','puntaje','emails_web','socials','direccion','created_at']
                        df = pd.DataFrame(rows, columns=cols)
                        with pd.ExcelWriter(excel_path) as writer:
                            df.to_excel(writer, sheet_name='Datos', index=False)
                        json.dump({'filters': {'from': from_d, 'to': to_d, 'csv_sub': csv_sub}, 'total': len(rows), 'data': df.to_dict(orient='records')}, open(json_path,'w',encoding='utf-8'), ensure_ascii=False, indent=2)
                        print(f"Export SQLite creado:\n  Excel: {excel_path}\n  JSON:  {json_path}\n  Registros: {len(rows)}")
                    export_sqlite_rango(from_date, to_date, csv_filter)
                except Exception as e:
                    print(f"Error exportando desde SQLite: {e}")
                continue
            if act == '9':
                # Ejecutar sync manual de la hoja general
                try:
                    from todos_expo_push import rebuild_unprocessed_from_csv, sync_processed_from_etapa1, sync_from_db_etapa1_sites, rebuild_filtered_unprocessed_from_csv, update_stats_tab  # type: ignore
                    # Refrescar primero Keywords_map → config_map.json para aplicar filtros actuales
                    try:
                        sync_master_mapa_sheet()
                        print("[MAPA] Keywords_map refrescado → config_map.json actualizado (previo al sync).")
                    except Exception as _me:
                        print(f"[MAPA] Aviso: no se pudo refrescar Keywords_map previo al sync: {_me}")
                    # Primero reconstruir TAB4 con hits de DOM/Nombre para excluirlos de TAB2
                    print("\nReconstruyendo 'Filtrados_sin_procesar_bot' (tab 4) desde ./csv (aplicando DOM/Nombre) ...")
                    n4 = rebuild_filtered_unprocessed_from_csv('csv')
                    print(f"Listo. Filtrados (tab4): {n4}")
                    # Luego reconstruir TAB2 excluyendo dominios presentes en TAB4
                    print("Reconstruyendo 'sin_procesar_bot' (tab 2) desde ./csv (excluyendo filtrados) ...")
                    n = rebuild_unprocessed_from_csv('csv')
                    print(f"Listo. Filas en tab2: {n}")
                    # 1) Sync desde etapa1_v1.json (compatibilidad)
                    print("Sincronizando procesados (tab 1) y descartados (tab 3) desde etapa1_v1.json ...")
                    a1_e1, a3_e1 = sync_processed_from_etapa1('out/etapa1_v1.json')
                    print(f"Agregados etapa1 -> tab1: {a1_e1} | tab3: {a3_e1}")
                    # 2) Backfill desde BD (etapa1_sites) para asegurar que TAB3 tenga TODOS los procesados
                    #    Usa data.db por defecto (db_manager.DB_PATH)
                    db_path = str((Path(__file__).resolve().parent / 'data.db'))
                    print(f"Backfill desde BD ({db_path}) → TAB3=todo y TAB1 si pasa filtros ...")
                    a1_db, a3_db = sync_from_db_etapa1_sites(db_path)
                    print(f"Agregados DB -> tab1: {a1_db} | tab3: {a3_db}")
                    # Resumen
                    print(f"Totales agregados → TAB1: {a1_e1 + a1_db} | TAB3: {a3_e1 + a3_db}")
                    # Actualizar pestaña 'stats' al final del flujo de sync
                    try:
                        res = update_stats_tab()
                        if isinstance(res, dict) and res.get('ok'):
                            print(f"[STATS] Actualizada pestaña 'stats' (rubros={res.get('rubros',0)}, total_items={res.get('total_items',0)}).")
                        else:
                            print(f"[STATS] Aviso: no se pudo actualizar stats: {res}")
                    except Exception as _se:
                        print(f"[STATS] Error actualizando 'stats': {_se}")
                except Exception as e:
                    print(f"[TODOS-EXPO] Error en sincronización manual: {e}")
                continue
            if act == '10':
                # Submenú Short_descrip desde cache
                print("\nShort_descrip desde cache:")
                print("1. Exportar JSONL (out/short_descrip_cache.jsonl)")
                print("2. Exportar CSV (out/short_descrip_cache.csv)")
                print("3. Backfill TAB1 (solo vacíos)")
                print("4. Muestra por pantalla (10 dominios)")
                print("0. Cancelar")
                sub = input("Seleccione: ").strip()
                out_path = None
                limit_val = None
                if sub == '0':
                    continue
                if sub == '1':
                    out_path = 'out/short_descrip_cache.jsonl'
                elif sub == '2':
                    out_path = 'out/short_descrip_cache.csv'
                elif sub == '3':
                    pass  # backfill sin out
                elif sub == '4':
                    limit_val = 10
                else:
                    print('Opción inválida')
                    continue
                try:
                    cmd = [sys.executable, 'gen_short_descrip_from_cache.py']
                    if out_path:
                        # Preguntar opcionalmente un límite
                        try:
                            raw_lim = input('Límite de dominios (Enter = todos): ').strip()
                            if raw_lim:
                                limit_val = int(raw_lim)
                        except Exception:
                            pass
                        cmd += ['--out', out_path]
                    if sub == '3':
                        cmd += ['--update-tab1', '--from-tab1']
                        try:
                            raw_lim = input('Límite de dominios para backfill (Enter = todos): ').strip()
                            if raw_lim:
                                limit_val = int(raw_lim)
                        except Exception:
                            pass
                    if limit_val is not None:
                        cmd += ['--limit', str(limit_val)]
                    print('Ejecutando:', ' '.join(shlex.quote(p) for p in cmd))
                    res = subprocess.run(cmd, check=False)
                    if res.returncode == 0:
                        print('[SHORTDESC] Listo.')
                    else:
                        print(f'[SHORTDESC] Terminó con código {res.returncode}')
                except Exception as e:
                    print(f'[SHORTDESC] Error: {e}')
                continue
            if act == '5':
                # Exportar solo rubro seleccionado + diff sheet
                sel = choose_csv()
                if not sel:
                    continue
                csv_path, _, _ = sel
                print("\nExportando SOLO rubro actual + hoja diferencial (si IDs configurados)...")
                diff_id = os.getenv('ENRICH_DIFF_SHEET_ID')
                excl_id = os.getenv('ENRICH_EXCLUDE_SHEET_ID')
                if not diff_id or not excl_id:
                    print("[WARN] Faltan ENRICH_DIFF_SHEET_ID o ENRICH_EXCLUDE_SHEET_ID; la hoja diferencial puede omitirse.")
                # Pasar archivo etapa1 filtrado temporalmente
                temp_path = Path('out/_etapa1_subset_tmp.json')
                try:
                    full = json.load(open('out/etapa1_v1.json','r',encoding='utf-8'))
                except Exception as e:
                    print(f"No se pudo leer etapa1_v1.json: {e}")
                    continue
                sites_all = full.get('sites', []) if isinstance(full, dict) else []
                norm_csv_sel = csv_path.replace('/', os.sep).replace('\\', os.sep)
                subset_sites = []
                for s in sites_all:
                    src = s.get('source_csv') or {}
                    if not isinstance(src, dict):
                        continue
                    f = (src.get('file') or '').replace('/', os.sep).replace('\\', os.sep)
                    if f.endswith(norm_csv_sel) or f == norm_csv_sel:
                        subset_sites.append(s)
                print(f"[SUBSET] Sitios en rubro seleccionado: {len(subset_sites)}")
                temp_doc = {'version': full.get('version',1), 'sites': subset_sites}
                temp_path.write_text(json.dumps(temp_doc, ensure_ascii=False, indent=2), encoding='utf-8')
                cmd_parts = [sys.executable, 'exportar_etapa1.py', '--push-diff-sheet', '--input', str(temp_path)]
                if args.extra_args:
                    cmd_parts += shlex.split(args.extra_args)
                print("Comando:")
                print(" ".join(shlex.quote(p) for p in cmd_parts))
                try:
                    res = subprocess.run(cmd_parts, check=False)
                    if res.returncode == 0:
                        print("Export rubro + (posible) diff finalizada (exit=0)")
                    else:
                        print(f"Advertencia: terminó con código {res.returncode}")
                except Exception as e:
                    print(f"Error ejecutando exportar_etapa1.py subset: {e}")
                finally:
                    try:
                        temp_path.unlink()
                    except Exception:
                        pass
                continue
            if act == '6':
                # Exportar GLOBAL marcando como USA_NIC los presentes en hoja exclusión
                print("\nExportación GLOBAL etiquetando items presentes en hoja (añadiendo Venue=USA_NIC si coincide)...")
                os.environ['TAG_USA_NIC_FROM_EXCLUSIONS'] = '1'
                # Activar push a hoja global si ID configurado
                if os.getenv('ENRICH_GLOBAL_EXPORT_SHEET_ID'):
                    os.environ['PUSH_GLOBAL_SHEET'] = '1'
                    print('[GLOBAL-SHEET] PUSH_GLOBAL_SHEET=1 activado (se intentará subir a hoja global).')
                else:
                    print('[GLOBAL-SHEET] No se configuró ENRICH_GLOBAL_EXPORT_SHEET_ID; se omitirá subida global.')
                # No pasar args.extra_args aquí: exportar_etapa1.py no soporta flags ajenos como --sheets-rubros-map-json
                cmd_parts = [sys.executable, 'exportar_etapa1.py']
                print("Comando:")
                print(" ".join(shlex.quote(p) for p in cmd_parts))
                try:
                    res = subprocess.run(cmd_parts, check=False)
                    if res.returncode == 0:
                        print("Export global + etiquetado finalizada (exit=0)")
                        # Mostrar resumen meta global si existe
                        meta_path = Path('out/global_sheet_meta.json')
                        if meta_path.exists():
                            try:
                                meta = json.load(open(meta_path,'r',encoding='utf-8'))
                                print('[GLOBAL-SHEET] Resumen subida:')
                                for k,v in meta.items():
                                    print(f'  {k}: {v}')
                            except Exception as me:
                                print(f'[GLOBAL-SHEET] Error leyendo meta: {me}')
                    else:
                        print(f"Advertencia: terminó con código {res.returncode}")
                except Exception as e:
                    print(f"Error ejecutando exportar_etapa1.py global tag: {e}")
                finally:
                    os.environ.pop('TAG_USA_NIC_FROM_EXCLUSIONS', None)
                    os.environ.pop('PUSH_GLOBAL_SHEET', None)
                continue
            if act == '7':
                # Exportar TODOS los CSVs (batch) usando la misma lógica que el submenú 2 → opción "todos"
                try:
                    from exportar_etapa1 import collect_all_web_emails, load_busquedas_externas, load_json_or_empty
                except Exception:
                    print("No se pudo importar utilidades de exportación, se exportará solo campos básicos.")
                    collect_all_web_emails = None  # type: ignore
                etapa1_path = 'out/etapa1_v1.json'
                if not Path(etapa1_path).exists():
                    print("No existe etapa1_v1.json, no se puede exportar.")
                    continue
                try:
                    etapa1_data = json.load(open(etapa1_path, 'r', encoding='utf-8')).get('sites', [])
                except Exception as e:
                    print(f"Error leyendo etapa1: {e}")
                    continue
                target_csvs = [str(p) for p in Path('csv').glob('*.csv')]
                if not target_csvs:
                    print('No hay CSVs en ./csv para exportar.')
                    continue
                print(f"Export batch de {len(target_csvs)} CSVs...")
                # Reusar helper local de la opción 2: definimos aquí un export_one_csv equivalente
                def export_one_csv_all(one_csv_path: str):
                    # Subset directo por source_csv o fallback por dominio si falta source
                    subset_local = _build_subset_for_csv_from_etapa1(etapa1_data, one_csv_path)
                    if not subset_local:
                        print(f"[EXPORT] No hay registros procesados aún para {Path(one_csv_path).name}.")
                        return
                    print(f"[EXPORT] {Path(one_csv_path).name}: {len(subset_local)} registros procesados. Generando export...")
                    out_dir_local = Path('out/exports_parciales')
                    out_dir_local.mkdir(parents=True, exist_ok=True)
                    ts_local = datetime.now().strftime('%Y%m%d_%H%M%S')
                    base_name_local = Path(one_csv_path).stem
                    excel_path_local = out_dir_local / f'export_parcial_{base_name_local}_{ts_local}.xlsx'
                    json_path_local = out_dir_local / f'export_parcial_{base_name_local}_{ts_local}.json'
                    # cargar externos si tenemos helpers
                    if collect_all_web_emails:
                        try:
                            busq_ext_local = load_busquedas_externas('out/busquedas_externas.json')  # type: ignore
                            v2_local = load_json_or_empty('out/etapa1_2_V2_V3.json')  # type: ignore
                            enriq_local = load_json_or_empty('out/enriquecidos.json')  # type: ignore
                            enrv3_local = load_json_or_empty('out/enriquecidov3.json')  # type: ignore
                        except Exception:
                            busq_ext_local = {}
                            v2_local = {}
                            enriq_local = {}
                            enrv3_local = {}
                    records_local = []
                    for item in subset_local:
                        src = item.get('source_csv') or {}
                        if not isinstance(src, dict):
                            src = {}
                        row = src.get('row') or {}
                        if not isinstance(row, dict):
                            row = {}
                        if collect_all_web_emails:
                            try:
                                emw, emx = collect_all_web_emails(item, busq_ext_local, v2_local, enriq_local, enrv3_local)  # type: ignore
                            except Exception:
                                emw = collect_all_web_emails(item, busq_ext_local, v2_local, enriq_local, enrv3_local)  # type: ignore
                                emx = ''
                            emails_web_local = emw
                            emails_ext_local = emx
                        else:
                            emails_raw_local = []
                            for e in (item.get('emails') or []):
                                if isinstance(e, dict):
                                    emails_raw_local.append(e.get('value',''))
                                else:
                                    emails_raw_local.append(str(e))
                            emails_web_local = ', '.join(sorted(set(emails_raw_local)))
                            emails_ext_local = ''
                        socials_list_local = []
                        for s in (item.get('socials') or []):
                            url = s.get('url') if isinstance(s, dict) else s
                            if url:
                                socials_list_local.append(str(url))
                        socials_joined_local = format_socials(socials_list_local, args.socials_max_len)
                        record_local = {
                            'Nombre': row.get('Nombre', row.get('Name', '')),
                            'Venue': src.get('rubro', ''),
                            'WEB': row.get('Sitio Web', row.get('Website', row.get('URL', ''))),
                            'puntaje_web': item.get('band_score') or (item.get('band') or {}).get('score') or 0,
                            'emails_Web': emails_web_local,
                            'emails_externos': emails_ext_local,
                            'socials': socials_joined_local,
                            'Direccion_Completa': (item.get('addresses') or [{}])[0].get('value','') if item.get('addresses') else row.get('Dirección', row.get('Address',''))
                        }
                        records_local.append(record_local)
                    # Guardar Excel + JSON
                    try:
                        with pd.ExcelWriter(excel_path_local) as writer:
                            pd.DataFrame(records_local).to_excel(writer, sheet_name='Datos', index=False)
                        json.dump({'data': records_local, 'total': len(records_local)}, open(json_path_local, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
                        print(f"[EXPORT] Parcial creado:\n  Excel: {excel_path_local}\n  JSON:  {json_path_local}")
                    except Exception as e:
                        print(f"Error generando export: {e}")
                    # Persistir en SQLite
                    try:
                        db_path_local = Path('out/adt_procesado.db')
                        conn_local = sqlite3.connect(db_path_local)
                        cur_local = conn_local.cursor()
                        cur_local.execute("""
                        CREATE TABLE IF NOT EXISTS procesado (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            csv_file TEXT,
                            nombre TEXT,
                            venue TEXT,
                            web TEXT,
                            puntaje REAL,
                            emails_web TEXT,
                            socials TEXT,
                            direccion TEXT,
                            item_valido TEXT,
                            comentario TEXT,
                            created_at TEXT,
                            updated_at TEXT,
                            UNIQUE(csv_file, web)
                        )
                        """)
                        # Migración: añadir columnas que puedan faltar en tablas antiguas
                        try:
                            cur_local.execute("SELECT item_valido FROM procesado LIMIT 1")
                        except Exception:
                            try:
                                cur_local.execute("ALTER TABLE procesado ADD COLUMN item_valido TEXT")
                            except Exception:
                                pass
                        try:
                            cur_local.execute("SELECT comentario FROM procesado LIMIT 1")
                        except Exception:
                            try:
                                cur_local.execute("ALTER TABLE procesado ADD COLUMN comentario TEXT")
                            except Exception:
                                pass
                        try:
                            cur_local.execute("SELECT updated_at FROM procesado LIMIT 1")
                        except Exception:
                            try:
                                cur_local.execute("ALTER TABLE procesado ADD COLUMN updated_at TEXT")
                            except Exception:
                                pass
                        now_local = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00','Z')
                        rows_local = [(
                            one_csv_path,
                            r['Nombre'],
                            r['Venue'],
                            r['WEB'],
                            r['puntaje_web'],
                            r['emails_Web'],
                            r['socials'],
                            r['Direccion_Completa'],
                            now_local,
                            now_local
                        ) for r in records_local]
                        cur_local.executemany("""
                            INSERT INTO procesado (csv_file, nombre, venue, web, puntaje, emails_web, socials, direccion, created_at, updated_at)
                            VALUES (?,?,?,?,?,?,?,?,?,?)
                            ON CONFLICT(csv_file, web) DO UPDATE SET
                                nombre=excluded.nombre,
                                venue=excluded.venue,
                                puntaje=excluded.puntaje,
                                emails_web=excluded.emails_web,
                                socials=excluded.socials,
                                direccion=excluded.direccion,
                                updated_at=excluded.updated_at
                        """, rows_local)
                        conn_local.commit()
                        conn_local.close()
                        print(f"[EXPORT] SQLite actualizado: {len(rows_local)} filas (out/adt_procesado.db)")
                    except Exception as e:
                        print(f"Error guardando en SQLite: {e}")
                    # Export adicional a Google Sheet por rubro
                    try:
                        _export_rubro_google_sheet(one_csv_path, subset_local)
                    except Exception as e:
                        print(f"[GSHEET] Aviso: exportación Google Sheet por rubro falló ({Path(one_csv_path).stem}): {e}")
                # Ejecutar
                for one in target_csvs:
                    export_one_csv_all(one)
                # Anotar etapa1 una vez al final del batch
                try:
                    _apply_comentarios_to_etapa1()
                except Exception as e:
                    print(f"[ETAPA1] Aviso: anotación post-export (todos) falló: {e}")
                continue
            if act == '8':
                # Submenú para elegir extractor: Python nativo o Node (ADT)
                print("\nExtracción Google Maps:")
                print("1. Python (recomendado, sin dependencias externas)")
                print("2. Node (ADT-ULTIMATE VERSION-OK-ApiMaps)")
                print("0. Cancelar")
                choice = input("Seleccione extractor: ").strip()
                try:
                    base = Path(__file__).resolve().parent
                    if choice == '1':
                        # Submenú Python extractor
                        print("\nPython extractor:")
                        print("  1) Ejecutar batch completo (rubros.json x florida_ciudades.json)")
                        print("  2) Solo un rubro + una ciudad")
                        print("  3) Una consulta exacta (query completa)")
                        print("  0) Volver")
                        sub = input("Seleccione: ").strip()
                        if sub == '0':
                            continue
                        env = os.environ.copy()
                        env['OUTPUT_CSV_DIR'] = str((base / 'csv').resolve())
                        # Aviso si falta API key (hará sólo URLs de búsqueda)
                        if not (env.get('GOOGLE_API_KEY') or os.getenv('GOOGLE_API_KEY')):
                            print('[MAPS] Aviso: no se detectó GOOGLE_API_KEY en el entorno. Se generarán solo URLs de búsqueda (sin websites).')
                            print('       Para resultados completos, configure GOOGLE_API_KEY en .env o variables de entorno.')
                            # Activar Selenium como fallback para intentar extraer websites del UI
                            env['MAPS_USE_SELENIUM'] = '1'
                        cmd = [sys.executable, str(base / 'tools' / 'maps_extractor.py')]
                        # Allow temporary API key override if user pastes one
                        try:
                            if not (env.get('GOOGLE_API_KEY') or os.getenv('GOOGLE_API_KEY')):
                                key_in = input('Pegue una GOOGLE_API_KEY para esta sesión (o Enter para omitir): ').strip()
                                if key_in:
                                    # Prefer passing via CLI to avoid mutating env globally
                                    cmd += ['--api-key', key_in]
                        except Exception:
                            pass
                        if sub == '1':
                            # batch por archivos por defecto
                            pass
                        elif sub == '2':
                            pr = input('Rubro/prompt (ej. amphitheaters of live music): ').strip()
                            ct = input('Ciudad (ej. Gainesville): ').strip()
                            if not pr or not ct:
                                print('Se requieren rubro y ciudad.')
                                continue
                            cmd += ['--prompt', pr, '--city', ct]
                        elif sub == '3':
                            q = input('Query completa (ej. amphitheaters of live music in Gainesville, Florida): ').strip()
                            if not q:
                                print('Se requiere query.')
                                continue
                            cmd += ['--query', q]
                        else:
                            print('Opción inválida')
                            continue
                        print('[MAPS] Ejecutando extractor Python…')
                        print(' '.join(shlex.quote(c) for c in cmd))
                        res = subprocess.run(cmd, env=env, check=False)
                        if res.returncode == 0:
                            print('[MAPS] Extracción Python finalizada. CSVs en ./csv')
                            # Preguntar si desea aplicar filtro de dominios únicos ahora
                            try:
                                ans = input('¿Aplicar filtro de dominios únicos ahora? [y/N]: ').strip().lower()
                                if ans in ('y','yes','s','si'):
                                    print('[CSV] Ejecutando filtro de dominios únicos…')
                                    res2 = subprocess.run([sys.executable, 'filtrar_csv_dominios_unicos.py'], check=False)
                                    if res2.returncode == 0:
                                        print('[CSV] Filtro aplicado correctamente.')
                                    else:
                                        print(f'[CSV] Filtro terminó con código {res2.returncode}')
                            except Exception:
                                pass
                        else:
                            print(f'[MAPS] Extractor Python terminó con código {res.returncode}')
                    elif choice == '2':
                        node_dir = base / 'ADT-ULTIMATE VERSION-OK-ApiMaps'
                        if not node_dir.exists():
                            print('[MAPS] No se encontró la carpeta ADT-ULTIMATE VERSION-OK-ApiMaps')
                        else:
                            env = os.environ.copy()
                            env['OUTPUT_CSV_DIR'] = str((base / 'csv').resolve())
                            cmd = [sys.executable, '-c', 'import os,subprocess,sys; subprocess.run(["node","index.js"], cwd=os.getenv("NODE_CWD"), check=False)']
                            env['NODE_CWD'] = str(node_dir)
                            print('[MAPS] Ejecutando Node extractor… (esto puede tardar)')
                            res = subprocess.run(cmd, env=env, check=False)
                            if res.returncode == 0:
                                print('[MAPS] Extracción Node finalizada. CSVs en ./csv')
                            else:
                                print(f'[MAPS] Node extractor terminó con código {res.returncode}')
                    else:
                        print('Cancelado')
                except Exception as e:
                    print(f'[MAPS] Error ejecutando extractor: {e}')
                continue
            print("Opción inválida")