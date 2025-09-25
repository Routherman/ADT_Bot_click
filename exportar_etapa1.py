#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import re
import json
from urllib.parse import urlparse
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Set, Optional
from statistics import mode, mean
from pathlib import Path

def clean_email(email: str) -> str:
    """Limpia y normaliza un email."""
    # Quitar prefijos %20 o 20
    email = email.strip()
    while email.lower().startswith(("%20", "20")):
        email = email[2:] if email.startswith("20") else email[3:]
        email = email.strip()
    
    # Quitar cualquier texto antes de @ que contenga ... o que no sea alfanumérico/_/./-/+
    if "@" in email:
        local, domain = email.split("@", 1)
        if "..." in local:
            local = local.split("...")[-1]
        # Solo permitir caracteres válidos en la parte local
        local = re.sub(r'[^a-zA-Z0-9._+-]', '', local)
        email = f"{local}@{domain}"
    
    return email.lower()

def is_valid_email(email: str) -> bool:
    """Verifica si un email tiene formato válido."""
    pattern = r'^[a-zA-Z0-9._+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def normalize_social_url(url: str) -> str:
    """Normaliza URLs de redes sociales para deduplicación."""
    if not url:
        return ""
    # Normalizar protocolo
    url = url.lower().replace('http://', 'https://')
    if not url.startswith('https://'):
        url = 'https://' + url
    
    # Normalizar www
    url = url.replace('https://www.', 'https://')
    
    # Limpiar parámetros y fragmentos
    url = url.split('?')[0].split('#')[0]
    
    # Remover slash final
    while url.endswith('/'):
        url = url[:-1]
        
    return url

def get_valid_unique_emails(emails_data: List[Dict[str, Any]]) -> str:
    """Extrae emails válidos y únicos de la lista de emails."""
    valid_emails = set()
    
    for email_item in emails_data:
        email = clean_email(email_item.get('value', ''))
        if email and is_valid_email(email):
            valid_emails.add(email)
    
    return ','.join(sorted(valid_emails))

# --- Extensiones para sumar correos desde búsquedas externas y etapa2_cache ---
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

def extract_domain(value: str) -> str:
    if not value:
        return ""
    v = value.strip().lower()
    # si ya parece un dominio
    if v and '://' not in v and '/' not in v and '@' not in v:
        return v.lstrip('www.')
    try:
        p = urlparse(v if "://" in v else ("https://" + v))
        host = (p.netloc or p.path or '').lower()
        return host.lstrip('www.')
    except Exception:
        return v.lstrip('www/')

def load_busquedas_externas(path: str = 'out/busquedas_externas.json') -> Dict[str, Any]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}

def load_json_or_empty(path: str) -> Any:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}

def emails_from_v2_for_domain(v2: Dict[str, Any], domain: str) -> Set[str]:
    out: Set[str] = set()
    try:
        node = v2.get(domain) if isinstance(v2, dict) else None
        if isinstance(node, dict):
            for c in (node.get('contacts') or []):
                if isinstance(c, dict):
                    em = clean_email(str(c.get('email') or ''))
                    if em and is_valid_email(em):
                        out.add(em)
    except Exception:
        pass
    return out

def emails_from_enriquecidos_for_domain(enriq: Dict[str, Any], domain: str) -> Set[str]:
    out: Set[str] = set()
    try:
        node = enriq.get(domain) if isinstance(enriq, dict) else None
        if isinstance(node, dict):
            # contacts_enriched -> source_person.email
            for item in (node.get('contacts_enriched') or []):
                if isinstance(item, dict):
                    sp = item.get('source_person')
                    if isinstance(sp, dict):
                        em = clean_email(str(sp.get('email') or ''))
                        if em and is_valid_email(em):
                            out.add(em)
                    # Also scan enrich provider blocks (e.g., contactout_people.profile.email/personal_email)
                    enrich_blk = item.get('enrich')
                    if isinstance(enrich_blk, dict):
                        for em2 in collect_emails_from_enrich(enrich_blk):
                            if em2 and is_valid_email(em2):
                                out.add(em2)
            # optional root emails list
            for em in (node.get('emails') or []):
                em2 = clean_email(str(em))
                if em2 and is_valid_email(em2):
                    out.add(em2)
    except Exception:
        pass
    return out

def collect_emails_from_enrich(obj: Any) -> Set[str]:
    """Recorre recursivamente un bloque de 'enrich' para extraer emails.
    Busca claves que contengan la palabra 'email' y extrae valores string/lista.
    """
    found: Set[str] = set()
    try:
        if isinstance(obj, dict):
            for k, v in obj.items():
                lk = str(k).lower()
                if 'email' in lk:
                    # v puede ser str, list, dict
                    if isinstance(v, str):
                        for m in EMAIL_RE.findall(v or ''):
                            em = clean_email(m)
                            if em:
                                found.add(em)
                    elif isinstance(v, list):
                        for it in v:
                            if isinstance(it, str):
                                for m in EMAIL_RE.findall(it or ''):
                                    em = clean_email(m)
                                    if em:
                                        found.add(em)
                            elif isinstance(it, dict):
                                found.update(collect_emails_from_enrich(it))
                    elif isinstance(v, dict):
                        found.update(collect_emails_from_enrich(v))
                else:
                    # seguir recorriendo
                    if isinstance(v, (dict, list)):
                        found.update(collect_emails_from_enrich(v))
        elif isinstance(obj, list):
            for it in obj:
                found.update(collect_emails_from_enrich(it))
    except Exception:
        pass
    return found

def domain_from_site_like(site: Dict[str, Any]) -> str:
    dom = (site.get('domain') or '').strip().lower()
    if not dom:
        u = (site.get('site_url') or '').strip()
        dom = extract_domain(u)
    return extract_domain(dom)

def emails_from_enriquecidov3_for_domain(enrv3: Dict[str, Any], domain: str) -> Set[str]:
    out: Set[str] = set()
    try:
        if not isinstance(enrv3, dict):
            return out
        sites = enrv3.get('sites')
        if not isinstance(sites, list):
            return out
        target = domain.strip().lower()
        for s in sites:
            if not isinstance(s, dict):
                continue
            d = domain_from_site_like(s)
            if d != target:
                continue
            # site-level emails
            for e in (s.get('emails') or []):
                val = ''
                if isinstance(e, dict):
                    val = str(e.get('value') or e.get('email') or e.get('mail') or '')
                elif isinstance(e, str):
                    val = e
                em = clean_email(val)
                if em and is_valid_email(em):
                    out.add(em)
            # people emails
            for p in (s.get('people') or []):
                if isinstance(p, dict):
                    em = clean_email(str(p.get('email') or ''))
                    if em and is_valid_email(em):
                        out.add(em)
    except Exception:
        pass
    return out

def extract_emails_from_etapa2_cache(domain: str, cache_dir: str = 'out/etapa2_cache', max_files: int = 200) -> Set[str]:
    emails: Set[str] = set()
    d = Path(cache_dir)
    if not d.exists():
        return emails
    key = (domain or '').lower().lstrip('www.')
    count = 0
    # Buscar archivos cuyo nombre contenga el dominio (recursivo por si hay subcarpetas)
    for fp in d.rglob('*'):
        if not fp.is_file():
            continue
        name = fp.name.lower()
        if key and key not in name:
            continue
        try:
            # limitar número de archivos a revisar por dominio
            if count >= max_files:
                break
            text = ''
            try:
                # intentar leer como texto
                text = fp.read_text(encoding='utf-8', errors='ignore')
            except Exception:
                # si falla, intentar como binario y decodificar de forma laxa
                text = fp.read_bytes().decode('utf-8', errors='ignore')
            for m in EMAIL_RE.finditer(text or ''):
                em = clean_email(m.group(0))
                if em and is_valid_email(em):
                    emails.add(em)
            count += 1
        except Exception:
            continue
    return emails

def collect_all_web_emails(item: Dict[str, Any], busq_ext: Dict[str, Any], v2: Dict[str, Any], enriq: Dict[str, Any], enrv3: Dict[str, Any]) -> str:
    """Combina emails de etapa1(item), busquedas_externas.json y etapa2_cache para 'emails_Web'."""
    collected: Set[str] = set()
    # 1) etapa1 (como hasta ahora)
    for email_item in item.get('emails', []) or []:
        # soportar tanto dicts {'value': ...} como strings
        em_raw = ''
        if isinstance(email_item, dict):
            em_raw = email_item.get('value', '')
        elif isinstance(email_item, str):
            em_raw = email_item
        em = clean_email(em_raw)
        if em and is_valid_email(em):
            collected.add(em)
    # 1.b) incluir emails que quedaron en people[] (ej. páginas de contacto)
    for person in item.get('people', []) or []:
        try:
            em_person = ''
            if isinstance(person, dict):
                em_person = person.get('email', '') or ''
            elif isinstance(person, str):
                # si por alguna razón hay strings, intentar extraer email vía regex
                m = EMAIL_RE.search(person)
                em_person = m.group(0) if m else ''
            em_person = clean_email(em_person)
            if em_person and is_valid_email(em_person):
                collected.add(em_person)
        except Exception:
            pass
    # 2) dominio para lookup externos
    dom = (item.get('domain') or '').strip().lower()
    if not dom:
        # como fallback, intentar desde la URL del registro
        src = item.get('source_csv') or {}
        if isinstance(src, dict):
            row = src.get('row') or {}
            if isinstance(row, dict):
                web = row.get('Sitio Web') or row.get('Website') or row.get('URL') or ''
                dom = extract_domain(web)
    # fallback adicional: site_url del item
    if not dom:
        dom = extract_domain(item.get('site_url') or '')
    dom = extract_domain(dom)
    # 3) busquedas_externas.json
    if dom and isinstance(busq_ext, dict):
        ent = busq_ext.get(dom) or {}
        if isinstance(ent, dict):
            for em in (ent.get('emails') or []):
                em2 = clean_email(str(em))
                if em2 and is_valid_email(em2):
                    collected.add(em2)
    # 4) etapa2_cache (búsqueda bruta en cache PDFs/HTML etapa2)
    if dom:
        for em3 in extract_emails_from_etapa2_cache(dom):
            if em3 and is_valid_email(em3):
                collected.add(em3)
    # 5) etapa1_2_V2_V3.json (contactos v2)
    if dom and isinstance(v2, dict):
        for em in emails_from_v2_for_domain(v2, dom):
            collected.add(em)
    # 6) enriquecidos.json
    if dom and isinstance(enriq, dict):
        for em in emails_from_enriquecidos_for_domain(enriq, dom):
            collected.add(em)
    # 7) enriquecidov3.json (sites list)
    if dom and isinstance(enrv3, dict):
        for em in emails_from_enriquecidov3_for_domain(enrv3, dom):
            collected.add(em)
    return ','.join(sorted(collected))

def calculate_score_stats(scores: List[float]) -> Dict[str, float]:
    """Calcula estadísticas de scores."""
    if not scores:
        return {
            "promedio": 0,
            "moda": 0,
            "maximo": 0,
            "minimo": 0
        }
    
    # Filtrar scores válidos y convertir a float
    valid_scores = [float(s) for s in scores if s is not None]
    if not valid_scores:
        return {
            "promedio": 0,
            "moda": 0,
            "maximo": 0,
            "minimo": 0
        }
    
    try:
        score_mode = mode(valid_scores)
    except:
        score_mode = valid_scores[0]  # Si no hay moda, usar el primer valor
    
    return {
        "promedio": mean(valid_scores),
        "moda": score_mode,
        "maximo": max(valid_scores),
        "minimo": min(valid_scores)
    }

def export_etapa1(etapa1_path: str, output_path: str):
    """
    Exporta los datos de etapa1 a Excel y JSON con estadísticas.
    
    Args:
        etapa1_path: Ruta al archivo etapa1_v1.json
        output_path: Ruta donde guardar las exportaciones
    """
    # Asegurar que existe el directorio de salida
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    # Obtener el mínimo score desde variables de entorno
    MIN_SCORE = float(os.getenv('SCRORE_MIN_EXPORT_STAGE1', '5.0'))
    
    # Cargar datos de etapa1
    with open(etapa1_path, 'r', encoding='utf-8') as f:
        data = json.load(f).get('sites', [])
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    excel_filename = f'exportacion_etapa1_{timestamp}.xlsx'
    json_filename = f'exportacion_etapa1_{timestamp}.json'
    
    # Preparar estadísticas
    total_records = len(data)
    valid_florida = [item for item in data if item.get('florida_ok', False)]
    valid_florida_count = len(valid_florida)
    
    # Conteos y estadísticas
    failed_acquisition = len([item for item in valid_florida if not item.get('band', {}).get('score')])
    scores = [item.get('band', {}).get('score', 0) for item in valid_florida]
    score_stats = calculate_score_stats(scores)
    
    # Contar emails válidos únicos
    sites_with_valid_emails = 0
    sites_without_emails = 0
    total_valid_emails = 0
    
    for item in valid_florida:
        valid_emails = set()
        for email_item in item.get('emails', []):
            email = clean_email(email_item.get('value', ''))
            if email and is_valid_email(email):
                valid_emails.add(email)
        
        if valid_emails:
            sites_with_valid_emails += 1
            total_valid_emails += len(valid_emails)
        else:
            sites_without_emails += 1
    
    # Crear resumen
    summary = {
        "Total registros": total_records,
        "Registros válidos Florida": valid_florida_count,
        "Fallaron adquisición": failed_acquisition,
        "Sin emails válidos": sites_without_emails,
        "Con emails válidos": sites_with_valid_emails,
        "Total emails válidos": total_valid_emails,
        "Score promedio": score_stats["promedio"],
        "Score moda": score_stats["moda"],
        "Score máximo": score_stats["maximo"],
        "Score mínimo": score_stats["minimo"]
    }
    
    # Cargar fuentes externas una sola vez
    busq_ext = load_busquedas_externas('out/busquedas_externas.json')
    v2 = load_json_or_empty('out/etapa1_2_V2_V3.json')
    enriq = load_json_or_empty('out/enriquecidos.json')
    enrv3 = load_json_or_empty('out/enriquecidov3.json')

    # Crear DataFrame para hoja 2
    records = []
    for item in valid_florida:
        if item.get('band', {}).get('score', 0) >= MIN_SCORE:
            # Asegurar estructuras seguras aunque falten datos
            source = item.get('source_csv') or {}
            if not isinstance(source, dict):
                source = {}
            # Obtener datos de source_csv.row si existe
            source_row = source.get('row') or {}
            if not isinstance(source_row, dict):
                source_row = {}
            record = {
                'Nombre': source_row.get('Nombre', source_row.get('Name', '')),
                'Venue': source.get('rubro', ''),
                'WEB': source_row.get('Sitio Web', source_row.get('Website', source_row.get('URL', ''))),
                'puntaje_web': item.get('band', {}).get('score', 0),
                # emails_Web ahora combina: etapa1 + people + externos + etapa2_cache + V2 + enriquecidos + enriquecidov3
                'emails_Web': collect_all_web_emails(item, busq_ext, v2, enriq, enrv3),
                'socials': ','.join(sorted(set(
                    normalize_social_url(s.get('url', '')) 
                    for s in item.get('socials', [])
                    if s.get('url')
                ))),
                'Direccion_Completa': (
                    (item.get('addresses') or [{}])[0].get('value', '') if item.get('addresses') else (
                        source_row.get('Dirección', source_row.get('Address', ''))
                    )
                )
            }
            records.append(record)
    
    # Crear Excel
    with pd.ExcelWriter(os.path.join(output_path, excel_filename)) as writer:
        # Hoja 1 - Resumen
        pd.DataFrame([summary]).to_excel(writer, sheet_name='Resumen', index=False)
        # Hoja 2 - Datos
        pd.DataFrame(records).to_excel(writer, sheet_name='Datos', index=False)
    
    # Crear JSON
    export_data = {
        'summary': summary,
        'data': records
    }
    
    with open(os.path.join(output_path, json_filename), 'w', encoding='utf-8') as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2)
    
    # Imprimir resumen en consola
    print("\nResumen de exportación:")
    print("-" * 40)
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
    
    print("\nArchivos exportados:")
    print(f"Excel: {excel_filename}")
    print(f"JSON: {json_filename}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Exportar datos de etapa1 a Excel y JSON')
    parser.add_argument('--input', '-i', 
                      default='out/etapa1_v1.json',
                      help='Ruta al archivo etapa1_v1.json')
    parser.add_argument('--output', '-o',
                      default='out/exports',
                      help='Carpeta donde guardar las exportaciones')
    args = parser.parse_args()
    
    export_etapa1(args.input, args.output)