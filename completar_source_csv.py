#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List

def load_json(path: str) -> Dict[str, Any]:
    """Carga un archivo JSON"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(path: str, data: Dict[str, Any]):
    """Guarda un archivo JSON"""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_csv_data() -> Dict[str, Dict[str, Any]]:
    """
    Carga todos los CSVs de la carpeta csv/ y construye un mapa de URLs a datos
    """
    csv_dir = Path("csv")
    url_data = {}
    
    for csv_file in csv_dir.glob("*.csv"):
        try:
            df = pd.read_csv(csv_file, encoding='utf-8')
        except:
            try:
                df = pd.read_csv(csv_file, encoding='latin1')
            except:
                print(f"Error leyendo {csv_file}")
                continue
        
        rubro = csv_file.stem.replace('rubro_', '')
        
        # Intentar diferentes nombres de columna para la URL
        url_cols = ['Sitio Web', 'Website', 'URL', 'url', 'web', 'Web']
        found_col = None
        for col in url_cols:
            if col in df.columns:
                found_col = col
                break
        
        if not found_col:
            print(f"No se encontró columna URL en {csv_file}")
            continue
            
        for _, row in df.iterrows():
            url = row[found_col]
            if pd.isna(url) or not url:
                continue
                
            url = url.strip().lower()
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
                
            # Guardar toda la fila y el rubro
            url_data[url] = {
                'row': row.to_dict(),
                'rubro': rubro,
                'file': str(csv_file)
            }
    
    return url_data

def normalize_domain(url: str) -> str:
    """Normaliza un dominio para matching"""
    domain = url.lower().strip()
    if domain.startswith(('http://', 'https://')):
        domain = domain.split('://', 1)[1]
    if domain.startswith('www.'):
        domain = domain[4:]
    return domain.split('/')[0]

def complete_source_csv():
    """Completa la información de source_csv en etapa1_v1.json"""
    # Cargar etapa1
    etapa1_path = "out/etapa1_v1.json"
    etapa1_data = load_json(etapa1_path)
    
    # Cargar datos de CSVs
    print("Cargando datos de CSVs...")
    url_data = load_csv_data()
    
    # Contar cuántos items necesitan actualización
    sites_updated = 0
    total_sites = len(etapa1_data.get('sites', []))
    
    # Actualizar cada sitio
    for site in etapa1_data.get('sites', []):
        needs_update = False
        
        # Verificar si tiene source_csv vacío o incompleto (mirando row interno)
        source = site.get('source_csv') or {}
        if not isinstance(source, dict):
            source = {}
        source_row = source.get('row') or {}
        if not isinstance(source_row, dict):
            source_row = {}
        if not (source_row.get('Sitio Web') or source_row.get('Website') or source_row.get('URL')):
            needs_update = True
        
        if needs_update:
            # Buscar por URL o dominio
            site_url = (site.get('site_url') or '').strip().lower()
            domain = (site.get('domain') or '').strip().lower()

            # normalizar site_url al estilo del índice (https:// ...)
            if site_url and not site_url.startswith(('http://', 'https://')):
                site_url = 'https://' + site_url
            
            # Intentar matchear por URL exacta primero
            matched_data = url_data.get(site_url)
            if not matched_data and site_url:
                # probar variantes http/https y www
                variants = set()
                if site_url.startswith('https://'):
                    variants.add('http://' + site_url[len('https://'):])
                if site_url.startswith('http://'):
                    variants.add('https://' + site_url[len('http://'):])
                # agregar/quitar www.
                base = site_url.split('://', 1)[1]
                if base.startswith('www.'):
                    variants.add(site_url.split('://', 1)[0] + '://' + base[4:])
                else:
                    variants.add(site_url.split('://', 1)[0] + '://' + 'www.' + base)
                for v in variants:
                    if v in url_data:
                        matched_data = url_data[v]
                        break
            
            # Si no, buscar por dominio
            if not matched_data:
                norm_domain = normalize_domain(domain)
                for url, data in url_data.items():
                    if normalize_domain(url) == norm_domain:
                        matched_data = data
                        break
            
            # Actualizar si encontramos datos
            if matched_data:
                site['source_csv'] = {
                    'file': matched_data['file'],
                    'rubro': matched_data['rubro'],
                    'row': matched_data['row']
                }
                sites_updated += 1
                
    # Guardar cambios
    print(f"\nActualizados {sites_updated} de {total_sites} sitios")
    save_json(etapa1_path, etapa1_data)
    print(f"Cambios guardados en {etapa1_path}")

if __name__ == "__main__":
    complete_source_csv()