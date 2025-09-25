#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import subprocess
import shlex
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from urllib.parse import urlparse

# Asegurar que podamos importar desde el directorio actual
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def load_json(path: str) -> Dict[str, Any]:
    """Carga un archivo JSON"""
    if not os.path.exists(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def normalize_url(url: str) -> str:
    """Normaliza una URL para comparación"""
    url = url.lower().strip()
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url
    return url

def get_csv_progress_stats(csv_path: Path) -> dict:
    """Obtiene estadísticas de progreso para un CSV específico"""
    try:
        # Leer CSV
        try:
            df = pd.read_csv(csv_path, encoding='utf-8')
        except:
            df = pd.read_csv(csv_path, encoding='latin1')
            
        # Obtener URLs del CSV
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
            
        # Normalizar URLs del CSV
        urls = set(normalize_url(url) for url in df[url_col].dropna())
        total_urls = len(urls)
        
        # Cargar datos procesados de la última exportación
        exports_dir = Path('out/exports')
        if exports_dir.exists():
            # Obtener el archivo de exportación más reciente
            export_files = list(exports_dir.glob('exportacion_etapa1_*.json'))
            if export_files:
                latest_export = max(export_files, key=lambda x: x.stat().st_mtime)
                processed_data = load_json(str(latest_export))
            else:
                processed_data = {'data': []}
        processed_urls = {normalize_url(site.get('WEB', '')) 
                         for site in processed_data.get('data', [])}
        
        # Obtener URLs procesadas que están en este CSV
        processed_in_csv = urls.intersection(processed_urls)
        
        # Contar sitios válidos en zona
        valid_in_zone = sum(1 for site in processed_data.get('data', []) 
                         if normalize_url(site.get('WEB', '')) in urls 
                         and site.get('score_zona', 0) > 0)
        
        return {
            'total': total_urls,
            'procesados': len(processed_in_csv),
            'pendientes': total_urls - len(processed_in_csv),
            'validos_zona': valid_in_zone,
            'urls_pendientes': urls - processed_urls
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
    """Muestra menú con progreso de cada CSV y permite seleccionar uno para continuar"""
    csv_dir = Path("csv")
    if not csv_dir.exists():
        print("Error: no existe directorio csv/")
        return None
        
    csvs = list(csv_dir.glob("*.csv"))
    if not csvs:
        print("Error: no hay archivos CSV en csv/")
        return None
        
    print("\n=== RESUMEN DE PROGRESO POR CSV ===\n")
    
    for i, csv_file in enumerate(csvs, 1):
        stats = get_csv_progress_stats(csv_file)
        rubro = csv_file.stem.replace('rubro_', '')
        
        print(f"{i}. {rubro}")
        print(f"   Total URLs: {stats['total']}")
        if stats['total'] > 0:
            print(f"   Procesados: {stats['procesados']} ({stats['procesados']/stats['total']*100:.1f}%)")
        else:
            print(f"   Procesados: {stats['procesados']} (0%)")
        print(f"   Pendientes: {stats['pendientes']}")
        print(f"   Válidos en zona: {stats['validos_zona']}")
        if 'error' in stats:
            print(f"   ⚠️ {stats['error']}")
        print()
    
    print("\nOpciones:")
    print("0. Salir")
    print("1-N. Seleccionar CSV para continuar procesamiento")
    
    while True:
        try:
            choice = int(input("\nSeleccione una opción (0-{}): ".format(len(csvs))))
            if choice == 0:
                return None
            if 1 <= choice <= len(csvs):
                csv_file = csvs[choice-1]
                stats = get_csv_progress_stats(csv_file)
                
                if stats['pendientes'] == 0:
                    print("\n⚠️ Este CSV ya está completamente procesado")
                    continue
                    
                print(f"\nContinuar procesando {csv_file.name}")
                print(f"URLs pendientes: {stats['pendientes']}")
                
                # Retornar ruta al CSV, índice de inicio y lista de URLs pendientes
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
    for i, csv_file in enumerate(csvs, 1):
        stats = get_csv_progress_stats(csv_file)
        rubro = csv_file.stem.replace('rubro_', '')

        print(f"{i}. {rubro}")
        print(f"   Total URLs: {stats['total']}")
        pct = (stats['procesados']/stats['total']*100) if stats['total'] else 0
        print(f"   Procesados: {stats['procesados']} ({pct:.1f}%)")
        print(f"   Pendientes: {stats['pendientes']}")
        print(f"   Válidos en zona: {stats['validos_zona']}")
        if 'error' in stats:
            print(f"   ⚠️ {stats['error']}")
        print()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Menú de progreso por CSV: interactivo o resumen no interactivo")
    parser.add_argument("--summary", action="store_true", help="Mostrar resumen no interactivo y salir")
    parser.add_argument("--no-run", action="store_true", help="No ejecutar nav_pro automáticamente tras seleccionar un CSV")
    parser.add_argument("--extra-args", type=str, default="", help="Argumentos extra a pasar a nav_pro (en una sola cadena)")
    args = parser.parse_args()

    if args.summary:
        print_progress_summary()
    else:
        selection = show_progress_menu()
        if not selection:
            sys.exit(0)

        csv_path, start_idx, pending_urls = selection
        print(f"\nSeleccionado: {csv_path}")
        print(f"Inicio sugerido (desde índice): {start_idx} — Pendientes: {len(pending_urls)}")

        # Permitir fijar un límite interactivo
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
            print(cmd_preview)
            sys.exit(0)

        # Construir comando para ejecutar nav_pro
        cmd_parts = [sys.executable, "nav_pro.py", "--csv", csv_path, "--start", str(start_idx)]
        if user_limit is not None:
            cmd_parts += ["--limit", str(user_limit)]
        if args.extra_args:
            # Respetar comillas del usuario en extra-args
            cmd_parts += shlex.split(args.extra_args)

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