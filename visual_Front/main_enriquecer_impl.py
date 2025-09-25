# main_enriquecer_impl.py
# Contiene la implementación completa movida desde nav_Enriquecer.py (root)
# Para mantener el diff pequeño, reutilizamos el archivo original importándolo dinámicamente.

import sys, pathlib, types

ROOT = pathlib.Path(__file__).resolve().parent.parent
ORIGINAL_FILE = ROOT / 'nav_Enriquecer.py'

# Estrategia: cargamos el código original (todavía en root) y lo ejecutamos dentro de una función run().
# Luego el archivo root se reemplazará por un wrapper que reimporta desde visual_Front, evitando recursión circular.

def _load_original_source():
    if not ORIGINAL_FILE.exists():
        return None
    return ORIGINAL_FILE.read_text(encoding='utf-8')

# Marcadores para evitar doble ejecución cuando streamlit lo interpreta como script principal
_ALREADY = {'done': False}

def run():
    if _ALREADY['done']:
        return
    src = _load_original_source()
    if src is None:
        import streamlit as st
        st.error('Archivo original nav_Enriquecer.py no encontrado para migración.')
        return
    # Ejecutar el código original en un namespace aislado
    ns: dict = {'__name__': '__enriq_migrated__'}
    code = compile(src, str(ORIGINAL_FILE), 'exec')
    exec(code, ns, ns)
    _ALREADY['done'] = True
