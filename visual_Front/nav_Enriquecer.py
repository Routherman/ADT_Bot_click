"""Wrapper Streamlit para interfaz Enriquecer (migrada a visual_Front).

Ejecutar: streamlit run visual_Front/nav_Enriquecer.py

La implementación completa está en visual_Front/main_enriquecer_impl.py
Este wrapper sólo importa y ejecuta run().
"""
import sys, pathlib, importlib
import streamlit as st

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

try:
	impl = importlib.import_module('visual_Front.main_enriquecer_impl')
	impl.run()
except ModuleNotFoundError:
	st.error("No se encontró visual_Front.main_enriquecer_impl. Se debe crear moviendo la lógica original.")
except Exception as ex:
	st.error(f"Error al cargar implementación: {ex}")
