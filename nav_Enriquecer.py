# -*- coding: utf-8 -*-
# nav_Enriquecer.py
# GUI Streamlit para enriquecer contactos con Lusha / ContactOut / RocketReach
# - Lee input con estructura {"version": X, "sites": [...]}
# - Muestra emails/people existentes por dominio
# - Prospecting por dominio (Lusha / RocketReach) + Enrichment por persona (Lusha / ContactOut / RocketReach)
# - Persiste:
#   (1) out/enriquecidos.json (dump crudo por dominio)
#   (2) out/enriquecidov3.json (MISMA estructura que el input) con people: [{name, role, email, source, department?, seniority?}]
# - Logs en logs/logs_apis.json
# - Tabla final (department ∈ dept_filter) OR (seniority ∈ seniority_ok)
# - Panel para editar diccionarios y guardarlos en .env (ENRICH_*)

import os, json, time, pathlib, io, csv, re
import math
import uuid
from urllib.parse import quote_plus
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import requests
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv, dotenv_values
import subprocess, sys
import importlib
try:
    import navegar_secciones as navsec  # Selenium-based section navigator
except Exception:
    navsec = None

# ----------------- Paths & Config -----------------
# Base paths
ROOT: pathlib.Path = pathlib.Path(__file__).resolve().parent
OUT_DIR: pathlib.Path = ROOT / "out"
LOGS_DIR: pathlib.Path = ROOT / "logs"
PROCS_LOGS_DIR: pathlib.Path = LOGS_DIR / "processes"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
PROCS_LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Files
ENRIQ_FILE_V3: pathlib.Path = OUT_DIR / "enriquecidov3.json"
ENRIQ_FILE_RAW: pathlib.Path = OUT_DIR / "enriquecidos.json"
LOGS_FILE: pathlib.Path = LOGS_DIR / "logs_apis.json"
BG_RUNS_FILE: pathlib.Path = LOGS_DIR / "bg_runs.json"
ENV_PATH: pathlib.Path = ROOT / ".env"

# API Keys from environment (loaded below after login as well)
LUSHA_API_KEY = os.getenv("LUSHA_API_KEY", "")
CONTACTOUT_API_KEY = os.getenv("CONTACTOUT_API_KEY", "")
ROCKETREACH_API_KEY = os.getenv("ROCKETREACH_API_KEY", "")

# ----------------- Setup -----------------
st.set_page_config(page_title="V4 Enriquecer • Lusha | ContactOut | RocketReach", layout="wide")
if "auth_user" not in st.session_state:
    st.session_state["auth_user"] = None  # {'username':..., 'role':...}
if "admin_logged_in" not in st.session_state:  # legacy control panel guard (will map to supervisor role)
    st.session_state["admin_logged_in"] = False

# --- DB Auth Integration ---
try:
    import db_manager as dbm
    dbm.init_db()
except Exception as _db_ex:
    st.sidebar.warning(f"DB no inicializada: {_db_ex}")
    dbm = None  # fallback to legacy single user

ROLE_LABELS = {
    'lector': 'Lector',
    'editor': 'Editor',
    'supervisor': 'Supervisor'
}

def current_perms():
    if not st.session_state.get('auth_user') or not dbm:
        # legacy: only allow read until login
        return {'read': False,'export': False,'modify': False,'control': False,'user_admin': False}
    return dbm.role_permissions(st.session_state['auth_user']['role'])

# Safe rerun helper for Streamlit (handles versions without experimental_rerun)
def _safe_rerun():
    try:
        st.rerun()
    except Exception:
        try:
            # Backwards compatibility for older Streamlit
            st.experimental_rerun()  # type: ignore[attr-defined]
        except Exception:
            pass

def login_form():
    st.title("Login requerido")
    if dbm is None:
        st.info("Modo legacy: usar usuario fijo.")
    user = st.text_input("Usuario", value="", key="login_user")
    pwd = st.text_input("Contraseña", value="", type="password", key="login_pwd")
    if st.button("Ingresar"):
        if dbm:
            auth = dbm.authenticate(user, pwd)
            if auth:
                st.session_state['auth_user'] = auth
                if auth['role'] == 'supervisor':
                    st.session_state['admin_logged_in'] = True
                st.success(f"Bienvenido {auth['username']} ({ROLE_LABELS.get(auth['role'], auth['role'])})")
                _safe_rerun()
            else:
                st.error("Credenciales inválidas o usuario inactivo")
        else:
            # fallback legacy credentials
            if user == "CapitanNic" and pwd == "Chula2004":
                st.session_state['auth_user'] = {'username': user, 'role': 'supervisor'}
                st.session_state['admin_logged_in'] = True
                st.success("Acceso concedido (legacy)")
                _safe_rerun()
            else:
                st.error("Usuario o contraseña incorrectos")
    st.caption("Rol determina permisos: lector, editor, supervisor.")

def logout_button():
    if st.session_state.get('auth_user'):
        if st.sidebar.button("Cerrar sesión"):
            st.session_state['auth_user'] = None
            st.session_state['admin_logged_in'] = False
            _safe_rerun()

if not st.session_state.get('auth_user'):
    login_form()
    st.stop()
else:
    logout_button()
load_dotenv()

# Reload API keys after loading .env
LUSHA_API_KEY = os.getenv("LUSHA_API_KEY", LUSHA_API_KEY)
CONTACTOUT_API_KEY = os.getenv("CONTACTOUT_API_KEY", CONTACTOUT_API_KEY)
ROCKETREACH_API_KEY = os.getenv("ROCKETREACH_API_KEY", ROCKETREACH_API_KEY)

# ------------- Sidebar: Sección selector -------------
available_sections = ["Enriquecer"]
if current_perms().get('control'):
    available_sections.append("Control")
if current_perms().get('user_admin'):
    available_sections.append("Usuarios")
    # Sección especial de migraciones a DB sólo para supervisor
    available_sections.append("Migraciones")
section = st.sidebar.radio("Sección", available_sections, index=0)

# Supervisor user admin panel (basic)
if section == "Usuarios":
    if not current_perms().get('user_admin'):
        st.error("No tiene permiso para administrar usuarios")
        st.stop()
    st.title("Administración de Usuarios")
    if not dbm:
        st.error("DB no disponible")
        st.stop()
    users = dbm.list_users()
    st.subheader("Usuarios existentes")
    valid_roles_local = getattr(dbm, 'VALID_ROLES', ['lector','editor','supervisor'])
    role_to_index = {r:i for i,r in enumerate(valid_roles_local)}
    for u in users:
        cols = st.columns([2,2,1,2,1])
        cols[0].markdown(f"**{u['username']}**")
        with cols[1]:
            role_sel = st.selectbox("Rol", valid_roles_local, index=role_to_index.get(u['role'],0), key=f"role_{u['username']}")
        with cols[2]:
            active_toggle = st.checkbox("Activo", value=u['active'], key=f"act_{u['username']}")
        with cols[3]:
            new_pass = st.text_input("Nuevo pass", value="", type="password", key=f"np_{u['username']}")
        with cols[4]:
            if st.button("Aplicar", key=f"apply_{u['username']}"):
                changed = []
                if role_sel != u['role']:
                    if dbm.set_user_role(u['username'], role_sel):
                        changed.append("rol")
                if active_toggle != u['active']:
                    if dbm.deactivate_user(u['username'], active_toggle):
                        changed.append("estado")
                if new_pass.strip():
                    if dbm.reset_password(u['username'], new_pass.strip()):
                        changed.append("password")
                if changed:
                    st.success("Actualizado: " + ", ".join(changed))
                else:
                    st.info("Sin cambios")
    st.markdown("---")
    st.subheader("Crear nuevo usuario")
    colc1, colc2, colc3, colc4 = st.columns([2,2,2,1])
    with colc1:
        nu_user = st.text_input("Usuario", key="nu_user")
    with colc2:
        nu_pass = st.text_input("Contraseña", key="nu_pass", type="password")
    with colc3:
        nu_role = st.selectbox("Rol", valid_roles_local, key="nu_role")
    with colc4:
        if st.button("Crear", key="nu_create"):
            if nu_user.strip() and nu_pass.strip():
                ok = dbm.create_user(nu_user.strip(), nu_pass.strip(), nu_role)
                if ok:
                    st.success("Usuario creado")
                    _safe_rerun()
                else:
                    st.error("No se pudo crear (¿usuario duplicado?)")
            else:
                st.warning("Completar usuario y contraseña")
    st.stop()

if section == "Migraciones":
    # Solo supervisor
    if not current_perms().get('user_admin'):
        st.error("No autorizado")
        st.stop()
    st.title("Migraciones a SQLite")
    if not dbm:
        st.error("DB no disponible")
        st.stop()
    counts_before = dbm.counts_summary()
    st.write("Conteos actuales:", counts_before)
    colm1, colm2, colm3 = st.columns(3)
    with colm1:
        if st.button("Migrar etapa1_v1.json"):
            r = dbm.migrate_etapa1()
            st.success(f"Migrados sitios: {r}")
    with colm2:
        if st.button("Migrar enriquecidov3.json"):
            r = dbm.migrate_enriquecidov3()
            st.success(f"Migrados people v3: {r}")
    with colm3:
        if st.button("Migrar enriquecidos.json (raw)"):
            r = dbm.migrate_enriquecidos_raw()
            st.success(f"Migrados raw: {r}")
    if st.button("Refrescar conteos"):
        st.write("Conteos:", dbm.counts_summary())
    st.stop()

def _render_control_dashboard():
    """Panel de Control con pestañas: Logs, Procesos lanzados, Procesamiento manual."""
    if not current_perms().get('control'):
        st.error("No autorizado para Control")
        return
    t1, t2, t3, t4, t5 = st.tabs(["Resumen", "Búsquedas", "Logs", "Procesos", "Manual"])
    with t3:
        st.subheader("Logs")
        logs = load_json(LOGS_FILE) or []
        if not isinstance(logs, list):
            logs = []
        providers = sorted({ (x.get("provider") or "").strip() for x in logs if isinstance(x, dict) })
        prov = st.selectbox("Filtrar por proveedor", ["(todos)"] + providers)
        last_n = st.number_input("Mostrar últimos N", min_value=10, max_value=5000, value=200, step=10)
        rows = []
        for ent in reversed(logs):
            if not isinstance(ent, dict):
                continue
            if prov != "(todos)" and (ent.get("provider") or "").strip() != prov:
                continue
            rows.append(ent)
            if len(rows) >= last_n:
                break
        if rows:
            df_logs = pd.DataFrame(rows)
            def _safe(v):
                if isinstance(v, (dict, list)):
                    try:
                        return json.dumps(v, ensure_ascii=False)
                    except Exception:
                        return str(v)
                if isinstance(v, (set, tuple)):
                    return ", ".join(map(str, v))
                return v
            df_logs = df_logs.applymap(_safe)
            st.dataframe(df_logs, use_container_width=True)
            ms_vals = [x.get("ms") for x in rows if isinstance(x, dict) and isinstance(x.get("ms"), int)]
            if ms_vals:
                st.caption(f"latencia ms • min: {min(ms_vals)} • p50: {sorted(ms_vals)[len(ms_vals)//2]} • max: {max(ms_vals)} • avg: {int(sum(ms_vals)/len(ms_vals))}")
        else:
            st.info("Sin logs para el filtro actual.")

    with t4:
        st.subheader("Procesos lanzados")
        st.caption("Ejecuciones de nav_pro lanzadas en background desde este panel")
        if "bg_runs" not in st.session_state:
            st.session_state["bg_runs"] = _load_bg_runs()
        bg_runs = st.session_state.get("bg_runs", [])

        def _is_pid_running_windows(pid: int) -> bool:
            try:
                res = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True)
                return str(pid) in (res.stdout or "")
            except Exception:
                return False

        col1, col2 = st.columns([1,1])
        with col1:
            if st.button("Refrescar estado"):
                _safe_rerun()
        with col2:
            if st.button("Limpiar finalizados"):
                alive = []
                for r in bg_runs:
                    if _is_pid_running_windows(int(r.get("pid",0))):
                        alive.append(r)
                st.session_state["bg_runs"] = alive
                _save_bg_runs(st.session_state["bg_runs"])
                st.success("Se limpiaron los procesos finalizados.")

        if bg_runs:
            for i, r in enumerate(list(bg_runs)):
                pid = int(r.get("pid", 0) or 0)
                alive = _is_pid_running_windows(pid) if pid else False
                cols = st.columns([2,3,2,2,2,2,2])
                cols[0].markdown(f"**PID**: {pid}")
                cols[1].markdown(f"**CSV**: {(r.get('csv') or '(manual)')} ")
                cols[2].markdown(f"start={r.get('start',0)} • limit={r.get('limit',0)}")
                cols[3].markdown(f"args: {r.get('args','') or '-'}")
                cols[4].markdown(f"lanzado: {r.get('ts','')}")
                cols[5].markdown(f"estado: {'Activo' if alive else 'Finalizado'}")
                with cols[6]:
                    if alive:
                        if st.button("Terminar", key=f"kill_{pid}"):
                            try:
                                subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True)
                                st.success(f"Enviado kill al PID {pid}")
                            except Exception as ex:
                                st.error(f"No se pudo terminar {pid}: {ex}")
                    else:
                        st.caption("—")

                logf = r.get("logfile")
                if logf and os.path.isfile(logf):
                    with st.expander(f"Consola PID {pid}"):
                        max_lines = st.number_input("Últimas N líneas", min_value=50, max_value=5000, value=200, step=50, key=f"tail_n_{pid}")
                        def _tail_file(path: str, lines: int = 200) -> str:
                            try:
                                with open(path, "rb") as fh:
                                    fh.seek(0, os.SEEK_END)
                                    size = fh.tell()
                                    block = 4096
                                    data = b""
                                    while size > 0 and data.count(b"\n") <= lines:
                                        read_size = min(block, size)
                                        size -= read_size
                                        fh.seek(size)
                                        chunk = fh.read(read_size)
                                        data = chunk + data
                                    txt = data.decode("utf-8", errors="replace")
                                    return "\n".join(txt.splitlines()[-lines:])
                            except Exception as ex:
                                return f"(no se pudo leer el log) {ex}"
                        st.code(_tail_file(logf, int(max_lines)), language="bash")
                        colv1, colv2, colv3 = st.columns([1,1,2])
                        with colv1:
                            if st.button("Actualizar", key=f"refresh_log_{pid}"):
                                _safe_rerun()
                        with colv2:
                            try:
                                with open(logf, "rb") as fh_all:
                                    st.download_button("Descargar log", data=fh_all.read(), file_name=os.path.basename(logf), key=f"dl_log_{pid}")
                            except Exception:
                                st.caption("No se pudo preparar descarga")
                        with colv3:
                            auto_key = f"auto_refresh_{pid}"
                            if auto_key not in st.session_state:
                                st.session_state[auto_key] = bool(alive)
                            elif not alive and st.session_state.get(auto_key):
                                st.session_state[auto_key] = False
                            auto = st.checkbox("Auto-actualizar", value=st.session_state.get(auto_key, False), key=auto_key)
                            ivl = st.number_input("cada (seg)", min_value=1, max_value=60, value=3, step=1, key=f"auto_ivl_{pid}")
                            if auto:
                                try:
                                    ms = int(ivl) * 1000
                                except Exception:
                                    ms = 3000
                                components.html(f"<script>setTimeout(function(){{window.parent.location.reload();}}, {ms});</script>", height=0)
        else:
            st.info("Aún no hay procesos lanzados desde este panel.")

    with t5:
        st.subheader("Procesamiento manual (1 dominio)")
        st.caption("Elegí un CSV y un dominio, o ingresá uno manualmente. Cada ejecución se lanza en una instancia separada.")
        csv_dir = ROOT / "csv"
        csv_files = sorted([p for p in csv_dir.glob("*.csv")]) if csv_dir.exists() else []
        sel_csv = None
        if csv_files:
            choice = st.selectbox("Seleccioná un CSV", [p.name for p in csv_files], key="ctrl_csv_sel_csv_manual_tab")
            sel_csv = next((p for p in csv_files if p.name == choice), None)
        else:
            st.info("No hay CSVs en la carpeta csv/")

        def _launch_bg(cmd: List[str], track: Dict[str, Any]):
            try:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_path = PROCS_LOGS_DIR / f"navpro_{ts}_manual.log"
                log_fh = open(log_path, "ab")
                DETACHED = getattr(subprocess, 'DETACHED_PROCESS', 0)
                NEW_GROUP = getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0)
                NO_WINDOW = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
                creationflags = NEW_GROUP | DETACHED | NO_WINDOW
                p = subprocess.Popen(
                    cmd,
                    cwd=str(ROOT),
                    stdout=log_fh,
                    stderr=log_fh,
                    creationflags=creationflags,
                )
                try:
                    log_fh.close()
                except Exception:
                    pass
                track.update({"pid": p.pid, "ts": datetime.now().isoformat(timespec='seconds')})
                track["logfile"] = str(log_path)
                st.session_state.setdefault("bg_runs", []).append(track)
                st.success(f"Lanzado (PID {p.pid})")
                _save_bg_runs(st.session_state.get("bg_runs", []))
            except Exception as ex:
                st.error(f"No se pudo lanzar: {ex}")

        if sel_csv:
            doms = _domains_from_csv(sel_csv)
            q = st.text_input("Buscar dominio en el CSV", value="", key="search_dom_manual")
            filtered = [d for d in doms if (q.strip().lower() in d.lower())] if q.strip() else doms
            preview = filtered[:500]
            sel_dom = st.selectbox("Elegí dominio (del CSV)", ["(ninguno)"] + preview, key="ctrl_sel_dom_manual")
            colm1, colm2 = st.columns([1,1])
            with colm1:
                if sel_dom != "(ninguno)":
                    if st.button("Procesar dominio (del CSV)", key="btn_proc_dom_from_csv_once"):
                        try:
                            idx = doms.index(sel_dom)
                            cmd = [sys.executable, "nav_pro.py", "--csv", str(sel_csv), "--start", str(idx), "--limit", "1"]
                            _launch_bg(cmd, {"csv": str(sel_csv), "start": idx, "limit": 1, "args": f"--domain={sel_dom}"})
                        except ValueError:
                            st.error("No se encontró el dominio en el CSV")
            with colm2:
                st.caption(f"Dominios en CSV: {len(doms)}  •  Mostrando: {len(preview)}")

        st.markdown("---")
        st.subheader("Ingresar dominio manual")
        manual_dom = st.text_input("Dominio (ej: acme.com)", value="", key="manual_dom_input")
        if st.button("Procesar dominio (manual)", key="btn_proc_dom_manual_once"):
            d = (manual_dom or "").strip()
            if not d:
                st.warning("Ingresá un dominio")
            else:
                try:
                    tmp_dir = OUT_DIR / "csv_tmp"
                    tmp_dir.mkdir(parents=True, exist_ok=True)
                    safe = re.sub(r"[^A-Za-z0-9._-]", "_", _base_domain(d) or d)
                    tmp_path = tmp_dir / f"manual_{safe}_{int(time.time())}.csv"
                    with tmp_path.open("w", encoding="utf-8", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=["domain"]) 
                        writer.writeheader()
                        writer.writerow({"domain": d})
                    cmd = [sys.executable, "nav_pro.py", "--csv", str(tmp_path), "--start", "0", "--limit", "1"]
                    _launch_bg(cmd, {"csv": str(tmp_path), "start": 0, "limit": 1, "args": "(manual)"})
                except Exception as ex:
                    st.error(f"No se pudo preparar el CSV temporal: {ex}")

## Defer rendering of Control until helpers are defined (moved below)

# ... many helper and API functions defined above ...

# (Control section rendering moved further down, after helper definitions)

# Defaults (se sobreescriben desde .env si existen)
DEFAULT_JOB_TITLES = [
    "marketing director","head of marketing","marketing manager",
    "event manager","event coordinator","entertainment director",
    "owner","president","ceo","cmo","vp marketing","venue manager","talent buyer"
]
DEFAULT_DEPARTMENTS  = ["marketing","events","public relations","communications","operations"]
DEFAULT_DEPT_FILTER  = {"marketing","events","communications"}
DEFAULT_SENIORITY    = ["owner","c_level","vp","director","manager","president","ceo"]
DEFAULT_SENIORITY_OK = {"owner","c_level","vp","president","ceo"}

# ----------------- Helpers .env -----------------
def _read_env_dict() -> Dict[str, str]:
    if ENV_PATH.exists():
        try:
            return {k: str(v) for k, v in (dotenv_values(ENV_PATH) or {}).items() if k}
        except Exception:
            return {}
    return {}

def _write_env_dict(env_map: Dict[str, str]) -> None:
    # Reescribimos el archivo entero con kv actuales + sistema
    # Tomamos también las variables actuales del proceso para no perder claves
    merged = dict(env_map)
    for k, v in os.environ.items():
        if k not in merged:
            merged[k] = v
    lines = [f"{k}={merged[k]}" for k in sorted(merged.keys()) if merged[k] is not None]
    ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

def set_env_list(key: str, items: List[str]) -> None:
    env_map = _read_env_dict()
    # normalizamos dedupe
    seen, clean = set(), []
    for x in (items or []):
        x = (x or "").strip()
        if x and x.lower() not in seen:
            seen.add(x.lower())
            clean.append(x)
    env_map[key] = ",".join(clean)
    _write_env_dict(env_map)

def get_env_list(key: str, fallback: List[str]) -> List[str]:
    val = os.getenv(key, "")
    if not val:
        return fallback
    return [x.strip() for x in val.split(",") if x.strip()]

# Sobrescribir defaults si el usuario ya guardó en .env
DEFAULT_JOB_TITLES = get_env_list("ENRICH_JOB_TITLES", DEFAULT_JOB_TITLES)
DEFAULT_DEPARTMENTS = get_env_list("ENRICH_DEPARTMENTS", DEFAULT_DEPARTMENTS)
DEFAULT_DEPT_FILTER = set(get_env_list("ENRICH_DEPT_FILTER", list(DEFAULT_DEPT_FILTER)))
DEFAULT_SENIORITY = get_env_list("ENRICH_SENIORITY", DEFAULT_SENIORITY)
DEFAULT_SENIORITY_OK = set(get_env_list("ENRICH_SENIORITY_OK", list(DEFAULT_SENIORITY_OK)))

# ----------------- Text Normalizers -----------------
def clean_role(role: str) -> str:
    if not role:
        return ""
    r = (role or "").replace("\u2014", "-").replace("\u2013", "-")
    r = r.replace("—", "-").replace("–", "-")
    # Cortes comunes por scrape truncado
    for sep in ["\n", " | ", "  ", "“", "”", "\"", "'", " li", " li "]:
        if sep in r:
            r = r.split(sep)[0]
    # Limpiar descripciones largas "We love having ..."
    if len(r) > 80 and "," in r:
        r = r.split(",")[0]
    return r.strip().strip("-:")

def is_probably_person_name(name: str) -> bool:
    """Heurística simple para evitar falsos positivos (ej: 'Read More', ciudades, etc.).
    Requiere al menos 2 tokens alfabéticos, cada uno >=2 chars, y que el total tenga letras.
    """
    n = (name or "").strip()
    if not n:
        return False
    # Demasiado corto o sin letras
    if len(re.sub(r"[^A-Za-z]", "", n)) < 3:
        return False
    parts = [p for p in re.split(r"\s+", n) if p]
    if len(parts) < 2:
        return False
    good_tokens = sum(1 for p in parts if re.search(r"[A-Za-z]", p) and len(re.sub(r"[^A-Za-z]","",p)) >= 2)
    return good_tokens >= 2

# ----------------- LinkedIn helpers -----------------
def _linkedin_from_people(people: List[Dict[str, Any]]) -> List[str]:
    links: List[str] = []
    for p in (people or []):
        for k in ("linkedin", "linkedin_url", "profile", "linkedinProfile"):
            url = (p.get(k) or "").strip()
            if url and "linkedin.com" in url.lower():
                links.append(url)
    # dedupe preserving order
    seen, out = set(), []
    for u in links:
        lu = u.lower()
        if lu not in seen:
            seen.add(lu); out.append(u)
    return out

def _google_linkedin_search(nombre: str, domain: str, max_hits: int = 5, timeout: int = 12) -> List[str]:
    """
    Best-effort: intenta Google y, si falla/bloquea, cae a DuckDuckGo HTML.
    Usa varias variantes de consulta y devuelve primeras URLs válidas de linkedin.com.
    """
    queries = [
        f"site:linkedin.com {nombre} {domain}",
        f"site:linkedin.com/company {nombre}",
        f"site:linkedin.com {domain}",
        f"{domain} linkedin"
    ]

    def extract_li(html: str) -> List[str]:
        candidates = re.findall(r'https?://[a-zA-Z0-9.-]*linkedin\.com/[^"\s<>]+', html or "")
        cleaned = []
        for u in candidates:
            u = u.split("&")[0]
            if any(seg in u for seg in ["/share", "/feed/", "/learning/", "/help/"]):
                continue
            if "/company/" in u or "/in/" in u or "/school/" in u:
                cleaned.append(u)
        seen, out = set(), []
        for u in cleaned:
            lu = u.lower()
            if lu not in seen:
                seen.add(lu); out.append(u)
            if len(out) >= max_hits:
                break
        return out

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9"
    }

    # Intento Google primero
    for q in queries:
        try:
            r = requests.get("https://www.google.com/search", params={"q": q, "hl": "en"}, headers=headers, timeout=timeout)
            if r.status_code == 200:
                out = extract_li(r.text)
                if out:
                    return out
        except Exception:
            pass

    # Fallback: DuckDuckGo HTML (sin JS)
    for q in queries:
        try:
            r = requests.post(
                "https://duckduckgo.com/html/",
                data={"q": q},
                headers=headers,
                timeout=timeout
            )
            if r.status_code == 200:
                out = extract_li(r.text)
                if out:
                    return out
        except Exception:
            pass
    return []

def _puppeteer_linkedin_search(nombre: str, domain: str, engine: str = "google", results: int = 50, timeout_ms: int = 25000, manual_query: Optional[str] = None) -> List[str]:
    """
    Ejecuta tools/puppeteer_linkedin_search.js y devuelve links a linkedin.com.
    Requiere Node y dependencias en tools (puppeteer). Usa env PUPPETEER_HEADLESS si está seteada.
    """
    try:
        tools_dir = str((ROOT / "tools").resolve())
        script = str((ROOT / "tools" / "puppeteer_linkedin_search.js").resolve())
        base_queries = [
            f"site:linkedin.com {nombre} {domain}",
            f"site:linkedin.com/company {nombre}",
            f"site:linkedin.com/company {domain}",
            f"{domain} linkedin",
            f"{nombre} linkedin"
        ]
        queries = ([manual_query] if manual_query else []) + base_queries
        engines = [engine] if engine else ["google", "yahoo"]

        def run_once(q: str, eng: str) -> List[str]:
            cmd = ["node", script, f"--engine={eng}", f"--query={q}", f"--results={results}", f"--timeout={timeout_ms}"]
            env = dict(os.environ)
            proc = subprocess.run(cmd, cwd=tools_dir, capture_output=True, text=True, env=env, timeout=(timeout_ms/1000+10))
            txt = proc.stdout.strip()
            data = {}
            if txt.startswith("{"):
                try:
                    data = json.loads(txt)
                except Exception as jex:
                    append_log({"provider":"puppeteer","fn":"linkedin.search","domain":domain,
                                "error":"json_parse_error","stdout_head": txt[:400], "stderr_head": (proc.stderr or "")[:400]})
                    return []
            else:
                append_log({"provider":"puppeteer","fn":"linkedin.search","domain":domain,
                            "error":"non_json_output","stdout_head": txt[:400], "stderr_head": (proc.stderr or "")[:400]})
                return []
            links = data.get("links") or []
            # Filtrar linkedin útil
            cleaned = []
            for u in links:
                u = (u or "").split("&")[0]
                if not u: continue
                if any(seg in u for seg in ["/share", "/feed/", "/learning/", "/help/"]):
                    continue
                if "/company/" in u or "/in/" in u or "/school/" in u:
                    cleaned.append(u)
            # Dedupe manteniendo orden
            seen, out = set(), []
            for u in cleaned:
                lu = u.lower()
                if lu not in seen:
                    seen.add(lu); out.append(u)
            return out

        # Ejecutar combinaciones hasta obtener algo
        for eng in engines:
            for q in queries:
                out = run_once(q, eng)
                if out:
                    return out
        return []
    except Exception as ex:
        append_log({"provider":"puppeteer","fn":"linkedin.search","domain":domain, "error": str(ex)})
        return []

def _navigate_sections_linkedin(domain: str, max_pages: int = 20, also_serp: bool = True) -> List[str]:
    """Navega la home y secciones del sitio y extrae URLs de linkedin.com encontradas en el HTML.
    Requiere navegar_secciones.py (Selenium + ChromeDriver)."""
    if not navsec:
        append_log({"provider":"nav","fn":"linkedin.sections","domain":domain, "error": "navegar_secciones_not_available"})
        return []
    base_url = f"https://{domain}"
    links_found: List[str] = []
    driver = None
    try:
        driver = navsec.build_driver()
        driver.get(base_url)
        try:
            navsec.accept_cookies_if_any(driver, None)
        except Exception:
            pass
        try:
            navsec.scroll_page(driver)
        except Exception:
            pass

        pages = [driver.current_url]
        try:
            nav_links = navsec.collect_nav_links(driver, base_url)[: max(0, max_pages - 1)]
        except Exception:
            nav_links = []
        pages.extend(nav_links)

        def extract_li_from_html(html: str) -> List[str]:
            candidates = re.findall(r'https?://[a-zA-Z0-9.-]*linkedin\\.com/[^"\s<>]+', html or "")
            cleaned = []
            for u in candidates:
                u = u.split("&")[0].split("#")[0]
                if any(seg in u for seg in ["/share", "/feed/", "/learning/", "/help/"]):
                    continue
                if "/company/" in u or "/in/" in u or "/school/" in u:
                    cleaned.append(u)
            seen, out = set(), []
            for u in cleaned:
                lu = (u or "").lower()
                if lu and lu not in seen:
                    seen.add(lu); out.append(u)
            return out

        for i, url in enumerate(pages[:max_pages], start=1):
            try:
                driver.get(url)
            except Exception:
                continue
            try:
                navsec.scroll_page(driver)
            except Exception:
                pass
            try:
                html = driver.page_source
            except Exception:
                html = ""
            hits = extract_li_from_html(html)
            if hits:
                links_found.extend(hits)
            time.sleep(0.3)

        # Opcional: también lanzar una búsqueda SERP (dom + linkedin) con Puppeteer (Google/Yahoo)
        serp_hits: List[str] = []
        if also_serp:
            try:
                # Priorizar "{domain} linkedin" y dejar que el buscador interno agregue queries de company
                serp_hits = _puppeteer_linkedin_search(domain, domain, engine=None, results=30, timeout_ms=25000, manual_query=f"{domain} linkedin")
                append_log({"provider":"puppeteer","fn":"linkedin.search_from_nav","domain":domain, "serp_count": len(serp_hits)})
            except Exception as ex:
                append_log({"provider":"puppeteer","fn":"linkedin.search_from_nav","domain":domain, "error": str(ex)})

        # Merge + dedupe final preservando orden
        merged = list(links_found) + list(serp_hits)
        seen, out = set(), []
        for u in merged:
            lu = (u or "").lower()
            if lu and lu not in seen:
                seen.add(lu); out.append(u)
        append_log({"provider":"nav","fn":"linkedin.sections","domain":domain, "count": len(out)})
        return out
    except Exception as ex:
        append_log({"provider":"nav","fn":"linkedin.sections","domain":domain, "error": str(ex)})
        return []
    finally:
        try:
            if driver:
                driver.quit()
        except Exception:
            pass

# ----------------- Helpers E/S -----------------
def load_json(path: pathlib.Path) -> Any:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: pathlib.Path, data: Any) -> None:
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)

def _load_bg_runs() -> List[Dict[str, Any]]:
    try:
        data = load_json(BG_RUNS_FILE)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []

def _save_bg_runs(runs: List[Dict[str, Any]]):
    try:
        save_json(BG_RUNS_FILE, runs or [])
    except Exception:
        pass

def append_log(entry: Dict[str, Any]) -> None:
    entry["_ts"] = datetime.now().isoformat()
    try:
        cur = load_json(LOGS_FILE) or []
        if not isinstance(cur, list):
            cur = []
    except Exception:
        cur = []
    cur.append(entry)
    save_json(LOGS_FILE, cur)

def normalize_records(raw: Any) -> Tuple[int, List[Dict[str, Any]]]:
    """
    Devuelve (version, lista_normalizada).
    Estructura esperada: {"version": X, "sites": [...]}  (como tu etapa1_v1.json)
    """
    version = raw.get("version", 3) if isinstance(raw, dict) else 3
    if isinstance(raw, dict) and isinstance(raw.get("sites"), list):
        sites = raw["sites"]
    elif isinstance(raw, list):
        # soporte alternativo
        sites = raw
    else:
        return version, []

    norm: List[Dict[str, Any]] = []
    for r in sites:
        if not isinstance(r, dict):
            # Skip invalid entries
            continue
        site_url = r.get("site_url") or ""
        domain = (r.get("domain") or site_url.replace("https://", "").replace("http://", "").split("/")[0]).strip()
        # Guard against None/non-dict for nested access
        sc = r.get("source_csv")
        row = sc.get("row") if isinstance(sc, dict) else None
        nombre = (
            (row.get("Nombre") if isinstance(row, dict) else None)
            or r.get("site_name")
            or domain
        )
        # emails normalizados
        emails: List[str] = []
        for e in (r.get("emails") or []):
            if isinstance(e, dict):
                val = e.get("value") or e.get("email") or e.get("mail")
                if isinstance(val, str) and val:
                    emails.append(val)
            elif isinstance(e, str) and e:
                emails.append(e)
        emails = sorted(set([em.strip() for em in emails if em]))
        people = r.get("people") or []  # [{name, role, email?, source?, pages?}]
        if not isinstance(people, list):
            people = []
        norm.append({
            "domain": domain,
            "nombre": nombre,
            "emails": emails,
            "people": people,
            "raw": r
        })
    return version, norm

def _etapa1_path() -> pathlib.Path:
    return OUT_DIR / "etapa1_v1.json"

def _get_site_slot_by_domain(data: Dict[str, Any], domain: str) -> Tuple[Optional[int], Optional[Dict[str, Any]]]:
    sites = data.get("sites") or []
    for i, s in enumerate(sites):
        d = (s.get("domain") or "").strip().lower()
        if d == (domain or "").strip().lower():
            return i, s
    return None, None

def _current_etapa1_socials_linkedin(domain: str, fallback_data: Optional[Dict[str, Any]]) -> List[str]:
    # Lee desde out/etapa1_v1.json si existe; si no, usa fallback_data (lo cargado en memoria)
    data = load_json(_etapa1_path()) if _etapa1_path().exists() else (fallback_data or {})
    if not isinstance(data, dict):
        return []
    _, site = _get_site_slot_by_domain(data, domain)
    if not site:
        return []
    urls = []
    for s in (site.get("socials") or []):
        if isinstance(s, dict):
            u = (s.get("url") or s.get("link") or "").strip()
            if u and "linkedin.com" in u.lower():
                urls.append(u)
    # dedupe
    seen, out = set(), []
    for u in urls:
        lu = u.lower()
        if lu not in seen:
            seen.add(lu); out.append(u)
    return out

def upsert_linkedin_to_etapa1(domain: str, urls: List[str], fallback_data: Optional[Dict[str, Any]]) -> int:
    """Inserta URLs de LinkedIn en socials para el dominio en out/etapa1_v1.json.
    Si el archivo no existe, parte de fallback_data (base_input) y lo crea.
    Devuelve la cantidad de URLs nuevas agregadas.
    """
    path = _etapa1_path()
    if path.exists():
        data = load_json(path)
    else:
        data = fallback_data or {"version": 3, "sites": []}
    if not isinstance(data, dict):
        data = {"version": 3, "sites": []}
    if not isinstance(data.get("sites"), list):
        data["sites"] = []

    idx, site = _get_site_slot_by_domain(data, domain)
    if site is None:
        site = {"domain": domain, "socials": []}
        data["sites"].append(site)
    if not isinstance(site.get("socials"), list):
        site["socials"] = []

    existing = set()
    for s in site["socials"]:
        if isinstance(s, dict):
            u = (s.get("url") or s.get("link") or "").strip().lower()
            if u:
                existing.add(u)

    added = 0
    for u in (urls or []):
        uu = (u or "").strip()
        if not uu:
            continue
        lu = uu.lower()
        if lu in existing:
            continue
        site["socials"].append({"platform": "linkedin", "url": uu})
        existing.add(lu)
        added += 1

    # Sanitizar NaN para no escribir JSON inválido
    def _replace_nans(obj: Any):
        if isinstance(obj, dict):
            return {k: _replace_nans(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [ _replace_nans(x) for x in obj ]
        try:
            if isinstance(obj, float) and math.isnan(obj):
                return None
        except Exception:
            pass
        if isinstance(obj, str) and obj.strip().lower() == "nan":
            return None
        return obj
    data = _replace_nans(data)
    save_json(path, data)
    return added

def upsert_emails_to_etapa1(domain: str, emails: List[str], fallback_data: Optional[Dict[str, Any]]) -> int:
    """Inserta emails en el campo 'emails' del dominio en out/etapa1_v1.json (dedupe, case-insensitive).
    Si el archivo no existe, parte de fallback_data (base_input) y lo crea."""
    path = _etapa1_path()
    if path.exists():
        data = load_json(path)
    else:
        data = fallback_data or {"version": 3, "sites": []}
    if not isinstance(data, dict):
        data = {"version": 3, "sites": []}
    if not isinstance(data.get("sites"), list):
        data["sites"] = []

    idx, site = _get_site_slot_by_domain(data, domain)
    if site is None:
        site = {"domain": domain, "emails": []}
        data["sites"].append(site)
    if not isinstance(site.get("emails"), list):
        site["emails"] = []

    # construir set de existentes (soportar strings o dicts con 'value')
    existing_lower = set()
    for e in site["emails"]:
        if isinstance(e, str):
            v = e.strip().lower()
            if v:
                existing_lower.add(v)
        elif isinstance(e, dict):
            v = (e.get("value") or "").strip().lower()
            if v:
                existing_lower.add(v)

    added = 0
    for e in (emails or []):
        ee = (e or "").strip()
        if not ee or "@" not in ee:
            continue
        eel = ee.lower()
        if eel in existing_lower:
            continue
        site["emails"].append(ee)
        existing_lower.add(eel)
        added += 1

    # Sanitizar NaN
    def _replace_nans(obj: Any):
        if isinstance(obj, dict):
            return {k: _replace_nans(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_replace_nans(x) for x in obj]
        try:
            if isinstance(obj, float) and math.isnan(obj):
                return None
        except Exception:
            pass
        if isinstance(obj, str) and obj.strip().lower() == "nan":
            return None
        return obj
    data = _replace_nans(data)
    save_json(path, data)
    return added

# ----------------- Socials helper -----------------
def _infer_platform(url: str, platform_hint: str = "") -> str:
    u = (url or "").lower()
    ph = (platform_hint or "").lower()
    if ph:
        return ph
    if "facebook.com" in u: return "facebook"
    if "instagram.com" in u: return "instagram"
    if "twitter.com" in u or "x.com" in u: return "twitter"
    if "youtube.com" in u or "youtu.be" in u: return "youtube"
    if "tiktok.com" in u: return "tiktok"
    if "threads.net" in u: return "threads"
    if "linktr.ee" in u: return "linktree"
    if "linkedin.com" in u: return "linkedin"
    return "other"

def ensure_v3_structure(base_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Crea o carga out/enriquecidov3.json con MISMA estructura que input.
    """
    v3 = load_json(ENRIQ_FILE_V3)
    if v3 and isinstance(v3, dict) and isinstance(v3.get("sites"), list):
        return v3
    version = base_input.get("version", 3)
    sites_in = base_input.get("sites", [])
    primed_sites = []
    for s in sites_in:
        primed_sites.append({
            "site_url": s.get("site_url"),
            "domain": s.get("domain") or (s.get("site_url","").replace("https://","").replace("http://","").split("/")[0]),
            "florida_ok": s.get("florida_ok", False),
            "band": s.get("band", {}),
            "emails": s.get("emails", []),
            "phones": s.get("phones", []),
            "addresses": s.get("addresses", []),
            "socials": s.get("socials", []),
            "people": s.get("people", []),
            "pages_scanned": s.get("pages_scanned"),
            "last_updated": s.get("last_updated"),
            "source_csv": s.get("source_csv", {}),
            "site_name": s.get("site_name"),
            "city": s.get("city")
        })
    out = {"version": version, "sites": primed_sites}
    save_json(ENRIQ_FILE_V3, out)
    return out

def upsert_raw_by_domain(domain: str, payload: Dict[str, Any]) -> None:
    """Guarda incremental en enriquecidos.json (histórico/diagnóstico)."""
    data = load_json(ENRIQ_FILE_RAW) or {}
    if domain not in data:
        data[domain] = {}
    # merge superficial
    for k, v in payload.items():
        if isinstance(v, list):
            cur = data[domain].get(k, [])
            if not isinstance(cur, list):
                cur = []
            seen = set(json.dumps(x, sort_keys=True) for x in cur)
            for it in v:
                key = json.dumps(it, sort_keys=True)
                if key not in seen:
                    cur.append(it); seen.add(key)
            data[domain][k] = cur
        elif isinstance(v, dict):
            cur = data[domain].get(k, {})
            if not isinstance(cur, dict):
                cur = {}
            cur.update(v)
            data[domain][k] = cur
        else:
            data[domain][k] = v
    save_json(ENRIQ_FILE_RAW, data)

def _person_key(p: Dict[str, Any]) -> str:
    return f"{(p.get('name') or '').strip().lower()}|{(p.get('email') or '').strip().lower()}|{(p.get('role') or '').strip().lower()}"

# ----------------- Emails de búsqueda externa -----------------
def _norm_domain(d: str) -> str:
    return (d or "").strip().lower()

def _domain_variants(d: str) -> List[str]:
    n = _norm_domain(d)
    if not n:
        return []
    if n.startswith("www."):
        bare = n[4:]
        return [n, bare]
    return [n, f"www.{n}"]

@st.cache_data(ttl=60)
def load_external_maps() -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Carga emails y names externos desde out/busquedas_externas.json y/o out/busqueda_externa.json.
    Retorna dos dicts: (emails_map, names_map) con claves de dominio y variantes con/sin www.
    """
    emails_map: Dict[str, set] = {}
    names_map: Dict[str, set] = {}
    candidates = [OUT_DIR / "busquedas_externas.json", OUT_DIR / "busqueda_externa.json"]
    for path in candidates:
        try:
            if not path.exists():
                continue
            data = load_json(path)
            if not isinstance(data, dict):
                continue
            for dom, payload in data.items():
                if not isinstance(payload, dict):
                    continue
                # Emails
                emails_set: set = set()
                for key in ["emails", "emails_found", "found_emails", "site_emails", "emails_web"]:
                    vals = payload.get(key)
                    if isinstance(vals, list):
                        for item in vals:
                            if isinstance(item, str):
                                v = item.strip()
                                if v:
                                    emails_set.add(v)
                            elif isinstance(item, dict):
                                v = (item.get("email") or item.get("value") or item.get("addr") or "").strip()
                                if v:
                                    emails_set.add(v)
                emails_set = {e for e in emails_set if "@" in e}
                # Names
                names_set: set = set()
                vals_n = payload.get("names")
                if isinstance(vals_n, list):
                    for item in vals_n:
                        if isinstance(item, str):
                            v = item.strip()
                            if v:
                                names_set.add(v)
                        elif isinstance(item, dict):
                            v = (item.get("name") or item.get("value") or "").strip()
                            if v:
                                names_set.add(v)
                # Guardar por variantes de dominio
                if emails_set:
                    for dv in _domain_variants(dom):
                        emails_map.setdefault(dv, set()).update(emails_set)
                if names_set:
                    for dv in _domain_variants(dom):
                        names_map.setdefault(dv, set()).update(names_set)
        except Exception:
            continue
    return (
        {d: sorted(list(es)) for d, es in emails_map.items()},
        {d: sorted(list(ns)) for d, ns in names_map.items()}
    )

def _collect_external_values(map_by_domain: Dict[str, List[str]], record_domain: str) -> List[str]:
    """Obtiene valores externos (emails o names) para un dominio record, considerando:
    - variantes exactas (con/sin www)
    - subdominios que terminen en .record_domain
    """
    out: List[str] = []
    seen = set()
    for dv in _domain_variants(record_domain):
        for val in map_by_domain.get(dv, []):
            lv = (val or "").strip()
            if not lv:
                continue
            lvn = lv.lower()
            if lvn in seen:
                continue
            seen.add(lvn); out.append(lv)
    # Subdominios (ej: shop.frostscience.org para frostscience.org)
    suf = f".{_norm_domain(record_domain)}"
    for dom_key, vals in map_by_domain.items():
        dk = _norm_domain(dom_key)
        if dk.endswith(suf):
            for v in vals:
                lv = (v or "").strip()
                if not lv:
                    continue
                lvn = lv.lower()
                if lvn in seen:
                    continue
                seen.add(lvn); out.append(lv)
    return out

# ----------------- CSV Rubros: dominios y métricas -----------------
def _extract_domain_from_str(s: str) -> str:
    if not s:
        return ""
    v = (str(s) or "").strip()
    if not v:
        return ""
    # Quitar mailto:
    v = re.sub(r"^mailto:", "", v, flags=re.IGNORECASE)
    # Si parece un email, tomar dominio
    if "@" in v and "/" not in v:
        try:
            return v.split("@")[1].strip().lower()
        except Exception:
            pass
    # Quitar esquema y parámetros
    v = v.replace("https://", "").replace("http://", "").replace("ftp://", "")
    v = v.split("/")[0].split("?")[0].split("#")[0]
    return v.strip().lower()

def _base_domain(dom: str) -> str:
    d = (dom or "").lower().strip()
    if not d:
        return ""
    if d.startswith("www."):
        d = d[4:]
    parts = d.split(".")
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return d

def _ok_zona(rec: Dict[str, Any]) -> bool:
    raw = rec.get("raw", {}) if isinstance(rec, dict) else {}
    if isinstance(raw.get("florida_ok"), bool):
        return bool(raw.get("florida_ok"))
    city = (raw.get("city") or "").strip().lower()
    # heurística básica
    return bool(re.search(r"\bfl\b|florida", city))

def _is_valido_rec(rec: Dict[str, Any]) -> bool:
    try:
        score = int((((rec.get("raw") or {}).get("band") or {}).get("score") or 0))
    except Exception:
        score = 0
    return (score > 5) and _ok_zona(rec)

# Score helper used across Control and UI
## _band_score defined earlier (shared by Control and UI)

def _domains_from_csv(path: pathlib.Path) -> List[str]:
    cols_domain_candidates = [
        "domain","dominio","web","website","site","site_url","url","URL","pagina","Pagina"
    ]
    try:
        df = pd.read_csv(path, dtype=str, low_memory=False)
    except Exception:
        try:
            df = pd.read_csv(path, dtype=str, low_memory=False, sep=";")
        except Exception:
            return []
    if df is None or df.empty:
        return []
    df_cols = {str(c).strip() for c in df.columns}
    chosen_col = None
    for c in cols_domain_candidates:
        if c in df_cols:
            chosen_col = c
            break
    domains: List[str] = []
    if chosen_col:
        vals = df[chosen_col].dropna().astype(str).tolist()
        for v in vals:
            d = _extract_domain_from_str(v)
            if d:
                domains.append(_base_domain(d))
    else:
        # fallback: inspeccionar todas las columnas buscando valores con dominio/URL
        for _, row in df.iterrows():
            for v in row.values:
                d = _extract_domain_from_str(v)
                if d:
                    domains.append(_base_domain(d))
    # dedupe
    seen, out = set(), []
    for d in domains:
        if d and d not in seen:
            seen.add(d); out.append(d)
    return out

def scan_csv_rubros(csv_dir: pathlib.Path) -> Dict[str, List[str]]:
    """Retorna {rubro: [dominios_unicos]} leyendo todos los .csv en csv_dir.
    Rubro = nombre del archivo (stem)."""
    rubros: Dict[str, List[str]] = {}
    if not csv_dir.exists() or not csv_dir.is_dir():
        return rubros
    for p in csv_dir.glob("*.csv"):
        rubro = p.stem
        try:
            doms = _domains_from_csv(p)
        except Exception:
            doms = []
        rubros[rubro] = doms
    return rubros

def upsert_v3_people(domain: str, new_people: List[Dict[str, Any]]) -> None:
    """
    Inserta/actualiza en enriquecidov3.json (misma estructura que input),
    sólo personas [{name, role, email, source, department?, seniority?}] (merge incremental por name+email+role).
    """
    v3 = load_json(ENRIQ_FILE_V3)
    if not v3 or not isinstance(v3, dict) or not isinstance(v3.get("sites"), list):
        return
    sites = v3["sites"]
    found = None
    for s in sites:
        if (s.get("domain") or "").strip().lower() == domain.strip().lower():
            found = s; break
    if not found:
        # si el dominio no estaba, lo agregamos mínimamente
        found = {"domain": domain, "people": []}
        sites.append(found)

    existing = found.get("people", [])
    idx = {_person_key(p): i for i, p in enumerate(existing)}
    for np in (new_people or []):
        # normalizamos shape
        merged = {
            "name": (np.get("name") or "").strip(),
            "role": (np.get("role") or "").strip(),
            "email": (np.get("email") or "").strip(),
            "source": (np.get("source") or "").strip()
        }
        # opcionales
        if np.get("department"):
            merged["department"] = (np.get("department") or "").strip()
        if np.get("seniority"):
            merged["seniority"] = (np.get("seniority") or "").strip()

        key = _person_key(merged)
        if key in idx:
            # merge suave: si el nuevo trae email/source/attrs, las mantenemos
            pos = idx[key]
            for k, v in merged.items():
                if v and not existing[pos].get(k):
                    existing[pos][k] = v
        else:
            existing.append(merged)
            idx[key] = len(existing) - 1

    found["people"] = existing
    v3["sites"] = sites
    save_json(ENRIQ_FILE_V3, v3)

# Persistencia de emails externos en enriquecidov3.json
def upsert_v3_external_emails(domain: str, ext_emails: List[str], merge_into_emails: bool = False) -> int:
    """Inserta emails externos en el campo emails_web del sitio correspondiente en enriquecidov3.json.
    Si merge_into_emails=True, también los agrega al arreglo emails (evitando duplicados).
    Devuelve cuántos emails nuevos fueron agregados a emails_web.
    """
    v3 = load_json(ENRIQ_FILE_V3)
    if not v3 or not isinstance(v3, dict) or not isinstance(v3.get("sites"), list):
        return 0
    sites = v3["sites"]
    found = None
    for s in sites:
        if (s.get("domain") or "").strip().lower() == (domain or "").strip().lower():
            found = s; break
    if not found:
        found = {"domain": domain, "emails": [], "emails_web": []}
        sites.append(found)

    # Asegurar campos
    if not isinstance(found.get("emails"), list):
        found["emails"] = []
    if not isinstance(found.get("emails_web"), list):
        found["emails_web"] = []

    # Conjuntos para dedupe
    cur_web = { (e or "").strip().lower() for e in found["emails_web"] if isinstance(e, str) }
    cur_emails = { (e or "").strip().lower() for e in found["emails"] if isinstance(e, str) }

    added_count = 0
    for e in (ext_emails or []):
        ee = (e or "").strip()
        if not ee or "@" not in ee:
            continue
        eel = ee.lower()
        if eel not in cur_web:
            found["emails_web"].append(ee)
            cur_web.add(eel)
            added_count += 1
        if merge_into_emails and eel not in cur_emails:
            found["emails"].append(ee)
            cur_emails.add(eel)

    v3["sites"] = sites
    save_json(ENRIQ_FILE_V3, v3)
    return added_count

def _get_v3_emails_web(domain: str) -> List[str]:
    """Devuelve emails_web ya guardados en enriquecidov3.json para el dominio (si existen)."""
    try:
        v3 = load_json(ENRIQ_FILE_V3)
        if not v3 or not isinstance(v3, dict) or not isinstance(v3.get("sites"), list):
            return []
        for s in v3["sites"]:
            if (s.get("domain") or "").strip().lower() == (domain or "").strip().lower():
                vals = s.get("emails_web") or []
                if isinstance(vals, list):
                    return [ (x or "").strip() for x in vals if isinstance(x, str) and x ]
                return []
    except Exception:
        return []
    return []

# ----------------- Enriquecidos (crudo) helpers -----------------
@st.cache_data(ttl=30)
def _get_enriched_items_for_domain(domain: str) -> List[Dict[str, Any]]:
    """Lee out/enriquecidos.json y devuelve items con emails enriquecidos por dominio.
    Cada item: {source: str, url: str|None, emails: List[str]}"""
    data = load_json(ENRIQ_FILE_RAW) or {}
    site = data.get(domain) or {}
    out: List[Dict[str, Any]] = []
    clist = site.get("contacts_enriched") or []
    for it in clist:
        if not isinstance(it, dict):
            continue
        enrich = it.get("enrich") or {}
        # ContactOut (dos vías posibles en esta app)
        for key in ("contactout_people", "contactout_linkedin"):
            co = enrich.get(key)
            if isinstance(co, dict):
                status = co.get("status_code")
                profile = co.get("profile") if status == 200 else None
                if isinstance(profile, dict):
                    url = (profile.get("url") or "").strip()
                    emails_raw: List[str] = []
                    for fld in ("email", "personal_email", "work_email"):
                        vals = profile.get(fld)
                        if isinstance(vals, list):
                            emails_raw.extend([v for v in vals if isinstance(v, str)])
                        elif isinstance(vals, str):
                            emails_raw.append(vals)
                    # Dedupe y sanity
                    seen: set = set()
                    emails: List[str] = []
                    for e in emails_raw:
                        ee = (e or "").strip()
                        if ee and "@" in ee and ee.lower() not in seen:
                            seen.add(ee.lower()); emails.append(ee)
                    if emails:
                        out.append({"source": key, "url": url, "emails": emails})
        # Otros proveedores si aportan emails claros (ajustable en el futuro)
        # Lusha person / RocketReach podrían incluir email directo en otras variantes de respuesta
        lp = enrich.get("lusha_person")
        if isinstance(lp, dict):
            # Formatos comunes: {'data': {...}} o plano
            candidates = []
            if isinstance(lp.get("data"), list):
                candidates = lp["data"]
            elif isinstance(lp.get("data"), dict):
                candidates = [lp["data"]]
            elif isinstance(lp, list):
                candidates = lp
            else:
                candidates = [lp]
            emails = []
            for c in candidates:
                if not isinstance(c, dict):
                    continue
                for k in ("email","workEmail","emailAddress"):
                    v = c.get(k)
                    if isinstance(v, str) and "@" in v:
                        emails.append(v.strip())
            if emails:
                # Lusha no expone URL de perfil en este flujo
                seen = set(); uniq = []
                for e in emails:
                    el = e.lower()
                    if el not in seen:
                        seen.add(el); uniq.append(e)
                out.append({"source": "lusha_person", "url": "", "emails": uniq})
        rr = enrich.get("rocketreach")
        if isinstance(rr, dict):
            emails = []
            items = rr.get("results") or rr.get("data") or rr.get("profiles") or []
            for it2 in (items or []):
                if not isinstance(it2, dict):
                    continue
                for k in ("email","work_email","current_work_email"):
                    v = it2.get(k)
                    if isinstance(v, str) and "@" in v:
                        emails.append(v.strip())
            if emails:
                seen = set(); uniq = []
                for e in emails:
                    el = e.lower()
                    if el not in seen:
                        seen.add(el); uniq.append(e)
                out.append({"source": "rocketreach", "url": "", "emails": uniq})
    return out

def _get_enriched_emails_list(domain: str) -> List[str]:
    items = _get_enriched_items_for_domain(domain)
    out: List[str] = []
    seen = set()
    for it in items:
        for e in (it.get("emails") or []):
            ee = (e or "").strip()
            if ee and ee.lower() not in seen:
                seen.add(ee.lower()); out.append(ee)
    return out

# ----------------- Clientes API -----------------
def lusha_prospect(domain: str, state: Optional[str], job_titles: List[str],
                   departments: List[str], seniority: List[str], per_page: int = 25) -> Dict[str, Any]:
    url = "https://api.lusha.com/prospecting/contact/search"
    # Esquema A recomendado (varía por plan/versión de Lusha):
    #   { companyDomains: [domain], contactFilters: { jobTitles, departments, seniority, locations }, page, perPage }
    contact_filters = {
        "jobTitles": job_titles,
        "departments": departments,
        "seniority": seniority
    }
    if state:
        contact_filters["locations"] = [{"country": "US", "region": state}]
    # Intento A1
    body_a1 = {
        "companyDomains": [domain],
        "contactFilters": contact_filters
    }
    t0 = time.time()
    try:
        r = requests.post(url, json=body_a1, headers={
            "api_key": LUSHA_API_KEY, "Content-Type": "application/json", "Accept": "application/json"
        }, timeout=40)
        ok = r.status_code
        if ok == 200 and "application/json" in (r.headers.get("Content-Type","")):
            data = r.json()
            append_log({"provider":"lusha","fn":"prospecting.search","domain":domain,
                        "status":ok,"ms":int((time.time()-t0)*1000)})
            return data

        # Intento A2 (alternativa company key)
        body_a2 = {
            "companiesDomains": [domain],
            "contactFilters": contact_filters
        }
        r_a2 = requests.post(url, json=body_a2, headers={
            "api_key": LUSHA_API_KEY, "Content-Type": "application/json", "Accept": "application/json"
        }, timeout=40)
        ok_a2 = r_a2.status_code
        if ok_a2 == 200 and "application/json" in (r_a2.headers.get("Content-Type","")):
            data_a2 = r_a2.json()
            append_log({"provider":"lusha","fn":"prospecting.search","domain":domain,
                        "status":ok_a2,"ms":int((time.time()-t0)*1000),"fallbackA2_no_pagination":True})
            return data_a2

        # Intento B1: filters.companies = [domain]
        body_b = {
            "filters": {
                "companies": [domain],
                "contacts": {
                    "job_titles": job_titles,
                    "departments": departments,
                    "seniority": seniority,
                }
            }
        }
        if state:
            body_b["filters"]["contacts"]["locations"] = [{"country": "US", "region": state}]
        r2 = requests.post(url, json=body_b, headers={
            "api_key": LUSHA_API_KEY, "Content-Type": "application/json", "Accept": "application/json"
        }, timeout=40)
        ok2 = r2.status_code
        if ok2 == 200 and "application/json" in (r2.headers.get("Content-Type","")):
            data2 = r2.json()
            append_log({"provider":"lusha","fn":"prospecting.search","domain":domain,
                        "status":ok2,"ms":int((time.time()-t0)*1000),"fallbackB":True})
            return data2
        append_log({"provider":"lusha","fn":"prospecting.search","domain":domain,
                    "status":ok2,"ms":int((time.time()-t0)*1000),"fallbackB":True,
                    "error_preview": (r2.text or "")[:400]})

        # Intento B2: filters.companies.domains = [domain]
        body_b2 = {
            "filters": {
                "companies": {"domains": [domain]},
                "contacts": {
                    "job_titles": job_titles,
                    "departments": departments,
                    "seniority": seniority,
                }
            }
        }
        if state:
            body_b2["filters"]["contacts"]["locations"] = [{"country": "US", "region": state}]
        r3 = requests.post(url, json=body_b2, headers={
            "api_key": LUSHA_API_KEY, "Content-Type": "application/json", "Accept": "application/json"
        }, timeout=40)
        ok3 = r3.status_code
        if ok3 == 200 and "application/json" in (r3.headers.get("Content-Type","")):
            data3 = r3.json()
            append_log({"provider":"lusha","fn":"prospecting.search","domain":domain,
                        "status":ok3,"ms":int((time.time()-t0)*1000),"fallbackB2":True})
            return data3
        append_log({"provider":"lusha","fn":"prospecting.search","domain":domain,
                    "status":ok3,"ms":int((time.time()-t0)*1000),"fallbackB2":True,
                    "error_preview": (r3.text or "")[:400]})

        # Intento B3: contacts camelCase
        body_b3 = {
            "filters": {
                "companies": {"domains": [domain]},
                "contacts": {
                    "jobTitles": job_titles,
                    "departments": departments,
                    "seniority": seniority,
                }
            }
        }
        if state:
            body_b3["filters"]["contacts"]["locations"] = [{"country": "US", "region": state}]
        r4 = requests.post(url, json=body_b3, headers={
            "api_key": LUSHA_API_KEY, "Content-Type": "application/json", "Accept": "application/json"
        }, timeout=40)
        ok4 = r4.status_code
        if ok4 == 200 and "application/json" in (r4.headers.get("Content-Type","")):
            data4 = r4.json()
            append_log({"provider":"lusha","fn":"prospecting.search","domain":domain,
                        "status":ok4,"ms":int((time.time()-t0)*1000),"fallbackB3":True})
            return data4
        # Nada funcionó: devolver último error con pista
        append_log({"provider":"lusha","fn":"prospecting.search","domain":domain,
                    "status":ok4,"ms":int((time.time()-t0)*1000),"fallbackB3":True,
                    "error_non200":True, "error_preview": (r4.text or "")[:400]})
        return {"error": (r4.text or r3.text or r2.text or r.text) or "lusha_prospect_non_200"}
    except Exception as ex:
        append_log({"provider":"lusha","fn":"prospecting.search","domain":domain,
                    "status":"EXC","error":str(ex)})
        return {"error": str(ex)}

def lusha_person_enrich(full_name: Optional[str]=None,
                        company_domain: Optional[str]=None,
                        linkedin_url: Optional[str]=None,
                        email: Optional[str]=None) -> Dict[str, Any]:
    """
    Lusha Person por POST usando un objeto 'contact' (según error reportado):
      contact debe tener una combinación válida, p.ej. firstName + lastName + company.domain
      Alternativamente linkedin_url o email.
    """
    url = "https://api.lusha.com/v2/person"
    # Construcción del contacto (formato contact + formatos aplanados para fallbacks)
    contact: Dict[str, Any] = {}
    if linkedin_url:
        contact["linkedinUrl"] = linkedin_url
    if email:
        contact["email"] = email
    # split de nombre si viene full_name
    if full_name and (" " in full_name):
        parts = [x for x in full_name.split(" ") if x]
        contact["firstName"] = parts[0]
        contact["lastName"] = parts[-1] if len(parts) > 1 else ""
    elif full_name:
        contact["firstName"] = full_name
    if company_domain:
        contact.setdefault("company", {})
        contact["company"]["domain"] = company_domain

    # Contacto aplanado (camelCase)
    contact_flat = {}
    if full_name:
        contact_flat["fullName"] = full_name
    if company_domain:
        contact_flat["companyDomain"] = company_domain
    if linkedin_url:
        contact_flat["linkedinUrl"] = linkedin_url
    if email:
        contact_flat["email"] = email
    # Variante snake_case
    contact_flat_snake = {}
    if full_name:
        contact_flat_snake["full_name"] = full_name
    if company_domain:
        contact_flat_snake["company_domain"] = company_domain
    if linkedin_url:
        contact_flat_snake["linkedin_url"] = linkedin_url
    if email:
        contact_flat_snake["email"] = email

    # Variante preferida de Lusha en algunos tenants: contacts[] con firstName/lastName + company.domain
    first_name = contact.get("firstName") or (full_name.split(" ")[0] if full_name else None)
    last_name = contact.get("lastName") if contact.get("lastName") is not None else (
        (full_name.split(" ")[-1] if full_name and len(full_name.split(" ")) > 1 else "")
    )
    contact_contacts_obj = {}
    if first_name:
        contact_contacts_obj["firstName"] = first_name
    if last_name is not None:
        contact_contacts_obj["lastName"] = last_name
    if company_domain:
        contact_contacts_obj["company"] = {"domain": company_domain}
    if linkedin_url:
        contact_contacts_obj["linkedinUrl"] = linkedin_url
    if email:
        contact_contacts_obj["email"] = email

    body = {"contact": contact}
    t0 = time.time()
    try:
        r = requests.post(url, json=body, headers={
            "api_key": LUSHA_API_KEY,
            "Content-Type": "application/json",
            "Accept": "application/json"
        }, timeout=40)
        ok = r.status_code
        ctype = r.headers.get("Content-Type", "")
        # 400: probar variantes de esquema de forma agresiva (según hints del tenant)
        if ok == 400:
            # 1) contacts[] con objeto anidado 'contact' + 'contactId' requerido por algunos tenants
            nested_contact: Dict[str, Any] = {}
            if linkedin_url:
                nested_contact["linkedinUrl"] = linkedin_url
            elif email:
                nested_contact["email"] = email
            elif full_name and company_domain:
                nested_contact["fullName"] = full_name
                nested_contact["companyDomain"] = company_domain
            else:
                # usar la mejor combinación disponible
                if full_name:
                    nested_contact["fullName"] = full_name
                if company_domain:
                    nested_contact["companyDomain"] = company_domain
                if contact.get("firstName") or contact.get("lastName"):
                    # algunos tenants aceptan first/last dentro de contact
                    if contact.get("firstName"):
                        nested_contact["firstName"] = contact["firstName"]
                    if contact.get("lastName") is not None:
                        nested_contact["lastName"] = contact.get("lastName")
                    if company_domain:
                        nested_contact["company"] = {"domain": company_domain}

            body2c = {"contacts": [{"contactId": str(uuid.uuid4()), "contact": nested_contact}]}
            r2c = requests.post(url, json=body2c, headers={
                "api_key": LUSHA_API_KEY,
                "Content-Type": "application/json",
                "Accept": "application/json"
            }, timeout=40)
            ok2c = r2c.status_code
            ctype2c = r2c.headers.get("Content-Type", "")
            if ok2c == 200 and "application/json" in ctype2c:
                data2c = r2c.json()
                append_log({"provider":"lusha","fn":"person.enrich","body":body2c,
                            "status":ok2c,"ms":int((time.time()-t0)*1000),"fallback_contacts_nested":True})
                return data2c
            else:
                append_log({"provider":"lusha","fn":"person.enrich","body":body2c,
                            "status":ok2c,"error":"fallback_contacts_nested_failed","raw": r2c.text})

            # 2) contacts[] con objeto estructurado firstName/lastName + company.domain
            body2a = {"contacts": [contact_contacts_obj]}
            r2a = requests.post(url, json=body2a, headers={
                "api_key": LUSHA_API_KEY,
                "Content-Type": "application/json",
                "Accept": "application/json"
            }, timeout=40)
            ok2a = r2a.status_code
            ctype2a = r2a.headers.get("Content-Type", "")
            if ok2a == 200 and "application/json" in ctype2a:
                data2a = r2a.json()
                append_log({"provider":"lusha","fn":"person.enrich","body":body2a,
                            "status":ok2a,"ms":int((time.time()-t0)*1000),"fallback_contacts_structured":True})
                return data2a
            else:
                append_log({"provider":"lusha","fn":"person.enrich","body":body2a,
                            "status":ok2a,"error":"fallback_contacts_structured_failed","raw": r2a.text})

            # 3) contacts[] con objeto APLANADO camelCase (fullName/companyDomain)
            body2 = {"contacts": [contact_flat]}
            r2 = requests.post(url, json=body2, headers={
                "api_key": LUSHA_API_KEY,
                "Content-Type": "application/json",
                "Accept": "application/json"
            }, timeout=40)
            ok2 = r2.status_code
            ctype2 = r2.headers.get("Content-Type", "")
            if ok2 == 200 and "application/json" in ctype2:
                data2 = r2.json()
                append_log({"provider":"lusha","fn":"person.enrich","body":body2,
                            "status":ok2,"ms":int((time.time()-t0)*1000),"fallback_contacts_array":True})
                return data2
            else:
                append_log({"provider":"lusha","fn":"person.enrich","body":body2,
                            "status":ok2,"error":"fallback_contacts_failed","raw": r2.text})

            # 4) contacts[] snake_case
            body2b = {"contacts": [contact_flat_snake]}
            r2b = requests.post(url, json=body2b, headers={
                "api_key": LUSHA_API_KEY,
                "Content-Type": "application/json",
                "Accept": "application/json"
            }, timeout=40)
            ok2b = r2b.status_code
            ctype2b = r2b.headers.get("Content-Type", "")
            if ok2b == 200 and "application/json" in ctype2b:
                data2b = r2b.json()
                append_log({"provider":"lusha","fn":"person.enrich","body":body2b,
                            "status":ok2b,"ms":int((time.time()-t0)*1000),"fallback_contacts_snake":True})
                return data2b
            else:
                append_log({"provider":"lusha","fn":"person.enrich","body":body2b,
                            "status":ok2b,"error":"fallback_contacts_snake_failed","raw": r2b.text})

            # 5) endpoint plural /v2/persons con contacts[]
            url3 = "https://api.lusha.com/v2/persons"
            r3 = requests.post(url3, json=(body2c if body2c else body2), headers={
                "api_key": LUSHA_API_KEY,
                "Content-Type": "application/json",
                "Accept": "application/json"
            }, timeout=40)
            ok3 = r3.status_code
            ctype3 = r3.headers.get("Content-Type", "")
            if ok3 == 200 and "application/json" in ctype3:
                data3 = r3.json()
                append_log({"provider":"lusha","fn":"persons.enrich","body":(body2c if body2c else body2),
                            "status":ok3,"ms":int((time.time()-t0)*1000),"fallback_persons_endpoint":True})
                return data3
            else:
                append_log({"provider":"lusha","fn":"persons.enrich","body":(body2c if body2c else body2),
                            "status":ok3,"error":"fallback_persons_failed","raw": r3.text})
        data = r.json() if (ok == 200 and "application/json" in ctype) else {"error": r.text}
        append_log({"provider":"lusha","fn":"person.enrich","body":body,
                    "status":ok,"ms":int((time.time()-t0)*1000)})
        return data
    except Exception as ex:
        append_log({"provider":"lusha","fn":"person.enrich","body":body,
                    "status":"EXC","error":str(ex)})
        return {"error": str(ex)}

def contactout_people_enrich(full_name: Optional[str]=None,
                             company_domain: Optional[str]=None,
                             linkedin_url: Optional[str]=None,
                             include_emails=True) -> Dict[str, Any]:
    url = "https://api.contactout.com/v1/people/enrich"
    body = {}
    if full_name:      body["full_name"] = full_name
    if company_domain:
        if isinstance(company_domain, str):
            body["company_domain"] = [company_domain]
        elif isinstance(company_domain, list):
            body["company_domain"] = company_domain
    if linkedin_url:   body["linkedin_url"] = linkedin_url
    if include_emails:
        body["include"] = ["work_email","personal_email","phone"]
    t0 = time.time()
    try:
        r = requests.post(url, json=body, headers={
            "token": CONTACTOUT_API_KEY,
            "Content-Type":"application/json",
            "Accept":"application/json"
        }, timeout=40)
        ok = r.status_code
        ctype = r.headers.get("Content-Type","")
        data = r.json() if "application/json" in ctype else {"raw": r.text}
        append_log({"provider":"contactout","fn":"people.enrich","body":body,
                    "status":ok,"ms":int((time.time()-t0)*1000)})
        return data
    except Exception as ex:
        append_log({"provider":"contactout","fn":"people.enrich","body":body,
                    "status":"EXC","error":str(ex)})
        return {"error": str(ex)}

def contactout_linkedin_enrich(linkedin_url: str) -> Dict[str, Any]:
    url = "https://api.contactout.com/v1/linkedin/enrich"
    params = {"profile": linkedin_url}
    t0 = time.time()
    try:
        r = requests.get(url, params=params, headers={
            "token": CONTACTOUT_API_KEY,
            "Content-Type":"application/json",
            "Accept":"application/json"
        }, timeout=40)
        ok = r.status_code
        ctype = r.headers.get("Content-Type","")
        data = r.json() if "application/json" in ctype else {"raw": r.text}
        append_log({"provider":"contactout","fn":"linkedin.enrich","profile":linkedin_url,
                    "status":ok,"ms":int((time.time()-t0)*1000)})
        return data
    except Exception as ex:
        append_log({"provider":"contactout","fn":"linkedin.enrich","profile":linkedin_url,
                    "status":"EXC","error":str(ex)})
        return {"error": str(ex)}

def rocketreach_people_search(name: Optional[str]=None,
                              company_domain: Optional[str]=None,
                              current_title: Optional[str]=None,
                              linkedin_url: Optional[str]=None) -> Dict[str, Any]:
    """
    Placeholder típico de RocketReach. Ajusta la ruta según tu plan.
    """
    url = "https://api.rocketreach.co/v2/api/search/person"
    body = {}
    if name:           body["name"] = name
    if company_domain: body["company_domain"] = company_domain
    if current_title:  body["current_title"] = current_title
    if linkedin_url:   body["linkedin_url"] = linkedin_url
    t0 = time.time()
    try:
        r = requests.post(url, json=body, headers={
            "X-Api-Key": ROCKETREACH_API_KEY,
            "Content-Type":"application/json"
        }, timeout=40)
        ok = r.status_code
        ctype = r.headers.get("Content-Type","")
        if "application/json" not in ctype:
            data = {
                "error": "rocketreach_html_response_or_404",
                "hint": "Verificá la base URL/endpoint del plan y el header de auth",
                "status": ok,
                "content_type": ctype,
                "raw_preview": (r.text or "")[:400]
            }
        else:
            data = r.json()
        append_log({"provider":"rocketreach","fn":"search.person","body":body,
                    "status":ok,"ms":int((time.time()-t0)*1000)})
        return data
    except Exception as ex:
        append_log({"provider":"rocketreach","fn":"search.person","body":body,
                    "status":"EXC","error":str(ex)})
        return {"error": str(ex)}

# ----------------- Parsers de respuestas API -> people items -----------------
def _get_company_domain_from_lusha_item(item: Dict[str, Any]) -> str:
    # Best-effort: distintos tenants pueden usar claves diferentes
    # Intentamos varias rutas comunes
    for k in ("companyDomain", "currentCompanyDomain", "domain"):
        v = (item.get(k) or "").strip()
        if v:
            return v
    # Objetos anidados
    for k in ("company", "organization", "currentCompany", "employer"):
        obj = item.get(k)
        if isinstance(obj, dict):
            for kk in ("domain", "companyDomain", "website"):
                v = (obj.get(kk) or "").strip()
                if v:
                    # website podría venir con esquema
                    v = v.replace("https://",""").replace("http://",""")
                    v = v.split("/")[0]
                    return v
    return ""

def extract_people_from_lusha_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convierte un contacto Lusha a {name, role, email, source, department?, seniority?}
    """
    full_name = item.get("fullName") or item.get("name") or ""
    title = item.get("title") or item.get("jobTitle") or ""
    email = (item.get("email") or item.get("workEmail") or item.get("emailAddress") or "").strip()
    dept = item.get("department") or ""
    seniority = item.get("seniority") or ""
    company_domain = _get_company_domain_from_lusha_item(item)
    return {
        "name": full_name,
        "role": title,
        "email": email,
        "source": "lusha",
        "department": dept,
        "seniority": seniority,
        "company_domain": company_domain
    }

def parse_lusha_prospect(resp: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    if not isinstance(resp, dict):
        return out
    # Ajustar según forma exacta de tu cuenta; común: {"contacts": [{...}, ...]}
    contacts = resp.get("contacts") or resp.get("data") or []
    if isinstance(contacts, dict) and "items" in contacts:
        contacts = contacts["items"]
    for c in contacts or []:
        out.append(extract_people_from_lusha_item(c))
    return out

def parse_lusha_person(resp: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    if not isinstance(resp, dict):
        return out
    # Puede venir como objeto único, lista en 'data', o lista en 'contacts', o plano
    items = None
    if isinstance(resp.get("data"), list):
        items = resp["data"]
    elif isinstance(resp.get("data"), dict):
        items = [resp["data"]]
    elif isinstance(resp.get("contacts"), list):
        items = resp["contacts"]
    elif isinstance(resp, list):
        items = resp
    else:
        items = [resp]

    for item in items:
        if not isinstance(item, dict):
            continue
        full_name = item.get("fullName") or item.get("name") or ""
        title = item.get("title") or item.get("jobTitle") or ""
        email = (item.get("email") or item.get("workEmail") or item.get("emailAddress") or "").strip()
        dept = item.get("department") or ""
        seniority = item.get("seniority") or ""
        cd = _get_company_domain_from_lusha_item(item)
        if full_name or title or email:
            out.append({
                "name": full_name,
                "role": title,
                "email": email,
                "source": "lusha",
                "department": dept,
                "seniority": seniority,
                "company_domain": cd
            })
    return out

def parse_contactout_enrich(resp: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    if not isinstance(resp, dict):
        return out
    # People Enrich suele devolver un objeto con campos email(s) y title
    # Estructura varía; soportamos keys comunes:
    name = resp.get("full_name") or resp.get("name") or ""
    title = resp.get("job_title") or resp.get("title") or ""
    dept = resp.get("department") or ""
    seniority = resp.get("seniority") or ""
    emails = []
    for k in ("work_email", "personal_email", "email"):
        v = resp.get(k)
        if isinstance(v, str) and v:
            emails.append(v)
    if not emails and isinstance(resp.get("emails"), list):
        emails = [e for e in resp["emails"] if isinstance(e, str)]
    if name or title or emails:
        if emails:
            for e in emails:
                out.append({
                    "name": name, "role": title, "email": e,
                    "source": "contactout", "department": dept, "seniority": seniority
                })
        else:
            out.append({
                "name": name, "role": title, "email": "",
                "source": "contactout", "department": dept, "seniority": seniority
            })
    return out

def parse_rocketreach(resp: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    if not isinstance(resp, dict):
        return out
    items = resp.get("results") or resp.get("data") or resp.get("profiles") or []
    for it in items or []:
        name = it.get("name") or f"{it.get('first_name','')} {it.get('last_name','')}".strip()
        title = it.get("current_title") or it.get("title") or ""
        dept = it.get("department") or ""
        seniority = it.get("seniority") or ""
        emails = []
        for k in ("email", "work_email", "current_work_email"):
            v = it.get(k)
            if isinstance(v, str) and v:
                emails.append(v)
        if name or title or emails:
            if emails:
                for e in emails:
                    out.append({
                        "name": name, "role": title, "email": e,
                        "source": "rocketreach", "department": dept, "seniority": seniority
                    })
            else:
                out.append({
                    "name": name, "role": title, "email": "",
                    "source": "rocketreach", "department": dept, "seniority": seniority
                })
    return out

# ----------------- Acciones de enriquecimiento -----------------
def enrich_from_people_list(domain: str, people: List[Dict[str, Any]],
                            use_lusha: bool, use_contactout: bool, use_rocketreach: bool) -> Dict[str, Any]:
    results = []
    for p in people:
        name = p.get("name") or ""
        role = clean_role(p.get("role") or "")
        linkedin = p.get("linkedin") or p.get("linkedin_url") or ""
        # Evitar llamados con nombres dudosos (ej: 'Read More', ciudades, etc.)
        if not is_probably_person_name(name):
            results.append({"source_person": p, "enrich": {"skip_reason": "not_probable_person_name"}})
            continue
        item = {"source_person": p, "enrich": {}}
        if use_contactout and linkedin:
            co = contactout_linkedin_enrich(linkedin)
            item["enrich"]["contactout_linkedin"] = co
            items = parse_contactout_enrich(co)
            if items:
                upsert_v3_people(domain, items)
        if use_contactout and name:
            co2 = contactout_people_enrich(full_name=name, company_domain=domain)
            item["enrich"]["contactout_people"] = co2
            items = parse_contactout_enrich(co2)
            if items:
                upsert_v3_people(domain, items)
        if use_lusha and name:
            lu = lusha_person_enrich(full_name=name, company_domain=domain)
            item["enrich"]["lusha_person"] = lu
            items = parse_lusha_person(lu)
            if items:
                upsert_v3_people(domain, items)
        if use_rocketreach and name:
            rr = rocketreach_people_search(name=name, company_domain=domain, current_title=role)
            item["enrich"]["rocketreach"] = rr
            items = parse_rocketreach(rr)
            if items:
                upsert_v3_people(domain, items)
        results.append(item)
    return {"contacts_enriched": results}

def _domain_matches(candidate: str, target: str) -> bool:
    c = (candidate or "").lower().strip()
    t = (target or "").lower().strip()
    if not c:
        return False
    # match por igualdad o por sufijo (maneja subdominios)
    return c == t or c.endswith("." + t)

def enrich_from_domain(domain: str, state_filter: Optional[str], per_page: int,
                       job_titles: List[str], departments: List[str], seniority: List[str],
                       use_lusha: bool, use_rocketreach: bool) -> Dict[str, Any]:
    out = {"prospecting": {}}
    if use_lusha:
        resp = lusha_prospect(domain, state_filter, job_titles, departments, seniority, per_page)
        out["prospecting"]["lusha"] = resp
        items = parse_lusha_prospect(resp)
        # Filtramos por company_domain cuando esté disponible
        filtered = []
        for it in items:
            cd = (it.get("company_domain") or "").strip()
            if not cd or _domain_matches(cd, domain):
                filtered.append(it)
        items = filtered
        if items:
            upsert_v3_people(domain, items)
    if use_rocketreach:
        rr = rocketreach_people_search(company_domain=domain)
        out["prospecting"]["rocketreach"] = rr
        items = parse_rocketreach(rr)
        if items:
            upsert_v3_people(domain, items)
    return out

# --- Section switch: render Control after all helpers are defined ---
if section == "Control":
    _render_control_dashboard()
    st.stop()

# ----------------- UI -----------------
st.title("V4 • Enriquecer contactos (Lusha / ContactOut / RocketReach)")
if not current_perms().get('modify'):
    st.info("Modo lectura: tu rol no permite modificar ni enriquecer (solo lectura / export restringido).")
st.caption("Lee tu JSON, muestra emails/personas por dominio y permite enriquecer (por dominio o persona). Guarda incrementalmente en out/enriquecidov3.json (estructura input) y out/enriquecidos.json (crudo). Logs en logs/logs_apis.json.")

# Fuente de datos
st.sidebar.header("Fuente de datos")
file_input = st.sidebar.file_uploader("Subí el archivo .json (estructura 'version' + 'sites')", type=["json"])
state_filter = st.sidebar.text_input("Filtro región (US state, ej. FL)", value="FL")
per_page = st.sidebar.number_input("Resultados por página (Lusha)", min_value=1, max_value=100, value=25, step=1)

# Proveedores
st.sidebar.header("Proveedores")
use_lusha       = st.sidebar.checkbox("Lusha", value=bool(LUSHA_API_KEY))
use_contactout  = st.sidebar.checkbox("ContactOut", value=bool(CONTACTOUT_API_KEY))
use_rocketreach = st.sidebar.checkbox("RocketReach", value=False)
if not use_rocketreach:
    st.sidebar.caption("RocketReach deshabilitado por defecto hasta validar endpoint y autenticación del plan.")

# Opciones por lote
st.sidebar.header("Ejecución")
select_all   = st.sidebar.checkbox("Seleccionar todos por defecto", value=False)
auto_mode    = st.sidebar.checkbox("Modo automático (por registro)", value=False)
run_auto     = st.sidebar.button("Procesar TODOS (Auto Global)")

# Diccionarios
st.sidebar.markdown("---")
st.sidebar.header("Diccionarios de búsqueda")
jt_text = st.sidebar.text_area("Job Titles (coma-separados)", value=", ".join(DEFAULT_JOB_TITLES), height=100)
dept_text = st.sidebar.text_area("Departments (coma-separados)", value=", ".join(DEFAULT_DEPARTMENTS), height=80)
dept_filter_text = st.sidebar.text_input("Dept Filter (tabla final; coma-separados)", value=", ".join(sorted(DEFAULT_DEPT_FILTER)))
seniority_text = st.sidebar.text_area("Seniority (coma-separados)", value=", ".join(DEFAULT_SENIORITY), height=80)
seniority_ok_text = st.sidebar.text_input("Seniority OK (tabla final; coma-separados)", value=", ".join(sorted(DEFAULT_SENIORITY_OK)))
can_modify = current_perms().get('modify')
save_dicts = st.sidebar.button("Guardar diccionarios en .env", disabled=not can_modify)
if not can_modify:
    st.sidebar.caption("Tu rol no permite modificar diccionarios (.env)")

if save_dicts and can_modify:
    set_env_list("ENRICH_JOB_TITLES", [x for x in jt_text.split(",")])
    set_env_list("ENRICH_DEPARTMENTS", [x for x in dept_text.split(",")])
    set_env_list("ENRICH_DEPT_FILTER", [x for x in dept_filter_text.split(",")])
    set_env_list("ENRICH_SENIORITY", [x for x in seniority_text.split(",")])
    set_env_list("ENRICH_SENIORITY_OK", [x for x in seniority_ok_text.split(",")])
    st.sidebar.success("Guardado en .env. Valores aplicados en memoria.")

    # Refrescar en memoria
    DEFAULT_JOB_TITLES[:] = get_env_list("ENRICH_JOB_TITLES", DEFAULT_JOB_TITLES)
    DEFAULT_DEPARTMENTS[:] = get_env_list("ENRICH_DEPARTMENTS", DEFAULT_DEPARTMENTS)
    DEFAULT_DEPT_FILTER.clear(); DEFAULT_DEPT_FILTER.update(get_env_list("ENRICH_DEPT_FILTER", list(DEFAULT_DEPT_FILTER)))
    DEFAULT_SENIORITY[:] = get_env_list("ENRICH_SENIORITY", DEFAULT_SENIORITY)
    DEFAULT_SENIORITY_OK.clear(); DEFAULT_SENIORITY_OK.update(get_env_list("ENRICH_SENIORITY_OK", list(DEFAULT_SENIORITY_OK)))

# ----------------- Cargar dataset -----------------
base_input = None
if file_input is not None:
    try:
        base_input = json.load(file_input)
    except Exception as ex:
        st.error(f"No se pudo leer el JSON: {ex}")

if base_input is None:
    # Intento de carga por defecto
    default_candidates = [OUT_DIR / "etapa1_v1.json", ROOT / "etapa1_v1.json"]
    for p in default_candidates:
        try:
            if p.exists():
                base_input = load_json(p)
                if isinstance(base_input, dict):
                    st.sidebar.info(f"Cargado por defecto: {p}")
                    break
        except Exception:
            pass
    if base_input is None:
        st.info("Subí el archivo .json en la barra lateral para comenzar.")
        st.stop()

version, records = normalize_records(base_input)
if not records:
    st.warning("No se encontraron registros en 'sites'. Verifica la estructura del JSON.")
    st.stop()

#! Resumen se moverá más abajo para mostrar primero los filtros

# Asegurar estructura enriquecidov3.json
_ = ensure_v3_structure(base_input)

# ----------------- Filtros/orden por score -----------------
def _band_score(rec: Dict[str, Any]) -> int:
    try:
        return int((((rec.get("raw") or {}).get("band") or {}).get("score") or 0))
    except Exception:
        return 0

scores_all = [ _band_score(r) for r in records ]
min_s = int(min(scores_all)) if scores_all else 0
max_s = int(max(scores_all)) if scores_all else 0

st.markdown("### Filtros")
if min_s < max_s:
    score_min, score_max = st.slider(
        "Filtrar por score",
        min_value=min_s,
        max_value=max_s,
        value=(min_s, max_s)
    )
else:
    score_min, score_max = min_s, max_s
    st.caption(f"Todos los items tienen el mismo score: {min_s}")

search_query = st.text_input(
    "Buscar (nombre, dominio, email)",
    value="",
    placeholder="Ej: acme.com, Juan Perez, info@empresa.com"
)

sort_dir = st.selectbox("Ordenar por score", ["Descendente", "Ascendente"], index=0)
recs_scored = [ (r, _band_score(r)) for r in records ]
recs_filtered = [ (r, s) for (r, s) in recs_scored if s >= score_min and s <= score_max ]

def _matches_search(rec: Dict[str, Any], q: str) -> bool:
    if not q:
        return True
    ql = q.strip().lower()
    if not ql:
        return True
    haystack: List[str] = []
    haystack.append((rec.get("domain") or ""))
    haystack.append((rec.get("nombre") or ""))
    for e in (rec.get("emails") or []):
        haystack.append(e or "")
    for p in (rec.get("people") or []):
        if isinstance(p, dict):
            haystack.append((p.get("name") or ""))
            haystack.append((p.get("email") or ""))
    return any(ql in (h or "").lower() for h in haystack)

if search_query.strip():
    recs_filtered = [ (r, s) for (r, s) in recs_filtered if _matches_search(r, search_query) ]

recs_sorted = sorted(recs_filtered, key=lambda x: x[1], reverse=(sort_dir == "Descendente"))
display_records = [ r for (r, _) in recs_sorted ]

# Ahora mostramos el resumen debajo de los filtros
st.markdown("---")
st.subheader("Resumen")
st.write(f"Registros cargados: **{len(records)}**")
st.write("Resultados se guardan en: `out/enriquecidov3.json` (estructura input) y `out/enriquecidos.json` (crudo). Logs en `logs/logs_apis.json`.")

# Resumen extendido por rubro (CSV)
csv_dir = ROOT / "csv"
rubros_map = scan_csv_rubros(csv_dir)
if rubros_map:
    # Mapear registros actuales por dominio base -> record
    by_domain_base: Dict[str, Dict[str, Any]] = {}
    for r in records:
        d = _base_domain(r.get("domain") or "")
        if d:
            by_domain_base[d] = r

    total_procesados = 0
    total_validos = 0
    rubros_rows: List[Dict[str, Any]] = []
    total_explorar = 0
    total_zona_ok = 0
    total_score_ok = 0
    for rubro, doms in sorted(rubros_map.items()):
        explorar = len(doms)  # dominios únicos a explorar desde CSV
        procesados = 0        # dominios presentes en etapa1_v1 (records cargados)
        zona_ok = 0
        score_ok = 0
        validos = 0
        for d in doms:
            rec = by_domain_base.get(d)
            if rec:
                procesados += 1
                if _ok_zona(rec):
                    zona_ok += 1
                try:
                    s = int((((rec.get("raw") or {}).get("band") or {}).get("score") or 0))
                except Exception:
                    s = 0
                if s > 5:
                    score_ok += 1
                if (s > 5) and _ok_zona(rec):
                    validos += 1
        pct = (validos / procesados * 100.0) if procesados > 0 else 0.0
        rubros_rows.append({
            "Rubro": rubro,
            "A explorar (CSV)": explorar,
            "Procesados (etapa1)": procesados,
            "Zona OK": zona_ok,
            "Score > 5": score_ok,
            "Válidos (ambas)": validos,
            "% Válidos": round(pct, 1)
        })
        total_explorar += explorar
        total_procesados += procesados
        total_zona_ok += zona_ok
        total_score_ok += score_ok
        total_validos += validos

    pct_total = (total_validos / total_procesados * 100.0) if total_procesados > 0 else 0.0

    # Estimado al fin de etapa 1 = % total × Total a explorar (proyecta si aún quedan por procesar)
    estimado = int(round((pct_total / 100.0) * total_explorar))

    # Cantidad de mails extraídos etapa 1 = base + people + enriquecidos (únicos)
    mails_stage1_set = set()
    for r in records:
        for e in (r.get("emails") or []):
            if isinstance(e, str) and e.strip() and "@" in e:
                mails_stage1_set.add(e.strip().lower())
        for p in (r.get("people") or []):
            pe = (p.get("email") or "").strip()
            if pe and "@" in pe:
                mails_stage1_set.add(pe.lower())
        for e2 in _get_enriched_emails_list(_base_domain(r.get("domain") or r.get("domain") or "")):
            if e2 and "@" in e2:
                mails_stage1_set.add(e2.strip().lower())

    # Mostrar tablas
    st.markdown("**Detalle por rubro (CSV)**")
    df_rubros = pd.DataFrame(rubros_rows)
    st.dataframe(df_rubros, use_container_width=True)

    st.markdown("**Totales**")
    df_totales = pd.DataFrame([{
        "A explorar (CSV)": total_explorar,
        "Procesados (etapa1)": total_procesados,
        "Zona OK": total_zona_ok,
        "Score > 5": total_score_ok,
        "Válidos (ambas)": total_validos,
        "% Total": round(pct_total, 1),
        "Estimado fin Etapa 1": estimado,
        "Mails Etapa 1 (únicos)": len(mails_stage1_set)
    }])
    st.table(df_totales)

# ----------------- Paginación -----------------
st.markdown("---")
st.subheader("Paginación")
page_size_label = "Mostrar resultados"
page_size_options = [50, 100, 200, 500, "Todos"]
default_idx = 0
page_size_choice = st.selectbox(page_size_label, page_size_options, index=default_idx, format_func=lambda x: str(x))

total_items = len(display_records)
if page_size_choice == "Todos":
    page_size = total_items if total_items > 0 else 1
else:
    page_size = int(page_size_choice)

total_pages = max(1, (total_items + page_size - 1) // page_size)
if "page_num" not in st.session_state:
    st.session_state["page_num"] = 1

# Reset page if filters changed drastically could be handled; keep simple for now
page_num = st.number_input("Página", min_value=1, max_value=total_pages, value=min(st.session_state["page_num"], total_pages), step=1)
st.session_state["page_num"] = page_num

start_idx = (page_num - 1) * page_size
end_idx = min(start_idx + page_size, total_items)
page_records = display_records[start_idx:end_idx]
st.caption(f"Página {page_num}/{total_pages} • Mostrando {end_idx - start_idx} de {total_items} registros")

# ========== Exportar a Excel ========== #
st.markdown("---")
st.header("Exportar a Excel")

# Score range selection (uses sidebar slider)
export_score_min, export_score_max = score_min, score_max

# File name/location input
default_filename = f"base2025FL_Todos_{datetime.now().strftime('%Y%m%d')}.xlsx"
filename = st.text_input("Nombre de archivo Excel", value=default_filename)

# Export button
can_export = current_perms().get('export')
export_btn = st.button("Exportar a Excel", disabled=not can_export)
if not can_export:
    st.caption("Tu rol no permite exportar (requiere rol editor o supervisor).")

# Build export dataframe
def get_social(rec, plat):
    socials = (rec.get("raw", {}).get("socials") or [])
    for s in socials:
        if isinstance(s, dict):
            p = (s.get("platform") or "").lower()
            url = s.get("url") or s.get("link") or ""
            if plat in p or plat in url.lower():
                return url
    return ""

def get_mails_todos(rec):
    emails = rec.get("emails") or []
    people = rec.get("people") or []
    ext_emails_map, _ = load_external_maps()
    ext_all = _collect_external_values(ext_emails_map, rec["domain"])
    emails_web_saved = _get_v3_emails_web(rec["domain"])
    enriched_emails = _get_enriched_emails_list(rec["domain"])  # nuevos de enriquecidos.json
    people_emails = [p.get("email") for p in people if p.get("email")]
    combined = []
    seen = set()
    for source in [emails, people_emails, ext_all, emails_web_saved, enriched_emails]:
        for e in source:
            if e and "@" in e and e.lower() not in seen:
                seen.add(e.lower())
                combined.append(e)
    return ", ".join(combined)

def build_export_df(records, score_min, score_max):
    rows = []
    for r in records:
        score = _band_score(r)
        if score < score_min or score > score_max:
            continue
        nombre = r.get("nombre") or r.get("domain")
        web = r.get("domain")
        mails_todos = get_mails_todos(r)
        facebook = get_social(r, "facebook")
        linkedin = get_social(r, "linkedin")
        # Resto: redes, direcciones, telefonos, etc
        redes = []
        for plat in ["instagram","youtube","tiktok","x","twitter","bandcamp","soundcloud"]:
            u = get_social(r, plat)
            if u:
                redes.append(f"{plat}: {u}")
        direcciones = "; ".join([a.get("value") for a in (r.get("addresses") or []) if a.get("value")])
        telefonos = ", ".join([p.get("value") for p in (r.get("phones") or []) if p.get("value")])
        rows.append({
            "Score": score,
            "Nombre": nombre,
            "web": web,
            "mails_todos": mails_todos,
            "red_social_facebook": facebook,
            "red_social_linkedin": linkedin,
            "redes": "; ".join(redes),
            "direcciones": direcciones,
            "telefonos": telefonos,
        })
    df = pd.DataFrame(rows)
    return df

if export_btn:
    with st.spinner("Exportando registros..."):
        # Exportar la vista actual (respeta score + búsqueda)
        df_todos = build_export_df(display_records, export_score_min, export_score_max)
        df_conmails = df_todos[df_todos["mails_todos"].str.strip() != ""]
        # Save both files
        out_path_todos = filename if filename else "base2025FL_Todos.xlsx"
        out_path_conmails = out_path_todos.replace("Todos", "conMails")
        try:
            df_todos.to_excel(out_path_todos, index=False)
            df_conmails.to_excel(out_path_conmails, index=False)
            # Descarga directa usando buffer en memoria
            buf_all = io.BytesIO()
            df_todos.to_excel(buf_all, index=False)
            buf_mail = io.BytesIO()
            df_conmails.to_excel(buf_mail, index=False)
        except Exception as ex:
            st.error(f"Error al exportar: {ex}")
            buf_all = None
            buf_mail = None
    # Fuera del spinner mostramos resultado y descargas
    if buf_all is not None and buf_mail is not None:
        st.success(f"Exportado: {out_path_todos} ({len(df_todos)} registros), {out_path_conmails} ({len(df_conmails)} con mails)")
        st.download_button(
            "Descargar Excel (Todos)",
            data=buf_all.getvalue(),
            file_name=os.path.basename(out_path_todos),
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        st.download_button(
            "Descargar Excel (conMails)",
            data=buf_mail.getvalue(),
            file_name=os.path.basename(out_path_conmails),
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

st.caption(f"Mostrando {len(page_records)} de {len(records)} registros (filtrados: {len(display_records)})")

# ----------------- Render por registro -----------------
st.subheader("Registros")
filtered_contacts_for_table = []  # acumulamos para la tabla final

for idx, rec in enumerate(page_records, start=start_idx + 1):
    domain = rec["domain"]
    nombre = rec["nombre"]
    emails = rec["emails"]
    people = rec["people"]
    raw = rec.get("raw", {})
    # Puntaje (band.score) para mostrar en el título
    try:
        band_score = int(((raw.get("band") or {}).get("score") or 0))
    except Exception:
        band_score = 0
    raw_socials = raw.get("socials") or []
    linkedin_links = []
    for s in raw_socials:
        if not isinstance(s, dict):
            continue
        plat = (s.get("platform") or "").lower()
        url = s.get("url") or s.get("link") or ""
        if ("linkedin" in plat) or ("linkedin.com" in url.lower()):
            if url:
                linkedin_links.append(url)

    with st.expander(f"[{idx}] {nombre}  •  {domain}  •  score: {band_score}", expanded=False):
        # Mails Todos (combinados) al inicio del item
        try:
            ext_emails_map2, _ = load_external_maps()
        except Exception:
            ext_emails_map2 = {}
        # Emails desde people
        people_emails = []
        for p in (people or []):
            pe = (p.get("email") or "").strip()
            if pe and "@" in pe:
                people_emails.append(pe)
        # Emails externos via mapas + emails_web guardados
        ext_all = _collect_external_values(ext_emails_map2, domain)
        emails_web_saved = _get_v3_emails_web(domain)
        enriched_emails_from_raw = _get_enriched_emails_list(domain)
        combined = []
        seen_all = set()
        for source_list in [emails or [], people_emails, ext_all, emails_web_saved, enriched_emails_from_raw]:
            for e in source_list:
                ee = (e or "").strip()
                if not ee or "@" not in ee:
                    continue
                eel = ee.lower()
                if eel in seen_all:
                    continue
                seen_all.add(eel)
                combined.append(ee)
        all_emails_csv = ", ".join(combined)
        if combined:
            st.markdown("**Mails Todos**")
            html_id = f"all_emails_{idx}"
            html_block = f'''
                <div style="display:flex; gap:8px; align-items:center; flex-wrap:wrap;">
                  <div id="{html_id}" style="flex:1; padding:8px; background:#f7f7f9; border:1px solid #e1e1e8; border-radius:6px; font-family:monospace; font-size:12px; white-space:normal; word-break:break-word;">
                    {all_emails_csv}
                  </div>
                  <button onclick="navigator.clipboard.writeText(document.getElementById('{html_id}').innerText)" style="padding:6px 10px; cursor:pointer;">
                    Copiar
                  </button>
                </div>
            '''
            components.html(html_block, height=80)

        # Ingreso manual y guardado a etapa 1
        manual_key = f"manual_emails_{domain}"
        manual_val = st.text_input(
        "Ingreso manual de mails (separados por coma)",
        value="",
        key=manual_key,
        placeholder="ej: contacto@dominio.com, otra@dominio.com"
        )
        if st.button("Guardar a etapa 1", key=f"save_manual_{domain}", disabled=not current_perms().get('modify')):
            emails_new = []
            for part in (manual_val or "").split(","):
                e = part.strip()
                if e and "@" in e:
                    emails_new.append(e)
            if emails_new:
                if current_perms().get('modify'):
                    added = upsert_emails_to_etapa1(domain, emails_new, base_input)
                    if added:
                        st.success(f"Guardados {added} email(s) en etapa1_v1.json")
                    else:
                        st.info("No había emails nuevos para guardar.")
                else:
                    st.warning("No tenés permiso para guardar (rol lector).")
            else:
                st.warning("Ingresá al menos un email válido.")

        cols_top = st.columns([2,2,1,1,1,1])
        with cols_top[0]:
            st.markdown("**Emails (existentes + people + externos)**")
            # base emails del JSON
            base_emails = [ (e or '').strip() for e in (emails or []) if (e or '').strip() ]
            base_set = { e.lower() for e in base_emails }
            # emails desde people
            people_emails_inline = []
            for p in (people or []):
                pe = (p.get("email") or "").strip()
                if pe and "@" in pe and pe.lower() not in base_set:
                    base_set.add(pe.lower())
                    people_emails_inline.append(pe)
            # Agregar emails y names desde búsquedas externas (si existen)
            try:
                ext_emails_map, ext_names_map = load_external_maps()
            except Exception:
                ext_emails_map, ext_names_map = {}, {}
            # Emails externos (considerando variantes y subdominios)
            ext_emails = _collect_external_values(ext_emails_map, domain)
            # Limpiar y excluir duplicados ya listados arriba
            seen = set()
            ext_clean = []
            for e in ext_emails:
                ee = (e or "").strip()
                if not ee:
                    continue
                eel = ee.lower()
                if "@" not in eel:
                    continue
                if eel in base_set or eel in seen:
                    continue
                seen.add(eel)
                ext_clean.append(ee)
            # Mostrar lista combinada (base + people + externos)
            combined_inline = base_emails + people_emails_inline + ext_clean
            if combined_inline:
                st.write("\n".join(f"- {e}" for e in combined_inline))
            else:
                st.write("_Sin emails para este item_")
            if ext_clean:
                st.caption("(Incluye correos detectados por búsquedas externas)")
                if st.button("Guardar mails externos en enriquecidov3", key=f"btn_save_ext_emails_{domain}", disabled=not current_perms().get('modify')):
                    added = upsert_v3_external_emails(domain, ext_clean, merge_into_emails=False)
                    if added:
                        st.success(f"Guardados {added} email(s) en emails_web de enriquecidov3.json")
                    else:
                        st.info("No había emails nuevos para guardar en emails_web")

            # Names externos (si existieran)
            ext_names = _collect_external_values(ext_names_map, domain)
            if ext_names:
                st.markdown("**names de busqueda externa**")
                st.write("\n".join(f"- {n}" for n in ext_names[:50]))
        with cols_top[1]:
            st.markdown("**People (existentes en input)**")
            if people:
                for p in people[:100]:
                    line = f"- {p.get('name','(s/nombre)')} — {p.get('role','(s/rol)')}  {('• '+p.get('email','')) if p.get('email') else ''}"
                    st.write(line)
            else:
                st.write("_Sin people en el JSON_")

        # Bloque: Enriquecidos (muestra fuente, URL y emails detectados en enriquecidos.json)
        enriched_items = _get_enriched_items_for_domain(domain)
        if enriched_items:
            st.markdown("**Enriquecidos**")
            for it in enriched_items:
                source = it.get("source") or ""
                url = it.get("url") or ""
                emails_list = it.get("emails") or []
                # Etiqueta de fuente amigable
                source_label = "contactout_people" if source == "contactout_people" else ("contactout_linkedin" if source == "contactout_linkedin" else source)
                # Render
                if url:
                    st.write(f"- enriquecidos ({source_label}) • URL: {url}")
                else:
                    st.write(f"- enriquecidos ({source_label})")
                if emails_list:
                    st.write("  emails: " + ", ".join(emails_list))

        # (Se movió "Mails Todos" al inicio del item)

        # Socials detectadas (todas las plataformas)
        social_map: Dict[str, List[str]] = {}
        for s in raw_socials:
            if not isinstance(s, dict):
                continue
            url = (s.get("url") or s.get("link") or "").strip()
            if not url:
                continue
            plat = _infer_platform(url, s.get("platform") or s.get("name") or s.get("type") or "")
            social_map.setdefault(plat, [])
            if url not in social_map[plat]:
                social_map[plat].append(url)
        if social_map:
            st.markdown("**Socials detectadas**")
            for plat in sorted(social_map.keys()):
                links_str = "  ".join(f"[{u}]({u})" for u in social_map[plat][:5])
                st.write(f"- {plat}: {links_str}")
        # Agregar LinkedIn desde people también
        ppl_linkedin = _linkedin_from_people(people)
        all_linkedin = sorted(set(list(linkedin_links) + ppl_linkedin))
        if all_linkedin:
            st.markdown("**LinkedIn detectado**")
            for lk in all_linkedin:
                st.markdown(f"- [{lk}]({lk})")
            if st.button("Guardar LinkedIn en etapa 1", key=f"btn_save_li_{domain}", disabled=not current_perms().get('modify')):
                added = upsert_linkedin_to_etapa1(domain, all_linkedin, base_input)
                if added:
                    st.success(f"Guardado en etapa1_v1.json ({added} URL(s) nuevas).")
                else:
                    st.info("No había URLs nuevas para guardar.")
        else:
            # Único botón: Navegar secciones y GUARDAR automáticamente el primer resultado
            if st.button("Buscar y guardar LinkedIn (Navegar secciones)", key=f"btn_find_li_nav_autosave_{domain}", disabled=not current_perms().get('modify')):
                with st.spinner("Navegando secciones del sitio y buscando LinkedIn..."):
                    hits = _navigate_sections_linkedin(domain)
                if hits:
                    first = hits[0]
                    added = upsert_linkedin_to_etapa1(domain, [first], base_input)
              
                    st.success(f"Detectado y guardado: {first}  •  (+{added} nueva(s))")
                else:
                    st.info("No se detectaron enlaces de LinkedIn al navegar secciones.")
        with cols_top[2]:
            selected_all = st.checkbox("Seleccionar todos", key=f"selall_{domain}", value=select_all, disabled=not current_perms().get('modify'))
        with cols_top[3]:
            q_people = st.button("Consultar People→APIs", key=f"btn_people_{domain}", disabled=not current_perms().get('modify'))
        with cols_top[4]:
            q_domain = st.button("Prospecting por Dominio", key=f"btn_dom_{domain}", disabled=not current_perms().get('modify'))
        with cols_top[5]:
            save_btn = st.button("Guardar (crudo)", key=f"btn_save_{domain}", disabled=not current_perms().get('modify'))

        # Selección por persona
        st.markdown("**Seleccionar personas para enriquecer**")
        sel_people = []
        for i, p in enumerate(people):
            chk = st.checkbox(
                f"{p.get('name','(s/nombre)')} — {p.get('role','(s/rol)')}",
                key=f"chk_{domain}_{i}",
                value=selected_all
            )
            if chk:
                sel_people.append(p)

        domain_results = {}

        if q_people and current_perms().get('modify'):
            targets = sel_people if sel_people else people
            if not targets:
                st.warning("No hay personas seleccionadas ni listadas para enriquecer.")
            else:
                with st.spinner("Consultando APIs para personas..."):
                    outp = enrich_from_people_list(
                        domain, targets, use_lusha, use_contactout, use_rocketreach
                    )
                    domain_results.update(outp)
                    upsert_raw_by_domain(domain, outp)
                st.success("Listo (personas).")
                st.json(domain_results)

        if q_domain and current_perms().get('modify'):
            with st.spinner("Prospecting por dominio..."):
                outp = enrich_from_domain(
                    domain, state_filter or None, int(per_page),
                    DEFAULT_JOB_TITLES, DEFAULT_DEPARTMENTS, DEFAULT_SENIORITY,
                    use_lusha, use_rocketreach
                )
                domain_results.update(outp)
                upsert_raw_by_domain(domain, outp)
            st.success("Listo (dominio).")
            st.json(domain_results)
            # Mensaje defensivo si RocketReach devolvió HTML/404
            rr_resp = (domain_results.get("prospecting") or {}).get("rocketreach")
            if isinstance(rr_resp, dict) and rr_resp.get("error") == "rocketreach_html_response_or_404":
                st.warning("RocketReach devolvió HTML/404. Revisá la URL del endpoint y el header de autenticación del plan.")

        if save_btn and domain_results and current_perms().get('modify'):
            upsert_raw_by_domain(domain, domain_results)
            st.success(f"Guardado crudo en {ENRIQ_FILE_RAW.name} (clave '{domain}').")

        # Modo automático por registro
        if auto_mode and current_perms().get('modify'):
            with st.spinner("Modo automático: dominio + personas..."):
                auto_out = {}
                auto_out.update(enrich_from_domain(
                    domain, state_filter or None, int(per_page),
                    DEFAULT_JOB_TITLES, DEFAULT_DEPARTMENTS, DEFAULT_SENIORITY,
                    use_lusha, use_rocketreach
                ))
                if people:
                    auto_out.update(enrich_from_people_list(
                        domain, people, use_lusha, use_contactout, use_rocketreach
                    ))
                upsert_raw_by_domain(domain, auto_out)
            st.info("Auto: guardado crudo incremental.")

# Auto Global
if run_auto and current_perms().get('modify'):
    with st.spinner("Procesando TODOS los registros en modo automático..."):
        for rec in records:
            domain = rec["domain"]
            people = rec.get("people") or []
            auto_out = {}
            auto_out.update(enrich_from_domain(
                domain, state_filter or None, int(per_page),
                DEFAULT_JOB_TITLES, DEFAULT_DEPARTMENTS, DEFAULT_SENIORITY,
                use_lusha, use_rocketreach
            ))
            if people:
                auto_out.update(enrich_from_people_list(
                    domain, people, use_lusha, use_contactout, use_rocketreach
                ))
            upsert_raw_by_domain(domain, auto_out)
    st.success("Auto Global finalizado.")
st.markdown("---")
