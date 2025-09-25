# db_manager.py
# SQLite backend for multi-user auth + consolidated data
# Provides:
#  - init_db(): create tables if not exist
#  - create_user(username, password, role)
#  - authenticate(username, password)
#  - list_users(), set_user_role(), deactivate_user(), reset_password()
#  - migration helpers: load JSON sources into structured tables
#  - data access helpers used by Streamlit UI (role-based)

import os, json, sqlite3, hashlib, hmac, binascii, time, pathlib
from typing import Optional, List, Dict, Any, Tuple

DB_PATH = pathlib.Path('data.db')
PBKDF_ITER = 48000
HASH_ALG = 'sha256'

VALID_ROLES = ['lector', 'editor', 'supervisor']  # ascending privilege

# ---------- Low level ----------

def _connect():
    return sqlite3.connect(DB_PATH)

# ---------- Password hashing ----------

def _hash_password(password: str, salt: Optional[bytes] = None) -> Tuple[str, str]:
    if salt is None:
        salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac(HASH_ALG, password.encode('utf-8'), salt, PBKDF_ITER)
    return binascii.hexlify(salt).decode(), binascii.hexlify(dk).decode()

def _verify_password(password: str, salt_hex: str, hash_hex: str) -> bool:
    salt = binascii.unhexlify(salt_hex.encode())
    dk_check = hashlib.pbkdf2_hmac(HASH_ALG, password.encode('utf-8'), salt, PBKDF_ITER)
    return hmac.compare_digest(binascii.hexlify(dk_check).decode(), hash_hex)

# ---------- Schema ----------

SCHEMA = {
    'users': '''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        salt TEXT NOT NULL,
        pw_hash TEXT NOT NULL,
        role TEXT NOT NULL,
        active INTEGER NOT NULL DEFAULT 1,
        created_at INTEGER NOT NULL
    );''',
    'etapa1_sites': '''CREATE TABLE IF NOT EXISTS etapa1_sites (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        domain TEXT,
        nombre TEXT,
        score INTEGER,
        raw_json TEXT
    );''',
    'external_emails': '''CREATE TABLE IF NOT EXISTS external_emails (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        domain TEXT,
        email TEXT
    );''',
    'enriched_people': '''CREATE TABLE IF NOT EXISTS enriched_people (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        domain TEXT,
        name TEXT,
        role TEXT,
        email TEXT,
        source TEXT,
        department TEXT,
        seniority TEXT
    );''',
    'enriched_raw': '''CREATE TABLE IF NOT EXISTS enriched_raw (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        domain TEXT,
        payload_json TEXT
    );''',
    'emails_web': '''CREATE TABLE IF NOT EXISTS emails_web (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        domain TEXT,
        email TEXT
    );'''
}

def init_db():
    with _connect() as cx:
        cur = cx.cursor()
        for ddl in SCHEMA.values():
            cur.execute(ddl)
        # seed supervisor if none
        cur.execute('SELECT COUNT(*) FROM users')
        n = cur.fetchone()[0]
        if n == 0:
            create_user('admin', 'Admin123$', 'supervisor')
        cx.commit()

# ---------- User management ----------

def create_user(username: str, password: str, role: str) -> bool:
    role = role.lower()
    if role not in VALID_ROLES:
        raise ValueError('Invalid role')
    salt, pw_hash = _hash_password(password)
    try:
        with _connect() as cx:
            cx.execute('INSERT INTO users (username,salt,pw_hash,role,created_at) VALUES (?,?,?,?,?)',
                       (username, salt, pw_hash, role, int(time.time())))
        return True
    except sqlite3.IntegrityError:
        return False

def authenticate(username: str, password: str) -> Optional[Dict[str, Any]]:
    with _connect() as cx:
        cur = cx.execute('SELECT id, salt, pw_hash, role, active FROM users WHERE username=?', (username,))
        row = cur.fetchone()
        if not row:
            return None
        uid, salt, pw_hash, role, active = row
        if not active:
            return None
        if _verify_password(password, salt, pw_hash):
            return {'id': uid, 'username': username, 'role': role}
        return None

def list_users() -> List[Dict[str, Any]]:
    with _connect() as cx:
        cur = cx.execute('SELECT id, username, role, active, created_at FROM users ORDER BY id')
        return [ {'id':r[0],'username':r[1],'role':r[2],'active':bool(r[3]),'created_at':r[4]} for r in cur.fetchall() ]

def set_user_role(username: str, role: str) -> bool:
    role = role.lower()
    if role not in VALID_ROLES:
        raise ValueError('Invalid role')
    with _connect() as cx:
        cur = cx.execute('UPDATE users SET role=? WHERE username=?', (role, username))
        return cur.rowcount > 0

def deactivate_user(username: str, active: bool) -> bool:
    with _connect() as cx:
        cur = cx.execute('UPDATE users SET active=? WHERE username=?', (1 if active else 0, username))
        return cur.rowcount > 0

def reset_password(username: str, new_password: str) -> bool:
    salt, pw_hash = _hash_password(new_password)
    with _connect() as cx:
        cur = cx.execute('UPDATE users SET salt=?, pw_hash=? WHERE username=?', (salt, pw_hash, username))
        return cur.rowcount > 0

# ---------- Migration helpers ----------

def load_json(path: pathlib.Path) -> Any:
    if not path.exists():
        return None
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)

OUT_DIR = pathlib.Path('out')

# Migration: etapa1_v1.json -> etapa1_sites + enriched_people (people) + emails_web (emails + people emails)

def migrate_etapa1(etapa1_path: Optional[pathlib.Path] = None) -> Dict[str, int]:
    etapa1_path = etapa1_path or (OUT_DIR / 'etapa1_v1.json')
    data = load_json(etapa1_path)
    if not isinstance(data, dict):
        return {'sites':0}
    sites = data.get('sites') or []
    inserted_sites = 0
    with _connect() as cx:
        cur = cx.cursor()
        for s in sites:
            if not isinstance(s, dict):
                continue
            domain = (s.get('domain') or '').strip().lower()
            if not domain:
                continue
            score = 0
            try:
                score = int((((s.get('band') or {}).get('score')) or 0))
            except Exception:
                pass
            nombre = s.get('site_name') or domain
            cur.execute('INSERT INTO etapa1_sites (domain,nombre,score,raw_json) VALUES (?,?,?,?)',
                        (domain, nombre, score, json.dumps(s, ensure_ascii=False)))
            inserted_sites += 1
            # emails
            for e in (s.get('emails') or []):
                if isinstance(e, dict):
                    e_val = e.get('value') or e.get('email') or ''
                else:
                    e_val = e
                e_val = (e_val or '').strip()
                if e_val and '@' in e_val:
                    cur.execute('INSERT INTO emails_web (domain,email) VALUES (?,?)', (domain, e_val.lower()))
            # people
            for p in (s.get('people') or []):
                if not isinstance(p, dict):
                    continue
                name = (p.get('name') or '').strip()
                role = (p.get('role') or '').strip()
                email = (p.get('email') or '').strip()
                dept = (p.get('department') or '').strip()
                seniority = (p.get('seniority') or '').strip()
                if email and '@' in email:
                    cur.execute('INSERT INTO emails_web (domain,email) VALUES (?,?)', (domain, email.lower()))
                cur.execute('INSERT INTO enriched_people (domain,name,role,email,source,department,seniority) VALUES (?,?,?,?,?,?,?)',
                            (domain,name,role,email,'input',dept,seniority))
        cx.commit()
    return {'sites': inserted_sites}

# Migration: enriquecidov3.json -> enriched_people (only new) + emails_web

def migrate_enriquecidov3(path: Optional[pathlib.Path] = None) -> Dict[str,int]:
    path = path or (OUT_DIR / 'enriquecidov3.json')
    data = load_json(path)
    if not isinstance(data, dict):
        return {'people':0}
    sites = data.get('sites') or []
    inserted_people = 0
    with _connect() as cx:
        cur = cx.cursor()
        for s in sites:
            domain = (s.get('domain') or '').strip().lower()
            if not domain:
                continue
            for p in (s.get('people') or []):
                if not isinstance(p, dict):
                    continue
                name = (p.get('name') or '').strip()
                role = (p.get('role') or '').strip()
                email = (p.get('email') or '').strip()
                dept = (p.get('department') or '').strip()
                seniority = (p.get('seniority') or '').strip()
                source = (p.get('source') or 'v3').strip()
                cur.execute('INSERT INTO enriched_people (domain,name,role,email,source,department,seniority) VALUES (?,?,?,?,?,?,?)',
                            (domain,name,role,email,source,dept,seniority))
                if email and '@' in email:
                    cur.execute('INSERT INTO emails_web (domain,email) VALUES (?,?)', (domain, email.lower()))
                inserted_people += 1
            # emails_web field
            for e in (s.get('emails_web') or []):
                e = (e or '').strip()
                if e and '@' in e:
                    cur.execute('INSERT INTO emails_web (domain,email) VALUES (?,?)', (domain, e.lower()))
        cx.commit()
    return {'people': inserted_people}

# Migration: enriquecidos.json (raw) -> enriched_raw + enriched_people (parsed emails inside contacts_enriched)

def migrate_enriquecidos_raw(path: Optional[pathlib.Path] = None) -> Dict[str,int]:
    path = path or (OUT_DIR / 'enriquecidos.json')
    data = load_json(path) or {}
    if not isinstance(data, dict):
        return {'raw':0,'emails':0}
    inserted_raw = 0
    inserted_people = 0
    with _connect() as cx:
        cur = cx.cursor()
        for domain, payload in data.items():
            domain_l = (domain or '').strip().lower()
            if not domain_l:
                continue
            cur.execute('INSERT INTO enriched_raw (domain,payload_json) VALUES (?,?)', (domain_l, json.dumps(payload, ensure_ascii=False)))
            inserted_raw += 1
            # parse emails from contacts_enriched -> naive scan
            clist = payload.get('contacts_enriched') or []
            for it in clist:
                if not isinstance(it, dict):
                    continue
                enrich = it.get('enrich') or {}
                for prov_key, prov_val in enrich.items():
                    # traverse dict looking for email-like strings
                    stack = [prov_val]
                    emails_found = set()
                    while stack:
                        obj = stack.pop()
                        if isinstance(obj, dict):
                            for k,v in obj.items():
                                if isinstance(v, (dict,list)):
                                    stack.append(v)
                                else:
                                    if isinstance(v,str) and '@' in v and len(v) < 120:
                                        emails_found.add(v.strip())
                        elif isinstance(obj, list):
                            for v in obj:
                                stack.append(v)
                    for email in emails_found:
                        cur.execute('INSERT INTO enriched_people (domain,name,role,email,source,department,seniority) VALUES (?,?,?,?,?,?,?)',
                                    (domain_l,'','',email,prov_key,'',''))
                        cur.execute('INSERT INTO emails_web (domain,email) VALUES (?,?)', (domain_l, email.lower()))
                        inserted_people += 1
        cx.commit()
    return {'raw': inserted_raw, 'emails': inserted_people}

# Utility: aggregate all unique emails per domain

def get_all_emails_for_domain(domain: str) -> List[str]:
    domain = domain.lower().strip()
    with _connect() as cx:
        cur = cx.execute('SELECT DISTINCT email FROM emails_web WHERE domain=?', (domain,))
        return [r[0] for r in cur.fetchall() if r[0]]

# Role helper
ROLE_PERMS = {
    'lector': {'read':True,'export':False,'modify':False,'control':False,'user_admin':False},
    'editor': {'read':True,'export':True,'modify':True,'control':False,'user_admin':False},
    'supervisor': {'read':True,'export':True,'modify':True,'control':True,'user_admin':True}
}

def role_permissions(role: str) -> Dict[str,bool]:
    return ROLE_PERMS.get(role.lower(), ROLE_PERMS['lector'])

# Convenience counts

def counts_summary() -> Dict[str,int]:
    with _connect() as cx:
        cur = cx.cursor()
        cur.execute('SELECT COUNT(*) FROM etapa1_sites'); sites = cur.fetchone()[0]
        cur.execute('SELECT COUNT(*) FROM enriched_people'); ppl = cur.fetchone()[0]
        cur.execute('SELECT COUNT(*) FROM emails_web'); emails = cur.fetchone()[0]
        return {'sites':sites,'people':ppl,'emails':emails}

if __name__ == '__main__':
    init_db()
    print('DB initialized. Users:', list_users())
