import os
import re
import threading
import time
import pandas as pd
import requests

try:
    import fcntl
except ImportError:
    fcntl = None

REGISTRATIONS_DF = None
REGISTRATIONS_LOCK = threading.Lock()

# Configuration
NJUKO_API_URL = os.getenv("NJUKO_API_URL", "https://api.njuko.com/profile-definition/export-public/695ed6b584a40eb05b4dc18f/UXM38196655")
SYNC_INTERVAL = int(os.getenv("SYNC_INTERVAL", 300))  # 5 minutes
REPORT_DIR = os.getenv("REPORT_DIR", "reportes_descargados")

def normalize_phone(phone):
    """Strips all non-digit characters and returns the last 9 digits."""
    if not phone:
        return ""
    digits = re.sub(r'\D', '', str(phone))
    return digits[-9:] if len(digits) >= 9 else digits

def download_report_logic():
    if not os.path.exists(REPORT_DIR):
        os.makedirs(REPORT_DIR, exist_ok=True)

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    try:
        print(f"[REGISTRATIONS] Fetching export metadata from {NJUKO_API_URL}")
        resp = requests.get(NJUKO_API_URL, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        
        s3_url = data.get("file")
        filename = data.get("filename", "njuko_export.xlsx")
        num_results = data.get("numberOfResults", "unknown")
        
        if not s3_url:
            print("[REGISTRATIONS] No file URL found in API response.")
            return False

        print(f"[REGISTRATIONS] Downloading registry file: {filename} (Results: {num_results})")
        report_resp = requests.get(s3_url, headers=headers, timeout=60)
        report_resp.raise_for_status()
        
        filepath = os.path.join(REPORT_DIR, "latest_registry.xlsx")
        with open(filepath, 'wb') as f:
            f.write(report_resp.content)
        
        print(f"[REGISTRATIONS] Registry downloaded successfully to {filepath}")
        return True
    except Exception as e:
        print(f"[REGISTRATIONS] Error downloading registry: {e}")
        return False

def update_registrations():
    global REGISTRATIONS_DF
    print(f"[REGISTRATIONS] Starting background update thread (PID: {os.getpid()})...")
    
    lock_path = os.path.join(REPORT_DIR, "download.lock")
    filepath = os.path.join(REPORT_DIR, "latest_registry.xlsx")
    last_loaded_mtime = None
    
    if not os.path.exists(REPORT_DIR):
        os.makedirs(REPORT_DIR, exist_ok=True)

    while True:
        try:
            # 1. Freshness check: If file exists and is < (SYNC_INTERVAL - 60) old, skip download
            needs_download = True
            if os.path.exists(filepath):
                if (time.time() - os.path.getmtime(filepath)) < (SYNC_INTERVAL - 60):
                    needs_download = False

            if needs_download:
                # 2. File-based lock to prevent multiple workers downloading simultaneously
                if fcntl:
                    with open(lock_path, 'w') as f_lock:
                        try:
                            fcntl.flock(f_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            download_report_logic()
                        except (BlockingIOError, OSError):
                            pass
                else:
                    download_report_logic()

            # 3. Load the latest local file into memory
            if os.path.exists(filepath):
                file_mtime = os.path.getmtime(filepath)
                if last_loaded_mtime is None or file_mtime != last_loaded_mtime or REGISTRATIONS_DF is None:
                    # Using engine='openpyxl' for .xlsx files
                    new_df = pd.read_excel(filepath)
                    
                    if new_df.empty:
                        print("[REGISTRATIONS] Warning: Downloaded registry is empty. Keeping old data.")
                    else:
                        # Normalize column names for flexible detection
                        cols = {c.strip(): c for c in new_df.columns}
                        cols_lower = {c.lower().strip(): c for c in new_df.columns}
                        
                        # Find Phone Column
                        phone_col = (
                            cols_lower.get('telefono') or 
                            cols_lower.get('phone') or 
                            cols_lower.get('celular') or 
                            cols_lower.get('mobile') or
                            'PHONE'
                        )
                        if phone_col in new_df.columns:
                            new_df['norm_phone'] = new_df[phone_col].apply(normalize_phone)
                        
                        # Find Name Columns
                        f_col = (
                            cols_lower.get('first name') or 
                            cols_lower.get('nombre') or 
                            cols_lower.get('first_name') or
                            'FIRST_NAME'
                        )
                        l_col = (
                            cols_lower.get('last name') or 
                            cols_lower.get('apellido') or 
                            cols_lower.get('last_name') or
                            'LAST_NAME'
                        )
                        
                        if f_col in new_df.columns and l_col in new_df.columns:
                            new_df['full_name'] = (new_df[f_col].fillna('') + ' ' + new_df[l_col].fillna('')).str.lower().str.strip()
                            new_df['full_name_rev'] = (new_df[l_col].fillna('') + ' ' + new_df[f_col].fillna('')).str.lower().str.strip()
                        elif f_col in new_df.columns: # Sometimes it's just one name column
                             new_df['full_name'] = new_df[f_col].fillna('').str.lower().str.strip()
                             new_df['full_name_rev'] = new_df['full_name']
                        
                        # Store identified columns for formatting
                        new_df.attrs['mapped_cols'] = {
                            'first_name': f_col,
                            'last_name': l_col,
                            'phone': phone_col,
                            'competition': cols_lower.get('competition') or cols_lower.get('carrera') or cols_lower.get('race') or 'Competition',
                            'cedula': cols_lower.get('cedula') or cols_lower.get('documento') or cols_lower.get('id') or 'Cedula'
                        }
                        
                        with REGISTRATIONS_LOCK:
                            REGISTRATIONS_DF = new_df
                        last_loaded_mtime = file_mtime
                        print(f"[REGISTRATIONS] Loaded {len(new_df)} registrations. (Columns mapped: {new_df.attrs['mapped_cols']})")
            else:
                print("[REGISTRATIONS] No local registry file found to load.")

        except Exception as e:
            print(f"[REGISTRATIONS] Error in background update: {e}")
        
        time.sleep(60) # Wake up every minute to check freshness

def format_user_data(row):
    """Helper to turn a dataframe row into a readable string using mapped columns if available."""
    # Attempt to get mapped columns from the dataframe attributes
    mapped = {}
    if REGISTRATIONS_DF is not None and hasattr(REGISTRATIONS_DF, 'attrs'):
        mapped = REGISTRATIONS_DF.attrs.get('mapped_cols', {})

    f_col = mapped.get('first_name', 'First name')
    l_col = mapped.get('last_name', 'Last name')
    race_col = mapped.get('competition', 'Competition')
    id_col = mapped.get('cedula', 'Cedula')

    f_name = row.get(f_col) or row.get('First Name') or ""
    l_name = row.get(l_col) or row.get('Last Name') or ""
    race = row.get(race_col) or row.get('Race') or "N/A"
    status = "Confirmado" # Njuko export is usually of confirmed entries
    cedula = row.get(id_col) or row.get('Documento') or "N/A"
    
    return (
        f"Nombre: {f_name} {l_name}. "
        f"Carrera: {race}. "
        f"Estado: {status}. "
        f"Cédula: {cedula}."
    )

def get_user_registration_info(sender_jid):
    if REGISTRATIONS_DF is None:
        return None
    
    user_phone = normalize_phone(sender_jid.split("@")[0])
    if not user_phone:
        return None

    with REGISTRATIONS_LOCK:
        if 'norm_phone' in REGISTRATIONS_DF.columns:
            match = REGISTRATIONS_DF[REGISTRATIONS_DF['norm_phone'] == user_phone]
            if not match.empty:
                return format_user_data(match.iloc[0])
    return None

def search_user_by_name(name_query):
    if REGISTRATIONS_DF is None:
        return None
    
    query = name_query.lower().strip()
    if not query:
        return None

    with REGISTRATIONS_LOCK:
        if 'full_name' in REGISTRATIONS_DF.columns:
            match = REGISTRATIONS_DF[
                (REGISTRATIONS_DF['full_name'] == query) | 
                (REGISTRATIONS_DF['full_name_rev'] == query) |
                (REGISTRATIONS_DF['full_name'].str.contains(query, na=False))
            ]
            if not match.empty:
                return format_user_data(match.iloc[0])
    return None
