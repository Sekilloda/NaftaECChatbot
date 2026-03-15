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

    try:
        print(f"[REGISTRATIONS] Fetching export metadata from {NJUKO_API_URL}")
        resp = requests.get(NJUKO_API_URL, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        
        s3_url = data.get("file")
        filename = data.get("filename", "njuko_export.xlsx")
        
        if not s3_url:
            print("[REGISTRATIONS] No file URL found in API response.")
            return False

        print(f"[REGISTRATIONS] Downloading registry file: {filename}")
        report_resp = requests.get(s3_url, timeout=60)
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
                # Using engine='openpyxl' for .xlsx files
                new_df = pd.read_excel(filepath)
                
                if new_df.empty:
                    print("[REGISTRATIONS] Warning: Downloaded registry is empty. Keeping old data.")
                else:
                    # Normalize columns to handle potential API changes
                    new_df.columns = [c.strip() for c in new_df.columns]
                    
                    # Phone normalization (Njuko uses 'Telefono')
                    phone_col = 'Telefono' if 'Telefono' in new_df.columns else 'PHONE'
                    if phone_col in new_df.columns:
                        new_df['norm_phone'] = new_df[phone_col].apply(normalize_phone)
                    
                    # Name normalization
                    f_col = 'First name' if 'First name' in new_df.columns else 'FIRST_NAME'
                    l_col = 'Last name' if 'Last name' in new_df.columns else 'LAST_NAME'
                    
                    if f_col in new_df.columns and l_col in new_df.columns:
                        new_df['full_name'] = (new_df[f_col].fillna('') + ' ' + new_df[l_col].fillna('')).str.lower().str.strip()
                        new_df['full_name_rev'] = (new_df[l_col].fillna('') + ' ' + new_df[f_col].fillna('')).str.lower().str.strip()
                    
                    with REGISTRATIONS_LOCK:
                        REGISTRATIONS_DF = new_df
                    print(f"[REGISTRATIONS] Loaded {len(new_df)} registrations.")
            else:
                print("[REGISTRATIONS] No local registry file found to load.")

        except Exception as e:
            print(f"[REGISTRATIONS] Error in background update: {e}")
        
        time.sleep(60) # Wake up every minute to check freshness

def format_user_data(row):
    """Helper to turn a dataframe row into a readable string."""
    f_name = row.get('First name') or row.get('First Name') or ""
    l_name = row.get('Last name') or row.get('Last Name') or ""
    race = row.get('Competition') or row.get('Race') or "N/A"
    dist = row.get('Competition') or row.get('Event') or "N/A" # In Njuko, Competition often includes distance
    status = "Confirmado" # Njuko export is usually of confirmed entries
    cedula = row.get('Cedula') or "N/A"
    
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
