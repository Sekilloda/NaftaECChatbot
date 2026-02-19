import os
import re
import threading
import time
import pandas as pd
import requests
import fcntl

REGISTRATIONS_DF = None
REGISTRATIONS_LOCK = threading.Lock()

def normalize_phone(phone):
    """Strips all non-digit characters and returns the last 9 digits."""
    if not phone:
        return ""
    digits = re.sub(r'\D', '', str(phone))
    return digits[-9:] if len(digits) >= 9 else digits

def download_report_logic():
    user = os.getenv("CHRONOTRACK_USER")
    pw = os.getenv("CHRONOTRACK_PASSWORD")
    report_id = os.getenv("CHRONOTRACK_REPORT_ID", "2309337")
    download_dir = os.path.join(os.getcwd(), "reportes_descargados")

    if not user or not pw:
        print("[CHRONOTRACK] Missing credentials for download.")
        return False

    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    session = requests.Session()
    session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})

    try:
        login_entry_url = "https://admin.chronotrack.com/admin"
        response = session.get(login_entry_url, allow_redirects=True)
        url_match = re.search(r'"loginAction":\s*"(https://.*?)"', response.text)
        if not url_match:
            print("[CHRONOTRACK] Could not find login action URL.")
            return False
        
        action_url = url_match.group(1).replace(r'\/', '/')
        payload = {"username": user, "password": pw}
        session.post(action_url, data=payload, allow_redirects=True)

        url_descarga = f"https://admin.chronotrack.com/report/index/view/reportID/{report_id}/format/csv"
        report_response = session.get(url_descarga)
        
        if report_response.status_code == 200:
            ctype = report_response.headers.get('Content-Type', '').lower()
            ext = "xls" if 'ms-excel' in ctype else "csv"
            filepath = os.path.join(download_dir, f"report_{report_id}.{ext}")
            with open(filepath, 'wb') as f:
                f.write(report_response.content)
            print(f"[CHRONOTRACK] Report downloaded successfully to {filepath}")
            return True
        else:
            print(f"[CHRONOTRACK] Download failed with status code: {report_response.status_code}")
    except Exception as e:
        print(f"[CHRONOTRACK] Download logic error: {e}")
    return False

def update_registrations():
    global REGISTRATIONS_DF
    print(f"[CHRONOTRACK] Starting background update thread (PID: {os.getpid()})...")
    
    report_id = os.getenv("CHRONOTRACK_REPORT_ID", "2309337")
    download_dir = os.path.join(os.getcwd(), "reportes_descargados")
    lock_path = os.path.join(download_dir, "download.lock")
    
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    while True:
        try:
            # 1. Freshness check: If file exists and is < 15 min old, don't even try to lock
            report_file_pattern = f"report_{report_id}"
            existing_files = [os.path.join(download_dir, f) for f in os.listdir(download_dir) if f.startswith(report_file_pattern)]
            
            needs_download = True
            if existing_files:
                latest_file = max(existing_files, key=os.path.getmtime)
                if (time.time() - os.path.getmtime(latest_file)) < 900: # 15 minutes
                    needs_download = False

            if needs_download:
                # 2. File-based lock to prevent multiple workers downloading simultaneously
                with open(lock_path, 'w') as f_lock:
                    try:
                        fcntl.flock(f_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        print(f"[CHRONOTRACK] Worker {os.getpid()} acquired lock. Downloading...")
                        download_report_logic()
                        # The flock is automatically released when the file is closed/with block ends
                    except BlockingIOError:
                        # Another worker has the lock
                        pass

            # 3. Load the latest local file into memory (Each worker has its own copy)
            existing_files = [os.path.join(download_dir, f) for f in os.listdir(download_dir) if f.startswith(report_file_pattern)]
            if existing_files:
                filepath = max(existing_files, key=os.path.getmtime)
                
                try:
                    new_df = pd.read_csv(filepath, sep='\t', encoding='utf-16', on_bad_lines='skip')
                    if len(new_df.columns) < 5:
                        raise ValueError("TSV failed")
                except:
                    new_df = pd.read_csv(filepath, on_bad_lines='skip')

                new_df.columns = [c.strip() for c in new_df.columns]
                
                if 'PHONE' in new_df.columns:
                    new_df['norm_phone'] = new_df['PHONE'].apply(normalize_phone)
                
                f_col = 'FIRST_NAME' if 'FIRST_NAME' in new_df.columns else 'First Name'
                l_col = 'LAST_NAME' if 'LAST_NAME' in new_df.columns else 'Last Name'
                
                if f_col in new_df.columns and l_col in new_df.columns:
                    new_df['full_name'] = (new_df[f_col].fillna('') + ' ' + new_df[l_col].fillna('')).str.lower().str.strip()
                    new_df['full_name_rev'] = (new_df[l_col].fillna('') + ' ' + new_df[f_col].fillna('')).str.lower().str.strip()
                
                with REGISTRATIONS_LOCK:
                    REGISTRATIONS_DF = new_df
                # print(f"[CHRONOTRACK] Worker {os.getpid()} loaded {len(new_df)} registrations.")
            else:
                print(f"[CHRONOTRACK] Worker {os.getpid()} - No local report found to load.")

        except Exception as e:
            print(f"[CHRONOTRACK] Error in background update (PID {os.getpid()}): {e}")
        
        time.sleep(3600)

def format_user_data(row):
    """Helper to turn a dataframe row into a readable string."""
    f_name = row.get('FIRST_NAME') or row.get('First Name') or ""
    l_name = row.get('LAST_NAME') or row.get('Last Name') or ""
    race = row.get('RACE_NAME') or row.get('Race') or "N/A"
    dist = row.get('REG_CHOICE') or row.get('Event') or "N/A"
    status = row.get('STATUS') or "N/A"
    bracket = row.get('BRACKET') or "N/A"
    
    return (
        f"Nombre: {f_name} {l_name}. "
        f"Carrera: {race}. "
        f"Distancia: {dist}. "
        f"Estado: {status}. "
        f"Categoría: {bracket}."
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
