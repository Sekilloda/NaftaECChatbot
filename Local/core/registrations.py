import os
import re
import threading
import time
import pandas as pd
import subprocess
import glob
import shutil

REGISTRATIONS_DF = None
REGISTRATIONS_LOCK = threading.Lock()

# Configuration
GDRIVE_FOLDER_URL = os.getenv("GDRIVE_FOLDER_URL", "https://drive.google.com/drive/u/0/folders/1cCOv0vqOkXhHcTWd0AQmSaqZ752g9kgt")
SYNC_INTERVAL = int(os.getenv("SYNC_INTERVAL", 300))  # 5 minutes by default

_DEFAULT_DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BASE_DIR = os.getenv("PERSISTENT_STORAGE_PATH", _DEFAULT_DATA_DIR)
REPORT_DIR = os.path.join(_BASE_DIR, "gdrive_sync")

def normalize_phone(phone):
    """Strips all non-digit characters and returns the full string of digits."""
    if pd.isna(phone) or not phone:
        return ""
    return "".join(filter(str.isdigit, str(phone)))

def download_gdrive_folder():
    """Uses gdown to sync the public Google Drive folder locally."""
    if not os.path.exists(REPORT_DIR):
        os.makedirs(REPORT_DIR, exist_ok=True)
    
    print(f"[REGISTRATIONS] Descargando/Sincronizando carpeta de Google Drive...")
    try:
        # Run gdown --folder <URL> -O <DIR>
        result = subprocess.run(
            ["gdown", "--folder", GDRIVE_FOLDER_URL, "-O", REPORT_DIR],
            capture_output=True, text=True, check=True
        )
        print("[REGISTRATIONS] Descarga completada exitosamente.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[REGISTRATIONS] Error descargando con gdown: {e.stderr}")
        return False
    except Exception as e:
        print(f"[REGISTRATIONS] Excepción inesperada al descargar: {e}")
        return False

def read_file_safely(file_path):
    """Reads a CSV or Excel file safely, handling encoding errors."""
    ext = file_path.lower().split('.')[-1]
    
    # Try reading headers first to map types (force IDs to strings)
    if ext == 'xlsx':
        try:
            peek = pd.read_excel(file_path, nrows=0)
            dtypes = {col: str for col in peek.columns if 'cédula' in col.lower() or 'teléfono' in col.lower() or 'cedula' in col.lower()}
            return pd.read_excel(file_path, dtype=dtypes)
        except Exception as e:
            print(f"Error reading excel {file_path}: {e}")
            return None
    elif ext == 'csv':
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        for enc in encodings:
            try:
                # Read 1 line to get columns
                peek = pd.read_csv(file_path, encoding=enc, nrows=0)
                dtypes = {col: str for col in peek.columns if 'cédula' in col.lower() or 'teléfono' in col.lower() or 'cedula' in col.lower()}
                
                df = pd.read_csv(file_path, encoding=enc, dtype=dtypes)
                return df
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"Error reading csv {file_path} with {enc}: {e}")
                return None
        print(f"Failed to read {file_path} with all attempted encodings.")
        return None
    return None

def process_and_combine_dataframes(dataframes):
    """Unifies multiple dataframes and extracts the standardized columns."""
    if not dataframes:
        return None
    
    combined_df = pd.concat(dataframes, ignore_index=True)
    if combined_df.empty:
        return combined_df

    # Normalize column names for flexible detection
    cols_lower = {c.lower().strip(): c for c in combined_df.columns}

    # Find Phone Column
    phone_col = (
        cols_lower.get('número de teléfono-(asistente)') or 
        cols_lower.get('telefono') or 
        cols_lower.get('phone') or 
        'Telefono'
    )
    if phone_col in combined_df.columns:
        combined_df['norm_phone'] = combined_df[phone_col].apply(normalize_phone)

    # Find Name Columns
    f_col = (
        cols_lower.get('nombres-(asistente)') or 
        cols_lower.get('first name') or 
        cols_lower.get('nombre') or 
        'First name'
    )
    l_col = (
        cols_lower.get('apellidos-(asistente)') or 
        cols_lower.get('last name') or 
        cols_lower.get('apellido') or 
        'Last name'
    )

    if f_col in combined_df.columns and l_col in combined_df.columns:
        combined_df['full_name'] = (combined_df[f_col].fillna('') + ' ' + combined_df[l_col].fillna('')).str.lower().str.strip()
        combined_df['full_name_rev'] = (combined_df[l_col].fillna('') + ' ' + combined_df[f_col].fillna('')).str.lower().str.strip()
    elif f_col in combined_df.columns:
        combined_df['full_name'] = combined_df[f_col].fillna('').str.lower().str.strip()
        combined_df['full_name_rev'] = combined_df['full_name']

    # Find Cedula Column
    id_col = (
        cols_lower.get('número de cédula-(asistente)') or 
        cols_lower.get('cedula') or 
        cols_lower.get('documento') or 
        'Cedula'
    )
    if id_col in combined_df.columns:
        def clean_id(x):
            if pd.isna(x): return ""
            s = str(x).strip()
            if '.0' in s: s = s.split('.')[0]
            return re.sub(r'\D', '', s).lstrip('0')
        combined_df['norm_cedula'] = combined_df[id_col].apply(clean_id)

    # Find Race/Competition Column
    race_col = (
        cols_lower.get('localidad') or 
        cols_lower.get('competition') or 
        cols_lower.get('carrera') or 
        'Competition'
    )

    combined_df.attrs['mapped_cols'] = {
        'first_name': f_col,
        'last_name': l_col,
        'phone': phone_col,
        'cedula': id_col,
        'competition': race_col,
        'status': cols_lower.get('status') or 'Status' # Fake column if it doesn't exist
    }

    return combined_df

def update_registrations():
    global REGISTRATIONS_DF
    print(f"[REGISTRATIONS] Iniciando hilo de sincronización de Google Drive (PID: {os.getpid()})...")

    while True:
        try:
            # 1. Download/sync the folder
            success = download_gdrive_folder()
            
            if success and os.path.exists(REPORT_DIR):
                # 2. Gather all excel and csv files
                all_files = glob.glob(os.path.join(REPORT_DIR, "*.csv")) + glob.glob(os.path.join(REPORT_DIR, "*.xlsx"))
                
                if not all_files:
                    print(f"[REGISTRATIONS] No se encontraron archivos CSV o Excel en la carpeta descargada.")
                else:
                    dataframes = []
                    for f in all_files:
                        df = read_file_safely(f)
                        if df is not None and not df.empty:
                            dataframes.append(df)
                    
                    if dataframes:
                        combined_df = process_and_combine_dataframes(dataframes)
                        with REGISTRATIONS_LOCK:
                            REGISTRATIONS_DF = combined_df
                        print(f"[REGISTRATIONS] Base de datos actualizada. Total inscritos: {len(combined_df)} de {len(dataframes)} archivo(s).")
                    else:
                        print(f"[REGISTRATIONS] No se pudo leer correctamente ningún archivo.")

        except Exception as e:
            print(f"[REGISTRATIONS] Error general en el hilo de actualización: {e}")
        
        # Esperar antes de la próxima sincronización
        time.sleep(SYNC_INTERVAL)

def format_user_data(row_or_df):
    """Helper to turn one or more dataframe rows into a readable string."""
    if row_or_df is None or (isinstance(row_or_df, pd.DataFrame) and row_or_df.empty):
        return None

    mapped = {}
    if REGISTRATIONS_DF is not None and hasattr(REGISTRATIONS_DF, 'attrs'):
        mapped = REGISTRATIONS_DF.attrs.get('mapped_cols', {})

    f_col = mapped.get('first_name', 'First name')
    l_col = mapped.get('last_name', 'Last name')
    race_col = mapped.get('competition', 'Competition')
    id_col = mapped.get('cedula', 'Cedula')

    def row_to_str(row):
        f_name = row.get(f_col, "")
        if pd.isna(f_name): f_name = ""
        l_name = row.get(l_col, "")
        if pd.isna(l_name): l_name = ""
        
        race = row.get(race_col, "N/A")
        if pd.isna(race): race = "N/A"
        
        cedula = row.get(id_col, "N/A")
        if pd.isna(cedula): cedula = "N/A"
        
        status = "Confirmado"
        return f"Registro: {f_name} {l_name} | Carrera: {race} | Cédula: {cedula} | Estado: {status}"

    if isinstance(row_or_df, pd.Series):
        return row_to_str(row_or_df)
    
    if isinstance(row_or_df, pd.DataFrame):
        lines = [row_to_str(row) for _, row in row_or_df.iterrows()]
        return "\n".join(lines)

    return None

def get_user_registration_info(sender_jid):
    """Checks registration by phone number (JID)."""
    if REGISTRATIONS_DF is None:
        return None
    
    user_phone = normalize_phone(sender_jid.split("@")[0])
    if not user_phone:
        return None
    
    user_suffix = user_phone[-9:] if len(user_phone) >= 9 else user_phone

    with REGISTRATIONS_LOCK:
        if 'norm_phone' in REGISTRATIONS_DF.columns:
            matches = REGISTRATIONS_DF[REGISTRATIONS_DF['norm_phone'].str.endswith(user_suffix, na=False)]
            if not matches.empty:
                return format_user_data(matches)
    return None

def search_registrations_by_cedula(cedula_query):
    """Checks registration by Cedula (strict match on digits)."""
    if REGISTRATIONS_DF is None:
        return None
    
    clean_cedula = re.sub(r'\D', '', str(cedula_query)).lstrip('0')
    if not clean_cedula:
        return None

    with REGISTRATIONS_LOCK:
        if 'norm_cedula' in REGISTRATIONS_DF.columns:
            matches = REGISTRATIONS_DF[REGISTRATIONS_DF['norm_cedula'] == clean_cedula]
            if not matches.empty:
                return format_user_data(matches)
    return None

def search_user_by_name(name_query):
    """Checks registration by name."""
    if REGISTRATIONS_DF is None:
        return None
    
    query = name_query.lower().strip()
    if len(query) < 4:
        return None

    with REGISTRATIONS_LOCK:
        if 'full_name' in REGISTRATIONS_DF.columns:
            exact_matches = REGISTRATIONS_DF[
                (REGISTRATIONS_DF['full_name'] == query) | 
                (REGISTRATIONS_DF['full_name_rev'] == query)
            ]
            if not exact_matches.empty:
                return format_user_data(exact_matches)
            
            if len(query) >= 4:
                query_parts = query.split()
                mask = pd.Series([True] * len(REGISTRATIONS_DF), index=REGISTRATIONS_DF.index)
                for part in query_parts:
                    if len(part) > 2:
                        mask &= REGISTRATIONS_DF['full_name'].str.contains(part, na=False)
                
                partial_matches = REGISTRATIONS_DF[mask]
                if not partial_matches.empty:
                    return format_user_data(partial_matches.head(3))
                    
    return None
