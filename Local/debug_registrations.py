import os
import sys

# Add the Local directory to the path so we can import core modules
sys.path.append(os.path.join(os.getcwd(), 'Local'))

import core.registrations as registrations

def debug_data():
    print("--- DEBUG REGISTRATIONS DATA ---")
    
    # 1. Download if missing
    registrations.download_report_logic()
    
    # 2. Replicate the loading logic exactly as in registrations.py
    filepath = os.path.join(registrations.REPORT_DIR, "latest_registry.xlsx")
    
    def clean_id(x):
        if registrations.pd.isna(x): return ""
        s = str(x).strip()
        if '.0' in s: s = s.split('.')[0]
        if 'e+' in s.lower():
            try:
                s = "{:.0f}".format(float(s))
            except: pass
        return s

    peek_df = registrations.pd.read_excel(filepath, nrows=0)
    dtype_map = {col: str for col in peek_df.columns if any(x in col.lower() for x in ['cedula', 'document', 'id', 'telefono', 'phone', 'celular'])}
    df = registrations.pd.read_excel(filepath, dtype=dtype_map)
    
    # Identify columns
    cols_lower = {c.lower().strip(): c for c in df.columns}
    phone_col = (cols_lower.get('telefono') or cols_lower.get('phone') or 'Telefono')
    id_col = (cols_lower.get('cedula') or cols_lower.get('documento') or 'Cedula')

    df['norm_phone'] = df[phone_col].apply(registrations.normalize_phone)
    df['norm_cedula'] = df[id_col].apply(clean_id)

    print(f"\nDetected phone_col: {phone_col}, id_col: {id_col}")
    print("\nSample of normalized data:")
    print(df[[phone_col, 'norm_phone', id_col, 'norm_cedula']].head(10))
    
    # Test specific lookups
    # Try Edison Paul
    target_cedula = "0601921257"
    print(f"\nTesting lookup for cedula: {target_cedula}")
    match = df[df['norm_cedula'] == target_cedula]
    print(f"Matches found: {len(match)}")
    if not match.empty:
        print(f"Match data:\n{match[['First name', 'Last name', phone_col, id_col]]}")

    # Try Phone for Edison Paul
    target_phone = "593987659958"
    norm_target_phone = registrations.normalize_phone(target_phone)
    print(f"\nTesting lookup for phone: {target_phone} (norm: {norm_target_phone})")
    match_ph = df[df['norm_phone'] == norm_target_phone]
    print(f"Matches found: {len(match_ph)}")
    if not match_ph.empty:
        print(f"Match data:\n{match_ph[['First name', 'Last name', phone_col, id_col]]}")

if __name__ == "__main__":
    debug_data()
