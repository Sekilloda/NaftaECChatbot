import os
import sys
import threading
import time
import pandas as pd

# Add the Local directory to the path so we can import core modules
sys.path.append(os.path.join(os.getcwd(), 'Local'))

from core.knowledge import responder
import core.registrations as registrations

def simulate_test():
    print("--- SIMULACIÓN DINÁMICA DE FLUJO DE REGISTRO ---")
    
    # 1. Start update in a background thread
    print("\n[STEP 1] Iniciando carga de base de datos de Njuko...")
    thread = threading.Thread(target=registrations.update_registrations, daemon=True)
    thread.start()
    
    # Wait for REGISTRATIONS_DF to be populated (max 30 seconds)
    timeout = 30
    start_time = time.time()
    while registrations.REGISTRATIONS_DF is None:
        if time.time() - start_time > timeout:
            print("ERROR: Tiempo de espera agotado cargando la base de datos.")
            return
        time.sleep(1)
        print(".", end="", flush=True)
    
    df = registrations.REGISTRATIONS_DF
    print(f"\nBase de datos cargada con {len(df)} registros.")

    # Find a user with a phone for Test 2
    # Ensure it's a valid row with First name and Telefono
    user_with_phone = df.dropna(subset=['First name', 'Telefono']).iloc[0]
    name_ph = f"{user_with_phone['First name']} {user_with_phone['Last name']}"
    phone_ph = user_with_phone['norm_phone']
    jid_ph = f"593{phone_ph}@s.whatsapp.net"

    # Find a user with multiple registrations for Test 3
    counts = df['norm_cedula'].value_counts()
    multi_id = counts[counts > 1].index[0]
    multi_user = df[df['norm_cedula'] == multi_id].iloc[0]
    name_multi = f"{multi_user['First name']} {multi_user['Last name']}"
    cedula_multi = multi_user['norm_cedula']
    num_entries = counts[multi_id]

    # 2. Test Case: Automatic detection by Phone (JID)
    print(f"\n[TEST 2] Verificación automática por Teléfono ({name_ph}):")
    response = responder("Hola, ¿estoy inscrito?", sender_jid=jid_ph)
    print(f"Usuario: Hola, ¿estoy inscrito? (JID: {jid_ph})\nBot: {response}")

    # 3. Test Case: Search by Cédula (Multiple Entries)
    print(f"\n[TEST 3] Búsqueda por Cédula con múltiples registros ({name_multi}, {num_entries} carreras):")
    jid_rand = "123456789@s.whatsapp.net"
    response = responder(f"Mi cédula es {cedula_multi}, ¿puedes revisar mi estado?", sender_jid=jid_rand)
    print(f"Usuario: Mi cédula es {cedula_multi}, ¿puedes revisar mi estado?\nBot: {response}")

    # 4. Test Case: Prevention of Hallucination (Fake Name)
    print("\n[TEST 4] Prevención de alucinación (Nombre falso):")
    response = responder("Soy Juanito Alcachofa de la Pradera, ¿estoy en la lista?", sender_jid=jid_rand)
    print(f"Usuario: Soy Juanito Alcachofa de la Pradera, ¿estoy en la lista?\nBot: {response}")

    # 5. Test Case: Robustness (Ask for Cédula when missing)
    print("\n[TEST 5] Solicitud de cédula cuando no hay datos:")
    response = responder("Quiero saber si mi inscripción fue exitosa.", sender_jid=jid_rand)
    print(f"Usuario: Quiero saber si mi inscripción fue exitosa.\nBot: {response}")

if __name__ == "__main__":
    simulate_test()
