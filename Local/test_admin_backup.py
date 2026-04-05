import os
import sys
import json
import time
from dotenv import load_dotenv

# Add the Local directory to the path
sys.path.append(os.path.join(os.getcwd(), 'Local'))

# Load environment variables before importing app
load_dotenv(os.path.join(os.getcwd(), 'Local', '.env'), override=True)

from app import app, BACKUP_TOKENS
import app as app_module
import core.whatsapp as whatsapp

# FORCE Admin Phone after import to override whatever was in .env
app_module.ADMIN_PHONE = "593991234567"
SECRET = os.getenv("WEBHOOK_SECRET")

# Mock send_whatsapp_document to avoid real API calls
whatsapp.send_whatsapp_document = lambda r, t, url, f: print(f"[MOCK] Sending document to {r}: {url}")

def test_backup_flow():
    client = app.test_client()
    admin_jid = "593991234567@s.whatsapp.net"

    print("\n--- TESTING #BACKUP ADMIN COMMAND ---")
    
    # 1. Simulate a #backup message from admin via webhook
    payload = {
        "event": "messages.upsert",
        "data": {
            "messages": [
                {
                    "key": {
                        "remoteJid": admin_jid,
                        "fromMe": False,
                        "id": "MSG_BACKUP_125"
                    },
                    "message": {
                        "conversation": "#backup"
                    },
                    "senderPn": "593991234567"
                }
            ]
        }
    }
    
    # Note: Using the secret in the header as required by app.py
    headers = {"X-Webhook-Secret": SECRET}
    
    response = client.post("/webhook", 
                           data=json.dumps(payload), 
                           content_type='application/json',
                           headers=headers)
    
    print(f"Webhook response: {response.status_code} - {response.get_data(as_text=True)}")
    
    if response.status_code == 200:
        print("SUCCESS: Webhook call returned 200.")
        
        # 2. Check if a token was generated
        if not BACKUP_TOKENS:
            # Maybe the logic returned 200 but didn't trigger backup
            print(f"ERROR: No backup token was generated. Response: {response.get_data(as_text=True)}")
            return

        token = list(BACKUP_TOKENS.keys())[0]
        print(f"Generated token: {token}")
        
        # 3. Test the download route
        print("\n--- TESTING DOWNLOAD ROUTE ---")
        download_res = client.get(f"/download_backup/{token}")
        print(f"Download response: {download_res.status_code}")
        
        if download_res.status_code == 200:
            print("SUCCESS: Backup file served correctly.")
            print(f"Content Type: {download_res.headers.get('Content-Type')}")
            # Clean up the generated zip file
            file_path = BACKUP_TOKENS[token]["file_path"]
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Cleaned up temporary zip: {file_path}")
        else:
            print(f"ERROR: Failed to download backup. {download_res.get_data(as_text=True)}")
    else:
        print(f"ERROR: Webhook did not return 200. Got {response.status_code} instead.")

if __name__ == "__main__":
    test_backup_flow()
