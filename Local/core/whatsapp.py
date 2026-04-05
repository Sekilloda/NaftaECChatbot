import base64
import os
import pathlib
import requests
from Crypto.Cipher import AES
from Crypto.Protocol.KDF import HKDF
from Crypto.Hash import SHA256
from Crypto.Util.Padding import unpad

WASENDER_API_TOKEN = os.getenv("WASENDER_API_TOKEN")
WASENDER_API_URL = "https://wasenderapi.com/api/send-message"

if not WASENDER_API_TOKEN:
    print("[WHATSAPP] WARNING: WASENDER_API_TOKEN is not set.")

def download_media(media_url):
    try:
        response = requests.get(media_url, timeout=15)
        response.raise_for_status()
        return response.content
    except requests.exceptions.RequestException as e:
        print(f"[WHATSAPP] Error downloading media from {media_url}: {e}")
        return None

def decrypt_and_save_media(media_key_b64, encrypted_data, output_path, media_type):
    """Decrypts WhatsApp media and saves it to output_path."""
    try:
        media_key = base64.b64decode(media_key_b64)
        app_info_map = {
            'image': b'WhatsApp Image Keys',
            'video': b'WhatsApp Video Keys',
            'audio': b'WhatsApp Audio Keys',
            'document': b'WhatsApp Document Keys',
        }
        app_info = app_info_map.get(media_type.lower(), b'WhatsApp Media Keys')

        if len(media_key) != 32:
            print(f"[WHATSAPP] Error: Media key length is not 32 bytes ({len(media_key)}).")
            return False

        expanded_keys = HKDF(master=media_key, key_len=48, salt=b'', hashmod=SHA256, context=app_info, num_keys=1)
        iv = expanded_keys[0:16]
        cipher_key = expanded_keys[16:48]

        if len(encrypted_data) <= 10:
            return False

        actual_ciphertext = encrypted_data[:-10]
        cipher = AES.new(cipher_key, AES.MODE_CBC, iv)
        decrypted_padded_data = cipher.decrypt(actual_ciphertext)

        try:
            decrypted_data = unpad(decrypted_padded_data, AES.block_size, style='pkcs7')
        except ValueError:
            return False

        output_dir = pathlib.Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'wb') as f:
            f.write(decrypted_data)
        return True

    except Exception as e:
        print(f"[WHATSAPP] Error decrypting {output_path}: {e}")
        return False

def normalize_phone(phone):
    """Strips all non-digit characters and returns the full string of digits."""
    if not phone:
        return ""
    return "".join(filter(str.isdigit, str(phone)))

def send_whatsapp_message(recipient, text):
    if not WASENDER_API_TOKEN:
        print("[WHATSAPP] Cannot send message: WASENDER_API_TOKEN is missing.")
        return False
        
    clean_recipient = str(recipient).split("@")[0]
    normalized_recipient = "".join(filter(str.isdigit, clean_recipient))
    
    if not normalized_recipient:
        print(f"[WHATSAPP] Error: Invalid recipient '{recipient}'")
        return False

    full_jid = f"{normalized_recipient}@s.whatsapp.net"
    
    payload = {"to": full_jid, "text": text}
    headers = {
        "Authorization": f"Bearer {WASENDER_API_TOKEN}",
        "Content-Type": "application/json"
    }
    try:
        print(f"[WHATSAPP] Sending message to {full_jid}...")
        response = requests.post(WASENDER_API_URL, json=payload, headers=headers, timeout=10)
        if not response.ok:
            print(f"[WHATSAPP] Failed to send message to {full_jid}: {response.status_code} - {response.text}")
        else:
            print(f"[WHATSAPP] Message sent successfully to {full_jid}")
        return response.ok
    except requests.exceptions.RequestException as e:
        print(f"[WHATSAPP] Error sending message to {full_jid}: {e}")
        return False

def send_whatsapp_document(recipient, text, document_url, file_name):
    """Sends a document via WASender API using a URL."""
    if not WASENDER_API_TOKEN:
        print("[WHATSAPP] Cannot send document: WASENDER_API_TOKEN is missing.")
        return False
        
    clean_recipient = str(recipient).split("@")[0]
    normalized_recipient = "".join(filter(str.isdigit, clean_recipient))
    
    if not normalized_recipient:
        return False

    full_jid = f"{normalized_recipient}@s.whatsapp.net"
    
    payload = {
        "to": full_jid,
        "text": text,
        "documentUrl": document_url,
        "fileName": file_name
    }
    headers = {
        "Authorization": f"Bearer {WASENDER_API_TOKEN}",
        "Content-Type": "application/json"
    }
    try:
        print(f"[WHATSAPP] Sending document ({file_name}) to {full_jid}...")
        response = requests.post(WASENDER_API_URL, json=payload, headers=headers, timeout=10)
        if not response.ok:
            print(f"[WHATSAPP] Failed to send document to {full_jid}: {response.status_code} - {response.text}")
        else:
            print(f"[WHATSAPP] Document sent successfully to {full_jid}")
        return response.ok
    except Exception as e:
        print(f"[WHATSAPP] Error sending document to {full_jid}: {e}")
        return False
