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

def download_media(media_url):
    try:
        response = requests.get(media_url, timeout=15)
        response.raise_for_status()
        return response.content
    except requests.exceptions.RequestException as e:
        print(f"Error downloading media from {media_url}: {e}")
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
            print(f"Error: Media key length is not 32 bytes ({len(media_key)}).")
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
        print(f"Error decrypting {output_path}: {e}")
        return False

def send_whatsapp_message(recipient, text):
    if "@s.whatsapp.net" in recipient:
        recipient = recipient.split("@")[0]
    payload = {"to": recipient, "text": text}
    headers = {
        "Authorization": f"Bearer {WASENDER_API_TOKEN}",
        "Content-Type": "application/json"
    }
    try:
        response = requests.post(WASENDER_API_URL, json=payload, headers=headers, timeout=10)
        return response.ok
    except requests.exceptions.RequestException as e:
        print(f"Error sending message: {e}")
        return False
