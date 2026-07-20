import os
import threading
import json
import re
import hmac
import hashlib
import zipfile
import io
import secrets
import time
import sqlite3
from flask import Flask, request, jsonify, send_file
from dotenv import load_dotenv

# Load environment variables once at entry point
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"), override=True)

# Core modules
from core.whatsapp import send_whatsapp_message, send_whatsapp_document, normalize_phone
from core.knowledge import responder
from core.registrations import update_registrations
from core.database import (
    init_db, save_message, get_last_messages,
    save_pending_confirmation, get_pending_confirmation, clear_pending_confirmation,
    set_user_status, get_user_status, reset_user_status
)

app = Flask(__name__)

# Security: One-time tokens for database download
BACKUP_TOKENS = {} # token: {"expires": timestamp, "file_path": path}

@app.route("/download_backup/<token>", methods=["GET"])
def download_backup(token):
    token_data = BACKUP_TOKENS.get(token)
    if not token_data:
        return "Token inválido o expirado", 404
    
    if time.time() > token_data["expires"]:
        del BACKUP_TOKENS[token]
        return "Token expirado", 403
    
    file_path = token_data["file_path"]
    if not os.path.exists(file_path):
        return "Archivo no encontrado", 404
        
    # We don't delete immediately because WASender needs to fetch it
    return send_file(file_path, as_attachment=True, download_name="naftaec_backup.zip")

# Inicializamos la base de datos aquí
init_db()

_SYNC_THREAD_LOCK = threading.Lock()
_SYNC_THREAD_STARTED = False

# Support for persistent storage on Render
DATA_DIR = os.getenv("PERSISTENT_STORAGE_PATH", os.path.dirname(os.path.abspath(__file__)))
MEDIA_DIR = os.path.join(DATA_DIR, 'media')
os.makedirs(MEDIA_DIR, exist_ok=True)

def cleanup_old_media(max_age_days=7):
    """Periodically cleans up the media folder to save space."""
    import time
    if not os.path.exists(MEDIA_DIR):
        return
        
    while True:
        try:
            now = time.time()
            for filename in os.listdir(MEDIA_DIR):
                file_path = os.path.join(MEDIA_DIR, filename)
                if os.path.isfile(file_path):
                    if os.stat(file_path).st_mtime < now - (max_age_days * 86400):
                        os.remove(file_path)
                        print(f"[APP] Cleaned up old media file: {filename}")
        except Exception as e:
            print(f"[APP] Error in media cleanup: {e}")
        time.sleep(3600) # Run every hour

def ensure_background_services():
    global _SYNC_THREAD_STARTED
    if _SYNC_THREAD_STARTED:
        return
    with _SYNC_THREAD_LOCK:
        if _SYNC_THREAD_STARTED:
            return
        init_db()
        threading.Thread(target=update_registrations, daemon=True, name="registrations-sync").start()
        threading.Thread(target=cleanup_old_media, daemon=True, name="media-cleanup").start()
        _SYNC_THREAD_STARTED = True
        print("[APP] Background services initialized.")

ADMIN_PHONE = os.getenv("ADMIN_PHONE")
if not ADMIN_PHONE:
    print("[APP] WARNING: ADMIN_PHONE not set. Ayuda status will not forward messages.")

WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "").strip()
ALLOW_UNAUTHENTICATED_WEBHOOK = os.getenv("ALLOW_UNAUTHENTICATED_WEBHOOK", "").strip().lower() in {"1", "true", "yes", "on"}
if not WEBHOOK_SECRET and not ALLOW_UNAUTHENTICATED_WEBHOOK:
    print("[APP] ERROR: WEBHOOK_SECRET not set. Webhook requests will be rejected until configured.")
elif not WEBHOOK_SECRET:
    print("[APP] WARNING: WEBHOOK_SECRET not set. ALLOW_UNAUTHENTICATED_WEBHOOK enabled for this environment.")


def classify_confirmation_reply(text):
    text_clean = (text or "").lower().strip()
    if not text_clean:
        return "unclear"

    tokens = set(re.findall(r"[a-záéíóúüñ]+", text_clean))
    affirmative_tokens = {"si", "sí", "sipi", "síp", "claro", "dale", "ok", "confirmo", "correcto", "afirmativo", "positivo"}
    negative_tokens = {"no", "nop", "negativo", "incorrecto", "falso", "nones", "jamas", "jamás", "nunca"}
    affirmative_phrases = {"esta bien", "está bien", "de acuerdo"}
    negative_phrases = {"para nada", "de ninguna manera"}

    has_affirmative = bool(tokens & affirmative_tokens) or any(phrase in text_clean for phrase in affirmative_phrases)
    has_negative = bool(tokens & negative_tokens) or any(phrase in text_clean for phrase in negative_phrases)

    if has_affirmative and not has_negative:
        return "affirmative"
    if has_negative and not has_affirmative:
        return "negative"
    return "unclear"

def is_authorized_webhook(req):
    if not WEBHOOK_SECRET:
        return ALLOW_UNAUTHENTICATED_WEBHOOK
    provided_values = [
        req.headers.get("X-Webhook-Secret", ""),
        req.headers.get("X-Webhook-Token", ""),
        req.args.get("secret", ""),
    ]
    auth_header = req.headers.get("Authorization", "")
    if auth_header.lower().startswith("bearer "):
        provided_values.append(auth_header[7:].strip())
    return any(
        candidate and hmac.compare_digest(candidate.strip(), WEBHOOK_SECRET)
        for candidate in provided_values
    )

def is_admin_sender(sender_jid):
    if not ADMIN_PHONE or not sender_jid:
        return False
    # sender_jid might be a string or a JID object, ensure it's a string
    sender_phone = normalize_phone(str(sender_jid).split("@")[0])
    if not sender_phone:
        return False
    admin_numbers = {
        normalize_phone(candidate)
        for candidate in re.split(r"[,;\s]+", ADMIN_PHONE)
        if candidate.strip()
    }
    admin_numbers.discard("")
    return sender_phone in admin_numbers


@app.route("/health", methods=["GET"])
def health():
    ensure_background_services()
    return jsonify({"status": "healthy", "user_status": "ready"}), 200

def get_effective_jid(message_container):
    """
    Consistent logic to get the most 'real' JID possible.
    Prioritizes cleanedSenderPn/senderPn (Phone Number) over remoteJid (which could be a LID).
    """
    key = message_container.get("key", {})
    # Check root and key for phone numbers
    pn = message_container.get("cleanedSenderPn") or message_container.get("senderPn") or \
         key.get("cleanedSenderPn") or key.get("senderPn")
    
    if pn:
        # Normalize to number@s.whatsapp.net
        clean_pn = "".join(filter(str.isdigit, str(pn)))
        return f"{clean_pn}@s.whatsapp.net"
    
    # Fallback to remoteJid if no PN is found
    jid = key.get("remoteJid")
    if jid:
        if "@" in jid:
            # Handle weird suffixes like the ones seen in simulation logs
            base, suffix = jid.split("@", 1)
            if "/" in suffix: # e.g. @Local/simulation/...
                return f"{base}@s.whatsapp.net"
            return jid
        return f"{jid}@s.whatsapp.net"
    
    return None

def _sanitize_media_component(value, fallback):
    clean = re.sub(r"[^A-Za-z0-9_.-]", "_", str(value or "")).strip("._")
    return clean or fallback

def build_media_output_path(message_id, extension):
    safe_id = _sanitize_media_component(message_id, "unknown")
    safe_ext = (extension or "").strip().lower()
    if safe_ext and not re.fullmatch(r"\.[a-z0-9]{1,10}", safe_ext):
        safe_ext = ""

    safe_filename = f"{safe_id}{safe_ext}"
    media_root = os.path.abspath(MEDIA_DIR)
    output_path = os.path.abspath(os.path.join(media_root, safe_filename))

    if not output_path.startswith(media_root + os.sep):
        raise ValueError("Invalid media path.")
    return output_path, safe_filename

def _message_sort_key(item):
    key = str(item[0])
    if key.isdigit():
        return (0, int(key))
    return (1, key)

def extract_message_containers(messages):
    if isinstance(messages, list):
        return [m for m in messages if isinstance(m, dict)]

    if isinstance(messages, dict):
        if any(str(key).isdigit() for key in messages.keys()):
            ordered_items = sorted(messages.items(), key=_message_sort_key)
            return [value for _, value in ordered_items if isinstance(value, dict)]
        return [messages]

    return []

def _process_single_message_container(message_container):
    if not message_container:
        return {"status": "skipped"}, 200

    key = message_container.get("key", {})
    is_from_me = bool(key.get("fromMe"))

    # Determine the user we are interacting with
    effective_user_jid = get_effective_jid(message_container)
    if not effective_user_jid:
        return {"status": "missing_sender"}, 400

    # For logging and is_admin_sender, we need to know who physically sent this specific packet
    sender = "bot@s.whatsapp.net" if is_from_me else effective_user_jid

    message_id = key.get("id", "unknown")
    msg_content = message_container.get("message") or {}

    # Extract text from various possible locations
    incoming_text = (
        msg_content.get("conversation") or
        msg_content.get("extendedTextMessage", {}).get("text") or
        msg_content.get("imageMessage", {}).get("caption") or
        ""
    ).strip()

    # Admin / Manual Override: Check this FIRST before skipping fromMe
    if incoming_text.lower().startswith(("#resuelto", "#resolver")):
        print(f"[ADMIN] Command detected: {incoming_text.split()[0]} | fromMe: {is_from_me} | Sender: {sender} | Target: {effective_user_jid}")
        # Check if this is an admin OR if the bot is sending it (meaning the account holder is typing)
        if is_from_me or is_admin_sender(sender):
            if not effective_user_jid.endswith("@g.us"):
                print(f"[ADMIN] Resetting status for {effective_user_jid}")
                reset_user_status(effective_user_jid)
                send_whatsapp_message(effective_user_jid, "Un representante ha marcado tu consulta como resuelta. El asistente virtual vuelve a estar activo.")
                return {"status": "admin_resuelto_success", "target": effective_user_jid}, 200

    if incoming_text.lower().startswith("#backup"):
        print(f"[ADMIN] Command detected: #backup | fromMe: {is_from_me} | Sender: {sender}")
        if is_from_me or is_admin_sender(sender):
            try:
                # 1. Create Zip in MEDIA_DIR
                zip_filename = f"backup_{int(time.time())}.zip"
                zip_path = os.path.join(MEDIA_DIR, zip_filename)

                from core.database import DB_PATH, get_db_connection
                from core.registrations import REPORT_DIR
                registry_path = os.path.join(REPORT_DIR, "latest_registry.xlsx")

                # Safer SQLite backup for consistency
                temp_db_path = f"{zip_path}.db"
                try:
                    with get_db_connection() as conn:
                        # Create a backup while the DB is in use
                        with sqlite3.connect(temp_db_path) as backup_conn:
                            conn.backup(backup_conn)
                except Exception as db_err:
                    print(f"[ADMIN] DB backup failed, copying file directly: {db_err}")
                    import shutil
                    shutil.copy2(DB_PATH, temp_db_path)

                with zipfile.ZipFile(zip_path, "w") as zipf:
                    if os.path.exists(temp_db_path):
                        zipf.write(temp_db_path, arcname="chat_history.db")
                    if os.path.exists(registry_path):
                        zipf.write(registry_path, arcname="latest_registry.xlsx")

                if os.path.exists(temp_db_path):
                    os.remove(temp_db_path)

                # 2. Generate Token
                token = secrets.token_urlsafe(32)
                BACKUP_TOKENS[token] = {
                    "expires": time.time() + 3600,  # 1 hour
                    "file_path": zip_path
                }

                # 3. Send via WhatsApp
                # Support for EXTERNAL_URL (e.g., https://mybot.onrender.com)
                base_url = os.getenv("EXTERNAL_URL") or request.host_url.rstrip("/")
                if not base_url.startswith("http"):
                    base_url = f"https://{base_url}"
                download_url = f"{base_url}/download_backup/{token}"

                target_for_backup = sender if not is_from_me else effective_user_jid
                send_whatsapp_document(
                    target_for_backup,
                    "Aquí tienes el respaldo de la base de datos y registros. El enlace expira en 1 hora.",
                    download_url,
                    "naftaec_backup.zip"
                )
                return {"status": "backup_sent", "token": token}, 200
            except Exception as e:
                import traceback
                traceback.print_exc()
                return {"status": "backup_error", "message": str(e)}, 500

    # Skip messages sent by the bot itself to avoid infinite loops
    if is_from_me:
        return {"status": "skipped_from_me"}, 200

    user_status = get_user_status(effective_user_jid)
    print(f"[WEBHOOK] messages.upsert | From: {effective_user_jid} | Text: {incoming_text[:50]}... | Status: {user_status}")

    if user_status == "ayuda":
        # Check for bot resume command
        if incoming_text and re.match(r"^(bot|b0t|b\.o\.t)$", incoming_text, re.IGNORECASE):
            reset_user_status(effective_user_jid)
            send_whatsapp_message(effective_user_jid, "Entendido. El asistente virtual vuelve a estar activo. ¿En qué puedo ayudarte?")
            if ADMIN_PHONE:
                clean_phone = effective_user_jid.split("@")[0]
                admin_msg = f"ℹ️ El usuario {clean_phone} ha cancelado la solicitud de ayuda y ha vuelto al modo bot."
                primary_admin = re.split(r"[,;\s]+", ADMIN_PHONE)[0].strip()
                send_whatsapp_message(primary_admin, admin_msg)
            return {"status": "user_reset_to_bot"}, 200

        # Silence in ayuda mode
        return {"status": "ayuda_active_silence"}, 200

    # --- STANDARD FLOW ---
    if not incoming_text:
        # Imagen u otro media sin texto
        send_whatsapp_message(effective_user_jid, "📸 Recibí tu archivo, pero actualmente solo puedo responder mensajes de texto. ¿En qué puedo ayudarte?")
        return {"status": "unsupported_type"}, 200

    save_message(effective_user_jid, "user", incoming_text)

    pending = get_pending_confirmation(effective_user_jid)
    if pending:
        if pending["state"] == "AWAITING_AYUDA_CONFIRMATION":
            classification = classify_confirmation_reply(incoming_text)
            if "affirmative" in classification:
                pending["state"] = "AWAITING_NAME_FOR_AYUDA"
                save_pending_confirmation(effective_user_jid, pending)
                send_whatsapp_message(effective_user_jid, "Entendido. Por favor, dime tu nombre para que un representante pueda atenderte mejor:")
                return {"status": "awaiting_name"}, 200
            if "negative" in classification:
                send_whatsapp_message(effective_user_jid, "Entendido. Continuamos con el asistente virtual. ¿En qué puedo ayudarte?")
                clear_pending_confirmation(effective_user_jid)
                return {"status": "ayuda_cancelled"}, 200

        elif pending["state"] == "AWAITING_NAME_FOR_AYUDA":
            user_name = incoming_text.strip()
            if len(user_name) < 2:
                send_whatsapp_message(effective_user_jid, "Por favor, ingresa un nombre válido.")
                return {"status": "invalid_name"}, 200

            set_user_status(effective_user_jid, "ayuda")
            send_whatsapp_message(effective_user_jid, f"Gracias, {user_name}. He notificado a un representante. Mientras un representante atiende tu caso, el asistente virtual no responderá a tus mensajes. Puedes escribir 'Bot' en cualquier momento para volver a hablar con el asistente.")

            if ADMIN_PHONE:
                clean_phone = effective_user_jid.split("@")[0]
                wa_link = f"https://wa.me/{clean_phone}"
                admin_msg = f"🚨 *SOLICITUD DE AYUDA HUMANA*\n\n👤 *Nombre:* {user_name}\n📱 *Teléfono:* {clean_phone}\n🔗 *Chat:* {wa_link}"

                primary_admin = re.split(r"[,;\s]+", ADMIN_PHONE)[0].strip()
                send_whatsapp_message(primary_admin, admin_msg)

            clear_pending_confirmation(effective_user_jid)
            return {"status": "ayuda_activated"}, 200

    # --- RAG RESPONSE ---
    history = get_last_messages(effective_user_jid, limit=20)
    response = responder(incoming_text, sender_jid=effective_user_jid, history=[(m["role"], m["content"]) for m in history])

    # Si responder() detectó necesidad de ayuda humana (prefijo [HELP])
    if response.startswith("[HELP]"):
        response_clean = response[6:].strip()
        save_message(effective_user_jid, "assistant", response_clean)
        save_pending_confirmation(effective_user_jid, {
            "message_id": message_id, "output_path": "", "original_filename": "",
            "state": "AWAITING_AYUDA_CONFIRMATION", "metadata": {}
        })
        send_whatsapp_message(effective_user_jid, response_clean)
        return {"status": "help_detected"}, 200

    save_message(effective_user_jid, "assistant", response)
    send_whatsapp_message(effective_user_jid, response)
    return {"status": "ok"}, 200

@app.route("/webhook", methods=["POST"])
def webhook():
    try:
        ensure_background_services()
        if not is_authorized_webhook(request):
            return jsonify({"status": "unauthorized"}), 401

        data = request.get_json()
        if not data:
            return jsonify({"status": "no_data"}), 400

        event_type = data.get("event")
        if event_type != "messages.upsert":
            return jsonify({"status": "ignored"})

        payload_data = data.get("data", {})
        messages = payload_data.get("messages") or payload_data.get("message")
        message_containers = extract_message_containers(messages)
        if not message_containers:
            return jsonify({"status": "unrecognized_structure"}), 400

        if len(message_containers) == 1:
            body, status_code = _process_single_message_container(message_containers[0])
            return jsonify(body), status_code

        results = []
        for message_container in message_containers:
            body, status_code = _process_single_message_container(message_container)
            result = {"status_code": status_code}
            result.update(body if isinstance(body, dict) else {"status": "unknown"})
            results.append(result)

        response_code = 500 if any(r["status_code"] >= 500 for r in results) else 200
        return jsonify({"status": "batch_processed", "results": results}), response_code

    except Exception as e:
        import traceback
        print(f"WEBHOOK ERROR: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    ensure_background_services()
    port = int(os.getenv("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)
