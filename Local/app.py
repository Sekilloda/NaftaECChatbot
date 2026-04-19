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
from core.whatsapp import send_whatsapp_message, send_whatsapp_document, download_media, decrypt_and_save_media, normalize_phone
from core.knowledge import responder, client
from core.registrations import update_registrations
from core.database import (
    init_db, save_message, get_last_messages, 
    save_pending_confirmation, get_pending_confirmation, clear_pending_confirmation,
    set_user_status, get_user_status, reset_user_status, save_validated_registry
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
_OCR_PROCESSOR = None

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

GEMINI_CLASSIFIER_MODEL = os.getenv("GEMINI_CLASSIFIER_MODEL", "gemini-2.5-flash-lite")
GEMINI_HELP_MODEL = os.getenv("GEMINI_HELP_MODEL", "gemini-2.5-flash-lite")

def hash_registry(data_dict):
    """Generates a unique hash for a registry entry."""
    encoded_str = json.dumps(data_dict, sort_keys=True).encode('utf-8')
    return hashlib.sha256(encoded_str).hexdigest()[:16]

def process_receipt_image_lazy(image_path, original_filename):
    global _OCR_PROCESSOR
    if _OCR_PROCESSOR is None:
        from core.ocr import process_receipt_image as _process_receipt_image
        _OCR_PROCESSOR = _process_receipt_image
    return _OCR_PROCESSOR(image_path, original_filename)

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

def format_ocr_data(data):
    return (
        f"🏦 Banco: {data.get('banco') or 'No detectado'}\n"
        f"💰 Monto: {data.get('monto') or 'No detectado'}\n"
        f"📅 Fecha: {data.get('fecha') or 'No detectado'}\n"
        f"🔢 Comprobante: {data.get('numero_comprobante') or 'No detectado'}\n"
        f"💳 Cuenta Origen: {data.get('cuenta_origen') or 'No detectado'}"
    )

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

    # --- DETERMINISTIC OCR MODE HANDLER ---
    pending = get_pending_confirmation(effective_user_jid)
    if pending and pending["state"].startswith("OCR_"):
        save_message(effective_user_jid, "user", incoming_text)

        # 1. OCR Edit Mode
        if pending["state"] == "OCR_EDIT_MODE":
            if incoming_text.lower() == "correcto":
                save_pending_confirmation(effective_user_jid, {**pending, "state": "OCR_AWAITING_RUNNER_COUNT"})
                send_whatsapp_message(effective_user_jid, "¿A cuántos corredores corresponde esta transacción?")
                return {"status": "ocr_confirmed"}, 200

            # Handle specific field updates
            match = re.match(r"(?i)^(banco|fecha|cuenta|cuenta origen)\s*:\s*(.*)", incoming_text)
            if match:
                field = match.group(1).lower().strip()
                value = match.group(2).strip()

                ocr_data = pending["metadata"].get("ocr_data", {})
                if "banco" in field:
                    ocr_data["banco"] = value
                elif "fecha" in field:
                    ocr_data["fecha"] = value
                elif "cuenta" in field:
                    ocr_data["cuenta_origen"] = value

                pending["metadata"]["ocr_data"] = ocr_data
                save_pending_confirmation(effective_user_jid, pending)

                msg = f"Dato actualizado.\n\n{format_ocr_data(ocr_data)}\n\n¿Deseas corregir algo más (Banco, Fecha, Cuenta Origen) o es 'Correcto'?"
                send_whatsapp_message(effective_user_jid, msg)
                return {"status": "ocr_field_updated"}, 200

            # Block sensitive field updates
            if any(x in incoming_text.lower() for x in ["monto", "total", "comprobante", "numero", "referencia"]):
                send_whatsapp_message(effective_user_jid, "Lo siento, el Monto y el Número de comprobante no pueden ser editados por seguridad. Si los datos son incorrectos, ¿deseas intentar subir la imagen de nuevo? (responde 'reintentar' o 'cancelar')")
                return {"status": "ocr_edit_blocked"}, 200

            if incoming_text.lower() == "reintentar":
                clear_pending_confirmation(effective_user_jid)
                send_whatsapp_message(effective_user_jid, "Entendido. Por favor, envía la imagen del comprobante de nuevo.")
                return {"status": "ocr_reset"}, 200

            send_whatsapp_message(effective_user_jid, "Por favor, para editar usa el formato 'Campo: Valor' (ej: Banco: Pichincha) o responde 'Correcto' si todo está bien.")
            return {"status": "ocr_waiting_valid_input"}, 200

        # 2. Awaiting Runner Count
        if pending["state"] == "OCR_AWAITING_RUNNER_COUNT":
            try:
                count = int(re.sub(r"\D", "", incoming_text))
                if count <= 0 or count > 10:
                    raise ValueError()

                pending["metadata"]["runner_count"] = count
                pending["metadata"]["cedulas_collected"] = []
                pending["state"] = "OCR_AWAITING_CEDULAS"
                save_pending_confirmation(effective_user_jid, pending)

                send_whatsapp_message(effective_user_jid, f"Entendido ({count} corredores). Por favor, ingresa el primer Número de cédula:")
                return {"status": "ocr_count_received"}, 200
            except Exception:
                send_whatsapp_message(effective_user_jid, "Por favor, ingresa un número válido de corredores (1-10).")
                return {"status": "ocr_invalid_count"}, 200

        # 3. Awaiting Cedulas
        if pending["state"] == "OCR_AWAITING_CEDULAS":
            cedula = re.sub(r"\D", "", incoming_text)
            if len(cedula) < 5:
                send_whatsapp_message(effective_user_jid, "El número de cédula parece inválido. Inténtalo de nuevo.")
                return {"status": "ocr_invalid_cedula"}, 200

            collected = pending["metadata"].get("cedulas_collected", [])
            collected.append(cedula)
            pending["metadata"]["cedulas_collected"] = collected

            target_count = pending["metadata"].get("runner_count", 1)
            if len(collected) < target_count:
                save_pending_confirmation(effective_user_jid, pending)
                send_whatsapp_message(effective_user_jid, f"Cédula recibida ({len(collected)}/{target_count}). Ingresa la siguiente:")
            else:
                pending["state"] = "OCR_FINAL_CONFIRMATION"
                save_pending_confirmation(effective_user_jid, pending)

                ocr_data = pending["metadata"]["ocr_data"]
                cedulas_str = "\n".join([f"- {c}" for c in collected])
                summary = (
                    f"📝 *RESUMEN FINAL DE REGISTRO*\n\n"
                    f"{format_ocr_data(ocr_data)}\n"
                    f"👥 Cédulas de corredores ({len(collected)}):\n{cedulas_str}\n\n"
                    "¿Confirmas que toda la información es correcta? Responde 'CONFIRMAR' para finalizar o 'REINTENTAR' para empezar de nuevo."
                )
                send_whatsapp_message(effective_user_jid, summary)
            return {"status": "ocr_all_cedulas_received"}, 200

        # 4. Final Confirmation Handler
        if pending["state"] == "OCR_FINAL_CONFIRMATION":
            if "confirmar" in incoming_text.lower() or "si" == incoming_text.lower():
                ocr_data = pending["metadata"].get("ocr_data", {})
                collected = pending["metadata"].get("cedulas_collected", [])

                inserted_count = 0
                total_count = len(collected)

                for cedula in collected:
                    comprobante = ocr_data.get("numero_comprobante", "")
                    monto = ocr_data.get("monto", "")
                    unique_id = hash_registry({"cedula": cedula, "num": comprobante, "monto": monto})
                    inserted = save_validated_registry({
                        "unique_id": unique_id,
                        "sender_jid": effective_user_jid,
                        "cedula": cedula,
                        "banco": ocr_data.get("banco", ""),
                        "monto": monto,
                        "fecha": ocr_data.get("fecha", ""),
                        "numero_comprobante": comprobante,
                        "cuenta_origen": ocr_data.get("cuenta_origen", "")
                    })
                    if inserted:
                        inserted_count += 1

                if total_count == 0:
                    response_message = "No encontré cédulas para registrar. Por favor, reintenta el proceso."
                elif inserted_count == total_count:
                    response_message = "Comprobante registrado exitosamente."
                elif inserted_count == 0:
                    response_message = "Este comprobante ya había sido registrado anteriormente."
                else:
                    response_message = f"Comprobante registrado parcialmente ({inserted_count}/{total_count} nuevos registros)."

                send_whatsapp_message(effective_user_jid, response_message)
                clear_pending_confirmation(effective_user_jid)
                return {"status": "ocr_final_success", "inserted": inserted_count, "total": total_count}, 200

            if "reintentar" in incoming_text.lower() or "no" == incoming_text.lower():
                clear_pending_confirmation(effective_user_jid)
                send_whatsapp_message(effective_user_jid, "Registro cancelado. Por favor, envía la imagen del comprobante de nuevo si deseas iniciar otro registro.")
                return {"status": "ocr_final_reset"}, 200

            send_whatsapp_message(effective_user_jid, "Por favor, responde 'CONFIRMAR' para guardar los datos o 'REINTENTAR' para cancelar.")
            return {"status": "ocr_final_waiting"}, 200

    # --- MEDIA HANDLING ---
    media_type = None
    ext = ""
    if "imageMessage" in msg_content:
        media_type, ext = "image", ".jpg"
    elif "documentMessage" in msg_content:
        media_type = "document"
        ext = os.path.splitext(msg_content["documentMessage"].get("fileName", ""))[1]

    if media_type:
        media_info = msg_content.get(f"{media_type}Message") or {}
        media_key, media_url = media_info.get("mediaKey"), media_info.get("url")
        if media_key and media_url:
            try:
                output_path, safe_filename = build_media_output_path(message_id, ext)
            except ValueError:
                print(f"[WEBHOOK] Rejected media path for message_id={message_id!r}")
                return {"status": "invalid_media_identifier"}, 400

            encrypted_data = download_media(media_url)
            if encrypted_data and decrypt_and_save_media(media_key, encrypted_data, output_path, media_type):
                if media_type == "image" or (media_type == "document" and ext.lower() in [".jpg", ".jpeg", ".png"]):
                    save_pending_confirmation(effective_user_jid, {
                        "message_id": safe_filename,
                        "output_path": output_path,
                        "original_filename": safe_filename,
                        "state": "AWAITING_RECEIPT_CONFIRMATION",
                        "metadata": {}
                    })
                    send_whatsapp_message(effective_user_jid, "¿Esta imagen corresponde a un comprobante de pago?")
                    return {"status": "media_received_awaiting_confirmation"}, 200
            return {"status": "media_processed"}, 200

    # --- STANDARD FLOW ---
    if not incoming_text:
        return {"status": "unsupported_type"}, 200

    save_message(effective_user_jid, "user", incoming_text)

    if pending:
        if pending["state"] == "AWAITING_RECEIPT_CONFIRMATION":
            classification = classify_confirmation_reply(incoming_text)
            if "affirmative" in classification:
                send_whatsapp_message(effective_user_jid, "Procesando el recibo, un momento...")
                ocr_data = process_receipt_image_lazy(pending["output_path"], pending["original_filename"])
                if ocr_data:
                    pending["state"] = "OCR_EDIT_MODE"
                    pending["metadata"]["ocr_data"] = ocr_data
                    save_pending_confirmation(effective_user_jid, pending)

                    msg = f"Datos extraídos:\n\n{format_ocr_data(ocr_data)}\n\n¿Deseas corregir algún campo (Banco, Fecha, Cuenta Origen)? Responde con 'Campo: Valor' o escribe 'Correcto'."
                    send_whatsapp_message(effective_user_jid, msg)
                else:
                    send_whatsapp_message(effective_user_jid, "Lo siento, no pude procesar la imagen.")
                    clear_pending_confirmation(effective_user_jid)
                return {"status": "ocr_started"}, 200
            if "negative" in classification:
                send_whatsapp_message(effective_user_jid, "Entendido. No realizaré acciones con esta imagen.")
                clear_pending_confirmation(effective_user_jid)
                return {"status": "ocr_cancelled"}, 200

        elif pending["state"] == "AWAITING_AYUDA_CONFIRMATION":
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

    # 4. Help Detection
    # 4. Help Detection
    needs_help = False
    if client:
        try:
           help_prompt = (
            f"Analiza este mensaje de WhatsApp: '{incoming_text}'\n\n"
            
            "Tu tarea es clasificar si el caso debe derivarse a un asesor humano.\n\n"
            
            "Responde HELP si el usuario:\n"
            "- Quiere hablar con una persona, asesor, agente, representante o humano\n"
            "- Da a entender que necesita atención personalizada\n"
            "- Está confundido y necesita más ayuda de la que el bot puede dar\n"
            "- Expresa molestia, frustración, enojo, desconfianza o reclama un problema\n"
            "- Dice que el bot no le ayuda, no entiende la respuesta, o quiere salir del bot\n"
            "- Insiste en el mismo problema sin resolverse\n\n"
            
            "Responde OK si el usuario:\n"
            "- Solo hace una consulta normal\n"
            "- Pide información general\n"
            "- Pregunta por precios, pagos, eventos, horarios, inscripciones o requisitos\n"
            "- Hace preguntas que el bot sí podría responder\n\n"
            
            "Ejemplos:\n"
            "- 'quiero hablar con alguien' -> HELP\n"
            "- 'me pasan con un asesor?' -> HELP\n"
            "- 'no entiendo' -> HELP\n"
            "- 'esto no me sirve' -> HELP\n"
            "- 'cuánto cuesta?' -> OK\n"
            "- 'qué requisitos hay?' -> OK\n\n"
            
            "Reglas:\n"
            "- Si hay duda entre HELP y OK, responde HELP\n"
            "- Responde una sola palabra, sin explicación\n"
            "- Responde SOLO con: HELP o OK"
        )
            help_res = client.models.generate_content(model=GEMINI_HELP_MODEL, contents=help_prompt)
            response_text = help_res.text.strip().upper()
            needs_help = response_text.startswith("HELP")
        except Exception:
            pass
        

    if needs_help:
        save_pending_confirmation(effective_user_jid, {"message_id": message_id, "output_path": "", "original_filename": "", "state": "AWAITING_AYUDA_CONFIRMATION", "metadata": {}})
        send_whatsapp_message(effective_user_jid, "¿Necesitas ayuda de un representante? Confirma si es así.")
        return {"status": "help_confirmation_sent"}, 200

    # 5. General Query
    history = get_last_messages(effective_user_jid, limit=20)
    response = responder(incoming_text, sender_jid=effective_user_jid, history=[(m["role"], m["content"]) for m in history])
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
