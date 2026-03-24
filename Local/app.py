import os
import threading
import json
import re
import hmac
import hashlib
from flask import Flask, request, jsonify
from dotenv import load_dotenv

# Load environment variables once at entry point
load_dotenv()

# Core modules
from core.whatsapp import send_whatsapp_message, download_media, decrypt_and_save_media, normalize_phone
from core.knowledge import responder, client
from core.registrations import update_registrations
from core.database import (
    init_db, save_message, get_last_messages, 
    save_pending_confirmation, get_pending_confirmation, clear_pending_confirmation,
    set_user_status, get_user_status, reset_user_status, save_validated_registry
)

app = Flask(__name__)

# Inicializamos la base de datos aquí
init_db()

_SYNC_THREAD_LOCK = threading.Lock()
_SYNC_THREAD_STARTED = False
_OCR_PROCESSOR = None

def cleanup_old_media(max_age_days=7):
    """Periodically cleans up the media folder to save space."""
    import time
    media_dir = 'media'
    if not os.path.exists(media_dir):
        return
        
    while True:
        try:
            now = time.time()
            for filename in os.listdir(media_dir):
                file_path = os.path.join(media_dir, filename)
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

@app.route("/webhook", methods=["POST"])
def webhook():
    try:
        ensure_background_services()
        if not is_authorized_webhook(request):
            return jsonify({"status": "unauthorized"}), 401

        data = request.get_json()
        if not data:
            return jsonify({"status": "no_data"}), 400
            
        if data.get("event") != "messages.upsert":
            return jsonify({"status": "ignored"})

        payload_data = data.get("data", {})
        messages = payload_data.get("messages") or payload_data.get("message")
        
        if isinstance(messages, list) and len(messages) > 0:
            message_container = messages[0]
        elif isinstance(messages, dict):
            message_container = messages.get("0") or messages
        else:
            return jsonify({"status": "unrecognized_structure"}), 400

        if not message_container or message_container.get("key", {}).get("fromMe"):
            return jsonify({"status": "skipped"})

        sender = message_container.get("key", {}).get("remoteJid")
        message_id = message_container.get("key", {}).get("id", "unknown")
        msg_content = message_container.get("message", {})

        user_status = get_user_status(sender)
        incoming_text = (msg_content.get("conversation") or msg_content.get("extendedTextMessage", {}).get("text") or "").strip()
        
        # Admin Override
        if is_admin_sender(sender):
            if incoming_text.lower().startswith("#resolver"):
                parts = incoming_text.split()
                if len(parts) > 1:
                    target_phone = normalize_phone(parts[1])
                    target_jid = f"{target_phone}@s.whatsapp.net"
                    reset_user_status(target_jid)
                    send_whatsapp_message(sender, f"✅ Estado de {target_phone} reseteado a 'bot'.")
                    send_whatsapp_message(target_jid, "Un representante ha marcado tu consulta como resuelta. El asistente virtual vuelve a estar activo.")
                    return jsonify({"status": "admin_reset_success"})

        if user_status == 'ayuda':
            if incoming_text.lower() == "#bot":
                reset_user_status(sender)
                send_whatsapp_message(sender, "Entendido. El asistente virtual vuelve a estar activo. ¿En qué puedo ayudarte?")
                return jsonify({"status": "user_reset_success"})
            if incoming_text:
                save_message(sender, "user", incoming_text)
                if ADMIN_PHONE:
                    push_name = message_container.get("pushName", "Desconocido")
                    clean_sender = sender.split('@')[0]
                    admin_msg = f"💬 Mensaje de {push_name} ({clean_sender}):\n{incoming_text}"
                    send_whatsapp_message(ADMIN_PHONE, admin_msg)
                return jsonify({"status": "forwarded_to_admin"})

        # --- DETERMINISTIC OCR MODE HANDLER ---
        pending = get_pending_confirmation(sender)
        if pending and pending['state'].startswith('OCR_'):
            save_message(sender, "user", incoming_text)
            
            # 1. OCR Edit Mode
            if pending['state'] == 'OCR_EDIT_MODE':
                if incoming_text.lower() == 'correcto':
                    save_pending_confirmation(sender, {**pending, "state": "OCR_AWAITING_RUNNER_COUNT"})
                    send_whatsapp_message(sender, "¿A cuántos corredores corresponde esta transacción?")
                    return jsonify({'status': 'ocr_confirmed'})

                # Handle specific field updates: "banco: Pichincha", "fecha: 01/01/2024", "cuenta: 12345"
                match = re.match(r"(?i)^(banco|fecha|cuenta|cuenta origen)\s*:\s*(.*)", incoming_text)
                if match:
                    field = match.group(1).lower().strip()
                    value = match.group(2).strip()
                    
                    ocr_data = pending['metadata'].get('ocr_data', {})
                    if 'banco' in field: ocr_data['banco'] = value
                    elif 'fecha' in field: ocr_data['fecha'] = value
                    elif 'cuenta' in field: ocr_data['cuenta_origen'] = value
                    
                    pending['metadata']['ocr_data'] = ocr_data
                    save_pending_confirmation(sender, pending)
                    
                    msg = f"Dato actualizado.\n\n{format_ocr_data(ocr_data)}\n\n¿Deseas corregir algo más (Banco, Fecha, Cuenta Origen) o es 'Correcto'?"
                    send_whatsapp_message(sender, msg)
                    return jsonify({'status': 'ocr_field_updated'})

                # Block sensitive field updates
                if any(x in incoming_text.lower() for x in ['monto', 'total', 'comprobante', 'numero', 'referencia']):
                    send_whatsapp_message(sender, "Lo siento, el Monto y el Número de comprobante no pueden ser editados por seguridad. Si los datos son incorrectos, ¿deseas intentar subir la imagen de nuevo? (responde 'reintentar' o 'cancelar')")
                    return jsonify({'status': 'ocr_edit_blocked'})

                if incoming_text.lower() == 'reintentar':
                    clear_pending_confirmation(sender)
                    send_whatsapp_message(sender, "Entendido. Por favor, envía la imagen del comprobante de nuevo.")
                    return jsonify({'status': 'ocr_reset'})
                
                send_whatsapp_message(sender, "Por favor, para editar usa el formato 'Campo: Valor' (ej: Banco: Pichincha) o responde 'Correcto' si todo está bien.")
                return jsonify({'status': 'ocr_waiting_valid_input'})

            # 2. Awaiting Runner Count
            if pending['state'] == 'OCR_AWAITING_RUNNER_COUNT':
                try:
                    count = int(re.sub(r'\D', '', incoming_text))
                    if count <= 0 or count > 10: raise ValueError()
                    
                    pending['metadata']['runner_count'] = count
                    pending['metadata']['cedulas_collected'] = []
                    pending['state'] = 'OCR_AWAITING_CEDULAS'
                    save_pending_confirmation(sender, pending)
                    
                    send_whatsapp_message(sender, f"Entendido ({count} corredores). Por favor, ingresa el primer Número de cédula:")
                    return jsonify({'status': 'ocr_count_received'})
                except:
                    send_whatsapp_message(sender, "Por favor, ingresa un número válido de corredores (1-10).")
                    return jsonify({'status': 'ocr_invalid_count'})

            # 3. Awaiting Cedulas
            if pending['state'] == 'OCR_AWAITING_CEDULAS':
                cedula = re.sub(r'\D', '', incoming_text)
                if len(cedula) < 5:
                    send_whatsapp_message(sender, "El número de cédula parece inválido. Inténtalo de nuevo.")
                    return jsonify({'status': 'ocr_invalid_cedula'})
                
                collected = pending['metadata'].get('cedulas_collected', [])
                collected.append(cedula)
                pending['metadata']['cedulas_collected'] = collected
                
                target_count = pending['metadata'].get('runner_count', 1)
                if len(collected) < target_count:
                    save_pending_confirmation(sender, pending)
                    send_whatsapp_message(sender, f"Cédula recibida ({len(collected)}/{target_count}). Ingresa la siguiente:")
                else:
                    # Transition to final confirmation
                    pending['state'] = 'OCR_FINAL_CONFIRMATION'
                    save_pending_confirmation(sender, pending)
                    
                    ocr_data = pending['metadata']['ocr_data']
                    cedulas_str = "\n".join([f"- {c}" for c in collected])
                    summary = (
                        f"📝 *RESUMEN FINAL DE REGISTRO*\n\n"
                        f"{format_ocr_data(ocr_data)}\n"
                        f"👥 Cédulas de corredores ({len(collected)}):\n{cedulas_str}\n\n"
                        "¿Confirmas que toda la información es correcta? Responde 'CONFIRMAR' para finalizar o 'REINTENTAR' para empezar de nuevo."
                    )
                    send_whatsapp_message(sender, summary)
                return jsonify({'status': 'ocr_all_cedulas_received'})

            # 4. Final Confirmation Handler
            if pending['state'] == 'OCR_FINAL_CONFIRMATION':
                if 'confirmar' in incoming_text.lower() or 'si' == incoming_text.lower():
                    ocr_data = pending['metadata']['ocr_data']
                    collected = pending['metadata']['cedulas_collected']
                    
                    for c in collected:
                        unique_id = hash_registry({"cedula": c, "num": ocr_data['numero_comprobante'], "monto": ocr_data['monto']})
                        save_validated_registry({
                            "unique_id": unique_id,
                            "sender_jid": sender,
                            "cedula": c,
                            "banco": ocr_data['banco'],
                            "monto": ocr_data['monto'],
                            "fecha": ocr_data['fecha'],
                            "numero_comprobante": ocr_data['numero_comprobante'],
                            "cuenta_origen": ocr_data['cuenta_origen']
                        })
                    
                    send_whatsapp_message(sender, "Comprobante registrado exitosamente.")
                    clear_pending_confirmation(sender)
                    return jsonify({'status': 'ocr_final_success'})
                
                elif 'reintentar' in incoming_text.lower() or 'no' == incoming_text.lower():
                    clear_pending_confirmation(sender)
                    send_whatsapp_message(sender, "Registro cancelado. Por favor, envía la imagen del comprobante de nuevo si deseas iniciar otro registro.")
                    return jsonify({'status': 'ocr_final_reset'})
                
                else:
                    send_whatsapp_message(sender, "Por favor, responde 'CONFIRMAR' para guardar los datos o 'REINTENTAR' para cancelar.")
                    return jsonify({'status': 'ocr_final_waiting'})

        # --- MEDIA HANDLING ---
        media_type = None
        if 'imageMessage' in msg_content: media_type, ext = 'image', '.jpg'
        elif 'documentMessage' in msg_content: 
            media_type = 'document'
            ext = os.path.splitext(msg_content['documentMessage'].get("fileName", ""))[1]

        if media_type:
            media_info = msg_content.get(f'{media_type}Message')
            media_key, media_url = media_info.get('mediaKey'), media_info.get('url')
            if media_key and media_url:
                output_path = os.path.join('media', f'{message_id}{ext}')
                encrypted_data = download_media(media_url)
                if encrypted_data and decrypt_and_save_media(media_key, encrypted_data, output_path, media_type):
                    if media_type == 'image' or (media_type == 'document' and ext.lower() in ['.jpg', '.jpeg', '.png']):
                        save_pending_confirmation(sender, {
                            "message_id": message_id, "output_path": output_path, 
                            "original_filename": f'{message_id}{ext}', "state": "AWAITING_RECEIPT_CONFIRMATION",
                            "metadata": {}
                        })
                        send_whatsapp_message(sender, "¿Esta imagen corresponde a un comprobante de pago?")
                        return jsonify({'status': 'media_received_awaiting_confirmation'})
                return jsonify({'status': 'media_processed'})

        # --- STANDARD FLOW ---
        if not incoming_text:
            return jsonify({"status": "unsupported_type"})

        save_message(sender, "user", incoming_text)

        if pending:
            if pending['state'] == 'AWAITING_RECEIPT_CONFIRMATION':
                classification = classify_confirmation_reply(incoming_text)
                if 'affirmative' in classification:
                    send_whatsapp_message(sender, "Procesando el recibo, un momento...")
                    ocr_data = process_receipt_image_lazy(pending['output_path'], pending['original_filename'])
                    if ocr_data:
                        pending['state'] = 'OCR_EDIT_MODE'
                        pending['metadata']['ocr_data'] = ocr_data
                        save_pending_confirmation(sender, pending)
                        
                        msg = f"Datos extraídos:\n\n{format_ocr_data(ocr_data)}\n\n¿Deseas corregir algún campo (Banco, Fecha, Cuenta Origen)? Responde con 'Campo: Valor' o escribe 'Correcto'."
                        send_whatsapp_message(sender, msg)
                    else:
                        send_whatsapp_message(sender, "Lo siento, no pude procesar la imagen.")
                        clear_pending_confirmation(sender)
                    return jsonify({'status': 'ocr_started'})
                elif 'negative' in classification:
                    send_whatsapp_message(sender, "Entendido. No realizaré acciones con esta imagen.")
                    clear_pending_confirmation(sender)
                    return jsonify({'status': 'ocr_cancelled'})

        # 4. Help Detection
        needs_help = False
        if client:
            try:
                help_prompt = f"Determina si el siguiente mensaje indica que el usuario necesita ayuda humana: '{incoming_text}'. Responde: HELP o OK."
                help_res = client.models.generate_content(model=GEMINI_HELP_MODEL, contents=help_prompt)
                needs_help = 'help' in help_res.text.strip().lower()
            except Exception: pass

        if needs_help:
            save_pending_confirmation(sender, {"message_id": message_id, "output_path": "", "original_filename": "", "state": "AWAITING_AYUDA_CONFIRMATION", "metadata": {}})
            send_whatsapp_message(sender, "¿Necesitas ayuda de un representante? Confirma si es así.")
            return jsonify({"status": "help_confirmation_sent"})

        # 5. General Query
        history = get_last_messages(sender, limit=20)
        response = responder(incoming_text, sender_jid=sender, history=[(m['role'], m['content']) for m in history])
        save_message(sender, "assistant", response)
        send_whatsapp_message(sender, response)
        return jsonify({"status": "ok"})

    except Exception as e:
        import traceback
        print(f"WEBHOOK ERROR: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    ensure_background_services()
    port = int(os.getenv("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)
