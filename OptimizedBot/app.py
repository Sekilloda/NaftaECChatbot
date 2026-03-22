import os
import threading
import json
import re
import hmac
from flask import Flask, request, jsonify
from dotenv import load_dotenv

# Load environment variables once at entry point
load_dotenv()

# Core modules
from core.whatsapp import send_whatsapp_message, download_media, decrypt_and_save_media
from core.knowledge import responder, client
from core.registrations import update_registrations
from core.database import (
    init_db, save_message, get_last_messages, 
    save_pending_confirmation, get_pending_confirmation, clear_pending_confirmation,
    set_user_status, get_user_status
)

app = Flask(__name__)

# Initialize database at import time as well (Gunicorn imports app module)
init_db()

_SYNC_THREAD_LOCK = threading.Lock()
_SYNC_THREAD_STARTED = False
_OCR_PROCESSOR = None

ADMIN_PHONE = os.getenv("ADMIN_PHONE")
if not ADMIN_PHONE:
    print("[APP] WARNING: ADMIN_PHONE not set. Ayuda status will not forward messages.")

WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "").strip()
if not WEBHOOK_SECRET:
    print("[APP] WARNING: WEBHOOK_SECRET not set. Webhook endpoint is unauthenticated.")

GEMINI_CLASSIFIER_MODEL = os.getenv("GEMINI_CLASSIFIER_MODEL", "gemini-2.5-flash-lite")

def ensure_background_services():
    global _SYNC_THREAD_STARTED
    if _SYNC_THREAD_STARTED:
        return
    with _SYNC_THREAD_LOCK:
        if _SYNC_THREAD_STARTED:
            return
        init_db()
        threading.Thread(target=update_registrations, daemon=True, name="registrations-sync").start()
        _SYNC_THREAD_STARTED = True
        print("[APP] Background services initialized.")

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
        return True

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

        # Robust extraction of the message object
        payload_data = data.get("data", {})
        messages = payload_data.get("messages") or payload_data.get("message")
        
        if isinstance(messages, list) and len(messages) > 0:
            message_container = messages[0]
        elif isinstance(messages, dict):
            # Sometimes APIs return dicts with string indices "0", "1"...
            message_container = messages.get("0") or messages
        else:
            print(f"WEBHOOK: Unrecognized message structure in data['data']: {json.dumps(data.get('data'))}")
            return jsonify({"status": "unrecognized_structure"}), 400

        if not message_container or message_container.get("key", {}).get("fromMe"):
            return jsonify({"status": "skipped"})

        sender = message_container.get("key", {}).get("remoteJid")
        message_id = message_container.get("key", {}).get("id", "unknown")
        msg_content = message_container.get("message", {})

        # Handle users in 'ayuda' status (Human Takeover)
        user_status = get_user_status(sender)
        if user_status == 'ayuda':
            incoming_text = msg_content.get("conversation") or msg_content.get("extendedTextMessage", {}).get("text")
            if incoming_text:
                save_message(sender, "user", incoming_text)
                if ADMIN_PHONE:
                    admin_msg = f"📩 AYUDA SOLICITADA:\nUsuario: {sender}\nMensaje: {incoming_text}"
                    if not send_whatsapp_message(ADMIN_PHONE, admin_msg):
                        return jsonify({"status": "admin_delivery_failed"}), 502
                return jsonify({"status": "forwarded_to_admin"})
            return jsonify({"status": "ayuda_active_media_ignored"})

        # 1. Media Handling (Detection and Download)
        media_type = None
        if 'imageMessage' in msg_content: media_type, ext = 'image', '.jpg'
        elif 'videoMessage' in msg_content: media_type, ext = 'video', '.mp4'
        elif 'audioMessage' in msg_content: media_type, ext = 'audio', '.ogg'
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
                    if media_type == 'image':
                        save_pending_confirmation(sender, {
                            "message_id": message_id, "output_path": output_path, 
                            "original_filename": f'{message_id}{ext}', "state": "awaiting_receipt_check"
                        })
                        send_whatsapp_message(sender, "¿Esta imagen corresponde a un comprobante de pago?")
                        return jsonify({'status': 'media_received_awaiting_confirmation'})
                return jsonify({'status': 'media_processed'})

        # 2. Text Handling
        incoming_text = msg_content.get("conversation") or msg_content.get("extendedTextMessage", {}).get("text")
        if not incoming_text:
            return jsonify({"status": "unsupported_type"})

        save_message(sender, "user", incoming_text)

        # 3. State Machine (Confirmations) - LOCAL PRE-PROCESSING
        pending = get_pending_confirmation(sender)
        if pending:
            # Local Fuzzy Match (Save API calls for simple yes/no)
            classification = classify_confirmation_reply(incoming_text)
            if classification == 'unclear' and client:
                # Only if local matching fails, we use Gemini (1.5-flash is cheaper/more robust)
                classify_prompt = f"Analiza el mensaje: '{incoming_text}'. El usuario está respondiendo a una confirmación. Responde ÚNICAMENTE: AFFIRMATIVE, NEGATIVE, o UNCLEAR."
                try:
                    res = client.models.generate_content(model=GEMINI_CLASSIFIER_MODEL, contents=classify_prompt)
                    classification = res.text.strip().lower()
                except Exception as e:
                    print(f"[APP] Confirmation classification fallback failed: {e}")
                    classification = 'unclear'

            if 'affirmative' in classification:
                if pending['state'] == 'awaiting_receipt_check':
                    send_whatsapp_message(sender, "Procesando el recibo, un momento...")
                    if process_receipt_image_lazy(pending['output_path'], pending['original_filename']):
                        txt_path = os.path.splitext(pending['output_path'])[0] + ".txt"
                        with open(txt_path, 'r', encoding='utf-8') as f:
                            receipt_data = f.read()
                        save_pending_confirmation(sender, {**pending, "state": "awaiting_data_validation"})
                        send_whatsapp_message(sender, f"Datos extraídos:\n\n{receipt_data}\n\n¿Son correctos estos datos?")
                    else:
                        send_whatsapp_message(sender, "Lo siento, no pude procesar la imagen.")
                        clear_pending_confirmation(sender)
                elif pending['state'] == 'awaiting_data_validation':
                    send_whatsapp_message(sender, "¡Excelente! Hemos validado tu comprobante. Pronto recibirás noticias nuestras.")
                    clear_pending_confirmation(sender)
                elif pending['state'] == 'awaiting_ayuda_confirmation':
                    set_user_status(sender, 'ayuda')
                    send_whatsapp_message(sender, "Entendido. He activado la atención personalizada. Tu siguiente mensaje será enviado directamente a un representante de NaftaEC.")
                    clear_pending_confirmation(sender)
            elif 'negative' in classification:
                send_whatsapp_message(sender, "Entendido. No realizaré más acciones con esta imagen.")
                clear_pending_confirmation(sender)
            else:
                send_whatsapp_message(sender, "Por favor, responde 'sí' o 'no'.")
            return jsonify({'status': 'state_handled'})

        # 4. General Query (Hybrid RAG + Registrations)
        # Note: Help Detection is now merged into the main responder in knowledge.py
        history = get_last_messages(sender, limit=20)
        
        # We now check if the response starts with a special [HELP_REQUESTED] flag
        response = responder(incoming_text, sender_jid=sender, history=[(m['role'], m['content']) for m in history])
        
        if "[HELP_REQUESTED]" in response:
            clean_msg = response.replace("[HELP_REQUESTED]", "").strip()
            save_pending_confirmation(sender, {
                "message_id": message_id, "output_path": "", 
                "original_filename": "", "state": "awaiting_ayuda_confirmation"
            })
            if not send_whatsapp_message(sender, clean_msg or "¿Necesitas ayuda de un representante de NaftaEC? Por favor confirma si es así."):
                return jsonify({"status": "delivery_failed", "stage": "help_confirmation"}), 502
            return jsonify({"status": "help_requested_confirmation_sent"})

        save_message(sender, "assistant", response)
        if not send_whatsapp_message(sender, response):
            return jsonify({"status": "delivery_failed", "stage": "assistant_reply"}), 502
        
        return jsonify({"status": "ok"})

    except Exception as e:
        import traceback
        print(f"WEBHOOK ERROR: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    ensure_background_services()
    
    port = int(os.getenv("PORT", 5001))
    debug_enabled = os.getenv("FLASK_DEBUG", "0").strip().lower() in {"1", "true", "yes"}
    print(f"[APP] Starting NaftaEC Chatbot on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=debug_enabled, use_reloader=debug_enabled)
