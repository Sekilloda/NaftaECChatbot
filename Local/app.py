import os
import threading
import json
from flask import Flask, request, jsonify
from dotenv import load_dotenv

# Load environment variables once at entry point
load_dotenv()

# Core modules
from core.whatsapp import send_whatsapp_message, download_media, decrypt_and_save_media
from core.ocr import process_receipt_image
from core.knowledge import responder, client
from core.registrations import update_registrations
from core.database import (
    init_db, save_message, get_last_messages, 
    save_pending_confirmation, get_pending_confirmation, clear_pending_confirmation,
    set_user_status, get_user_status
)

app = Flask(__name__)
ADMIN_PHONE = os.getenv("ADMIN_PHONE")
if not ADMIN_PHONE:
    print("[APP] WARNING: ADMIN_PHONE not set. Ayuda status will not forward messages.")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "user_status": "ready"}), 200

@app.route("/webhook", methods=["POST"])
def webhook():
    try:
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
                    send_whatsapp_message(ADMIN_PHONE, admin_msg)
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

        # 3. State Machine (Confirmations)
        pending = get_pending_confirmation(sender)
        if pending:
            # Classification of user response
            classify_prompt = f"Analiza el mensaje: '{incoming_text}'. El usuario está respondiendo a una confirmación. Responde ÚNICAMENTE: AFFIRMATIVE, NEGATIVE, o UNCLEAR."
            res = client.models.generate_content(model="gemini-flash-lite-latest", contents=classify_prompt)
            classification = res.text.strip().lower()

            if 'affirmative' in classification:
                if pending['state'] == 'awaiting_receipt_check':
                    send_whatsapp_message(sender, "Procesando el recibo, un momento...")
                    if process_receipt_image(pending['output_path'], pending['original_filename']):
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

        # 4. Help Detection (Edge Case Catcher)
        help_prompt = f"Determina si el siguiente mensaje indica que el usuario necesita ayuda humana, tiene un problema que el bot no puede resolver, o está frustrado: '{incoming_text}'. Responde ÚNICAMENTE: HELP o OK."
        help_res = client.models.generate_content(model="gemini-flash-lite-latest", contents=help_prompt)
        if 'help' in help_res.text.strip().lower():
            save_pending_confirmation(sender, {
                "message_id": message_id, "output_path": "", 
                "original_filename": "", "state": "awaiting_ayuda_confirmation"
            })
            send_whatsapp_message(sender, "¿Necesitas ayuda de un representante de NaftaEC? Por favor confirma si es así.")
            return jsonify({"status": "help_requested_confirmation_sent"})

        # 5. General Query (Hybrid RAG + Registrations)
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
    # Initialize DB and start sync thread here
    init_db()
    threading.Thread(target=update_registrations, daemon=True).start()
    
    port = int(os.getenv("PORT", 5001))
    print(f"[APP] Starting NaftaEC Chatbot on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=True)
