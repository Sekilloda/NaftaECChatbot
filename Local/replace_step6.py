import sys

def run():
    with open('/home/jocotoco3/Documentos/NaftaECChatbot/Local/app.py', 'r') as f:
        content = f.read()

    start_str = """    # --- DETERMINISTIC OCR MODE HANDLER ---
    pending = get_pending_confirmation(effective_user_jid)"""
    
    end_str = """            clear_pending_confirmation(effective_user_jid)
            return {"status": "ayuda_activated"}, 200"""
    
    start_idx = content.find(start_str)
    end_idx = content.find(end_str) + len(end_str)
    
    if start_idx == -1 or end_idx < len(end_str):
        print("Could not find start or end string.")
        return
        
    replacement = """    # --- STANDARD FLOW ---
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
            return {"status": "ayuda_activated"}, 200"""
    
    new_content = content[:start_idx] + replacement + content[end_idx:]
    
    with open('/home/jocotoco3/Documentos/NaftaECChatbot/Local/app.py', 'w') as f:
        f.write(new_content)
    print("Replaced successfully.")

run()
