import sys

def run():
    with open('/home/jocotoco3/Documentos/NaftaECChatbot/Local/app.py', 'r') as f:
        content = f.read()

    start_str = """    # 4. Help Detection
    needs_help = False"""
    
    end_str = """    save_message(effective_user_jid, "assistant", response)
    send_whatsapp_message(effective_user_jid, response)
    return {"status": "ok"}, 200"""
    
    start_idx = content.find(start_str)
    end_idx = content.find(end_str) + len(end_str)
    
    if start_idx == -1 or end_idx < len(end_str):
        print("Could not find start or end string.")
        return
        
    replacement = """    # --- RAG RESPONSE ---
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
    return {"status": "ok"}, 200"""
    
    new_content = content[:start_idx] + replacement + content[end_idx:]
    
    with open('/home/jocotoco3/Documentos/NaftaECChatbot/Local/app.py', 'w') as f:
        f.write(new_content)
    print("Replaced successfully.")

run()
