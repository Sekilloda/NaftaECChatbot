# Standard Python library imports
import base64
import json
import os
import pathlib
import re
import traceback
import configparser # Added
import sys # Added

# Third-party library imports
import cv2
from dotenv import load_dotenv
from flask import Flask, request, jsonify
import google.generativeai as genai
import numpy as np
from PIL import Image
import pytesseract
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from Crypto.Cipher import AES
from Crypto.Hash import HMAC, SHA256
from Crypto.Protocol.KDF import HKDF
from Crypto.Util.Padding import unpad

# Load environment variables
load_dotenv()

app = Flask(__name__)

pending_confirmations = {}

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
WASENDER_API_TOKEN = os.getenv("WASENDER_API_TOKEN")
WASENDER_API_URL = "https://wasenderapi.com/api/send-message"

# Setup Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

DEFAULT_PARAMS = {
    'clahe_clip_limit': '2.0',
    'clahe_tile_grid_size_x': '8',
    'clahe_tile_grid_size_y': '8',
    'tesseract_psm': '6',
    'tesseract_lang': 'spa',
    'save_intermediate_images': 'False' # Default to False for app.py
}

def load_params(config_file_path):
    config = configparser.ConfigParser()
    params = DEFAULT_PARAMS.copy()

    if not os.path.exists(config_file_path):
        print(f"Warning: Config file '{config_file_path}' not found. Using default parameters for OCR in app.py.")
        # Convert types for defaults
        params['clahe_clip_limit'] = float(params['clahe_clip_limit'])
        params['clahe_tile_grid_size_x'] = int(params['clahe_tile_grid_size_x'])
        params['clahe_tile_grid_size_y'] = int(params['clahe_tile_grid_size_y'])
        params['tesseract_psm'] = str(params['tesseract_psm'])
        params['save_intermediate_images'] = params['save_intermediate_images'].lower() == 'true'
        return params
    try:
        config.read(config_file_path)
        if 'Parameters' in config:
            config_params = config['Parameters']
            params['clahe_clip_limit'] = float(config_params.get('clahe_clip_limit', DEFAULT_PARAMS['clahe_clip_limit']))
            params['clahe_tile_grid_size_x'] = int(config_params.get('clahe_tile_grid_size_x', DEFAULT_PARAMS['clahe_tile_grid_size_x']))
            params['clahe_tile_grid_size_y'] = int(config_params.get('clahe_tile_grid_size_y', DEFAULT_PARAMS['clahe_tile_grid_size_y']))
            params['tesseract_psm'] = str(config_params.get('tesseract_psm', DEFAULT_PARAMS['tesseract_psm']))
            params['tesseract_lang'] = config_params.get('tesseract_lang', DEFAULT_PARAMS['tesseract_lang'])
            # save_intermediate_images is False by default in app.py, not read from file for app.py
            params['save_intermediate_images'] = False 
        else:
            print(f"Warning: '[Parameters]' section not found in '{config_file_path}'. Using default parameters for OCR in app.py.")
            params['clahe_clip_limit'] = float(params['clahe_clip_limit'])
            params['clahe_tile_grid_size_x'] = int(params['clahe_tile_grid_size_x'])
            params['clahe_tile_grid_size_y'] = int(params['clahe_tile_grid_size_y'])
            params['tesseract_psm'] = str(params['tesseract_psm'])
            params['save_intermediate_images'] = False
    except Exception as e:
        print(f"Error reading or parsing config file '{config_file_path}' in app.py: {e}. Using default parameters.")
        params['clahe_clip_limit'] = float(params['clahe_clip_limit'])
        params['clahe_tile_grid_size_x'] = int(params['clahe_tile_grid_size_x'])
        params['clahe_tile_grid_size_y'] = int(params['clahe_tile_grid_size_y'])
        params['tesseract_psm'] = str(params['tesseract_psm'])
        params['save_intermediate_images'] = False
    return params

def preprocess_image(cv_image, params, output_dir=None, base_filename=None): # output_dir/base_filename not used if not saving
    print("[PREPROC_APP] Starting preprocessing in app.py...")
    # 1. Grayscale
    gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    print("[PREPROC_APP] Converted to grayscale.")
    # No saving of intermediate images in app.py context

    # 2. CLAHE
    clahe_clip = params.get('clahe_clip_limit', 2.0) # Default if not in params
    clahe_tile_x = params.get('clahe_tile_grid_size_x', 8)
    clahe_tile_y = params.get('clahe_tile_grid_size_y', 8)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_tile_x, clahe_tile_y))
    processed_image = clahe.apply(gray_image)
    print(f"[PREPROC_APP] Applied CLAHE (Clip: {clahe_clip}, Tile: ({clahe_tile_x},{clahe_tile_y})).")
    
    # 3. Binarization (Otsu)
    _, processed_image = cv2.threshold(processed_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print("[PREPROC_APP] Applied Otsu's binarization.")
    
    return processed_image

faqs = [
    {
        "type": "faq",
        "question": "¿Hasta cuándo están abiertas las inscripciones de la carrera 1?",
        "answer": "Hola, las inscripciones están abiertas hasta la primera semana de abril o cuando llenemos los cupos, lo que suceda primero."
    },
    {
        "type": "faq",
        "question": "¿Hasta cuándo están abiertas las inscripciones de la carrera 2?",
        "answer": "Hola, las inscripciones están abiertas hasta la primera semana de septiembre o cuando llenemos los cupos, lo que suceda primero"
    },
    {
        "type": "faq",
        "question": "¿Algun codigo promocional/promociones/descuentos para grupos?",
        "answer": "Hola, envías la foto de tu comprobante de pago y te generan el código para que completes tu formulario de inscripción, si perteneces a un grupo de 10 o más personas, tienen el 10% de descuento"
    },
    {
        "type": "faq",
        "question": "Alguien me puede recomendar donde me puedo hospedar?",
        "answer": "El hospedaje oficial del evento es la Hostería Andaluza"
    },
    {
        "type": "faq",
        "question": "Cómo me puedo inscribir?",
        "answer": "Hola, primero haces el depósito o transferencia a los números de cuenta que aparecen en la publicación, hecho esto, envías la foto del comprobante de pago al WhatsApp 0990953113, para enviarte un link con un código para que completes tu formulario de inscripción"
    },
    {
        "type": "faq",
        "question": "Hola en qué fecha será la ruta del hielero?",
        "answer": "Hola, como consta en la publicación será el 4 de mayo"
    },
    {
        "type": "faq",
        "question": "Información donde registrarse una vez realizado el pago",
        "answer": "Hola, una vez hecho el depósito, envías la foto del comprobante de pago al WhatsApp 0990953113, para enviarte un link con un código para que completes tu formulario de inscripción"
    },
    {
        "type": "faq",
        "question": "Amigos de @naftaec disculpen es hasta hoy las promo de reyes, hasta que hora se puede hacer la transferencia?",
        "answer": "Hola, hoy cerramos la promo, hasta las 6 recibimos pagos y hasta la medianoche se deben completar los registros"
    },
    {
        "type": "faq",
        "question": "Hola quiero saber la inscripción para la otra carrera cuando iniciará",
        "answer": "hola, las inscripciones para el Altar Reto Trail ya están abiertas"
    },
    {
        "type": "faq",
        "question": "Donde se entregarán los kits?",
        "answer": "Hola. En el concesionario Nissan Renault ubicado a una cuadra del Supermaxi, sector norte de la ciudad, de 13h00 a 18h00"
    }
]

documentos = faqs
descripciones = [d["question"] if d["type"] == "faq" else d["description"] for d in documentos]

# Carga modelo de embeddings
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedder.encode(descripciones)

def responder(pregunta, k=2):
    pregunta_emb = embedder.encode([pregunta])
    similitudes = cosine_similarity(pregunta_emb, embeddings)[0]
    top_k_idx = similitudes.argsort()[-k:][::-1]

    contexto = []
    for idx in top_k_idx:
        item = documentos[idx]
        if item["type"] == "faq":
            contexto.append(f"Pregunta frecuente: {item['question']}\nRespuesta: {item['answer']}\n")

    prompt = "\n".join(contexto) + f"\n\nUsuario pregunta: {pregunta}"
    respuesta = model.generate_content(prompt).text.strip()
    return respuesta

def download_media(media_url):
    try:
        # print(f"Downloading media from: {media_url}") # Removed as per request
        response = requests.get(media_url, timeout=15) # Added timeout
        response.raise_for_status()  # Raises an HTTPError for bad responses (4XX or 5XX)
        # print(f"Successfully downloaded {len(response.content)} bytes.") # Removed as per request
        return response.content
    except requests.exceptions.RequestException as e:
        print(f"Error downloading media from {media_url}: {e}")
        return None

def decrypt_and_save_media(media_key_b64, encrypted_data, output_path, media_type):
    """Decrypts WhatsApp media and saves it to output_path."""
    try:
        # print(f"Starting decryption for {media_type} to save at {output_path}") # Removed as per request
        media_key = base64.b64decode(media_key_b64)

        # Standard app_info strings for HKDF
        # Reference: https://github.com/WhisperSystems/Signal-Android/wiki/WhatsApp-Media-Keys-and-Decryption
        # Although that link is old, the general principle for app_info holds for many WA media types
        # For a more generic approach if direct app_info mapping is unknown:
        # some libraries or implementations might just use a generic salt or empty salt if app_info is not critical
        # or if the key already accounts for it. The PHP didn't specify it.
        # Let's use common ones based on typical findings. If this fails, this is a key area to debug.

        app_info_map = {
            'image': b'WhatsApp Image Keys',
            'video': b'WhatsApp Video Keys',
            'audio': b'WhatsApp Audio Keys',
            'document': b'WhatsApp Document Keys',
            # stickers might use 'WhatsApp Image Keys' or a specific one.
            # 'ptt' (push-to-talk audio) might also use 'WhatsApp Audio Keys'
        }

        app_info = app_info_map.get(media_type.lower())
        if not app_info:
            print(f"Warning: No specific app_info for media type '{media_type}'. Using a generic/empty salt might be an alternative.")
            # Fallback or error, for now, let's proceed with a default or raise error
            # Forcing a generic one if not found for robustness, though decryption might fail if specific one is needed.
            app_info = b'WhatsApp Media Keys' # A generic fallback

        # HKDF derivation
        # Salt is usually not used or is an empty byte string for this specific WhatsApp scheme
        # The media_key itself is the IKM (Input Keying Material)
        # Output length: 16 (IV) + 32 (Cipher Key) + 32 (MAC Key) = 80 bytes
        # However, some sources say IV is first 16 of encrypted_data, and HKDF is for cipher+mac key (total 64 derived)
        # Let's assume media_key is 32 bytes, and we derive a longer key from it for IV, Cipher, MAC.
        # If media_key is 64 bytes (key + mac_key_material), then HKDF is used differently.
        # The PHP example implies mediaKey is THE key. Let's assume it's the master secret.        # The common pattern is that the mediaKey (32 bytes) is expanded.
        # If media_key is 32 bytes (IKM):
        #   derived_key = HKDF(master=media_key, key_len=80, salt=None, hashmod=SHA256, context=app_info)
        #   iv = derived_key[0:16]
        #   cipher_key = derived_key[16:48]
        #   mac_key_from_hkdf = derived_key[48:80]
        # And then encrypted_data is ciphertext || original_mac_value (last 32 bytes)
        # The MAC check is then HMAC(mac_key_from_hkdf, iv + ciphertext) == original_mac_value

        # Let's re-evaluate: The most common implementation detail found is:
        # 1. mediaKey (32 bytes) is the master secret.
        # 2. HKDF expands this using app_info to get 80 bytes.
        #    - First 16 bytes = IV
        #    - Next 32 bytes = Cipher Key
        #    - Next 32 bytes = MAC Key
        # 3. The downloaded encrypted_data is: CIPHERTEXT || MAC_FROM_SENDER (last 32 bytes)
        # 4. Decryption uses derived IV and Cipher Key on CIPHERTEXT.
        # 5. MAC check: HMAC(derived MAC_KEY, IV + CIPHERTEXT_BEFORE_DECRYPTION) vs MAC_FROM_SENDER.

        if len(media_key) != 32:
            print(f"Error: Media key length is not 32 bytes. Actual length: {len(media_key)}")
            return False

        # Derive keys using HKDF V3 (which is common for WhatsApp)
        # Salt is typically null or an empty byte string for this WhatsApp scheme
        # expanded_key_material = HKDF(media_key, 80, app_info, SHA256) # pycryptodome syntax
        # Pycryptodome HKDF: master_secret, key_len, salt, hashmod, num_keys, context, label
        # Derive 48 bytes: 16 for IV, 32 for Cipher Key, based on PHP example logic
        expanded_keys = HKDF(master=media_key, key_len=48, salt=b'', hashmod=SHA256, context=app_info, num_keys=1)

        iv = expanded_keys[0:16]
        cipher_key = expanded_keys[16:48]

        # Ciphertext is all but the last 10 bytes (potential MAC/padding not used in this scheme)
        if len(encrypted_data) <= 10:
            print(f"Error: Encrypted data is too short ({len(encrypted_data)} bytes). Needs to be more than 10 bytes.")
            return False

        actual_ciphertext = encrypted_data[:-10]
        # MAC verification is skipped in this scheme.

        # print("Proceeding with decryption (MAC verification skipped as per new strategy).") # Removed as per request

        # Decrypt
        cipher = AES.new(cipher_key, AES.MODE_CBC, iv)
        decrypted_padded_data = cipher.decrypt(actual_ciphertext)

        # Unpad
        try:
            decrypted_data = unpad(decrypted_padded_data, AES.block_size, style='pkcs7')
        except ValueError as e:
            # This can happen if decryption failed (wrong key/iv) or data is not correctly padded
            print(f"Error unpadding data: {e}. This might indicate a decryption error.")
            # Log parts of the data for debugging if necessary, carefully due to sensitivity
            # print(f"Padded data (first 64 bytes): {decrypted_padded_data[:64].hex()}")
            return False

        # print(f"Successfully decrypted {len(decrypted_data)} bytes.") # Removed as per request

        # Ensure output directory exists
        output_dir = pathlib.Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save decrypted media
        with open(output_path, 'wb') as f:
            f.write(decrypted_data)
        # print(f"Decrypted media saved to {output_path}") # Removed as per request
        return True

    except Exception as e:
        print(f"Error during decryption or saving for {output_path}: {e}")
        import traceback
        print(traceback.format_exc())
        return False

@app.route("/webhook", methods=["POST"])
def webhook():
    try:
        data = request.get_json()

        if data.get("event") != "messages.upsert":
            return jsonify({"status": "ignored", "reason": "wrong_event_type"})

        message_container = data.get("data", {}).get("messages", {})
        if not message_container:
            message_container = data.get("data", {}).get("message", {})

        if not message_container:
            print(f"No 'messages' or 'message' field found in data.get('data', {{}}). Received data: {json.dumps(data, indent=2)}")
            return jsonify({"status": "no_message_data_container"})

        if message_container.get("key", {}).get("fromMe"):
            return jsonify({"status": "self_message_skipped"})

        sender = message_container.get("key", {}).get("remoteJid")
        message_id = message_container.get("key", {}).get("id", "unknown")

        msg_content = message_container.get("message", {})
        if not msg_content and message_container:
            is_potential_content_node = any(k in message_container for k in ['imageMessage', 'videoMessage', 'audioMessage', 'documentMessage', 'conversation', 'extendedTextMessage'])
            if is_potential_content_node:
                msg_content = message_container
            else:
                print(f"Empty or unrecognized message content for message_id: {message_id}. Message container: {json.dumps(message_container, indent=2)}")
                return jsonify({"status": "empty_or_unrecognized_message_content", "id": message_id})
        elif not msg_content:
            print(f"Critical: msg_content is None or empty for message_id: {message_id}")
            return jsonify({"status": "critical_empty_message_content", "id": message_id})

        media_info = None
        media_type = None
        extension = None

        if 'imageMessage' in msg_content:
            media_info = msg_content['imageMessage']
            media_type = 'image'
            extension = '.jpg'
        elif 'videoMessage' in msg_content:
            media_info = msg_content['videoMessage']
            media_type = 'video'
            extension = '.mp4'
        elif 'audioMessage' in msg_content:
            media_info = msg_content['audioMessage']
            media_type = 'audio'
            extension = '.ogg'
        elif 'documentMessage' in msg_content:
            media_info = msg_content['documentMessage']
            media_type = 'document'
            file_name = media_info.get("fileName", "")
            _, ext = os.path.splitext(file_name)
            extension = ext if ext else ''

        if media_info:
            media_key = media_info.get('mediaKey')
            media_url = media_info.get('url')
            direct_path = media_info.get('directPath')

            if media_key and media_url:
                output_filename = f'{message_id}{extension}'
                output_dir = 'media' # Changed as per request
                output_path = os.path.join(output_dir, output_filename)

                # print(f"Attempting to download media: type={media_type}, url={media_url}, id={message_id}, output_path={output_path}") # Removed as per request
                encrypted_data = download_media(media_url)

                if encrypted_data:
                    # print(f"Download successful for {message_id}. Attempting decryption and save.") # Removed as per request
                    decryption_successful = decrypt_and_save_media(media_key, encrypted_data, output_path, media_type)
                    if decryption_successful:
                        print(f"Media decrypted and saved successfully: {output_path}")
                        if media_type == 'image':
                            # Handle user sending a new image while another is pending confirmation
                            if sender in pending_confirmations:
                                old_image_message_id = pending_confirmations[sender].get('message_id', 'anterior')
                                warning_message = (
                                    "Veo que enviaste una nueva imagen. Vamos a procesar esta última. "
                                    "Si querías confirmar una imagen anterior (ID aproximado: " + old_image_message_id + "), "
                                    "por favor respóndenos sobre esa primero o reenvíala después de confirmar esta."
                                )
                                send_whatsapp_message(sender, warning_message)
                                print(f"Sender {sender} sent a new image, superseding pending confirmation for message ID {old_image_message_id}.")

                            # Store for confirmation (or overwrite if already exists)
                            pending_confirmations[sender] = {
                                "message_id": message_id, # This is the new image's message_id
                                "output_path": output_path,
                                "original_filename": output_filename # output_filename is f'{message_id}{extension}'
                            }
                            confirmation_question = "¿Esta imagen corresponde a un comprobante de transferencia, depósito o cualquier otro recibo de un pago realizado?"
                            send_whatsapp_message(sender, confirmation_question)
                            print(f"Image awaiting confirmation from {sender} for message_id {message_id}")
                            return jsonify({'status': 'image_awaiting_confirmation', 'id': message_id})
                        else:
                            # For non-image media, no OCR and no confirmation needed for now
                            return jsonify({'status': 'media_processed_successfully_not_image', 'id': message_id, 'path': output_path})
                    else:
                        print(f"Media decryption failed for {message_id}, type: {media_type}")
                        return jsonify({'status': 'media_decryption_failed', 'id': message_id, 'type': media_type})
                else:
                    print(f"Media download failed for {message_id}, url: {media_url}")
                    return jsonify({'status': 'media_download_failed', 'id': message_id, 'url': media_url})
            else:
                print(f"Media message ({media_type}) for id {message_id}, but key/URL missing. URL: {media_url}, Key: {media_key}")
                return jsonify({"status": "media_info_incomplete", "type": media_type, "id": message_id})

        incoming_text = None
        if "conversation" in msg_content:
            incoming_text = msg_content["conversation"]
        elif "extendedTextMessage" in msg_content:
            incoming_text = msg_content["extendedTextMessage"].get("text")

        if not incoming_text:
            print(f"Unsupported message type for id {message_id}. msg_content: {json.dumps(msg_content, indent=2)}")
            return jsonify({"status": "unsupported_message_type", "id": message_id})

        # Check if this sender has a pending confirmation
        if sender in pending_confirmations:
            # This is potentially a response to the confirmation question
            image_info = pending_confirmations[sender]

            # Refined Gemini Prompt
            prompt_text = (
                f"Analiza el siguiente mensaje del usuario: '{incoming_text}'. "
                f"El usuario está respondiendo a la pregunta: '¿Esta imagen corresponde a un comprobante de transferencia, depósito o cualquier otro recibo de un pago realizado?'. "
                f"Tu tarea es clasificar la respuesta del usuario. Responde ÚNICAMENTE con UNA de las siguientes tres palabras: AFFIRMATIVE, NEGATIVE, o UNCLEAR. "
                f"No añadas ninguna explicación ni puntuación adicional."
            )
            
            raw_classification = "error_gemini" # Default in case of exception
            try:
                gemini_classification_response = model.generate_content(prompt_text)
                raw_classification = gemini_classification_response.text.strip()
            except Exception as gen_err:
                print(f"Error generating content with Gemini for classification: {gen_err}")
                # raw_classification remains "error_gemini"
            
            processed_classification = raw_classification.lower()

            print(f"Sender: {sender}, Incoming Text: \"{incoming_text}\", Raw Gemini Classification: \"{raw_classification}\", Processed Classification: \"{processed_classification}\"")
            
            image_details = pending_confirmations[sender] # Contains message_id, output_path, original_filename

            if processed_classification == 'affirmative':
                print(f"User {sender} confirmed (AFFIRMATIVE). Processing OCR for image: {image_details['output_path']}")
                ocr_success = process_receipt_image(image_details['output_path'], image_details['original_filename'])
                
                if ocr_success:
                    txt_filename = os.path.splitext(image_details['original_filename'])[0] + ".txt"
                    # image_details['output_path'] is like 'media/xyz.jpg'
                    # original_filename is like 'xyz.jpg'
                    # txt_filepath should be 'media/xyz.txt'
                    txt_filepath = os.path.join(os.path.dirname(image_details['output_path']), txt_filename)
                    
                    try:
                        with open(txt_filepath, 'r', encoding='utf-8') as f:
                            receipt_content = f.read()
                        send_whatsapp_message(sender, receipt_content)
                        print(f"Sent OCR results to {sender} for image {image_details['original_filename']}")
                        del pending_confirmations[sender]
                        return jsonify({'status': 'ocr_processed_and_sent', 'id': image_details['message_id']})
                    except FileNotFoundError:
                        print(f"Error: OCR output file {txt_filepath} not found, though ocr_success was true.")
                        send_whatsapp_message(sender, "Lo siento, hubo un error al leer el resultado del recibo.")
                        del pending_confirmations[sender]
                        return jsonify({'status': 'ocr_processed_file_read_error', 'id': image_details['message_id']})
                    except Exception as e_read:
                        print(f"Error reading OCR output file {txt_filepath}: {e_read}")
                        send_whatsapp_message(sender, "Lo siento, hubo un error al leer el resultado del recibo.")
                        del pending_confirmations[sender]
                        return jsonify({'status': 'ocr_processed_file_read_error', 'id': image_details['message_id']})
                else:
                    print(f"OCR processing failed for {image_details['original_filename']} for sender {sender}")
                    send_whatsapp_message(sender, "Lo siento, no pude procesar la imagen del recibo.")
                    del pending_confirmations[sender]
                    return jsonify({'status': 'ocr_processing_failed', 'id': image_details['message_id']})
            
            elif processed_classification == 'negative':
                print(f"User {sender} denied confirmation (NEGATIVE) for image {image_details['original_filename']}")
                send_whatsapp_message(sender, "Entendido. Si la imagen no era un recibo, no la procesaré. Puede enviar otra imagen si lo desea o hacerme una pregunta.")
                del pending_confirmations[sender]
                return jsonify({'status': 'ocr_cancelled_by_user', 'id': image_details['message_id']})
            
            else: # Covers 'unclear', 'error_gemini', or any other unexpected/malformed response from Gemini
                print(f"Confirmation unclear from {sender} for image {image_details['original_filename']}. Processed Classification: {processed_classification}")
                send_whatsapp_message(sender, "No entendí tu respuesta. Por favor, responde 'sí' o 'no' para el recibo anterior, o envía un nuevo mensaje si quieres que ignore la imagen.")
                # Entry NOT removed from pending_confirmations, user can try again.
                return jsonify({'status': 'confirmation_unclear', 'id': image_details['message_id']})
        
        else:
            # No pending confirmation, process as a general question
            gemini_response = responder(incoming_text)
            send_whatsapp_message(sender, gemini_response)
            return jsonify({"status": "text_message_processed", "id": message_id})

    except Exception as e:
        print(f"Error in webhook: {str(e)}\nTraceback: {traceback.format_exc()}")
        return jsonify({"status": "error", "message": str(e)}), 500

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
        print(f"WaSender response: {response.status_code} {response.text}")
        return response.ok
    except requests.exceptions.RequestException as e:
        print(f"Error sending message via WaSender: {e}")
        return False

import os # Ensure os is imported if not already at the top for this function specifically
import re # Ensure re is imported
from PIL import Image # Ensure Image is imported

import os       # Should be already imported
import re       # Should be already imported
from PIL import Image # Should be already imported
import pytesseract # Should be already imported
import cv2      # Should be already imported
import numpy as np  # Should be already imported

def process_receipt_image(image_path: str, original_filename: str) -> bool:
    print(f"Starting Tesseract OCR with preprocessing for image: {image_path} (original: {original_filename})")
    
    # Load OCR parameters
    # Assuming ocr_params.txt is in the 'Local' directory, relative to where app.py is (which is also Local/)
    # If app.py is moved, this path might need adjustment or be made absolute/configurable.
    params_file_path = os.path.join(os.path.dirname(__file__), "ocr_params.txt")
    params = load_params(params_file_path)

    try:
        # Load image with PIL
        pil_image = Image.open(image_path)

        # Convert PIL image to OpenCV format
        cv_image = np.array(pil_image.convert('RGB')) 
        cv_image = cv_image[:, :, ::-1].copy() # RGB to BGR

        # Preprocess the image using the loaded parameters
        preprocessed_cv_image = preprocess_image(cv_image, params)
        
        # Perform OCR using pytesseract on the preprocessed image
        custom_tesseract_config = f'--psm {params["tesseract_psm"]}'
        raw_text = pytesseract.image_to_string(preprocessed_cv_image, lang=params['tesseract_lang'], config=custom_tesseract_config)
        
        print(f"--- TESSERACT PREPROCESSED RAW OUTPUT for {original_filename} (PSM {params['tesseract_psm']}, lang='{params['tesseract_lang']}') ---")
        print(raw_text)
        print("--- END TESSERACT PREPROCESSED RAW OUTPUT ---")

        # Initialize fields
        banco = "Not found"
        total = "Not found"
        documento = "Not found"
        fecha = "Not found"

        lines = raw_text.split('\n')

        # --- BANCO ---
        banco_found_on_line = False
        for line in lines:
            if re.search(r"(?i)BANCO", line):
                # Try to get text after "BANCO" keyword
                match = re.search(r"(?i)BANCO\s*[:\-]?\s*(.*)", line)
                if match:
                    potential_banco = match.group(1).strip()
                    # Clean common suffixes and artifacts
                    potential_banco = re.sub(r"(?i)\s*(?:C\.A|S\.A|E\.A|CIA\. LTDA|LTDA)[\-\.]*$", "", potential_banco).strip()
                    potential_banco = re.sub(r"[€]", "", potential_banco).strip() # Remove stray symbols like Euro
                    if len(potential_banco) > 2 : # Check if it's a meaningful name
                        banco = potential_banco
                        banco_found_on_line = True
                        print(f"[TESSERACT_EXTRACT] BANCO (from line): {banco}")
                        break
        
        if not banco_found_on_line:
            # Fallback: search for known bank names in the whole text if not found on a "BANCO" line
            known_banks_pattern = r"(?i)(Pichincha|Produbanco|Guayaquil|Pacífico|Bolivariano|Internacional)"
            bank_keyword_match = re.search(known_banks_pattern, raw_text)
            if bank_keyword_match:
                banco = bank_keyword_match.group(1).strip()
                print(f"[TESSERACT_EXTRACT] BANCO (known keyword fallback): {banco}")
        
        # --- DOCUMENTO ---
        for line in lines:
            if re.search(r"(?i)Documento", line):
                numbers_in_line = re.findall(r'\d+', line)
                if numbers_in_line:
                    documento = str(max(int(n) for n in numbers_in_line))
                    print(f"[TESSERACT_EXTRACT] DOCUMENTO (from line, largest number): {documento}")
                    break
        
        # --- TOTAL ---
        for line in lines:
            if re.search(r"(?i)Total", line):
                # Regex for amount like xxx.xx or xxx,xx (allows for . or , as decimal)
                match = re.search(r"([\d]+[\.,][\d]{2})", line) 
                if match:
                    total_str = match.group(1).replace(',', '.') # Standardize to dot as decimal
                    # Validate it's a number before assigning
                    if re.match(r"^\d+\.\d{2}$", total_str):
                         total = total_str
                         print(f"[TESSERACT_EXTRACT] TOTAL (from line): {total}")
                         break
                    else:
                        print(f"[TESSERACT_EXTRACT_WARN] TOTAL candidate on line '{line.strip()}' was '{match.group(1)}', not valid x.xx format after cleaning.")
                else: # If first regex fails on total line, try to find any number on that line as last resort
                    numbers_on_total_line = re.findall(r"[\d.,]+", line)
                    if numbers_on_total_line:
                        # Get the number that looks most like a monetary value (e.g. ends in .xx or ,xx)
                        for num_str in numbers_on_total_line:
                            if re.search(r"[\.,]\d{2}$", num_str):
                                total = num_str.replace(',', '.')
                                if re.match(r"^\d+\.\d{2}$", total):
                                    print(f"[TESSERACT_EXTRACT] TOTAL (any number on 'Total' line): {total}")
                                    break
                        if total != "Not found": break


        # --- FECHA ---
        for line in lines:
            # User wants YYYY/MMM/DD, MMM being 3 letters
            # Tesseract output was "2020/NOV/02"
            if re.search(r"(?i)Fecha", line):
                match = re.search(r"(\d{4}/[A-Z0-9]{3}/\d{2})", line, re.IGNORECASE) # Added IGNORECASE here too
                if match:
                    fecha = match.group(1).upper() # Standardize month to uppercase
                    print(f"[TESSERACT_EXTRACT] FECHA (from line YYYY/MMM/DD): {fecha}")
                    break
        
        # Build output text
        parsed_info = [
            f"Banco: {banco}",
            f"Total: {total}",
            f"Documento: {documento}",
            f"Fecha: {fecha}"
        ]

        # Save to .txt file
        base_name = os.path.splitext(original_filename)[0]
        output_txt_dir = os.path.dirname(image_path)
        output_txt_path = os.path.join(output_txt_dir, base_name + ".txt")

        print(f"Saving extracted Tesseract (preprocessed) info to: {output_txt_path}")
        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(parsed_info))

        print(f"Successfully processed Tesseract OCR (preprocessed) for {original_filename}")
        return True

    except pytesseract.TesseractNotFoundError:
        print("ERROR: Tesseract executable not found. Please ensure Tesseract is installed and in your PATH.")
        return False
    except Exception as e:
        print(f"Error during OCR processing for {original_filename}: {e}")
        import traceback
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)# Standard Python library imports
import base64
import json
import os
import pathlib
import re
import traceback
import configparser # Added
import sys # Added

# Third-party library imports
import cv2
from dotenv import load_dotenv
from flask import Flask, request, jsonify
import google.generativeai as genai
import numpy as np
from PIL import Image
import pytesseract
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from Crypto.Cipher import AES
from Crypto.Hash import HMAC, SHA256
from Crypto.Protocol.KDF import HKDF
from Crypto.Util.Padding import unpad

# Load environment variables
load_dotenv()

app = Flask(__name__)

#pending_confirmations = {} # Replaced by file-based state management

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
WASENDER_API_TOKEN = os.getenv("WASENDER_API_TOKEN")
WASENDER_API_URL = "https://wasenderapi.com/api/send-message"

# Setup Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

DEFAULT_PARAMS = {
    'clahe_clip_limit': '2.0',
    'clahe_tile_grid_size_x': '8',
    'clahe_tile_grid_size_y': '8',
    'tesseract_psm': '6',
    'tesseract_lang': 'spa',
    'save_intermediate_images': 'False' # Default to False for app.py
}

# File-based state management for pending confirmations
CONFIRMATIONS_FILE = 'pending_confirmations.json'

def _load_confirmations():
    """Loads pending confirmations from the JSON file."""
    try:
        with open(CONFIRMATIONS_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Info: Confirmations file '{CONFIRMATIONS_FILE}' not found or invalid JSON. Returning empty dict. Error: {e}")
        return {}

def _save_confirmations(confirmations_data):
    """Saves the given confirmations data to the JSON file."""
    try:
        with open(CONFIRMATIONS_FILE, 'w') as f:
            json.dump(confirmations_data, f, indent=4)
    except IOError as e:
        print(f"Error: Could not write to confirmations file '{CONFIRMATIONS_FILE}'. Error: {e}")


def save_pending_confirmation(sender_jid, data):
    """Saves a single pending confirmation."""
    confirmations = _load_confirmations()
    confirmations[sender_jid] = data
    _save_confirmations(confirmations)

def get_pending_confirmation(sender_jid):
    """Retrieves a pending confirmation for a given sender."""
    confirmations = _load_confirmations()
    return confirmations.get(sender_jid)

def clear_pending_confirmation(sender_jid):
    """Clears a pending confirmation for a given sender."""
    confirmations = _load_confirmations()
    if sender_jid in confirmations:
        del confirmations[sender_jid]
        _save_confirmations(confirmations)
        return True
    return False

# End of file-based state management functions


def load_params(config_file_path):
    config = configparser.ConfigParser()
    params = DEFAULT_PARAMS.copy()

    if not os.path.exists(config_file_path):
        print(f"Warning: Config file '{config_file_path}' not found. Using default parameters for OCR in app.py.")
        # Convert types for defaults
        params['clahe_clip_limit'] = float(params['clahe_clip_limit'])
        params['clahe_tile_grid_size_x'] = int(params['clahe_tile_grid_size_x'])
        params['clahe_tile_grid_size_y'] = int(params['clahe_tile_grid_size_y'])
        params['tesseract_psm'] = str(params['tesseract_psm'])
        params['save_intermediate_images'] = params['save_intermediate_images'].lower() == 'true'
        return params
    try:
        config.read(config_file_path)
        if 'Parameters' in config:
            config_params = config['Parameters']
            params['clahe_clip_limit'] = float(config_params.get('clahe_clip_limit', DEFAULT_PARAMS['clahe_clip_limit']))
            params['clahe_tile_grid_size_x'] = int(config_params.get('clahe_tile_grid_size_x', DEFAULT_PARAMS['clahe_tile_grid_size_x']))
            params['clahe_tile_grid_size_y'] = int(config_params.get('clahe_tile_grid_size_y', DEFAULT_PARAMS['clahe_tile_grid_size_y']))
            params['tesseract_psm'] = str(config_params.get('tesseract_psm', DEFAULT_PARAMS['tesseract_psm']))
            params['tesseract_lang'] = config_params.get('tesseract_lang', DEFAULT_PARAMS['tesseract_lang'])
            # save_intermediate_images is False by default in app.py, not read from file for app.py
            params['save_intermediate_images'] = False 
        else:
            print(f"Warning: '[Parameters]' section not found in '{config_file_path}'. Using default parameters for OCR in app.py.")
            params['clahe_clip_limit'] = float(params['clahe_clip_limit'])
            params['clahe_tile_grid_size_x'] = int(params['clahe_tile_grid_size_x'])
            params['clahe_tile_grid_size_y'] = int(params['clahe_tile_grid_size_y'])
            params['tesseract_psm'] = str(params['tesseract_psm'])
            params['save_intermediate_images'] = False
    except Exception as e:
        print(f"Error reading or parsing config file '{config_file_path}' in app.py: {e}. Using default parameters.")
        params['clahe_clip_limit'] = float(params['clahe_clip_limit'])
        params['clahe_tile_grid_size_x'] = int(params['clahe_tile_grid_size_x'])
        params['clahe_tile_grid_size_y'] = int(params['clahe_tile_grid_size_y'])
        params['tesseract_psm'] = str(params['tesseract_psm'])
        params['save_intermediate_images'] = False
    return params

def preprocess_image(cv_image, params, output_dir=None, base_filename=None): # output_dir/base_filename not used if not saving
    print("[PREPROC_APP] Starting preprocessing in app.py...")
    # 1. Grayscale
    gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    print("[PREPROC_APP] Converted to grayscale.")
    # No saving of intermediate images in app.py context

    # 2. CLAHE
    clahe_clip = params.get('clahe_clip_limit', 2.0) # Default if not in params
    clahe_tile_x = params.get('clahe_tile_grid_size_x', 8)
    clahe_tile_y = params.get('clahe_tile_grid_size_y', 8)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_tile_x, clahe_tile_y))
    processed_image = clahe.apply(gray_image)
    print(f"[PREPROC_APP] Applied CLAHE (Clip: {clahe_clip}, Tile: ({clahe_tile_x},{clahe_tile_y})).")
    
    # 3. Binarization (Otsu)
    _, processed_image = cv2.threshold(processed_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print("[PREPROC_APP] Applied Otsu's binarization.")
    
    return processed_image

faqs = [
    {
        "type": "faq",
        "question": "¿Hasta cuándo están abiertas las inscripciones de la carrera 1?",
        "answer": "Hola, las inscripciones están abiertas hasta la primera semana de abril o cuando llenemos los cupos, lo que suceda primero."
    },
    {
        "type": "faq",
        "question": "¿Hasta cuándo están abiertas las inscripciones de la carrera 2?",
        "answer": "Hola, las inscripciones están abiertas hasta la primera semana de septiembre o cuando llenemos los cupos, lo que suceda primero"
    },
    {
        "type": "faq",
        "question": "¿Algun codigo promocional/promociones/descuentos para grupos?",
        "answer": "Hola, envías la foto de tu comprobante de pago y te generan el código para que completes tu formulario de inscripción, si perteneces a un grupo de 10 o más personas, tienen el 10% de descuento"
    },
    {
        "type": "faq",
        "question": "Alguien me puede recomendar donde me puedo hospedar?",
        "answer": "El hospedaje oficial del evento es la Hostería Andaluza"
    },
    {
        "type": "faq",
        "question": "Cómo me puedo inscribir?",
        "answer": "Hola, primero haces el depósito o transferencia a los números de cuenta que aparecen en la publicación, hecho esto, envías la foto del comprobante de pago al WhatsApp 0990953113, para enviarte un link con un código para que completes tu formulario de inscripción"
    },
    {
        "type": "faq",
        "question": "Hola en qué fecha será la ruta del hielero?",
        "answer": "Hola, como consta en la publicación será el 4 de mayo"
    },
    {
        "type": "faq",
        "question": "Información donde registrarse una vez realizado el pago",
        "answer": "Hola, una vez hecho el depósito, envías la foto del comprobante de pago al WhatsApp 0990953113, para enviarte un link con un código para que completes tu formulario de inscripción"
    },
    {
        "type": "faq",
        "question": "Amigos de @naftaec disculpen es hasta hoy las promo de reyes, hasta que hora se puede hacer la transferencia?",
        "answer": "Hola, hoy cerramos la promo, hasta las 6 recibimos pagos y hasta la medianoche se deben completar los registros"
    },
    {
        "type": "faq",
        "question": "Hola quiero saber la inscripción para la otra carrera cuando iniciará",
        "answer": "hola, las inscripciones para el Altar Reto Trail ya están abiertas"
    },
    {
        "type": "faq",
        "question": "Donde se entregarán los kits?",
        "answer": "Hola. En el concesionario Nissan Renault ubicado a una cuadra del Supermaxi, sector norte de la ciudad, de 13h00 a 18h00"
    }
]

documentos = faqs
descripciones = [d["question"] if d["type"] == "faq" else d["description"] for d in documentos]

# Carga modelo de embeddings
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedder.encode(descripciones)

def responder(pregunta, k=2):
    pregunta_emb = embedder.encode([pregunta])
    similitudes = cosine_similarity(pregunta_emb, embeddings)[0]
    top_k_idx = similitudes.argsort()[-k:][::-1]

    contexto = []
    for idx in top_k_idx:
        item = documentos[idx]
        if item["type"] == "faq":
            contexto.append(f"Pregunta frecuente: {item['question']}\nRespuesta: {item['answer']}\n")

    prompt = "Eres un asistente que le permite a la empresa de carreras NaftaEc comunicarse de manera efectiva con sus clientes. \
    Partiendo del siguiente contexto, responde a las preguntas del cliente. \n".join(contexto) + f"\n\nUsuario pregunta: {pregunta}"
    respuesta = model.generate_content(prompt).text.strip()
    return respuesta

def download_media(media_url):
    try:
        # print(f"Downloading media from: {media_url}") # Removed as per request
        response = requests.get(media_url, timeout=15) # Added timeout
        response.raise_for_status()  # Raises an HTTPError for bad responses (4XX or 5XX)
        # print(f"Successfully downloaded {len(response.content)} bytes.") # Removed as per request
        return response.content
    except requests.exceptions.RequestException as e:
        print(f"Error downloading media from {media_url}: {e}")
        return None

def decrypt_and_save_media(media_key_b64, encrypted_data, output_path, media_type):
    """Decrypts WhatsApp media and saves it to output_path."""
    try:
        # print(f"Starting decryption for {media_type} to save at {output_path}") # Removed as per request
        media_key = base64.b64decode(media_key_b64)

        # Standard app_info strings for HKDF
        # Reference: https://github.com/WhisperSystems/Signal-Android/wiki/WhatsApp-Media-Keys-and-Decryption
        # Although that link is old, the general principle for app_info holds for many WA media types
        # For a more generic approach if direct app_info mapping is unknown:
        # some libraries or implementations might just use a generic salt or empty salt if app_info is not critical
        # or if the key already accounts for it. The PHP didn't specify it.
        # Let's use common ones based on typical findings. If this fails, this is a key area to debug.

        app_info_map = {
            'image': b'WhatsApp Image Keys',
            'video': b'WhatsApp Video Keys',
            'audio': b'WhatsApp Audio Keys',
            'document': b'WhatsApp Document Keys',
            # stickers might use 'WhatsApp Image Keys' or a specific one.
            # 'ptt' (push-to-talk audio) might also use 'WhatsApp Audio Keys'
        }

        app_info = app_info_map.get(media_type.lower())
        if not app_info:
            print(f"Warning: No specific app_info for media type '{media_type}'. Using a generic/empty salt might be an alternative.")
            # Fallback or error, for now, let's proceed with a default or raise error
            # Forcing a generic one if not found for robustness, though decryption might fail if specific one is needed.
            app_info = b'WhatsApp Media Keys' # A generic fallback

        # HKDF derivation
        # Salt is usually not used or is an empty byte string for this specific WhatsApp scheme
        # The media_key itself is the IKM (Input Keying Material)
        # Output length: 16 (IV) + 32 (Cipher Key) + 32 (MAC Key) = 80 bytes
        # However, some sources say IV is first 16 of encrypted_data, and HKDF is for cipher+mac key (total 64 derived)
        # Let's assume media_key is 32 bytes, and we derive a longer key from it for IV, Cipher, MAC.
        # If media_key is 64 bytes (key + mac_key_material), then HKDF is used differently.
        # The PHP example implies mediaKey is THE key. Let's assume it's the master secret.        # The common pattern is that the mediaKey (32 bytes) is expanded.
        # If media_key is 32 bytes (IKM):
        #   derived_key = HKDF(master=media_key, key_len=80, salt=None, hashmod=SHA256, context=app_info)
        #   iv = derived_key[0:16]
        #   cipher_key = derived_key[16:48]
        #   mac_key_from_hkdf = derived_key[48:80]
        # And then encrypted_data is ciphertext || original_mac_value (last 32 bytes)
        # The MAC check is then HMAC(mac_key_from_hkdf, iv + ciphertext) == original_mac_value

        # Let's re-evaluate: The most common implementation detail found is:
        # 1. mediaKey (32 bytes) is the master secret.
        # 2. HKDF expands this using app_info to get 80 bytes.
        #    - First 16 bytes = IV
        #    - Next 32 bytes = Cipher Key
        #    - Next 32 bytes = MAC Key
        # 3. The downloaded encrypted_data is: CIPHERTEXT || MAC_FROM_SENDER (last 32 bytes)
        # 4. Decryption uses derived IV and Cipher Key on CIPHERTEXT.
        # 5. MAC check: HMAC(derived MAC_KEY, IV + CIPHERTEXT_BEFORE_DECRYPTION) vs MAC_FROM_SENDER.

        if len(media_key) != 32:
            print(f"Error: Media key length is not 32 bytes. Actual length: {len(media_key)}")
            return False

        # Derive keys using HKDF V3 (which is common for WhatsApp)
        # Salt is typically null or an empty byte string for this WhatsApp scheme
        # expanded_key_material = HKDF(media_key, 80, app_info, SHA256) # pycryptodome syntax
        # Pycryptodome HKDF: master_secret, key_len, salt, hashmod, num_keys, context, label
        # Derive 48 bytes: 16 for IV, 32 for Cipher Key, based on PHP example logic
        expanded_keys = HKDF(master=media_key, key_len=48, salt=b'', hashmod=SHA256, context=app_info, num_keys=1)

        iv = expanded_keys[0:16]
        cipher_key = expanded_keys[16:48]

        # Ciphertext is all but the last 10 bytes (potential MAC/padding not used in this scheme)
        if len(encrypted_data) <= 10:
            print(f"Error: Encrypted data is too short ({len(encrypted_data)} bytes). Needs to be more than 10 bytes.")
            return False

        actual_ciphertext = encrypted_data[:-10]
        # MAC verification is skipped in this scheme.

        # print("Proceeding with decryption (MAC verification skipped as per new strategy).") # Removed as per request

        # Decrypt
        cipher = AES.new(cipher_key, AES.MODE_CBC, iv)
        decrypted_padded_data = cipher.decrypt(actual_ciphertext)

        # Unpad
        try:
            decrypted_data = unpad(decrypted_padded_data, AES.block_size, style='pkcs7')
        except ValueError as e:
            # This can happen if decryption failed (wrong key/iv) or data is not correctly padded
            print(f"Error unpadding data: {e}. This might indicate a decryption error.")
            # Log parts of the data for debugging if necessary, carefully due to sensitivity
            # print(f"Padded data (first 64 bytes): {decrypted_padded_data[:64].hex()}")
            return False

        # print(f"Successfully decrypted {len(decrypted_data)} bytes.") # Removed as per request

        # Ensure output directory exists
        output_dir = pathlib.Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save decrypted media
        with open(output_path, 'wb') as f:
            f.write(decrypted_data)
        # print(f"Decrypted media saved to {output_path}") # Removed as per request
        return True

    except Exception as e:
        print(f"Error during decryption or saving for {output_path}: {e}")
        import traceback
        print(traceback.format_exc())
        return False

@app.route("/webhook", methods=["POST"])
def webhook():
    try:
        data = request.get_json()

        if data.get("event") != "messages.upsert":
            return jsonify({"status": "ignored", "reason": "wrong_event_type"})

        message_container = data.get("data", {}).get("messages", {})
        if not message_container:
            message_container = data.get("data", {}).get("message", {})

        if not message_container:
            print(f"No 'messages' or 'message' field found in data.get('data', {{}}). Received data: {json.dumps(data, indent=2)}")
            return jsonify({"status": "no_message_data_container"})

        if message_container.get("key", {}).get("fromMe"):
            return jsonify({"status": "self_message_skipped"})

        sender = message_container.get("key", {}).get("remoteJid")
        message_id = message_container.get("key", {}).get("id", "unknown")

        msg_content = message_container.get("message", {})
        if not msg_content and message_container:
            is_potential_content_node = any(k in message_container for k in ['imageMessage', 'videoMessage', 'audioMessage', 'documentMessage', 'conversation', 'extendedTextMessage'])
            if is_potential_content_node:
                msg_content = message_container
            else:
                print(f"Empty or unrecognized message content for message_id: {message_id}. Message container: {json.dumps(message_container, indent=2)}")
                return jsonify({"status": "empty_or_unrecognized_message_content", "id": message_id})
        elif not msg_content:
            print(f"Critical: msg_content is None or empty for message_id: {message_id}")
            return jsonify({"status": "critical_empty_message_content", "id": message_id})

        media_info = None
        media_type = None
        extension = None

        if 'imageMessage' in msg_content:
            media_info = msg_content['imageMessage']
            media_type = 'image'
            extension = '.jpg'
        elif 'videoMessage' in msg_content:
            media_info = msg_content['videoMessage']
            media_type = 'video'
            extension = '.mp4'
        elif 'audioMessage' in msg_content:
            media_info = msg_content['audioMessage']
            media_type = 'audio'
            extension = '.ogg'
        elif 'documentMessage' in msg_content:
            media_info = msg_content['documentMessage']
            media_type = 'document'
            file_name = media_info.get("fileName", "")
            _, ext = os.path.splitext(file_name)
            extension = ext if ext else ''

        if media_info:
            media_key = media_info.get('mediaKey')
            media_url = media_info.get('url')
            direct_path = media_info.get('directPath')

            if media_key and media_url:
                output_filename = f'{message_id}{extension}'
                output_dir = 'media' # Changed as per request
                output_path = os.path.join(output_dir, output_filename)

                # print(f"Attempting to download media: type={media_type}, url={media_url}, id={message_id}, output_path={output_path}") # Removed as per request
                encrypted_data = download_media(media_url)

                if encrypted_data:
                    # print(f"Download successful for {message_id}. Attempting decryption and save.") # Removed as per request
                    decryption_successful = decrypt_and_save_media(media_key, encrypted_data, output_path, media_type)
                    if decryption_successful:
                        print(f"Media decrypted and saved successfully: {output_path}")
                        if media_type == 'image':
                            # Handle user sending a new image while another is pending confirmation
                            old_confirmation_data = get_pending_confirmation(sender)
                            if old_confirmation_data:
                                old_image_message_id = old_confirmation_data.get('message_id', 'anterior')
                                warning_message = (
                                    "Veo que enviaste una nueva imagen. Vamos a procesar esta última. "
                                    "Si querías confirmar una imagen anterior (ID aproximado: " + old_image_message_id + "), "
                                    "por favor respóndenos sobre esa primero o reenvíala después de confirmar esta."
                                )
                                send_whatsapp_message(sender, warning_message)
                                print(f"Sender {sender} sent a new image, superseding pending confirmation for message ID {old_image_message_id}.")

                            # Store for confirmation (or overwrite if already exists)
                            new_confirmation_data = {
                                "message_id": message_id, # This is the new image's message_id
                                "output_path": output_path,
                                "original_filename": output_filename # output_filename is f'{message_id}{extension}'
                            }
                            save_pending_confirmation(sender, new_confirmation_data)
                            confirmation_question = "¿Esta imagen corresponde a un comprobante de transferencia, depósito o cualquier otro recibo de un pago realizado?"
                            send_whatsapp_message(sender, confirmation_question)
                            print(f"Image awaiting confirmation from {sender} for message_id {message_id}")
                            return jsonify({'status': 'image_awaiting_confirmation', 'id': message_id})
                        else:
                            # For non-image media, no OCR and no confirmation needed for now
                            return jsonify({'status': 'media_processed_successfully_not_image', 'id': message_id, 'path': output_path})
                    else:
                        print(f"Media decryption failed for {message_id}, type: {media_type}")
                        return jsonify({'status': 'media_decryption_failed', 'id': message_id, 'type': media_type})
                else:
                    print(f"Media download failed for {message_id}, url: {media_url}")
                    return jsonify({'status': 'media_download_failed', 'id': message_id, 'url': media_url})
            else:
                print(f"Media message ({media_type}) for id {message_id}, but key/URL missing. URL: {media_url}, Key: {media_key}")
                return jsonify({"status": "media_info_incomplete", "type": media_type, "id": message_id})

        incoming_text = None
        if "conversation" in msg_content:
            incoming_text = msg_content["conversation"]
        elif "extendedTextMessage" in msg_content:
            incoming_text = msg_content["extendedTextMessage"].get("text")

        if not incoming_text:
            print(f"Unsupported message type for id {message_id}. msg_content: {json.dumps(msg_content, indent=2)}")
            return jsonify({"status": "unsupported_message_type", "id": message_id})

        # Check if this sender has a pending confirmation
        image_details = get_pending_confirmation(sender)
        if image_details:
            # This is potentially a response to the confirmation question
            # image_info = pending_confirmations[sender] # Old way, now image_details is used

            # Refined Gemini Prompt
            prompt_text = (
                f"Analiza el siguiente mensaje del usuario: '{incoming_text}'. "
                f"El usuario está respondiendo a la pregunta: '¿Esta imagen corresponde a un comprobante de transferencia, depósito o cualquier otro recibo de un pago realizado?'. "
                f"Tu tarea es clasificar la respuesta del usuario. Responde ÚNICAMENTE con UNA de las siguientes tres palabras: AFFIRMATIVE, NEGATIVE, o UNCLEAR. "
                f"No añadas ninguna explicación ni puntuación adicional."
            )
            
            raw_classification = "error_gemini" # Default in case of exception
            try:
                gemini_classification_response = model.generate_content(prompt_text)
                raw_classification = gemini_classification_response.text.strip()
            except Exception as gen_err:
                print(f"Error generating content with Gemini for classification: {gen_err}")
                # raw_classification remains "error_gemini"
            
            processed_classification = raw_classification.lower()

            print(f"Sender: {sender}, Incoming Text: \"{incoming_text}\", Raw Gemini Classification: \"{raw_classification}\", Processed Classification: \"{processed_classification}\"")
            
            # image_details = pending_confirmations[sender] # Old way, already fetched as image_details

            if processed_classification == 'affirmative':
                print(f"User {sender} confirmed (AFFIRMATIVE). Processing OCR for image: {image_details['output_path']}")
                ocr_success = process_receipt_image(image_details['output_path'], image_details['original_filename'])
                
                if ocr_success:
                    txt_filename = os.path.splitext(image_details['original_filename'])[0] + ".txt"
                    # image_details['output_path'] is like 'media/xyz.jpg'
                    # original_filename is like 'xyz.jpg'
                    # txt_filepath should be 'media/xyz.txt'
                    txt_filepath = os.path.join(os.path.dirname(image_details['output_path']), txt_filename)
                    
                    try:
                        with open(txt_filepath, 'r', encoding='utf-8') as f:
                            receipt_content = f.read()
                        send_whatsapp_message(sender, receipt_content)
                        print(f"Sent OCR results to {sender} for image {image_details['original_filename']}")
                        clear_pending_confirmation(sender)
                        return jsonify({'status': 'ocr_processed_and_sent', 'id': image_details['message_id']})
                    except FileNotFoundError:
                        print(f"Error: OCR output file {txt_filepath} not found, though ocr_success was true.")
                        send_whatsapp_message(sender, "Lo siento, hubo un error al leer el resultado del recibo.")
                        clear_pending_confirmation(sender)
                        return jsonify({'status': 'ocr_processed_file_read_error', 'id': image_details['message_id']})
                    except Exception as e_read:
                        print(f"Error reading OCR output file {txt_filepath}: {e_read}")
                        send_whatsapp_message(sender, "Lo siento, hubo un error al leer el resultado del recibo.")
                        clear_pending_confirmation(sender)
                        return jsonify({'status': 'ocr_processed_file_read_error', 'id': image_details['message_id']})
                else:
                    print(f"OCR processing failed for {image_details['original_filename']} for sender {sender}")
                    send_whatsapp_message(sender, "Lo siento, no pude procesar la imagen del recibo.")
                    clear_pending_confirmation(sender)
                    return jsonify({'status': 'ocr_processing_failed', 'id': image_details['message_id']})
            
            elif processed_classification == 'negative':
                print(f"User {sender} denied confirmation (NEGATIVE) for image {image_details['original_filename']}")
                send_whatsapp_message(sender, "Entendido. Si la imagen no era un recibo, no la procesaré. Puede enviar otra imagen si lo desea o hacerme una pregunta.")
                clear_pending_confirmation(sender)
                return jsonify({'status': 'ocr_cancelled_by_user', 'id': image_details['message_id']})
            
            else: # Covers 'unclear', 'error_gemini', or any other unexpected/malformed response from Gemini
                print(f"Confirmation unclear from {sender} for image {image_details['original_filename']}. Processed Classification: {processed_classification}")
                send_whatsapp_message(sender, "No entendí tu respuesta. Por favor, responde 'sí' o 'no' para el recibo anterior, o envía un nuevo mensaje si quieres que ignore la imagen.")
                # Entry NOT removed from pending_confirmations, user can try again.
                return jsonify({'status': 'confirmation_unclear', 'id': image_details['message_id']})
        
        else:
            # No pending confirmation, process as a general question
            gemini_response = responder(incoming_text)
            send_whatsapp_message(sender, gemini_response)
            return jsonify({"status": "text_message_processed", "id": message_id})

    except Exception as e:
        print(f"Error in webhook: {str(e)}\nTraceback: {traceback.format_exc()}")
        return jsonify({"status": "error", "message": str(e)}), 500

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
        print(f"WaSender response: {response.status_code} {response.text}")
        return response.ok
    except requests.exceptions.RequestException as e:
        print(f"Error sending message via WaSender: {e}")
        return False

import os # Ensure os is imported if not already at the top for this function specifically
import re # Ensure re is imported
from PIL import Image # Ensure Image is imported

import os       # Should be already imported
import re       # Should be already imported
from PIL import Image # Should be already imported
import pytesseract # Should be already imported
import cv2      # Should be already imported
import numpy as np  # Should be already imported

def process_receipt_image(image_path: str, original_filename: str) -> bool:
    print(f"Starting Tesseract OCR with preprocessing for image: {image_path} (original: {original_filename})")
    
    # Load OCR parameters
    # Assuming ocr_params.txt is in the 'Local' directory, relative to where app.py is (which is also Local/)
    # If app.py is moved, this path might need adjustment or be made absolute/configurable.
    params_file_path = os.path.join(os.path.dirname(__file__), "ocr_params.txt")
    params = load_params(params_file_path)

    try:
        # Load image with PIL
        pil_image = Image.open(image_path)

        # Convert PIL image to OpenCV format
        cv_image = np.array(pil_image.convert('RGB')) 
        cv_image = cv_image[:, :, ::-1].copy() # RGB to BGR

        # Preprocess the image using the loaded parameters
        preprocessed_cv_image = preprocess_image(cv_image, params)
        
        # Perform OCR using pytesseract on the preprocessed image
        custom_tesseract_config = f'--psm {params["tesseract_psm"]}'
        raw_text = pytesseract.image_to_string(preprocessed_cv_image, lang=params['tesseract_lang'], config=custom_tesseract_config)
        
        print(f"--- TESSERACT PREPROCESSED RAW OUTPUT for {original_filename} (PSM {params['tesseract_psm']}, lang='{params['tesseract_lang']}') ---")
        print(raw_text)
        print("--- END TESSERACT PREPROCESSED RAW OUTPUT ---")

        # Initialize fields
        banco = "Not found"
        total = "Not found"
        documento = "Not found"
        fecha = "Not found"

        lines = raw_text.split('\n')

        # --- BANCO ---
        banco_found_on_line = False
        for line in lines:
            if re.search(r"(?i)BANCO", line):
                # Try to get text after "BANCO" keyword
                match = re.search(r"(?i)BANCO\s*[:\-]?\s*(.*)", line)
                if match:
                    potential_banco = match.group(1).strip()
                    # Clean common suffixes and artifacts
                    potential_banco = re.sub(r"(?i)\s*(?:C\.A|S\.A|E\.A|CIA\. LTDA|LTDA)[\-\.]*$", "", potential_banco).strip()
                    potential_banco = re.sub(r"[€]", "", potential_banco).strip() # Remove stray symbols like Euro
                    if len(potential_banco) > 2 : # Check if it's a meaningful name
                        banco = potential_banco
                        banco_found_on_line = True
                        print(f"[TESSERACT_EXTRACT] BANCO (from line): {banco}")
                        break
        
        if not banco_found_on_line:
            # Fallback: search for known bank names in the whole text if not found on a "BANCO" line
            known_banks_pattern = r"(?i)(Pichincha|Produbanco|Guayaquil|Pacífico|Bolivariano|Internacional)"
            bank_keyword_match = re.search(known_banks_pattern, raw_text)
            if bank_keyword_match:
                banco = bank_keyword_match.group(1).strip()
                print(f"[TESSERACT_EXTRACT] BANCO (known keyword fallback): {banco}")
        
        # --- DOCUMENTO ---
        for line in lines:
            if re.search(r"(?i)Documento", line):
                numbers_in_line = re.findall(r'\d+', line)
                if numbers_in_line:
                    documento = str(max(int(n) for n in numbers_in_line))
                    print(f"[TESSERACT_EXTRACT] DOCUMENTO (from line, largest number): {documento}")
                    break
        
        # --- TOTAL ---
        for line in lines:
            if re.search(r"(?i)Total", line):
                # Regex for amount like xxx.xx or xxx,xx (allows for . or , as decimal)
                match = re.search(r"([\d]+[\.,][\d]{2})", line) 
                if match:
                    total_str = match.group(1).replace(',', '.') # Standardize to dot as decimal
                    # Validate it's a number before assigning
                    if re.match(r"^\d+\.\d{2}$", total_str):
                         total = total_str
                         print(f"[TESSERACT_EXTRACT] TOTAL (from line): {total}")
                         break
                    else:
                        print(f"[TESSERACT_EXTRACT_WARN] TOTAL candidate on line '{line.strip()}' was '{match.group(1)}', not valid x.xx format after cleaning.")
                else: # If first regex fails on total line, try to find any number on that line as last resort
                    numbers_on_total_line = re.findall(r"[\d.,]+", line)
                    if numbers_on_total_line:
                        # Get the number that looks most like a monetary value (e.g. ends in .xx or ,xx)
                        for num_str in numbers_on_total_line:
                            if re.search(r"[\.,]\d{2}$", num_str):
                                total = num_str.replace(',', '.')
                                if re.match(r"^\d+\.\d{2}$", total):
                                    print(f"[TESSERACT_EXTRACT] TOTAL (any number on 'Total' line): {total}")
                                    break
                        if total != "Not found": break


        # --- FECHA ---
        for line in lines:
            # User wants YYYY/MMM/DD, MMM being 3 letters
            # Tesseract output was "2020/NOV/02"
            if re.search(r"(?i)Fecha", line):
                match = re.search(r"(\d{4}/[A-Z0-9]{3}/\d{2})", line, re.IGNORECASE) # Added IGNORECASE here too
                if match:
                    fecha = match.group(1).upper() # Standardize month to uppercase
                    print(f"[TESSERACT_EXTRACT] FECHA (from line YYYY/MMM/DD): {fecha}")
                    break
        
        # Build output text
        parsed_info = [
            f"Banco: {banco}",
            f"Total: {total}",
            f"Documento: {documento}",
            f"Fecha: {fecha}"
        ]

        # Save to .txt file
        base_name = os.path.splitext(original_filename)[0]
        output_txt_dir = os.path.dirname(image_path)
        output_txt_path = os.path.join(output_txt_dir, base_name + ".txt")

        print(f"Saving extracted Tesseract (preprocessed) info to: {output_txt_path}")
        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(parsed_info))

        print(f"Successfully processed Tesseract OCR (preprocessed) for {original_filename}")
        return True

    except pytesseract.TesseractNotFoundError:
        print("ERROR: Tesseract executable not found. Please ensure Tesseract is installed and in your PATH.")
        return False
    except Exception as e:
        print(f"Error during OCR processing for {original_filename}: {e}")
        import traceback
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
