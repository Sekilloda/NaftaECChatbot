# Standard Python library imports
import base64
import json
import os
import pathlib
import re
import traceback
import configparser # Added
import sys # Added
import random

# Third-party library imports
import cv2
from dotenv import load_dotenv
from flask import Flask, request, jsonify, g
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
import pandas as pd
# Mistral and Langchain related imports
from mistralai import Mistral # Using Mistral directly as per notebook
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import ChatPromptTemplate
# from langchain_core.messages import HumanMessage # Not strictly needed if using .to_messages()

# Load environment variables
load_dotenv()



app = Flask(__name__)


# --- SQLite: robusta para concurrencia ----------------------------------
import sqlite3

DB_PATH = "chat_history.db"

def get_db():
    """
    Abre una nueva conexión a la base de datos si no existe una en el contexto de la solicitud actual.
    """
    if 'db' not in g:
        g.db = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(e=None):
    """Cierra la conexión a la base de datos al final de la solicitud."""
    db = g.pop('db', None)
    if db is not None:
        db.close()

def init_db():
    """Inicializa la base de datos y crea las tablas si no existen."""
    db = get_db()
    db.executescript("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS conversation_params (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            model_name TEXT NOT NULL,
            context_window INTEGER NOT NULL,
            temperature REAL NOT NULL,
            status TEXT NOT NULL DEFAULT 'active', -- 'active', 'awaiting_rating', or 'finished'
            satisfaction_rating INTEGER,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS parameter_combination_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL,
            context_window INTEGER NOT NULL,
            temperature REAL NOT NULL,
            conversation_count INTEGER DEFAULT 0,
            UNIQUE(model_name, context_window, temperature)
        );
    """)
    db.commit()

# Initialize the database when the app starts
with app.app_context():
    init_db()


def canonicalize_jid(jid: str) -> str:
    """
    Normaliza los identificadores para que en la DB usemos siempre la misma forma.
    - Si ya contiene '@' se deja tal cual (ej: '573001112233@s.whatsapp.net' o 'whatsapp:+57300...')
    - Si no contiene '@' asumimos número y añadimos '@s.whatsapp.net'
    """
    if not jid:
        return jid
    jid = jid.strip()
    # Mantener formatos tipo 'whatsapp:+57...' o que ya contengan domain
    if jid.startswith("whatsapp:") or "@" in jid:
        return jid
    return f"{jid}@s.whatsapp.net"

def save_message(user_id: str, role: str, content: str):
    """Guarda un mensaje (user/assistant) en la base de datos. No lanza excepción si falla."""
    try:
        user_id_norm = canonicalize_jid(user_id)
        db = get_db()
        db.execute(
            "INSERT INTO conversations (user_id, role, content) VALUES (?, ?, ?)",
            (user_id_norm, role, content)
        )
        db.commit()
    except Exception as e:
        print(f"[DB] Error guardando mensaje para {user_id}: {e}")

def get_last_messages(user_id: str, limit: int = 5):
    """Devuelve los últimos 'limit' mensajes CRONOLÓGICOS (del más antiguo al más reciente)."""
    try:
        user_id_norm = canonicalize_jid(user_id)
        db = get_db()
        rows = db.execute(
            "SELECT role, content, created_at FROM conversations WHERE user_id=? ORDER BY created_at DESC LIMIT ?",
            (user_id_norm, limit)
        ).fetchall()
        # devolver en orden cronológico (más antiguo primero)
        return rows[::-1]
    except Exception as e:
        print(f"[DB] Error leyendo historial para {user_id}: {e}")
        return []
# ------------------------------------------------------------------------
def get_or_create_conversation_params(user_id: str) -> dict:
    """
    Obtiene los parámetros de A/B testing para un usuario.
    Si no existen, los crea aleatoriamente y los guarda en la DB.
    """
    user_id_norm = canonicalize_jid(user_id)
    db = get_db()
    params = db.execute("SELECT model_name, context_window, temperature FROM conversation_params WHERE user_id=? AND status='active'", (user_id_norm,)).fetchone()

    if params:
        return {"model_name": params["model_name"], "context_window": params["context_window"], "temperature": params["temperature"]}
    else:
        # Parameter generation loop
        while True:
            model_name = random.choice(["gemini-2.5-flash-lite", "mistral-small-2506"])
            context_window = random.choice([10, 20])
            
            if "gemini" in model_name:
                temperature = random.choice([1.0, 0.5])
            else: # mistral
                temperature = random.choice([0.15, 0.7])

            # Check usage count for this combination
            count_row = db.execute(
                "SELECT conversation_count FROM parameter_combination_stats WHERE model_name=? AND context_window=? AND temperature=?",
                (model_name, context_window, temperature)
            ).fetchone()
            
            count = count_row[0] if count_row else 0

            if count < 12:
                break  # Found a combination with less than 12 uses

            # If count is >= 12, check if all combinations have reached the threshold
            total_combinations = 2 * 2 * 2 # 2 models, 2 contexts, 2 temperatures
            fully_saturated_combinations = db.execute(
                "SELECT COUNT(*) FROM parameter_combination_stats WHERE conversation_count >= 12"
            ).fetchone()[0]

            if fully_saturated_combinations == total_combinations:
                break # All combinations are saturated, proceed with the current one

        new_params = {
            "model_name": model_name,
            "context_window": context_window,
            "temperature": temperature,
        }

        try:
            db.execute(
                "INSERT INTO conversation_params (user_id, model_name, context_window, temperature, status) VALUES (?, ?, ?, ?, 'active')",
                (user_id_norm, new_params["model_name"], new_params["context_window"], new_params["temperature"])
            )
            # Actualizamos las estadísticas
            db.execute(
                "INSERT INTO parameter_combination_stats (model_name, context_window, temperature, conversation_count) VALUES (?, ?, ?, 1) ON CONFLICT(model_name, context_window, temperature) DO UPDATE SET conversation_count = conversation_count + 1",
                (new_params["model_name"], new_params["context_window"], new_params["temperature"])
            )
            db.commit()
            print(f"[A/B Test] Nuevos parámetros para {user_id_norm}: {new_params}")
        except Exception as e:
            print(f"[DB] Error creando parámetros para {user_id_norm}: {e}")
            # Fallback a default si la DB falla
            return {"model_name": "gemini-2.5-flash-lite", "context_window": 10, "temperature": 1.0}
            
        return new_params


# ------------------ Funciones para leer historial y construir contexto ------------------

def fetch_last_messages_from_db(user_id: str, limit: int = 5):
    """
    Lee los últimos 'limit' mensajes del usuario desde la tabla 'conversations'.
    Devuelve una lista de dicts con keys: 'role', 'content', 'created_at'.
    El orden devuelto es cronológico: del más antiguo al más reciente.
    """
    try:
        if not user_id:
            return []

        user_norm = canonicalize_jid(user_id)
        db = get_db()
        rows = db.execute(
            "SELECT role, content, created_at FROM conversations WHERE user_id=? ORDER BY created_at DESC LIMIT ?",
            (user_norm, limit)
        ).fetchall()
        # rows están en orden DESC por created_at — invertimos para entregar cronológico ASC
        rows = rows[::-1]
        result = [{"role": r[0], "content": r[1], "created_at": r[2]} for r in rows]

        return result
    except Exception as e:
        print(f"[DB] Error en fetch_last_messages_from_db para {user_id}: {e}")
        return []

def build_context_from_db(user_id: str, limit: int = 6, include_system: bool = False):
    """
    Construye un string contexto con los últimos mensajes para enviar al LLM.
    - include_system: si True incluye eventos 'system' (ej. recibo de message_id)
    Formato de salida (ejemplo):
        Usuario: Hola, quiero inscribirme
        Asistente: Claro, ¿tienes tu comprobante?
        Usuario: Sí, lo envío ahora
    """
    rows = fetch_last_messages_from_db(user_id, limit=limit)
    if not rows:
        return ""

    lines = []
    for entry in rows:
        role = entry.get("role", "").lower()
        content = entry.get("content", "") or ""
        if role == "user":
            label = "Usuario"
        elif role == "assistant":
            label = "Asistente"
        elif role == "system":
            if not include_system:
                continue
            label = "Sistema"
        else:
            label = role.capitalize()
        # Normalizar newlines para que el prompt no quede roto
        content_clean = content.replace("\r\n", "\n").strip()
        lines.append(f"{label}: {content_clean}")
    return "\n".join(lines)



#pending_confirmations = {} # Replaced by file-based state management

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
WASENDER_API_TOKEN = os.getenv("WASENDER_API_TOKEN")
WASENDER_API_URL = "https://wasenderapi.com/api/send-message"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Setup Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash-lite") # MODIFICADO para usar un modelo más potente

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
    },
    {
        "type": "faq",
        "question": "Muchas gracias, qué hago ahora? Está mi registro completo? Eso es todo",
        "answer": "Si has recibido tu código y se ha validado con tu comprobante, el proceso está listo. Bienvenido a la familia Nafta!"
    },
    {
        "type": "faq",
        "question": "Hola, quiero información. Hola, buenas tardes, buenos días, buenas noches. Hola, info",
        "answer": "Hola! Soy el asistente virtual de NaftaEC, estoy disponible para ayudarte con tus requerimientos."
    }
]


df = pd.read_excel("faqs.xlsx") 

documentos = df.to_dict(orient="records")

descripciones = [d["question"] for d in documentos]



#documentos = faqs
#descripciones = [d["question"] if d["type"] == "faq" else d["description"] for d in documentos]

# Carga modelo de embeddings
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedder.encode(descripciones)

def responder(pregunta_actual, historial_conversacion="", k=2, model_name="gemini-2.5-flash-lite", temperature=0.2):
    """
    Finds relevant context and generates a response using a structured prompt.
    - pregunta_actual: The user's most recent message. Used for semantic search.
    - historial_conversacion: The string of previous turns in the conversation.
    - model_name: The name of the model to use for generation.
    - temperature: The temperature for the generation.
    """
    # STEP 1: RETRIEVAL (using ONLY the current question)
    pregunta_emb = embedder.encode([pregunta_actual])
    similitudes = cosine_similarity(pregunta_emb, embeddings)[0]
    
    # Get up to k results, but only if they are reasonably similar (e.g., similarity > 0.5)
    # This prevents adding irrelevant context for unrelated questions.
    min_similarity_threshold = 0.5
    top_k_idx = [idx for idx in similitudes.argsort()[-k:][::-1] if similitudes[idx] > min_similarity_threshold]

    # STEP 2: CONTEXT ASSEMBLY
    contexto_rag = []
    if not top_k_idx:
        contexto_rag.append("No se encontró información relevante en la base de conocimientos para esta pregunta.")
    else:
        for idx in top_k_idx:
            item = documentos[idx]
            if item["type"] == "faq":
                contexto_rag.append(f"Pregunta Frecuente: {item['question']}\nRespuesta: {item['answer']}")

    contexto_rag_str = "\n---\n".join(contexto_rag)

    # STEP 3: GENERATION (with a structured prompt)
    prompt_final = f"""Eres un asistente virtual amigable y eficiente para NaftaEC, una empresa que organiza carreras. Tu misión es resolver dudas y ayudar a los usuarios.

**INSTRUCCIONES IMPORTANTES:**
1.  **Prioriza la Base de Conocimientos**: La siguiente sección 'BASE DE CONOCIMIENTOS' contiene la información más precisa. Úsala como tu fuente principal para responder la 'PREGUNTA ACTUAL DEL USUARIO'.
2.  **Usa el Historial para Contexto**: La sección 'HISTORIAL DE CONVERSACIÓN' te ayuda a entender lo que se ha hablado antes (por ejemplo, si el usuario ya saludó o si están hablando de un tema específico). No busques respuestas en el historial, solo úsalo para que tu respuesta fluya naturalmente.
3.  **Sé Conciso**: Responde directamente a la pregunta del usuario.
4.  **Si no sabes, dilo**: Si la 'BASE DE CONOCIMIENTOS' no contiene la respuesta, es mejor decir "No tengo información sobre eso en este momento, pero puedo ayudarte con otras preguntas sobre nuestras carreras." No inventes información.
5.  **Finaliza la conversación**: Si el usuario indica que no necesita más ayuda, despídete amablemente y menciona que pronto recibirá su código si ha completado un pago.

---
**BASE DE CONOCIMIENTOS (Usa esto para responder):**
{contexto_rag_str}
---
**HISTORIAL DE CONVERSACIÓN (Úsalo para contexto):**
{historial_conversacion if historial_conversacion else "No hay historial previo."}
---
**PREGUNTA ACTUAL DEL USUARIO (Debes responder a esto):**
{pregunta_actual}
---

Asistente:"""

     # STEP 4: DYNAMIC GENERATION
    print(f"[LLM] Generating response using {model_name} with temp={temperature}")

    try:
        if "gemini" in model_name:
            gemini_model = genai.GenerativeModel(model_name)
            generation_config = genai.types.GenerationConfig(temperature=temperature)
            respuesta = gemini_model.generate_content(prompt_final, generation_config=generation_config).text.strip()
        elif "mistral" in model_name:
            mistral_client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
            messages = [{"role": "user", "content": prompt_final}]
            chat_response = mistral_client.chat.complete(
                model=model_name,
                messages=messages,
                temperature=temperature
            )
            respuesta = chat_response.choices[0].message.content.strip()
        else:
            print(f"[LLM_ERROR] Unknown model: {model_name}. Reverting to default.")
            # Fallback to the original global model
            respuesta = model.generate_content(prompt_final).text.strip()
    except Exception as e:
        print(f"[LLM] Error during dynamic generation with {model_name}: {e}")
        respuesta = "Lo siento, estoy teniendo dificultades para procesar tu solicitud en este momento."

    
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
        # The PHP example implies mediaKey is THE key. Let's assume it's the master secret.       # The common pattern is that the mediaKey (32 bytes) is expanded.
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

        message_container = data.get("data", {}).get("messages")
        if not message_container:
            message_container = data.get("data", {}).get("message")

        if not message_container:
            print(f"No 'messages' or 'message' field found in data.get('data', {{}}). Received data: {json.dumps(data, indent=2)}")
            return {"status": "no_message_data_container"}

        if isinstance(message_container, list):
            if not message_container:
                return {"status": "empty_messages_list"}
            message_container = message_container[0]

        if message_container.get("key", {}).get("fromMe"):
            return jsonify({"status": "self_message_skipped"})

        sender = message_container.get("key", {}).get("remoteJid")
        message_id = message_container.get("key", {}).get("id", "unknown")

        save_message(sender, "system", f"received_message_id:{message_id}")

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
                output_dir = 'media'
                output_path = os.path.join(output_dir, output_filename)
                encrypted_data = download_media(media_url)

                if encrypted_data:
                    decryption_successful = decrypt_and_save_media(media_key, encrypted_data, output_path, media_type)
                    if decryption_successful:
                        print(f"Media decrypted and saved successfully: {output_path}")
                        save_message(sender, "user", f"[media:{media_type}] {output_path}")
                        if media_type == 'image':
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

                            new_confirmation_data = {
                                "message_id": message_id,
                                "output_path": output_path,
                                "original_filename": output_filename,
                                "state": "awaiting_image_confirmation"
                            }
                            save_pending_confirmation(sender, new_confirmation_data)
                            confirmation_question = "¿Esta imagen corresponde a un comprobante de transferencia, depósito o cualquier otro recibo de un pago realizado?"
                            send_whatsapp_message(sender, confirmation_question)
                            print(f"Image awaiting confirmation from {sender} for message_id {message_id}")
                            return jsonify({'status': 'image_awaiting_confirmation', 'id': message_id})
                        else:
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

        if incoming_text:
            save_message(sender, "user", incoming_text)

        if not incoming_text:
            print(f"Unsupported message type for id {message_id}. msg_content: {json.dumps(msg_content, indent=2)}")
            return jsonify({"status": "unsupported_message_type", "id": message_id})

        if incoming_text.strip().lower() == "fin.":
            db = get_db()
            db.execute("UPDATE conversation_params SET status='awaiting_rating' WHERE user_id=? AND status='active'", (canonicalize_jid(sender),))
            db.commit()
            send_whatsapp_message(sender, "Gracias por tu tiempo. Por favor, califica esta conversación del 1 al 7 (7 es la mejor).")
            return jsonify({"status": "awaiting_rating"})

        # Check if there's a conversation awaiting a rating for this user
        db = get_db()
        rating_pending = db.execute("SELECT id FROM conversation_params WHERE user_id=? AND status='awaiting_rating'", (canonicalize_jid(sender),)).fetchone()

        if rating_pending:
            try:
                rating = int(incoming_text.strip())
                if 1 <= rating <= 7:
                    db.execute("UPDATE conversation_params SET satisfaction_rating=?, status='finished' WHERE id=?", (rating, rating_pending['id']))
                    db.commit()
                    send_whatsapp_message(sender, "¡Gracias por tu calificación!")
                    return jsonify({"status": "rating_received"})
                else:
                    raise ValueError("Rating out of range")
            except (ValueError, TypeError):
                send_whatsapp_message(sender, "Por favor, ingresa un número del 1 al 7.")
                return jsonify({"status": "invalid_rating"})

        pending_details = get_pending_confirmation(sender)
        if pending_details:
            if pending_details.get("state") == "awaiting_data_confirmation":
                print(f"User {sender} is responding to data confirmation. Message: '{incoming_text}'")
                prompt_text = (
                    f"Analiza la siguiente respuesta del usuario: '{incoming_text}'. "
                    f"El usuario está respondiendo a la pregunta de si los datos que extrajimos de su recibo son correctos. "
                    f"Tu tarea es clasificar su respuesta. Responde ÚNICAMENTE con UNA de las siguientes tres palabras: AFFIRMATIVE, NEGATIVE, o UNCLEAR. "
                    f"No añadas ninguna explicación ni puntuación."
                )
                try:
                    gemini_response = model.generate_content(prompt_text)
                    classification = gemini_response.text.strip().lower()
                except Exception as e:
                    print(f"Error classifying data confirmation response: {e}")
                    classification = "unclear"

                if classification == 'affirmative':
                    goodbye_message = "¡Perfecto! Hemos confirmado tus datos. Pronto recibirás tu código de inscripción. ¡Gracias por unirte a NaftaEC!"
                    send_whatsapp_message(sender, goodbye_message)
                    clear_pending_confirmation(sender)
                    return jsonify({'status': 'data_confirmed_conversation_ended', 'id': pending_details['message_id']})
                
                elif classification == 'negative':
                    request_correction_message = "Entendido. Por favor, ¿podrías indicarme los datos correctos o enviar una imagen más clara del comprobante para corregirlos?"
                    send_whatsapp_message(sender, request_correction_message)
                    clear_pending_confirmation(sender)
                    return jsonify({'status': 'data_denied_restarting_flow', 'id': pending_details['message_id']})

                else: # Unclear
                    clarification_message = "No entendí bien tu respuesta. ¿Son los datos que te mostré correctos? Por favor, responde con 'sí' o 'no'."
                    send_whatsapp_message(sender, clarification_message)
                    return jsonify({'status': 'data_confirmation_unclear', 'id': pending_details['message_id']})

            elif pending_details.get("state") == "awaiting_image_confirmation":
                prompt_text = (
                    f"Analiza el siguiente mensaje del usuario: '{incoming_text}'. "
                    f"El usuario está respondiendo a la pregunta: '¿Esta imagen corresponde a un comprobante de transferencia, depósito o cualquier otro recibo de un pago realizado?'. "
                    f"Tu tarea es clasificar la respuesta del usuario. Responde ÚNICAMENTE con UNA de las siguientes tres palabras: AFFIRMATIVE, NEGATIVE, o UNCLEAR. "
                    f"No añadas ninguna explicación ni puntuación adicional."
                )
                
                raw_classification = "error_gemini"
                try:
                    gemini_classification_response = model.generate_content(prompt_text)
                    raw_classification = gemini_classification_response.text.strip()
                except Exception as gen_err:
                    print(f"Error generating content with Gemini for classification: {gen_err}")
                
                processed_classification = raw_classification.lower()

                print(f"Sender: {sender}, Incoming Text: \"{incoming_text}\", Raw Gemini Classification: \"{raw_classification}\", Processed Classification: \"{processed_classification}\"")
                
                if processed_classification == 'affirmative':
                    print(f"User {sender} confirmed (AFFIRMATIVE). Processing OCR for image: {pending_details['output_path']}")
                    ocr_success = process_receipt_image(pending_details['output_path'], pending_details['original_filename'])
                    
                    if ocr_success:
                        txt_filename = os.path.splitext(pending_details['original_filename'])[0] + ".txt"
                        txt_filepath = os.path.join(os.path.dirname(pending_details['output_path']), txt_filename)
                        
                        try:
                            with open(txt_filepath, 'r', encoding='utf-8') as f:
                                receipt_content = f.read()
                            
                            prompt_for_data_confirmation = (
                                f"Eres un asistente de NaftaEC. Acabas de procesar un recibo de pago para un usuario. "
                                f"Los datos extraídos son:\n\n{receipt_content}\n\n"
                                f"Tu tarea es presentarle esta información al usuario de forma clara y amigable, y luego preguntarle explícitamente si los datos son correctos. "
                                f"Termina tu mensaje con una pregunta clara como '¿Son correctos estos datos?'."
                            )

                            confirmation_request_message = model.generate_content(prompt_for_data_confirmation).text.strip()
                            
                            send_whatsapp_message(sender, confirmation_request_message)
                            
                            pending_details['state'] = 'awaiting_data_confirmation'
                            pending_details['ocr_data'] = receipt_content
                            save_pending_confirmation(sender, pending_details)

                            print(f"Sent OCR data confirmation request to {sender} for image {pending_details['original_filename']}")
                            return jsonify({'status': 'ocr_processed_awaiting_data_confirmation', 'id': pending_details['message_id']})

                        except Exception as e_read:
                            print(f"Error reading OCR output file {txt_filepath}: {e_read}")
                            send_whatsapp_message(sender, "Lo siento, hubo un error al leer el resultado del recibo.")
                            clear_pending_confirmation(sender)
                            return jsonify({'status': 'ocr_processed_file_read_error', 'id': pending_details['message_id']})
                    else:
                        print(f"OCR processing failed for {pending_details['original_filename']} for sender {sender}")
                        send_whatsapp_message(sender, "Lo siento, no pude procesar la imagen del recibo.")
                        clear_pending_confirmation(sender)
                        return jsonify({'status': 'ocr_processing_failed', 'id': pending_details['message_id']})
                
                elif processed_classification == 'negative':
                    print(f"User {sender} denied confirmation (NEGATIVE) for image {pending_details['original_filename']}")
                    send_whatsapp_message(sender, "Entendido. Si la imagen no era un recibo, no la procesaré. Puede enviar otra imagen si lo desea o hacerme una pregunta.")
                    clear_pending_confirmation(sender)
                    return jsonify({'status': 'ocr_cancelled_by_user', 'id': pending_details['message_id']})
                
                else: 
                    print(f"Confirmation unclear from {sender} for image {pending_details['original_filename']}. Processed Classification: {processed_classification}")
                    send_whatsapp_message(sender, "No entendí tu respuesta. Por favor, responde 'sí' o 'no' para el recibo anterior, o envía un nuevo mensaje si quieres que ignore la imagen.")
                    return jsonify({'status': 'confirmation_unclear', 'id': pending_details['message_id']})
        
        else:
            # No hay confirmación pendiente, se procesa como una pregunta general con RAG y contexto
            print(f"Processing general question from {sender}: '{incoming_text}'")
            # Obtener parámetros de A/B testing
            conv_params = get_or_create_conversation_params(sender)
            
            context_str = build_context_from_db(sender, limit=conv_params["context_window"]) 
 

            try:
                # `incoming_text` (pregunta actual) se usa para la búsqueda semántica.
                # `context_str` (historial) se usa para que la conversación fluya.
                answer_text = responder(
                    pregunta_actual=incoming_text,
                    historial_conversacion=context_str,
                    model_name=conv_params["model_name"],
                    temperature=conv_params["temperature"]
                )
            except Exception as e:
                print(f"[LLM] Error in new responder function: {e}")
                answer_text = "Lo siento, ocurrió un error al procesar tu solicitud."

            send_whatsapp_message(sender, answer_text)
            return jsonify({"status": "text_message_processed_with_rag_and_context", "id": message_id})

    except Exception as e:
        print(f"Error in webhook: {str(e)}\nTraceback: {traceback.format_exc()}")
        return jsonify({"status": "error", "message": str(e)}), 500

def handle_webhook_data(data):
    try:
        if data.get("event") != "messages.upsert":
            return {"status": "ignored", "reason": "wrong_event_type"}

        message_container = data.get("data", {}).get("messages", {})
        if not message_container:
            message_container = data.get("data", {}).get("message", {})

        if not message_container:
            print(f"No 'messages' or 'message' field found in data.get('data', {{}}). Received data: {json.dumps(data, indent=2)}")
            return {"status": "no_message_data_container"}

        if isinstance(message_container, list):
            if not message_container:
                return {"status": "empty_messages_list"}
            message_container = message_container[0]

        if message_container.get("key", {}).get("fromMe"):
            return {"status": "self_message_skipped"}

        sender = message_container.get("key", {}).get("remoteJid")
        message_id = message_container.get("key", {}).get("id", "unknown")

        save_message(sender, "system", f"received_message_id:{message_id}")

        msg_content = message_container.get("message", {})
        if not msg_content and message_container:
            is_potential_content_node = any(k in message_container for k in ['imageMessage', 'videoMessage', 'audioMessage', 'documentMessage', 'conversation', 'extendedTextMessage'])
            if is_potential_content_node:
                msg_content = message_container
            else:
                print(f"Empty or unrecognized message content for message_id: {message_id}. Message container: {json.dumps(message_container, indent=2)}")
                return {"status": "empty_or_unrecognized_message_content", "id": message_id}
        elif not msg_content:
            print(f"Critical: msg_content is None or empty for message_id: {message_id}")
            return {"status": "critical_empty_message_content", "id": message_id}

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
                output_dir = 'media'
                output_path = os.path.join(output_dir, output_filename)

                if media_url.startswith('file://'):
                    file_path = media_url[7:]
                    try:
                        with open(file_path, 'rb') as f:
                            encrypted_data = f.read()
                        
                        os.makedirs(output_dir, exist_ok=True)
                        with open(output_path, 'wb') as f:
                            f.write(encrypted_data)
                        
                        print(f"Media saved successfully (simulation): {output_path}")
                        save_message(sender, "user", f"[media:{media_type}] {output_path}")
                        
                        if media_type == 'image':
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

                            new_confirmation_data = {
                                "message_id": message_id,
                                "output_path": output_path,
                                "original_filename": output_filename
                            }
                            save_pending_confirmation(sender, new_confirmation_data)
                            confirmation_question = "¿Esta imagen corresponde a un comprobante de transferencia, depósito o cualquier otro recibo de un pago realizado?"
                            send_whatsapp_message(sender, confirmation_question)
                            print(f"Image awaiting confirmation from {sender} for message_id {message_id}")
                            return {'status': 'image_awaiting_confirmation', 'id': message_id}
                        else:
                            return {'status': 'media_processed_successfully_not_image', 'id': message_id, 'path': output_path}
                            
                    except Exception as e:
                        print(f"Error processing local file {file_path}: {e}")
                        return {'status': 'local_file_processing_error', 'id': message_id}
                else:
                    encrypted_data = download_media(media_url)
                    if encrypted_data:
                        decryption_successful = decrypt_and_save_media(media_key, encrypted_data, output_path, media_type)
                        if decryption_successful:
                            print(f"Media decrypted and saved successfully: {output_path}")
                            save_message(sender, "user", f"[media:{media_type}] {output_path}")
                            if media_type == 'image':
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

                                new_confirmation_data = {
                                    "message_id": message_id,
                                    "output_path": output_path,
                                    "original_filename": output_filename
                                }
                                save_pending_confirmation(sender, new_confirmation_data)
                                confirmation_question = "¿Esta imagen corresponde a un comprobante de transferencia, depósito o cualquier otro recibo de un pago realizado?"
                                send_whatsapp_message(sender, confirmation_question)
                                print(f"Image awaiting confirmation from {sender} for message_id {message_id}")
                                return {'status': 'image_awaiting_confirmation', 'id': message_id}
                            else:
                                return {'status': 'media_processed_successfully_not_image', 'id': message_id, 'path': output_path}
                        else:
                            print(f"Media decryption failed for {message_id}, type: {media_type}")
                            return {'status': 'media_decryption_failed', 'id': message_id, 'type': media_type}
                    else:
                        print(f"Media download failed for {message_id}, url: {media_url}")
                        return {'status': 'media_download_failed', 'id': message_id, 'url': media_url}
            else:
                print(f"Media message ({media_type}) for id {message_id}, but key/URL missing. URL: {media_url}, Key: {media_key}")
                return {"status": "media_info_incomplete", "type": media_type, "id": message_id}

        incoming_text = None
        if "conversation" in msg_content:
            incoming_text = msg_content["conversation"]
        elif "extendedTextMessage" in msg_content:
            incoming_text = msg_content["extendedTextMessage"].get("text")

        if incoming_text:
            save_message(sender, "user", incoming_text)

        if not incoming_text:
            print(f"Unsupported message type for id {message_id}. msg_content: {json.dumps(msg_content, indent=2)}")
            return {"status": "unsupported_message_type", "id": message_id}

        image_details = get_pending_confirmation(sender)
        if image_details:
            prompt_text = (
                f"Analiza el siguiente mensaje del usuario: '{incoming_text}'. "
                f"El usuario está respondiendo a la pregunta: '¿Esta imagen corresponde a un comprobante de transferencia, depósito o cualquier otro recibo de un pago realizado?'. "
                f"Tu tarea es clasificar la respuesta del usuario. Responde ÚNICAMENTE con UNA de las siguientes tres palabras: AFFIRMATIVE, NEGATIVE, o UNCLEAR. "
                f"No añadas ninguna explicación ni puntuación adicional."
            )
            
            raw_classification = "error_gemini"
            try:
                gemini_classification_response = model.generate_content(prompt_text)
                raw_classification = gemini_classification_response.text.strip()
            except Exception as gen_err:
                print(f"Error generating content with Gemini for classification: {gen_err}")
            
            processed_classification = raw_classification.lower()

            print(f"Sender: {sender}, Incoming Text: \"{incoming_text}\", Raw Gemini Classification: \"{raw_classification}\", Processed Classification: \"{processed_classification}\"")
            
            if processed_classification == 'affirmative':
                print(f"User {sender} confirmed (AFFIRMATIVE). Processing OCR for image: {image_details['output_path']}")
                ocr_success = process_receipt_image(image_details['output_path'], image_details['original_filename'])
                
                if ocr_success:
                    txt_filename = os.path.splitext(image_details['original_filename'])[0] + ".txt"
                    txt_filepath = os.path.join(os.path.dirname(image_details['output_path']), txt_filename)
                    
                    try:
                        with open(txt_filepath, 'r', encoding='utf-8') as f:
                            receipt_content = f.read()
                        send_whatsapp_message(sender, receipt_content)
                        print(f"Sent OCR results to {sender} for image {image_details['original_filename']}")
                        clear_pending_confirmation(sender)
                        return {'status': 'ocr_processed_and_sent', 'id': image_details['message_id']}
                    except FileNotFoundError:
                        print(f"Error: OCR output file {txt_filepath} not found, though ocr_success was true.")
                        send_whatsapp_message(sender, "Lo siento, hubo un error al leer el resultado del recibo.")
                        clear_pending_confirmation(sender)
                        return {'status': 'ocr_processed_file_read_error', 'id': image_details['message_id']}
                    except Exception as e_read:
                        print(f"Error reading OCR output file {txt_filepath}: {e_read}")
                        send_whatsapp_message(sender, "Lo siento, hubo un error al leer el resultado del recibo.")
                        clear_pending_confirmation(sender)
                        return {'status': 'ocr_processed_file_read_error', 'id': image_details['message_id']}
                else:
                    print(f"OCR processing failed for {image_details['original_filename']} for sender {sender}")
                    send_whatsapp_message(sender, "Lo siento, no pude procesar la imagen del recibo.")
                    clear_pending_confirmation(sender)
                    return {'status': 'ocr_processing_failed', 'id': image_details['message_id']}
            
            elif processed_classification == 'negative':
                print(f"User {sender} denied confirmation (NEGATIVE) for image {image_details['original_filename']}")
                send_whatsapp_message(sender, "Entendido. Si la imagen no era un recibo, no la procesaré. Puede enviar otra imagen si lo desea o hacerme una pregunta.")
                clear_pending_confirmation(sender)
                return {'status': 'ocr_cancelled_by_user', 'id': image_details['message_id']}
            
            else:
                print(f"Confirmation unclear from {sender} for image {image_details['original_filename']}. Processed Classification: {processed_classification}")
                send_whatsapp_message(sender, "No entendí tu respuesta. Por favor, responde 'sí' o 'no' para el recibo anterior, o envía un nuevo mensaje si quieres que ignore la imagen.")
                return {'status': 'confirmation_unclear', 'id': image_details['message_id']}
        
        else:
            # No hay confirmación pendiente, se procesa como una pregunta general con RAG y contexto
            print(f"Processing general question from {sender} in simulator: '{incoming_text}'")
            context_str = build_context_from_db(sender, limit=10)

            try:
                # `incoming_text` se usa para la búsqueda, `context_str` para el flujo de la conversación.
                answer_text = responder(pregunta_actual=incoming_text, historial_conversacion=context_str)
            except Exception as e:
                print(f"[LLM] Error in new responder function (simulator): {e}")
                answer_text = "Lo siento, ocurrió un error al procesar tu solicitud."

            save_message(sender, "assistant", answer_text)
            send_whatsapp_message(sender, answer_text)
            return {"status": "text_message_processed_with_rag_and_context", "id": message_id}

    except Exception as e:
        print(f"Error in handle_webhook_data: {str(e)}\nTraceback: {traceback.format_exc()}")
        return {"status": "error", "message": str(e)}
def send_whatsapp_message(recipient, text):
    """
    Envía el mensaje y además lo persiste en SQLite como 'assistant'.
    Mantiene la compatibilidad con los formatos de recipient que uses.
    """
    # canonicalizamos el recipient para la DB (pero respetamos el formato que usa la API)
    recipient_for_db = recipient
    if "@s.whatsapp.net" in recipient:
        # la API original espera solo el número o a veces el formato sin domain;
        # guardamos en DB con domain para coherencia
        recipient_for_db = canonicalize_jid(recipient)
        # para la petición HTTP original, convertir a forma numérica
        recipient_payload = recipient.split("@")[0]
    else:
        recipient_for_db = canonicalize_jid(recipient)
        recipient_payload = recipient  # si ya es 'whatsapp:+57...' o '57300...'

    payload = {"to": recipient_payload, "text": text}
    headers = {
        "Authorization": f"Bearer {WASENDER_API_TOKEN}",
        "Content-Type": "application/json"
    }
    try:
        response = requests.post(WASENDER_API_URL, json=payload, headers=headers, timeout=10)
        print(f"WaSender response: {response.status_code} {response.text}")
        # Guardamos lo enviado por el asistente en la DB (aunque el envío falle, nos interesa el registro)
        try:
            save_message(recipient_for_db, "assistant", text)
        except Exception as e:
            print(f"[DB] Error guardando mensaje enviado: {e}")
        return response.ok
    except requests.exceptions.RequestException as e:
        print(f"Error sending message via WaSender: {e}")
        # Aun en fallo, intentamos guardar el mensaje para auditoría
        try:
            save_message(recipient_for_db, "assistant", f"[FAILED_TO_SEND] {text}")
        except Exception as e2:
            print(f"[DB] Error guardando mensaje de fallo: {e2}")
        return False


# Mistral OCR Pipeline functions
def encode_image_to_base64(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Error [MISTRAL_OCR]: Image file {image_path} not found for base64 encoding.")
        return None
    except Exception as e:
        print(f"Error [MISTRAL_OCR]: Failed to encode image {image_path} to base64: {e}")
        return None

def run_mistral_ocr_pipeline(image_path: str):
    print(f"[MISTRAL_OCR] Attempting Mistral OCR pipeline for {image_path}")
    mistral_api_key = os.getenv("MISTRAL_API_KEY")
    if not mistral_api_key:
        print("Error [MISTRAL_OCR]: MISTRAL_API_KEY environment variable not set.")
        return None

    base64_image = encode_image_to_base64(image_path)
    if not base64_image:
        return None

    try:
        # Using the direct Mistral client as per the notebook
        client = Mistral(api_key=mistral_api_key) 

        # Assuming client.ocr.process is the correct method based on the notebook.
        # This is a key point that might need adjustment if the SDK has a different structure.
        ocr_response = client.ocr.process(
            model="mistral-ocr-2503", # This model name might need verification
            document={
                "type": "image_url", 
                "image_url": f"data:image/jpeg;base64,{base64_image}"
            },
            include_image_base64=True # As per notebook
        )
        
        # Validate response structure (this is also based on notebook assumptions)
        if not hasattr(ocr_response, 'pages') or not ocr_response.pages or \
            not hasattr(ocr_response.pages[0], 'markdown') or not ocr_response.pages[0].markdown:
            print("Error [MISTRAL_OCR]: Mistral OCR returned no content or unexpected response structure.")
            # print(f"Full Mistral OCR response for debugging: {ocr_response}") # Potentially large
            return None
        
        ocr_text = ocr_response.pages[0].markdown
        print(f"[MISTRAL_OCR] Mistral OCR successful, got text (first 200 chars): {ocr_text[:200]}...")

    except Exception as e:
        print(f"Error [MISTRAL_OCR]: Call to Mistral OCR API failed: {e}")
        import traceback
        print(traceback.format_exc())
        return None

    # Langchain with Gemini for structured data extraction
    try:
        gemini_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0) # GEMINI_API_KEY is used by library

        response_schemas = [
            ResponseSchema(name="banco", description="Nombre del banco emisor del comprobante. Si no se encuentra, N/A."),
            ResponseSchema(name="hora", description="Hora de la transacción (HH:MM:SS o HH:MM). Si no se encuentra, N/A."),
            ResponseSchema(name="nombre", description="Nombre completo del titular o pagador. Si no se encuentra, N/A."),
            ResponseSchema(name="cedula", description="Cédula de identidad del titular o pagador. Si no se encuentra, N/A."),
            ResponseSchema(name="fecha", description="Fecha de la transacción (DD/MM/YYYY o YYYY/MM/DD u otro formato claro). Si no se encuentra, N/A."),
            ResponseSchema(name="monto", description="Monto total de la transacción (número con dos decimales). Si no se encuentra, N/A."),
            ResponseSchema(name="numero_transaccion", description="Número de la transacción o referencia. Si no se encuentra, N/A.")
        ]
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()

        prompt_template_str = """
Extrae la siguiente información del texto de un comprobante bancario que se te proporciona.
Sigue estas reglas estrictamente:
1. Extrae los valores para los campos: banco, hora, nombre, cedula, fecha, monto, numero_transaccion.
2. Si un campo no se encuentra de manera clara y específica en el texto, DEBES retornar 'N/A' para ese campo. No inventes información.
3. Para el campo 'monto', asegúrate que sea un número, preferiblemente con dos decimales (ej: 123.45). Si ves comas como separadores de miles y punto decimal (ej: 1,234.56), usa el punto como decimal. Si ves punto como separador de miles y coma decimal (ej: 1.234,56), usa la coma como decimal y luego reemplázala por punto.
4. Para el campo 'fecha', intenta normalizarla a DD/MM/YYYY si es posible, pero mantén el formato original si la normalización es ambigua.

Texto del comprobante:
```{text}```

{format_instructions}
"""
        prompt = ChatPromptTemplate.from_template(template=prompt_template_str)
        
        formatted_prompt_value = prompt.format_prompt(text=ocr_text, format_instructions=format_instructions)
        # Using invoke with .to_messages() for ChatModels
        response = gemini_llm.invoke(formatted_prompt_value.to_messages()) 
        
        parsed_data = output_parser.parse(response.content)
        
        print(f"[MISTRAL_OCR] Gemini structured data extraction successful: {parsed_data}")
        return parsed_data

    except Exception as e:
        print(f"Error [MISTRAL_OCR]: Gemini data extraction from Mistral text failed: {e}")
        import traceback
        print(traceback.format_exc())
        return None

# End of Mistral OCR Pipeline functions


import os # Ensure os is imported if not already at the top for this function specifically
import re # Ensure re is imported
from PIL import Image # Ensure Image is imported

import os       # Should be already imported
import re      # Should be already imported
from PIL import Image # Should be already imported
import pytesseract # Should be already imported
import cv2     # Should be already imported
import numpy as np  # Should be already imported

def process_receipt_image(image_path: str, original_filename: str) -> bool:
    print(f"Starting Tesseract OCR with preprocessing for image: {image_path} (original: {original_filename})")
    
    # Load OCR parameters
    # Assuming ocr_params.txt is in the 'Local' directory, relative to where app.py is (which is also Local/)
    # If app.py is moved, this path might need adjustment or be made absolute/configurable.
    params_file_path = "ocr_params.txt"
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
            known_banks_pattern = r"(?i)\b(Pichincha|Produbanco|Guayaquil|Pacífico|Bolivariano|Internacional)\b"
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
        
        # Mistral OCR Fallback Logic
        if any(val == "Not found" for val in [banco, total, documento, fecha]):
            print(f"[OCR_FALLBACK] Tesseract OCR results incomplete for {original_filename}. Fields: Banco='{banco}', Total='{total}', Documento='{documento}', Fecha='{fecha}'. Attempting Mistral OCR fallback.")
            mistral_output = run_mistral_ocr_pipeline(image_path) # image_path is the argument to process_receipt_image

            if mistral_output and isinstance(mistral_output, dict):
                print(f"[OCR_FALLBACK] Mistral pipeline output for {original_filename}: {mistral_output}")
                
                # Update 'banco'
                if banco == "Not found":
                    mistral_banco = mistral_output.get('banco')
                    if mistral_banco and mistral_banco.strip().upper() not in ["N/A", ""]:
                        banco = mistral_banco.strip()
                        print(f"[OCR_FALLBACK] Updated 'banco' from Mistral: {banco}")

                # Update 'total' (from 'monto')
                if total == "Not found":
                    mistral_monto = mistral_output.get('monto')
                    if mistral_monto and mistral_monto.strip().upper() not in ["N/A", ""]:
                        total = mistral_monto.strip() 
                        print(f"[OCR_FALLBACK] Updated 'total' from Mistral ('monto'): {total}")

                # Update 'documento' (from 'numero_transaccion')
                if documento == "Not found":
                    mistral_documento = mistral_output.get('numero_transaccion')
                    if mistral_documento and mistral_documento.strip().upper() not in ["N/A", ""]:
                        documento = mistral_documento.strip()
                        print(f"[OCR_FALLBACK] Updated 'documento' from Mistral ('numero_transaccion'): {documento}")
                
                # Update 'fecha'
                if fecha == "Not found":
                    mistral_fecha = mistral_output.get('fecha')
                    if mistral_fecha and mistral_fecha.strip().upper() not in ["N/A", ""]:
                        fecha = mistral_fecha.strip()
                        print(f"[OCR_FALLBACK] Updated 'fecha' from Mistral: {fecha}")
            else:
                print(f"[OCR_FALLBACK] Mistral OCR pipeline did not return valid data for {original_filename}.")
        else:
            print(f"[OCR_MAIN] Tesseract OCR results complete for {original_filename}. No fallback needed.")

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

def handle_webhook_data(data):
    """
    Función para manejar datos de webhook que puede ser llamada directamente
    desde el simulador sin necesidad del framework Flask
    """
    try:
        if data.get("event") != "messages.upsert":
            return {"status": "ignored", "reason": "wrong_event_type"}

        message_container = data.get("data", {}).get("messages")
        if not message_container:
            message_container = data.get("data", {}).get("message")

        if not message_container:
            print(f"No 'messages' or 'message' field found in data.get('data', {{}}). Received data: {json.dumps(data, indent=2)}")
            return {"status": "no_message_data_container"}

        # Si messages es lista, tomar el primer elemento
        if isinstance(message_container, list):
            if not message_container:
                return {"status": "empty_messages_list"}
            message_container = message_container[0]

        if message_container.get("key", {}).get("fromMe"):
            return {"status": "self_message_skipped"}

        sender = message_container.get("key", {}).get("remoteJid")
        message_id = message_container.get("key", {}).get("id", "unknown")

        save_message(sender, "system", f"received_message_id:{message_id}")

        msg_content = message_container.get("message", {})
        if not msg_content and message_container:
            is_potential_content_node = any(k in message_container for k in ['imageMessage', 'videoMessage', 'audioMessage', 'documentMessage', 'conversation', 'extendedTextMessage'])
            if is_potential_content_node:
                msg_content = message_container
            else:
                print(f"Empty or unrecognized message content for message_id: {message_id}. Message container: {json.dumps(message_container, indent=2)}")
                return {"status": "empty_or_unrecognized_message_content", "id": message_id}
        elif not msg_content:
            print(f"Critical: msg_content is None or empty for message_id: {message_id}")
            return {"status": "critical_empty_message_content", "id": message_id}

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

            if media_key and media_url:
                output_filename = f'{message_id}{extension}'
                output_dir = 'media'
                output_path = os.path.join(output_dir, output_filename)

                # Para simulaciones locales, manejamos archivos locales directamente
                if media_url.startswith('file://'):
                    file_path = media_url.replace('file:///', '').replace('file://', '') # Funciona para Windows y Unix
                    try:
                        os.makedirs(output_dir, exist_ok=True)
                        # Copiamos el archivo simulando la descarga
                        import shutil
                        shutil.copy(file_path, output_path)
                        decryption_successful = True
                    except Exception as e:
                        print(f"Error processing local file {file_path}: {e}")
                        return {'status': 'local_file_processing_error', 'id': message_id}
                else:
                    # Código para URLs reales de WhatsApp
                    encrypted_data = download_media(media_url)
                    if encrypted_data:
                        decryption_successful = decrypt_and_save_media(media_key, encrypted_data, output_path, media_type)
                    else:
                        print(f"Media download failed for {message_id}, url: {media_url}")
                        return {'status': 'media_download_failed', 'id': message_id, 'url': media_url}

                if decryption_successful:
                    print(f"Media processed and saved successfully: {output_path}")
                    save_message(sender, "user", f"[media:{media_type}] {output_path}")
                    if media_type == 'image':
                        # Lógica para solicitar confirmación de la imagen
                        new_confirmation_data = {
                            "message_id": message_id,
                            "output_path": output_path,
                            "original_filename": output_filename,
                            "state": "awaiting_image_confirmation"
                        }
                        save_pending_confirmation(sender, new_confirmation_data)
                        confirmation_question = "¿Esta imagen corresponde a un comprobante de transferencia, depósito o cualquier otro recibo de un pago realizado?"
                        send_whatsapp_message(sender, confirmation_question)
                        return {'status': 'image_awaiting_confirmation', 'id': message_id}
                    else:
                        return {'status': 'media_processed_successfully_not_image', 'id': message_id, 'path': output_path}
                else:
                    print(f"Media processing/decryption failed for {message_id}")
                    return {'status': 'media_processing_failed', 'id': message_id}
            else:
                print(f"Media message ({media_type}) for id {message_id}, but key/URL missing.")
                return {"status": "media_info_incomplete", "type": media_type, "id": message_id}

        incoming_text = None
        if "conversation" in msg_content:
            incoming_text = msg_content["conversation"]
        elif "extendedTextMessage" in msg_content:
            incoming_text = msg_content["extendedTextMessage"].get("text")

        if not incoming_text:
            print(f"Unsupported message type for id {message_id}.")
            return {"status": "unsupported_message_type", "id": message_id}

        save_message(sender, "user", incoming_text)

        pending_details = get_pending_confirmation(sender)
        if pending_details:
            # Flujo de confirmación de datos OCR
            if pending_details.get("state") == "awaiting_data_confirmation":
                prompt_text = (
                    f"Analiza la respuesta del usuario: '{incoming_text}'. El usuario está respondiendo si los datos de un recibo son correctos. "
                    "Clasifica su respuesta. Responde ÚNICAMENTE con: AFFIRMATIVE, NEGATIVE, o UNCLEAR."
                )
                gemini_response = model.generate_content(prompt_text)
                classification = gemini_response.text.strip().lower()

                if classification == 'affirmative':
                    goodbye_message = "¡Perfecto! Hemos confirmado tus datos. Pronto recibirás tu código de inscripción. ¡Gracias por unirte a NaftaEC!"
                    send_whatsapp_message(sender, goodbye_message)
                    clear_pending_confirmation(sender)
                    return {'status': 'data_confirmed_conversation_ended', 'id': pending_details['message_id']}
                elif classification == 'negative':
                    request_correction_message = "Entendido. Por favor, ¿podrías indicarme los datos correctos o enviar una imagen más clara del comprobante?"
                    send_whatsapp_message(sender, request_correction_message)
                    clear_pending_confirmation(sender)
                    return {'status': 'data_denied_restarting_flow', 'id': pending_details['message_id']}
                else: # Unclear
                    clarification_message = "No entendí tu respuesta. ¿Son los datos que te mostré correctos? Por favor, responde con 'sí' o 'no'."
                    send_whatsapp_message(sender, clarification_message)
                    return {'status': 'data_confirmation_unclear', 'id': pending_details['message_id']}

            # Flujo de confirmación de tipo de imagen
            elif pending_details.get("state") == "awaiting_image_confirmation":
                prompt_text = (
                    f"Analiza la respuesta del usuario: '{incoming_text}'. El usuario está respondiendo si una imagen es un comprobante de pago. "
                    "Clasifica su respuesta. Responde ÚNICAMENTE con: AFFIRMATIVE, NEGATIVE, o UNCLEAR."
                )
                gemini_response = model.generate_content(prompt_text)
                classification = gemini_response.text.strip().lower()

                if classification == 'affirmative':
                    ocr_success = process_receipt_image(pending_details['output_path'], pending_details['original_filename'])
                    if ocr_success:
                        txt_filepath = os.path.join(os.path.dirname(pending_details['output_path']), os.path.splitext(pending_details['original_filename'])[0] + ".txt")
                        with open(txt_filepath, 'r', encoding='utf-8') as f:
                            receipt_content = f.read()
                        
                        image_for_llm = Image.open(pending_details['output_path'])
                        prompt_for_data_confirmation = (
                            f"Basado en la imagen adjunta, extrajiste estos datos:\n\n{receipt_content}\n\n"
                            "Presenta esta información al usuario y pregúntale claramente si los datos son correctos."
                        )
                        response = model.generate_content([prompt_for_data_confirmation, image_for_llm])
                        confirmation_request_message = response.text.strip()
                        send_whatsapp_message(sender, confirmation_request_message)
                        
                        pending_details['state'] = 'awaiting_data_confirmation'
                        save_pending_confirmation(sender, pending_details)
                        return {'status': 'ocr_processed_awaiting_data_confirmation', 'id': pending_details['message_id']}
                    else:
                        send_whatsapp_message(sender, "Lo siento, no pude procesar la imagen del recibo.")
                        clear_pending_confirmation(sender)
                        return {'status': 'ocr_processing_failed', 'id': pending_details['message_id']}
                elif classification == 'negative':
                    send_whatsapp_message(sender, "Entendido. No procesaré la imagen. Puedes enviar otra o hacerme una pregunta.")
                    clear_pending_confirmation(sender)
                    return {'status': 'ocr_cancelled_by_user', 'id': pending_details['message_id']}
                else: # Unclear
                    send_whatsapp_message(sender, "No entendí. Por favor, responde 'sí' si es un recibo o 'no' si no lo es.")
                    return {'status': 'image_confirmation_unclear', 'id': pending_details['message_id']}
        else:
            # Flujo de conversación general
            context_str = build_context_from_db(sender, limit=10)
            answer_text = responder(pregunta_actual=incoming_text, historial_conversacion=context_str)
            save_message(sender, "assistant", answer_text)
            send_whatsapp_message(sender, answer_text)
            return {"status": "text_message_processed_with_rag_and_context", "id": message_id}

    except Exception as e:
        print(f"Error in handle_webhook_data: {str(e)}\nTraceback: {traceback.format_exc()}")
        return {"status": "error", "message": str(e)}


def send_whatsapp_message_simulator(recipient, text):
    """
    Versión simulada de send_whatsapp_message que solo muestra en consola
    """
    print(f"[ASISTENTE] {text}")
    
    # Guardar en la base de datos normalmente
    recipient_for_db = canonicalize_jid(recipient)
    try:
        save_message(recipient_for_db, "assistant", text)
    except Exception as e:
        print(f"[DB] Error guardando mensaje enviado: {e}")
    
    return True

# Detectar si estamos en modo simulación y reemplazar la función
if os.environ.get('SIMULATION_MODE') or '--simulate' in sys.argv:
    print("=== MODO SIMULACIÓN ACTIVADO ===")
    send_whatsapp_message = send_whatsapp_message_simulator

# --- Fin código simulador ---

if __name__ == "__main__":
    # Verificar si se ejecuta en modo simulación
    if '--simulate' in sys.argv:
        print("Ejecutando en modo simulación...")
        # No iniciar el servidor Flask en modo simulación
        print("Servidor Flask no iniciado (modo simulación)")
    else:
        app.run(host="0.0.0.0", port=5001, debug=True)
        # Ejecución normal - iniciar servidor Flask