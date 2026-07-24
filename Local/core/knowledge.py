import os
import re
import json
import requests
import pandas as pd
import numpy as np
from google import genai
from google.genai import types

# Load registration info from registrations module
from core.registrations import get_user_registration_info, search_user_by_name, search_registrations_by_cedula

# Robust environment loading
from dotenv import load_dotenv
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(base_dir, ".env"), override=True)

# Setup Clients
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
GEMINI_RESPONSE_MODEL = os.getenv("GEMINI_RESPONSE_MODEL", "gemini-2.5-flash")
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")

# Global variables for lazy loading
_EMBEDDINGS = None
_FAQS_DF = None
_DOCUMENTOS = None
_PROMPT_TEMPLATE = None

def get_mistral_api_key():
    return os.getenv("MISTRAL_API_KEY") or os.getenv("MISTRAL_API")

def call_mistral_chat(prompt, model_id, mistral_key):
    res = requests.post(
        "https://api.mistral.ai/v1/chat/completions",
        headers={"Authorization": f"Bearer {mistral_key}", "Content-Type": "application/json"},
        json={"model": model_id, "messages": [{"role": "user", "content": prompt}], "temperature": 0.7},
        timeout=30
    )
    res.raise_for_status()
    return res.json()["choices"][0]["message"]["content"].strip()

FAQ_DRIVE_URL = "https://docs.google.com/spreadsheets/d/1eGpP94CwTY4LlLz3AMZR2NyYp5kF3HHe/export?format=xlsx"
PROMPT_DRIVE_URL = "https://docs.google.com/document/d/1W5VMpeS7210SBg-KlTtJKkZp_naw-c4OlR6SVEzRaRM/export?format=txt"

def download_faqs():
    persistent_dir = os.getenv("PERSISTENT_STORAGE_PATH", "/var/lib/naftaec")
    # If not on render (no PERSISTENT_STORAGE_PATH), use the local directory to avoid permission issues
    faq_path = os.path.join(persistent_dir, "faqs_drive.xlsx") if os.getenv("PERSISTENT_STORAGE_PATH") else os.path.join(base_dir, "faqs_drive.xlsx")
    try:
        print(f"[KNOWLEDGE] Descargando FAQs desde Google Drive...")
        res = requests.get(FAQ_DRIVE_URL, timeout=15)
        res.raise_for_status()
        with open(faq_path, "wb") as f:
            f.write(res.content)
        print(f"[KNOWLEDGE] FAQs descargadas exitosamente en {faq_path}")
        return faq_path
    except Exception as e:
        print(f"[KNOWLEDGE] Error descargando FAQs: {e}")
        return None

def download_prompt():
    persistent_dir = os.getenv("PERSISTENT_STORAGE_PATH", "/var/lib/naftaec")
    prompt_path = os.path.join(persistent_dir, "prompt_drive.txt") if os.getenv("PERSISTENT_STORAGE_PATH") else os.path.join(base_dir, "prompt_drive.txt")
    try:
        print(f"[KNOWLEDGE] Descargando Prompt desde Google Drive...")
        res = requests.get(PROMPT_DRIVE_URL, timeout=15)
        res.raise_for_status()
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write(res.text.strip('\ufeff'))
        print(f"[KNOWLEDGE] Prompt descargado exitosamente en {prompt_path}")
        return prompt_path
    except Exception as e:
        print(f"[KNOWLEDGE] Error descargando Prompt: {e}")
        return None

def reset_knowledge_base():
    global _EMBEDDINGS, _FAQS_DF, _DOCUMENTOS, _PROMPT_TEMPLATE
    print("[KNOWLEDGE] Reseteando base de conocimientos y prompt...")
    _EMBEDDINGS = None
    _FAQS_DF = None
    _DOCUMENTOS = None
    _PROMPT_TEMPLATE = None
    download_faqs()
    download_prompt()
    _get_knowledge_base()
    return True

def _get_knowledge_base():
    global _EMBEDDINGS, _FAQS_DF, _DOCUMENTOS, _PROMPT_TEMPLATE
    
    if _PROMPT_TEMPLATE is None:
        persistent_dir = os.getenv("PERSISTENT_STORAGE_PATH", "/var/lib/naftaec")
        drive_prompt_path = os.path.join(persistent_dir, "prompt_drive.txt") if os.getenv("PERSISTENT_STORAGE_PATH") else os.path.join(base_dir, "prompt_drive.txt")
        if not os.path.exists(drive_prompt_path):
            download_prompt()
        if os.path.exists(drive_prompt_path):
            with open(drive_prompt_path, "r", encoding="utf-8") as f:
                _PROMPT_TEMPLATE = f.read()
        else:
            print("[KNOWLEDGE] Warning: No prompt template available.")
            _PROMPT_TEMPLATE = ""

    if _EMBEDDINGS is None:
        persistent_dir = os.getenv("PERSISTENT_STORAGE_PATH", "/var/lib/naftaec")
        drive_faq_path = os.path.join(persistent_dir, "faqs_drive.xlsx") if os.getenv("PERSISTENT_STORAGE_PATH") else os.path.join(base_dir, "faqs_drive.xlsx")
        
        if not os.path.exists(drive_faq_path):
            download_faqs()
            
        faq_path = drive_faq_path if os.path.exists(drive_faq_path) else "/app/faqs.xlsx"
        
        if not os.path.exists(faq_path):
            print(f"[KNOWLEDGE] Warning: {faq_path} not found.")
            return None, None, None
            
        try:
            _FAQS_DF = pd.read_excel(faq_path)
            _DOCUMENTOS = _FAQS_DF.to_dict(orient="records")
            descripciones = [str(d["question"] if d["type"] == "faq" else d["description"]) for d in _DOCUMENTOS]
            
            if client:
                print(f"[KNOWLEDGE] Generando embeddings en lotes para {len(descripciones)} filas...")
                all_embeddings = []
                # PROCESAR EN LOTES DE 100 (Límite estricto de Google)
                for i in range(0, len(descripciones), 100):
                    lote = descripciones[i:i + 100]
                    res = client.models.embed_content(
                        model=GEMINI_EMBEDDING_MODEL,
                        contents=lote,
                        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
                    )
                    all_embeddings.extend([e.values for e in res.embeddings])
                
                _EMBEDDINGS = np.array(all_embeddings)
                print(f"[KNOWLEDGE] ¡Éxito! {len(_EMBEDDINGS)} vectores generados.")
            else:
                print("[KNOWLEDGE] Error: Gemini client not initialized.")
        except Exception as e:
            print(f"[KNOWLEDGE] Error crítico cargando base: {e}")
            return None, None, None
            
    return _EMBEDDINGS, _FAQS_DF, _DOCUMENTOS

def responder(pregunta, sender_jid=None, history=None, k=2):
    """
    Hybrid Response Logic with Multi-Provider Fallback.
    """
    # --- 1. DETECCIÓN DE AYUDA HUMANA ---
    help_keywords = {"ayuda", "soporte", "humano", "persona", "agente", "asesor", "joder", "mierda", "estafa", "robo", "fraude"}
    pregunta_lower = pregunta.lower()
    needs_human = any(word in pregunta_lower for word in help_keywords)

    # --- 2. BÚSQUEDA DE REGISTRO (NJUKO) ---
    registration_info = None
    registration_info = get_user_registration_info(sender_jid) if sender_jid else None
    
    # Búsqueda por Cédula en el texto
    cedula_match = re.search(r"\b(\d{7,10})\b", pregunta)
    if not registration_info and cedula_match:
        potential_cedula = cedula_match.group(1)
        registration_info = search_registrations_by_cedula(potential_cedula)
        if registration_info:
            registration_info = f"[DATOS ENCONTRADOS POR CÉDULA {potential_cedula}]:\n{registration_info}"

    # Búsqueda por Nombre
    if not registration_info:
        name_match = re.search(r"(?i)(?:soy|me llamo|mi nombre es|habla|inscripción de)\s+([a-záéíóúüñ\s]{4,40})", pregunta)
        if name_match:
            potential_name = name_match.group(1).strip()
            registration_info = search_user_by_name(potential_name)
            if registration_info:
                registration_info = f"[COINCIDENCIA POR NOMBRE '{potential_name}']:\n{registration_info}"

    registration_context = "NO se encontró información de registro para este usuario."
    if registration_info:
        registration_context = f"SÍ se encontró información de registro en Njuko:\n{registration_info}\n"

    # --- 3. RAG LOOKUP (BÚSQUEDA EN EXCEL) ---
    faq_context = "--- INFORMACIÓN DE FAQs ---\n"
    embeddings, df, documentos = _get_knowledge_base()
    top_faq_answer = None
    best_score = 0
    
    if embeddings is not None and client is not None:
        try:
            res_query = client.models.embed_content(
                model=GEMINI_EMBEDDING_MODEL,
                contents=pregunta,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
            )
            pregunta_emb = np.array(res_query.embeddings[0].values)
            p_norm = np.linalg.norm(pregunta_emb)
            
            if p_norm > 0:
                e_norms = np.linalg.norm(embeddings, axis=1)
                e_norms[e_norms == 0] = 1.0 
                similitudes = np.dot(embeddings, pregunta_emb.T).flatten() / (e_norms * p_norm)
                similitudes = np.nan_to_num(similitudes)
                
                top_k_idx = similitudes.argsort()[-k:][::-1]
                found_faq = False
                for idx in top_k_idx:
                    score = similitudes[idx]
                    if score > 0.4: 
                        item = documentos[idx]
                        faq_context += f"Pregunta: {item['question']}\nRespuesta: {item['answer']}\n"
                        found_faq = True
                        if top_faq_answer is None:
                            top_faq_answer = str(item.get("answer", "")).strip()
                            best_score = score
                if not found_faq:
                    faq_context += "No hay una respuesta exacta en las FAQs.\n"
        except Exception as e:
            print(f"[KNOWLEDGE] RAG error: {e}")
            faq_context += "Error buscando en FAQs.\n"
    else:
        faq_context += "Base de conocimientos no disponible.\n"

    # --- 4. HISTORIAL ---
    history_str = ""
    if history:
        history_str = "--- HISTORIAL DE CHAT ---\n"
        for role, content in history[-10:]: # Tomamos los últimos 10 para no saturar
            label = "Usuario" if role == "user" else "Asistente"
            history_str += f"{label}: {content}\n"

    # --- 5. CONSTRUCCIÓN DEL PROMPT ---
    if not _PROMPT_TEMPLATE:
        print("[KNOWLEDGE] Warning: Fallback a prompt vacío.")
        prompt = f"{faq_context}\n{history_str}\n{registration_context}\nMENSAJE: {pregunta}"
    else:
        prompt = _PROMPT_TEMPLATE.replace("[FAQ_CONTEXT]", faq_context[:1500])\
                                 .replace("[HISTORY]", history_str)\
                                 .replace("[REGISTRATION_CONTEXT]", registration_context)\
                                 .replace("[USER_MESSAGE]", pregunta)
    # --- 6. GENERACIÓN DE RESPUESTA CON GEMINI ---
    try:
        if client:
            print(f"[KNOWLEDGE] Calling Gemini: {GEMINI_RESPONSE_MODEL}")
            res = client.models.generate_content(
                model=GEMINI_RESPONSE_MODEL, 
                contents=prompt, 
                config=types.GenerateContentConfig(temperature=0.7)
            )
            respuesta_final = res.text.strip()
            
            # Debug limitado para que WhatsApp no rechace el mensaje (máx 4096 caracteres)
            
            #debug_info = f"\n\n--- DEBUG CONTEXT ---\n{prompt}"[:800]
            return f"{respuesta_final}"
            
    except Exception as e:
        print(f"[KNOWLEDGE] Gemini failed: {e}")

    # --- 7. FALLBACKS (Si Gemini falla o casos especiales) ---
    if needs_human:
        return "[HELP] He notado que podrías necesitar ayuda de un representante. ¿Deseas que te ponga en contacto con un asesor humano?"

    if registration_info:
        return f"Hola runner, pude verificar tus datos:\n{registration_info}\n\n¿En qué más puedo ayudarte?"

    if top_faq_answer and best_score > 0.6:
        return top_faq_answer

    return "Lo siento, tuve un error técnico. Si deseas consultar tu inscripción, por favor envíame tu número de cédula."

def summarize_history(history):
    if not client or not history:
        return "No hay contexto de conversación disponible."
    
    history_str = ""
    for role, content in history:
        label = "Usuario" if role == "user" else "Asistente"
        history_str += f"{label}: {content}\n"
        
    prompt = (
        "Por favor, genera un resumen muy breve y conciso (máximo 3 oraciones) de la siguiente conversación "
        "entre un usuario y el asistente virtual de NaftaEC.\n"
        "El objetivo es que un agente humano pueda leer este resumen y entender rápidamente qué estaba "
        "intentando hacer el usuario y por qué pidió ayuda humana.\n\n"
        "Conversación:\n"
        f"{history_str}"
    )
    
    try:
        res = client.models.generate_content(
            model=GEMINI_RESPONSE_MODEL, 
            contents=prompt, 
            config=types.GenerateContentConfig(temperature=0.3)
        )
        return res.text.strip()
    except Exception as e:
        print(f"[KNOWLEDGE] Summarization failed: {e}")
        return "No se pudo generar el resumen debido a un error técnico."
