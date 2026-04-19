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
GEMINI_RESPONSE_MODEL = os.getenv("GEMINI_RESPONSE_MODEL", "gemini-2.5-flash-lite")
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")

# Global variables for lazy loading
_EMBEDDINGS = None
_FAQS_DF = None
_DOCUMENTOS = None

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

def _get_knowledge_base():
    global _EMBEDDINGS, _FAQS_DF, _DOCUMENTOS
    
    # Intentamos cargar si los embeddings están vacíos
    if _EMBEDDINGS is None:
        faq_path = "/app/faqs.xlsx"
        
        try:
            # 1. Cargar el DataFrame
            _FAQS_DF = pd.read_excel(faq_path)
            _DOCUMENTOS = _FAQS_DF.to_dict(orient="records")
            descripciones = [str(d["question"] if d["type"] == "faq" else d["description"]) for d in _DOCUMENTOS]
            
            if client:
                print(f"[KNOWLEDGE] Generando embeddings para {len(descripciones)} filas...")
                
                # 2. Llamada a Gemini (Aquí es donde sospecho que falla)
                # OJO: Asegúrate de que el modelo sea el correcto para tu región/API
                res = client.models.embed_content(
                    model=GEMINI_EMBEDDING_MODEL,
                    contents=descripciones,
                    config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
                )
                
                _EMBEDDINGS = np.array([e.values for e in res.embeddings])
                print(f"[KNOWLEDGE] Base cargada exitosamente: {len(_DOCUMENTOS)} items.")
            else:
                return None, None, None

        except Exception as e:
            # 3. CAPTURA DE ERROR CRÍTICO
            # Si falla, imprimimos el error para verlo en el WhatsApp de debug
            error_msg = f"ERROR EN EMBEDDINGS: {str(e)}"
            print(f"[KNOWLEDGE] {error_msg}")
            
            # IMPORTANTE: Reseteamos para que no se quede bloqueado en el error
            _FAQS_DF = None 
            _DOCUMENTOS = None
            
            # Devolvemos el error en lugar de None para que lo veas en el chat
            return error_msg, None, None
            
    return _EMBEDDINGS, _FAQS_DF, _DOCUMENTOS

def responder(pregunta, sender_jid=None, history=None, k=2):
    """
    Hybrid Response Logic with Multi-Provider Fallback.
    """
    
    try:
        if client:
            print(f"[KNOWLEDGE] Calling Gemini: {GEMINI_RESPONSE_MODEL}")
            res = client.models.generate_content(
                model=GEMINI_RESPONSE_MODEL, 
                contents=prompt, 
                config=types.GenerateContentConfig(temperature=0.7)
            )
            
            respuesta_final = res.text.strip()
            
            # --- BLOQUE DE DEBUG INMEDIATO ---
            # Esto añadirá el contexto exacto al final de cada mensaje de WhatsApp
            debug_info = f"\n\n--- DEBUG CONTEXT ---\n{prompt}"
            return f"{respuesta_final}{debug_info}"
            # ---------------------------------

    except Exception as e:
        print(f"[KNOWLEDGE] Gemini failed: {e}")

    
    # 0. Emergency check for human help keywords
    help_keywords = {"ayuda", "soporte", "humano", "persona", "agente", "asesor", "joder", "mierda", "estafa", "robo", "fraude"}
    pregunta_lower = pregunta.lower()
    needs_human = any(word in pregunta_lower for word in help_keywords)

    # 1. Registration Lookup
    registration_info = None
    
    # Check by Phone (Automatic)
    registration_info = get_user_registration_info(sender_jid) if sender_jid else None
    
    # Check by Cedula in text
    cedula_match = re.search(r"\b(\d{7,10})\b", pregunta)
    if not registration_info and cedula_match:
        potential_cedula = cedula_match.group(1)
        registration_info = search_registrations_by_cedula(potential_cedula)
        if registration_info:
            registration_info = f"[DATOS ENCONTRADOS POR CÉDULA {potential_cedula}]:\n{registration_info}"

    # Check by Name (Conservative)
    if not registration_info:
        name_match = re.search(r"(?i)(?:soy|me llamo|mi nombre es|habla|inscripción de)\s+([a-záéíóúüñ\s]{4,40})", pregunta)
        if name_match:
            potential_name = name_match.group(1).strip()
            registration_info = search_user_by_name(potential_name)
            if registration_info:
                registration_info = f"[COINCIDENCIA POR NOMBRE '{potential_name}']:\n{registration_info}"

    registration_context = "NO se encontró información de registro para este usuario."
    if registration_info:
        registration_context = (
            "SÍ se encontró información de registro en Njuko:\n"
            f"{registration_info}\n"
        )

    # 2. RAG Lookup
    faq_context = "--- INFORMACIÓN DE FAQs ---\n"
    embeddings, df, documentos = _get_knowledge_base()
    top_faq_answer = None
    best_score = 0
    
    if embeddings is not None and client is not None:
        try:
            # Generate query embedding
            res_query = client.models.embed_content(
                model=GEMINI_EMBEDDING_MODEL,
                contents=pregunta,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
            )
            pregunta_emb = np.array(res_query.embeddings[0].values)
            p_norm = np.linalg.norm(pregunta_emb)
            
            if p_norm > 0:
                # Manual Cosine Similarity
                e_norms = np.linalg.norm(embeddings, axis=1)
                e_norms[e_norms == 0] = 1.0 
                
                similitudes = np.dot(embeddings, pregunta_emb.T).flatten() / (e_norms * p_norm)
                similitudes = np.nan_to_num(similitudes)
                
                top_k_idx = similitudes.argsort()[-k:][::-1]
                found_faq = False
                for idx in top_k_idx:
                    score = similitudes[idx]
                    if score > 0.6: 
                        item = documentos[idx]
                        if item["type"] == "faq":
                            faq_context += f"Pregunta: {item['question']}\nRespuesta: {item['answer']}\n"
                            found_faq = True
                            if top_faq_answer is None:
                                top_faq_answer = str(item.get("answer", "")).strip() or None
                                best_score = score
                if not found_faq:
                    faq_context += "No hay una respuesta exacta en las FAQs.\n"
            else:
                faq_context += "No se pudo procesar la consulta (vector nulo).\n"
        except Exception as e:
            print(f"[KNOWLEDGE] RAG Similarity error: {e}")
            faq_context += "Error buscando en FAQs.\n"
    else:
        faq_context += "Base de conocimientos no disponible.\n"

    # 3. History Assembly
    history_str = ""
    if history:
        history_str = "--- HISTORIAL DE CHAT ---\n"
        for role, content in history:
            label = "Usuario" if role == "user" else "Asistente"
            history_str += f"{label}: {content}\n"

    prompt = (
        "Eres el asistente virtual de NaftaEC. Tu misión es ayudar a runners.\n\n"
        "REGLAS DE ORO:\n"
        "1. Si tienes 'INFORMACIÓN DE REGISTRO EN NJUKO', úsala para responder consultas sobre el estado de inscripción. ¡No inventes ni pidas datos que ya tienes!\n"
        "2. Si el usuario hace una pregunta general, saluda, o tiene dudas sobre el evento, responde usando ÚNICAMENTE la 'INFORMACIÓN DE FAQs'. No pidas la cédula para preguntas generales.\n"
        "3. SOLO si el usuario pregunta explícitamente por su estado de inscripción, cupo, o registro, Y el contexto dice 'NO se encontró', entonces dile que no lo hallas y pide su número de cédula.\n"
        "4. Sé extremadamente conciso, amable y profesional. Responde siempre en Español.\n\n"
        f"{faq_context}\n"
        f"{history_str}\n"
        "--- CONTEXTO DE REGISTRO ACTUAL ---\n"
        f"{registration_context}\n"
        "------------------------------------\n\n"
        f"MENSAJE DEL USUARIO: {pregunta}\n\n"
        "Instrucción final: Evalúa la intención del mensaje del usuario. Si es una duda general o saludo, responde naturalmente. Si está intentando consultar su registro y este 'NO se encontró', pídele su cédula."
    )

    # 4. Gemini Response
    try:
        if client:
            print(f"[KNOWLEDGE] Calling Gemini: {GEMINI_RESPONSE_MODEL}")
            res = client.models.generate_content(
                model=GEMINI_RESPONSE_MODEL, 
                contents=prompt, 
                config=types.GenerateContentConfig(temperature=0.7)
            )
            
            respuesta_final = res.text.strip()
            
            # --- BLOQUE DE DEBUG INMEDIATO ---
            # Esto añadirá el contexto exacto al final de cada mensaje de WhatsApp
            debug_info = f"\n\n--- DEBUG CONTEXT ---\n{prompt}"
            return f"{respuesta_final}{debug_info}"
            # ---------------------------------

    except Exception as e:
        print(f"[KNOWLEDGE] Gemini failed: {e}")

    # 5. Smart Fallback
    if needs_human:
        return "He notado que podrías necesitar ayuda de un representante. ¿Deseas que te ponga en contacto con un asesor humano de NaftaEC?"

    if registration_info:
        return f"Hola runner, pude verificar tus datos:\n{registration_info}\n\n¿En qué más puedo ayudarte?"

    if top_faq_answer and best_score > 0.6:
        return top_faq_answer

    return "Lo siento, en este momento tengo mucha demanda. Si deseas consultar tu inscripción, por favor envíame tu número de cédula."
