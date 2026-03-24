import os
import re
import json
import requests
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types
from huggingface_hub import login

# Load registration info from registrations module
from core.registrations import get_user_registration_info, search_user_by_name

# Setup Clients
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
GEMINI_RESPONSE_MODEL = os.getenv("GEMINI_RESPONSE_MODEL", "gemini-2.5-flash-lite")
# Legacy providers commented out for preDeployment
# GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
# OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemma-2-9b-it:free")
# MISTRAL_CHAT_MODEL = os.getenv("MISTRAL_CHAT_MODEL", "mistral-small-latest")

# Handle Hugging Face Authentication
HF_TOKEN = os.getenv("HF_TOKEN") or (os.getenv("MISTRAL_API") if str(os.getenv("MISTRAL_API")).startswith("hf_") else None)
if HF_TOKEN:
    try:
        login(token=HF_TOKEN)
    except Exception as e:
        print(f"[KNOWLEDGE] HF Login failed: {e}")

# Global variables for lazy loading
_EMBEDDER = None
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
    global _EMBEDDER, _EMBEDDINGS, _FAQS_DF, _DOCUMENTOS
    
    if _FAQS_DF is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        faq_path = os.path.join(base_dir, "faqs.xlsx")
        
        if not os.path.exists(faq_path):
            print(f"[KNOWLEDGE] Warning: {faq_path} not found.")
            return None, None, None, None
            
        try:
            _FAQS_DF = pd.read_excel(faq_path)
            _DOCUMENTOS = _FAQS_DF.to_dict(orient="records")
            descripciones = [str(d["question"] if d["type"] == "faq" else d["description"]) for d in _DOCUMENTOS]
            
            print("[KNOWLEDGE] Initializing SentenceTransformer...")
            _EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
            _EMBEDDINGS = _EMBEDDER.encode(descripciones)
            
            # SANITY CHECK: Detect zero-vectors or NaNs (common on Python 3.14/Mac)
            norms = np.linalg.norm(_EMBEDDINGS, axis=1)
            if np.any(norms == 0):
                print("[KNOWLEDGE] CRITICAL: Zero-vectors detected in embeddings! RAG effectiveness will be limited.")
            if np.any(np.isnan(_EMBEDDINGS)):
                print("[KNOWLEDGE] WARNING: NaNs detected in embeddings. Cleaning...")
                _EMBEDDINGS = np.nan_to_num(_EMBEDDINGS)
            
            print(f"[KNOWLEDGE] Knowledge base loaded with {len(_DOCUMENTOS)} items.")
        except Exception as e:
            print(f"[KNOWLEDGE] Error loading knowledge base: {e}")
            return None, None, None, None
            
    return _EMBEDDER, _EMBEDDINGS, _FAQS_DF, _DOCUMENTOS

def responder(pregunta, sender_jid=None, history=None, k=2):
    """
    Hybrid Response Logic with Multi-Provider Fallback.
    """
    # 0. Emergency check for human help keywords
    help_keywords = {"ayuda", "soporte", "humano", "persona", "agente", "asesor", "joder", "mierda"}
    pregunta_lower = pregunta.lower()
    needs_human = any(word in pregunta_lower for word in help_keywords)

    # 1. Registration Lookup
    registration_info = get_user_registration_info(sender_jid) if sender_jid else None
    if not registration_info:
        name_match = re.search(r"(?i)(?:soy|me llamo|mi nombre es|habla)\s+([a-záéíóúüñ\s]{4,40})", pregunta)
        if name_match:
            potential_name = name_match.group(1).strip()
            registration_info = search_user_by_name(potential_name)
            if registration_info:
                registration_info = f"[MATCH BY NAME: '{potential_name}'] {registration_info}"

    registration_context = ""
    if registration_info:
        registration_context = (
            "--- DATOS VERIFICADOS DEL USUARIO (REGISTRO NJUKO) ---\n"
            "Usa estos datos como verdad absoluta:\n"
            f"{registration_info}\n"
        )

    # 2. RAG Lookup
    faq_context = "--- BASE DE CONOCIMIENTOS (FAQs) ---\n"
    embedder, embeddings, df, documentos = _get_knowledge_base()
    top_faq_answer = None
    best_score = 0
    
    if embedder is not None:
        try:
            pregunta_emb = embedder.encode([pregunta])
            p_norm = np.linalg.norm(pregunta_emb)
            
            if p_norm > 0:
                # Manual Cosine Similarity for stability
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
        history_str = "--- HISTORIAL RECIENTE ---\n"
        for role, content in history:
            label = "Usuario" if role == "user" else "Asistente"
            history_str += f"{label}: {content}\n"

    prompt = (
        "¡Hola runner! Estás a un paso de tu próxima carrera de trail. ¿Quieres ver eventos cercanos, inscribirte o conocer distancias?\n"
        "Eres el asistente oficial de NaftaEC. Tienes acceso a inscripciones y FAQs.\n"
        "REGLAS:\n"
        "1. Sé amable, profesional y muy conciso.\n"
        "2. Usa los 'DATOS VERIFICADOS' si están presentes.\n"
        "3. Si no encuentras al usuario en los datos, pide su nombre completo.\n\n"
        f"{registration_context}\n"
        f"{faq_context}\n"
        f"{history_str}\n"
        f"MENSAJE ACTUAL DEL USUARIO: {pregunta}\n\n"
        "Asistente:"
    )

    # 4. Gemini Response (Streamlined for preDeployment)
    try:
        if client:
            print(f"[KNOWLEDGE] Calling Gemini: {GEMINI_RESPONSE_MODEL}")
            res = client.models.generate_content(
                model=GEMINI_RESPONSE_MODEL, 
                contents=prompt, 
                config=types.GenerateContentConfig(temperature=0.7)
            )
            return res.text.strip()
    except Exception as e:
        print(f"[KNOWLEDGE] Gemini failed: {e}")

    # Legacy Multi-Provider Fallback (Commented out for preDeployment)
    """
    models_to_try = [
        ("gemini", GEMINI_RESPONSE_MODEL),
        ("groq", GROQ_MODEL),
        ("openrouter", OPENROUTER_MODEL),
        ("mistral", MISTRAL_CHAT_MODEL)
    ]
    # ... rest of fallback logic preserved for easy restoration
    """

    # 5. Smart Fallback (if all AI providers failed)
    if needs_human:
        return "He notado que podrías necesitar ayuda de un representante. ¿Deseas que te ponga en contacto con un asesor humano de NaftaEC?"

    if registration_info:
        return f"Hola runner, pude verificar tus datos: {registration_info} ¿En qué más puedo ayudarte con tu inscripción?"

    if top_faq_answer and best_score > 0.6:
        return top_faq_answer

    return "Lo siento, en este momento tengo mucha demanda y no puedo procesar tu solicitud exacta. ¿Deseas consultar sobre inscripciones, pagos o hablar con un asesor?"
