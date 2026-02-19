import os
import re
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai
from google.genai import types
from huggingface_hub import login

# Load registration info from chronotrack module
from core.chronotrack import get_user_registration_info, search_user_by_name

# Setup Gemini Client
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

# Handle Hugging Face Authentication
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    try:
        login(token=HF_TOKEN)
    except Exception as e:
        print(f"HF Login failed: {e}")

# Load FAQs
df = pd.read_excel("faqs.xlsx")
documentos = df.to_dict(orient="records")
descripciones = [d["question"] if d["type"] == "faq" else d["description"] for d in documentos]

# Embedder for FAQ RAG
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedder.encode(descripciones)

def responder(pregunta, sender_jid=None, history=None, k=2):
    """
    Hybrid Response Logic:
    1. Deterministic Lookup (Registration Report) - HIGHEST PRIORITY
    2. Fuzzy Name Lookup (if user provided a name in their current question)
    3. RAG (FAQs) - SECONDARY PRIORITY
    4. LLM (History & Context)
    """
    
    # 1. Deterministic Lookup by Phone (WhatsApp ID)
    registration_info = get_user_registration_info(sender_jid) if sender_jid else None
    if registration_info:
        print(f"[HYBRID] Phone match found for {sender_jid}")
    
    # 2. Heuristic: If no phone match, see if the user mentioned a name
    if not registration_info:
        # Improved name extraction regex
        name_match = re.search(r"(?i)(?:soy|me llamo|mi nombre es|habla)\s+([a-záéíóúüñ\s]{4,40})", pregunta)
        if name_match:
            potential_name = name_match.group(1).strip()
            print(f"[HYBRID] Searching for name: '{potential_name}'")
            registration_info = search_user_by_name(potential_name)
            if registration_info:
                print(f"[HYBRID] Name match found for '{potential_name}'")
                registration_info = f"[MATCH BY NAME: '{potential_name}'] {registration_info}"
            else:
                print(f"[HYBRID] No name match found for '{potential_name}'")

    registration_context = ""
    if registration_info:
        registration_context = (
            "--- DATOS VERIFICADOS DEL USUARIO (CHRONOTRACK) ---\n"
            "Usa estos datos como verdad absoluta para este usuario específico:\n"
            f"{registration_info}\n"
        )
    else:
        print("[HYBRID] No registration info found for this user/message.")

    # 3. RAG Lookup from FAQs
    pregunta_emb = embedder.encode([pregunta])
    similitudes = cosine_similarity(pregunta_emb, embeddings)[0]
    top_k_idx = similitudes.argsort()[-k:][::-1]

    faq_context = "--- BASE DE CONOCIMIENTOS (FAQs) ---\n"
    found_faq = False
    for idx in top_k_idx:
        if similitudes[idx] > 0.45: 
            item = documentos[idx]
            if item["type"] == "faq":
                faq_context += f"Pregunta: {item['question']}\nRespuesta: {item['answer']}\n"
                found_faq = True
    
    if not found_faq:
        faq_context += "No hay una respuesta exacta en las FAQs. Responde con amabilidad general basándote en la marca NaftaEC.\n"

    # 4. History Assembly
    history_str = ""
    if history:
        history_str = "--- HISTORIAL RECIENTE ---\n"
        for role, content in history:
            label = "Usuario" if role == "user" else "Asistente"
            history_str += f"{label}: {content}\n"

    prompt = (
        "Eres el asistente oficial de NaftaEC. Tienes acceso a una base de datos local de inscripciones que se te proporciona en la sección 'DATOS VERIFICADOS'.\n"
        "REGLAS CRÍTICAS:\n"
        "1. NUNCA digas que no tienes acceso a la base de datos o que no puedes verificar registros en tiempo real. Tienes los datos aquí mismo.\n"
        "2. Si el usuario pregunta por su inscripción y la sección 'DATOS VERIFICADOS' está vacía o no coincide, responde exactamente: 'No encuentro un registro con esos datos en nuestro sistema.' y solicita su nombre completo o comprobante.\n"
        "3. Si los 'DATOS VERIFICADOS' están presentes, úsalos como la única verdad para confirmar su estado (ej. 'Veo que estás inscrito en la 10K').\n"
        "4. Para saludos o preguntas generales no relacionadas con el estado de inscripción, usa las FAQs sin mencionar registros.\n"
        "5. Sé amable, profesional y muy conciso.\n\n"
        f"{registration_context}\n"
        f"{faq_context}\n"
        f"{history_str}\n"
        f"MENSAJE ACTUAL DEL USUARIO: {pregunta}\n\n"
        "Asistente:"
    )
    
    try:
        response = client.models.generate_content(
            model="gemini-flash-lite-latest",
            contents=prompt,
            config=types.GenerateContentConfig(temperature=1.0)
        )
        return response.text.strip()
    except Exception as e:
        print(f"Error in hybrid responder: {e}")
        return "Lo siento, tuve un problema al procesar tu respuesta. Por favor, intenta de nuevo."
