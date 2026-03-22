import os
import sqlite3
import json
import sys
import time
import requests
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Ensure we are in the Local directory for imports and relative paths
SIM_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_DIR = os.path.dirname(SIM_DIR)
sys.path.append(LOCAL_DIR)

from core.knowledge import responder
from core.database import init_db as init_core_db

# Load environment variables
load_dotenv(os.path.join(LOCAL_DIR, ".env"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY") or os.getenv("MISTRAL_API")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
GEMINI_SIM_MODEL = os.getenv("GEMINI_SIM_MODEL", "gemini-2.5-flash-lite")

client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

SIM_DB_PATH = os.path.join(SIM_DIR, "test_dataset.db")
LOCAL_FALLBACK_PERSONAS = [
    {
        "name": "Carlos Ruiz",
        "personality": "Paciente pero confundido",
        "proficiency": 0.6,
        "goal": "Saber si su inscripción para la 10K está confirmada."
    },
    {
        "name": "Elena Gomez",
        "personality": "Muy apurada y directa",
        "proficiency": 0.9,
        "goal": "Preguntar por el precio de la 21K y descuentos para grupos."
    },
    {
        "name": "Ricardo Soria",
        "personality": "Frustrado y sarcastico",
        "proficiency": 0.4,
        "goal": "Quejarse de que pago y no recibio su codigo."
    },
]


def init_sim_db():
    conn = sqlite3.connect(SIM_DB_PATH)
    try:
        with conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS personas (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    personality TEXT,
                    proficiency REAL,
                    goal TEXT
                );
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    persona_id INTEGER,
                    start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                    end_time DATETIME,
                    evaluation_score INTEGER,
                    evaluation_reason TEXT,
                    FOREIGN KEY (persona_id) REFERENCES personas(id)
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id INTEGER,
                    role TEXT,
                    content TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                );
                """
            )
    finally:
        conn.close()


def reset_sim_db():
    conn = sqlite3.connect(SIM_DB_PATH)
    try:
        with conn:
            conn.execute("DELETE FROM messages")
            conn.execute("DELETE FROM conversations")
            conn.execute("DELETE FROM personas")
    finally:
        conn.close()


def clean_json_text(text):
    if not text:
        return ""
    text = text.strip()
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0]
    return text.strip()


def build_local_personas(count):
    personas = []
    for i in range(count):
        base = LOCAL_FALLBACK_PERSONAS[i % len(LOCAL_FALLBACK_PERSONAS)].copy()
        if i >= len(LOCAL_FALLBACK_PERSONAS):
            base["name"] = f"{base['name']} {i+1}"
        personas.append(base)
    return personas


def local_user_reply(persona, history):
    goal = str(persona.get("goal", "")).lower()
    templates = [
        f"Hola, soy {persona.get('name', 'usuario')}. {persona.get('goal', '')}",
        "Me puedes confirmar el estado por favor?",
        "Que pasos sigo ahora?",
        "Perfecto, gracias.",
        "Quedo atento."
    ]
    if "descuento" in goal:
        templates = [
            "Hola, quiero saber el precio y descuento por grupo.",
            "Somos varias personas, que descuento aplica?",
            "Hasta cuando aplica la promocion?",
            "Como hacemos el pago grupal?",
            "Gracias por la ayuda."
        ]
    elif "codigo" in goal or "pago" in goal:
        templates = [
            "Hola, pague y no recibi mi codigo.",
            "Sigo esperando respuesta.",
            "Necesito solucion hoy.",
            "Me pueden escalar con soporte humano?",
            "Gracias, quedo pendiente."
        ]

    user_turns = len([m for m in history if m["role"] == "user"])
    return templates[min(user_turns, len(templates) - 1)]


def local_evaluate_conversation(history, goal):
    assistant_msgs = [m["content"] for m in history if m["role"] == "assistant"]
    if not assistant_msgs:
        return {"score": 1, "reason": "No hubo respuestas del asistente."}
    error_hits = sum(
        1 for msg in assistant_msgs
        if any(token in msg.lower() for token in ["error_technical", "tuve un problema", "alta demanda"])
    )
    score = max(1, 8 - (error_hits * 2))
    if error_hits == 0:
        score = min(10, score + 2)
    return {"score": score, "reason": f"Evaluacion local fallback. Respuestas con error: {error_hits}."}


def call_mistral_chat(prompt, model_id, temperature=1.0):
    res = requests.post(
        "https://api.mistral.ai/v1/chat/completions",
        headers={"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"},
        json={"model": model_id, "messages": [{"role": "user", "content": prompt}], "temperature": temperature},
        timeout=30,
    )
    res.raise_for_status()
    return res.json()["choices"][0]["message"]["content"]


def call_ai_fallback(prompt, response_mime_type="text/plain", temperature=1.0):
    """Multi-model fallback for simulation tasks."""
    models_to_try = [
        ("groq", "llama-3.1-8b-instant"),
        ("gemini", GEMINI_SIM_MODEL),
        ("openrouter", "meta-llama/llama-3.1-8b-instruct:free"),
        ("mistral", "mistral-small-latest"),
    ]

    for provider, model_id in models_to_try:
        try:
            if provider == "groq" and GROQ_API_KEY:
                res = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                    json={"model": model_id, "messages": [{"role": "user", "content": prompt}], "temperature": temperature},
                    timeout=30,
                )
                res.raise_for_status()
                return res.json()["choices"][0]["message"]["content"]

            if provider == "gemini" and client:
                config = types.GenerateContentConfig(temperature=temperature)
                if response_mime_type == "application/json":
                    config.response_mime_type = "application/json"
                res = client.models.generate_content(model=model_id, contents=prompt, config=config)
                return res.text

            if provider == "openrouter" and OPENROUTER_API_KEY:
                res = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
                    json={"model": model_id, "messages": [{"role": "user", "content": prompt}]},
                    timeout=30,
                )
                res.raise_for_status()
                return res.json()["choices"][0]["message"]["content"]

            if provider == "mistral" and MISTRAL_API_KEY:
                return call_mistral_chat(prompt, model_id, temperature=temperature)

        except Exception as e:
            if any(err in str(e).lower() for err in ["429", "quota", "resource_exhausted"]):
                print(f"  [SIM-FALLBACK] {provider}/{model_id} hit limit. Trying next...")
                continue
            print(f"  [SIM-ERROR] {provider}/{model_id} failed: {e}")

    return None


def generate_personas(count=5):
    print(f"Generando {count} personas de prueba...")
    prompt = (
        f"Genera una lista de {count} personas (usuarios ficticios) en JSON para probar un chatbot de trail running (NaftaEC).\n"
        "Cada objeto debe tener: name (Nombre espanol), personality (ej: sarcastico, apurado), proficiency (0.0 a 1.0), goal (que quiere lograr).\n"
        "Responde UNICAMENTE con el JSON array."
    )

    personas = None
    res_text = call_ai_fallback(prompt, response_mime_type="application/json")
    if res_text:
        try:
            personas = json.loads(clean_json_text(res_text))
        except Exception as e:
            print(f"[SIM] Error parsing personas, using fallback: {e}")

    if not personas:
        personas = build_local_personas(count)
        print(f"[SIM] Using {len(personas)} fallback personas.")

    conn = sqlite3.connect(SIM_DB_PATH)
    try:
        with conn:
            for p in personas:
                conn.execute(
                    "INSERT INTO personas (name, personality, proficiency, goal) VALUES (?, ?, ?, ?)",
                    (p["name"], p["personality"], p["proficiency"], p["goal"]),
                )
    finally:
        conn.close()

    return personas


def get_user_reply(persona, history):
    history_str = "\n".join([f"{m['role']}: {m['content']}" for m in history])
    prompt = (
        f"Eres {persona['name']} (Personalidad: {persona['personality']}, Competencia: {persona['proficiency']}/1.0).\n"
        f"Tu objetivo: {persona['goal']}.\n"
        f"HISTORIAL:\n{history_str}\n\n"
        "Escribe tu siguiente mensaje corto al bot. RESPONDE SOLO EL TEXTO DEL MENSAJE."
    )
    reply = call_ai_fallback(prompt, temperature=1.1)
    if not reply:
        return local_user_reply(persona, history)
    return reply.strip() or local_user_reply(persona, history)


def evaluate_conversation(history, goal):
    history_str = "\n".join([f"{m['role']}: {m['content']}" for m in history])
    prompt = (
        f"Analiza la conversacion de NaftaEC. Objetivo del usuario: {goal}\n\n"
        f"CHAT:\n{history_str}\n\n"
        "El bot ayudo al usuario? Responde en JSON con: score (1-10) y reason (explicacion breve en espanol)."
    )
    res_text = call_ai_fallback(prompt, response_mime_type="application/json")
    if not res_text:
        return local_evaluate_conversation(history, goal)
    try:
        return json.loads(clean_json_text(res_text))
    except Exception:
        return local_evaluate_conversation(history, goal)


def run_simulation(num_personas=5, convs_per_persona=10):
    init_sim_db()
    init_core_db()
    reset_sim_db()
    generate_personas(num_personas)

    conn = sqlite3.connect(SIM_DB_PATH)
    conn.row_factory = sqlite3.Row

    try:
        personas = conn.execute("SELECT * FROM personas ORDER BY id ASC").fetchall()

        for persona in personas:
            p_dict = dict(persona)
            print(f"\n>>> Simulando para {p_dict['name']}")

            for c_idx in range(convs_per_persona):
                print(f"  Conv {c_idx+1}/{convs_per_persona}...")
                with conn:
                    cursor = conn.execute("INSERT INTO conversations (persona_id) VALUES (?)", (p_dict["id"],))
                    conv_id = cursor.lastrowid

                history = []
                sender_id = f"sim_{p_dict['id']}_{c_idx}@s.whatsapp.net"

                for m_idx in range(5):
                    user_text = get_user_reply(p_dict, history)
                    history.append({"role": "user", "content": user_text})
                    with conn:
                        conn.execute(
                            "INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)",
                            (conv_id, "user", user_text),
                        )

                    bot_resp = responder(
                        user_text,
                        sender_jid=sender_id,
                        history=[(m["role"], m["content"]) for m in history[:-1]],
                    )
                    bot_resp = (bot_resp or "").strip() or "ERROR_TECHNICAL: respuesta vacia"

                    history.append({"role": "assistant", "content": bot_resp})
                    with conn:
                        conn.execute(
                            "INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)",
                            (conv_id, "assistant", bot_resp),
                        )

                    print(f"    T{m_idx+1}: U: {user_text[:20]}.. | B: {bot_resp[:20]}..")
                    if "ERROR_TECHNICAL" in bot_resp:
                        print(f"    T{m_idx+1}: [BOT FAILED ALL MODELS] Ending conversation.")
                        break
                    time.sleep(0.2)

                ev = evaluate_conversation(history, p_dict["goal"])
                with conn:
                    conn.execute(
                        "UPDATE conversations SET end_time=CURRENT_TIMESTAMP, evaluation_score=?, evaluation_reason=? WHERE id=?",
                        (ev.get("score", 0), ev.get("reason", ""), conv_id),
                    )
                print(f"    Eval: {ev.get('score')}/10 - {ev.get('reason')}")
    finally:
        conn.close()


if __name__ == "__main__":
    run_simulation(num_personas=5, convs_per_persona=10)
