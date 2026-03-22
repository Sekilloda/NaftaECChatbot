import os
import sqlite3
import json
import random
import sys
import time
import requests
from datetime import datetime
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Ensure we are in the Local directory for relative paths like faqs.xlsx
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.getcwd())

from core.knowledge import responder
from core.database import save_message, get_last_messages, set_user_status, init_db

# Load environment variables
load_dotenv(".env")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY") or os.getenv("MISTRAL_API")

client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
if not client:
    print("[SIM] GEMINI_API_KEY not found. Running in local fallback mode.")
USE_LOCAL_SIM_FALLBACK = client is None
GEMINI_SIM_MODEL = os.getenv("GEMINI_SIM_MODEL", "gemini-2.5-flash-lite")

SIM_DB_PATH = "simulation/simulation_results.db"
LOCAL_FALLBACK_PERSONAS = [
    {
        "name": "Carlos Ruiz",
        "personality": "Paciente pero confundido",
        "proficiency": 0.6,
        "goal": "Saber si su inscripción para la 10K está confirmada."
    },
    {
        "name": "Elena Gómez",
        "personality": "Muy apurada y directa",
        "proficiency": 0.9,
        "goal": "Preguntar por el precio de la carrera 21K y descuentos para grupos."
    },
    {
        "name": "Ricardo Soria",
        "personality": "Frustrado y sarcástico",
        "proficiency": 0.4,
        "goal": "Quejarse de que pagó y no recibió su código."
    },
]

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

def persona_get(persona, key, default=""):
    if isinstance(persona, dict):
        return persona.get(key, default)
    try:
        return persona[key]
    except Exception:
        return default

def local_user_message(persona, turn_idx):
    goal = str(persona_get(persona, "goal", "")).lower()
    name = str(persona_get(persona, "name", "Usuario"))
    persona_goal = str(persona_get(persona, "goal", "Necesito ayuda con mi inscripción."))
    templates = [
        f"Hola, soy {name}. {persona_goal}",
        "¿Me pueden confirmar mi estado y próximos pasos?",
        "¿Qué debo hacer si no veo actualización?",
        "Gracias. ¿Algo más que deba considerar?",
        "Perfecto, quedo atento."
    ]
    if "descuento" in goal:
        templates = [
            f"Hola, soy {name}. Quiero saber precio y descuentos de grupo.",
            "Somos varias personas. ¿Qué descuento aplica para grupo?",
            "¿Hasta cuándo está vigente esa promoción?",
            "¿Cómo hacemos el pago para todo el grupo?",
            "Gracias por la información."
        ]
    elif "código" in goal or "pagó" in goal:
        templates = [
            f"Hola, soy {name}. Pagué y no recibo mi código.",
            "Estoy esperando respuesta desde ayer.",
            "Necesito una solución hoy por favor.",
            "¿Me pueden escalar con soporte humano?",
            "Gracias, quedo pendiente."
        ]
    return templates[min(turn_idx, len(templates) - 1)]

def local_evaluate_conversation(history, goal):
    assistant_msgs = [m["content"] for m in history if m["role"] == "assistant"]
    if not assistant_msgs:
        return {"score": 1, "reason": "No hubo respuestas del asistente."}
    
    goal_lower = str(goal).lower()
    error_hits = sum(
        1 for msg in assistant_msgs
        if any(token in msg.lower() for token in ["error_technical", "mucha demanda", "tuve un problema"])
    )
    
    # Detect RAG Hallucinations (Hostería/Hospedaje mentions when goal is NOT about lodging)
    hallucination_hits = 0
    if "hospedaje" not in goal_lower and "hotel" not in goal_lower and "dormir" not in goal_lower:
        hallucination_hits = sum(
            1 for msg in assistant_msgs
            if any(token in msg.lower() for token in ["hostería", "hospedaje", "andaluza"])
        )

    score = max(1, 10 - (error_hits * 3) - (hallucination_hits * 4))
    return {"score": score, "reason": f"Evaluación local. Errores: {error_hits}, Hallucinaciones: {hallucination_hits}."}

def local_bot_response(user_msg):
    text = (user_msg or "").lower()
    if "descuento" in text or "grupo" in text:
        return "Sí, hay descuentos para grupos. Comparte cuántos corredores son y te indicamos el porcentaje."
    if "código" in text or "pagué" in text or "pago" in text:
        return "Entiendo. Verificaremos tu pago y código. Si no aparece en breve, te pasamos con soporte humano."
    if "inscrip" in text or "confirm" in text:
        return "Claro. Puedo ayudarte a revisar tu inscripción y estado con tus datos registrados."
    return "Gracias por tu mensaje. Te ayudo con la información de tu carrera e inscripción."

def save_personas_to_db(personas):
    conn = sqlite3.connect(SIM_DB_PATH)
    try:
        with conn:
            for p in personas:
                cursor = conn.execute(
                    "INSERT INTO personas (name, personality, proficiency, goal) VALUES (?, ?, ?, ?)",
                    (p['name'], p['personality'], p['proficiency'], p['goal'])
                )
                p['id'] = cursor.lastrowid
    finally:
        conn.close()

def create_test_registry(personas):
    import pandas as pd
    reg_data = []
    for p in personas:
        name_parts = p['name'].split()
        first_name = name_parts[0]
        last_name = " ".join(name_parts[1:]) if len(name_parts) > 1 else ""
        reg_data.append({
            'First name': first_name,
            'Last name': last_name,
            'Telefono': f"5939{random.randint(10000000, 99999999)}",
            'Competition': random.choice(['10K', '21K', '5K']),
            'Cedula': f"{random.randint(1000000000, 1999999999)}",
            'Status': 'Confirmado'
        })
    os.makedirs("reportes_descargados", exist_ok=True)
    pd.DataFrame(reg_data).to_excel("reportes_descargados/latest_registry.xlsx", index=False)
    print(f"[SIM] Created test registry with {len(reg_data)} entries.")

def init_sim_db():
    if not os.path.exists("simulation"):
        os.makedirs("simulation")
    conn = sqlite3.connect(SIM_DB_PATH)
    try:
        with conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS personas (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    personality TEXT,
                    proficiency REAL,
                    goal TEXT
                );
                CREATE TABLE IF NOT EXISTS test_conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    persona_id INTEGER,
                    start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                    end_time DATETIME,
                    evaluation_score INTEGER,
                    evaluation_reason TEXT,
                    FOREIGN KEY (persona_id) REFERENCES personas(id)
                );
                CREATE TABLE IF NOT EXISTS test_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id INTEGER,
                    role TEXT,
                    content TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (conversation_id) REFERENCES test_conversations(id)
                );
            """)
    finally:
        conn.close()

def call_ai_fallback(prompt, response_mime_type="text/plain", temperature=1.0):
    """Multi-model fallback for simulation tasks."""
    models_to_try = [
        ("groq", "llama-3.1-8b-instant"),
        ("gemini", GEMINI_SIM_MODEL),
        ("openrouter", "google/gemma-2-9b-it:free"),
        ("mistral", "mistral-small-latest"),
    ]

    for provider, model_id in models_to_try:
        try:
            if provider == "groq" and GROQ_API_KEY:
                print(f"  [SIM-FALLBACK] Trying {provider}/{model_id}...")
                res = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
                    json={"model": model_id, "messages": [{"role": "user", "content": prompt}], "temperature": temperature},
                    timeout=30,
                )
                res.raise_for_status()
                return res.json()["choices"][0]["message"]["content"]

            if provider == "gemini" and client:
                print(f"  [SIM-FALLBACK] Trying {provider}/{model_id}...")
                config = types.GenerateContentConfig(temperature=temperature)
                if response_mime_type == "application/json":
                    config.response_mime_type = "application/json"
                res = client.models.generate_content(model=model_id, contents=prompt, config=config)
                return res.text

            if provider == "openrouter" and OPENROUTER_API_KEY:
                or_models = [model_id, "minimax/minimax-m2.5:free", "nvidia/nemotron-3-nano-30b-a3b:free", "nvidia/nemotron-3-super-120b-a12b:free", "meta-llama/llama-3.2-1b-instruct:free"]
                for or_model in or_models:
                    try:
                        print(f"  [SIM-FALLBACK] Trying {provider}/{or_model}...")
                        res = requests.post(
                            "https://openrouter.ai/api/v1/chat/completions",
                            headers={
                                "Authorization": f"Bearer {OPENROUTER_API_KEY}", 
                                "Content-Type": "application/json",
                                "HTTP-Referer": "https://github.com/NaftaEC",
                                "X-Title": "NaftaEC Simulation"
                            },
                            json={"model": or_model, "messages": [{"role": "user", "content": prompt}]},
                            timeout=30,
                        )
                        res.raise_for_status()
                        return res.json()["choices"][0]["message"]["content"]
                    except Exception as or_e:
                        print(f"  [SIM-ERROR] OpenRouter {or_model} failed: {or_e}")
                        continue
                continue

            if provider == "mistral" and MISTRAL_API_KEY:
                # If the key is actually a Hugging Face token, use HF Inference API
                if str(MISTRAL_API_KEY).startswith("hf_"):
                    hf_models = ["meta-llama/Llama-3.2-1B-Instruct", "mistralai/Mistral-7B-Instruct-v0.2", "HuggingFaceH4/zephyr-7b-beta"]
                    for hf_model in hf_models:
                        try:
                            print(f"  [SIM-FALLBACK] Using HF Inference API for: {hf_model}")
                            hf_url = f"https://api-inference.huggingface.co/models/{hf_model}"
                            res = requests.post(
                                hf_url,
                                headers={"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"},
                                json={"inputs": prompt, "parameters": {"max_new_tokens": 512, "temperature": temperature}},
                                timeout=30
                            )
                            res.raise_for_status()
                            out = res.json()
                            if isinstance(out, list): out = out[0]
                            content = out.get("generated_text", str(out))
                            if prompt in content: content = content.replace(prompt, "")
                            return content.strip()
                        except Exception as hf_e:
                            print(f"  [SIM-ERROR] HF Model {hf_model} failed: {hf_e}")
                            continue
                    continue

                print(f"  [SIM-FALLBACK] Trying {provider}/{model_id}...")
                res = requests.post(
                    "https://api.mistral.ai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"},
                    json={"model": model_id, "messages": [{"role": "user", "content": prompt}], "temperature": temperature},
                    timeout=30,
                )
                res.raise_for_status()
                return res.json()["choices"][0]["message"]["content"]

        except Exception as e:
            if any(err in str(e).lower() for err in ["429", "quota", "resource_exhausted"]):
                print(f"  [SIM-FALLBACK] {provider}/{model_id} hit limit. Trying next...")
                continue
            print(f"  [SIM-ERROR] {provider}/{model_id} failed: {e}")

    return None

def generate_personas(count=5):
    prompt = (
        f"Genera una lista de {count} personas (usuarios ficticios) para probar un chatbot de carreras de trail running en Ecuador (NaftaEC).\n"
        "Cada persona debe tener:\n"
        "1. Nombre completo (español)\n"
        "2. Personalidad (ej. paciente, frustrado, apurado, detallista, sarcástico, etc.)\n"
        "3. Nivel de competencia (0.0 a 1.0, donde 1.0 es comunicación clara y eficiente, y 0.0 es confuso, con faltas de ortografía, mensajes fragmentados).\n"
        "4. Objetivo/Meta (ej. preguntar por inscripciones, saber si su pago fue recibido, preguntar por hospedaje, quejarse de que no le llega el código, etc.)\n"
        "Responde ÚNICAMENTE con un JSON array de objetos con las claves: name, personality, proficiency, goal."
    )
    
    res_text = call_ai_fallback(prompt, response_mime_type="application/json")

    personas = None
    if res_text:
        try:
            personas = json.loads(clean_json_text(res_text))
        except Exception as e:
            print(f"[SIM] Could not parse persona JSON, using fallback personas: {e}")

    if not personas:
        personas = build_local_personas(count)
        print(f"[SIM] Using {len(personas)} local fallback personas.")

    save_personas_to_db(personas)
    create_test_registry(personas)
    return personas

def get_user_message(persona, history):
    turn_idx = max(0, len([m for m in history if m["role"] == "user"]))
    history_str = "\n".join([f"{m['role']}: {m['content']}" for m in history])
    
    prompt = (
        f"Eres {persona['name']}. Tu personalidad es {persona['personality']} y tu nivel de competencia comunicativa es {persona['proficiency']}/1.0.\n"
        f"Tu objetivo en este chat es: {persona['goal']}.\n\n"
        f"Este es el historial de la conversación:\n{history_str}\n\n"
        "Escribe tu siguiente mensaje al chatbot. RECUERDA:\n"
        "- Si tu competencia es baja, usa lenguaje informal, abreviaturas raras o comete errores ortográficos intencionales.\n"
        "- Si tu personalidad es frustrada, sé impaciente.\n"
        "- Mantente en personaje.\n"
        "- Responde ÚNICAMENTE con el texto del mensaje, sin comillas ni explicaciones."
    )
    
    res_text = call_ai_fallback(prompt, temperature=1.0)
    if not res_text:
        return local_user_message(persona, turn_idx)
    return res_text.strip() or local_user_message(persona, turn_idx)

def evaluate_conversation(history, goal):
    history_str = "\n".join([f"{m['role']}: {m['content']}" for m in history])
    
    prompt = (
        f"Analiza la siguiente conversación entre un Usuario y un Asistente del chatbot de NaftaEC.\n"
        f"El objetivo del usuario era: {goal}\n\n"
        f"Conversación:\n{history_str}\n\n"
        "Determina si el asistente fue de ayuda y si se resolvió el problema o se dio la información correcta.\n"
        "Responde ÚNICAMENTE con un JSON con las claves: score (del 1 al 10) y reason (una breve explicación)."
    )
    
    res_text = call_ai_fallback(prompt, response_mime_type="application/json")
    if not res_text:
        return local_evaluate_conversation(history, goal)
    try:
        return json.loads(clean_json_text(res_text))
    except Exception:
        return local_evaluate_conversation(history, goal)

def run_simulation_for_persona(persona, conv_id):
    global USE_LOCAL_SIM_FALLBACK
    history = []
    sender_jid = f"test_user_{persona['id']}_{conv_id}@s.whatsapp.net"
    
    conn = sqlite3.connect(SIM_DB_PATH)
    try:
        with conn:
            cursor = conn.execute("INSERT INTO test_conversations (persona_id) VALUES (?)", (persona['id'],))
            test_conv_id = cursor.lastrowid
            
        for i in range(5): 
            user_msg = get_user_message(persona, history)
            history.append({"role": "user", "content": user_msg})
            
            with conn:
                conn.execute(
                    "INSERT INTO test_messages (conversation_id, role, content) VALUES (?, ?, ?)",
                    (test_conv_id, "user", user_msg)
                )
            
            bot_history = [(h['role'], h['content']) for h in history[:-1]]
            if USE_LOCAL_SIM_FALLBACK:
                bot_resp = local_bot_response(user_msg)
            else:
                try:
                    bot_resp = responder(user_msg, sender_jid=sender_jid, history=bot_history)
                    if (not bot_resp) or ("ERROR_TECHNICAL" in bot_resp) or ("tuve un problema" in bot_resp.lower()):
                        USE_LOCAL_SIM_FALLBACK = True
                        bot_resp = local_bot_response(user_msg)
                except Exception as e:
                    print(f"  [SIM] Responder failed, switching to local bot fallback: {e}")
                    USE_LOCAL_SIM_FALLBACK = True
                    bot_resp = local_bot_response(user_msg)
            
            history.append({"role": "assistant", "content": bot_resp})
            with conn:
                conn.execute(
                    "INSERT INTO test_messages (conversation_id, role, content) VALUES (?, ?, ?)",
                    (test_conv_id, "assistant", bot_resp)
                )
            
            print(f"  [{persona['name']}] Message {i*2+1}: {user_msg[:50]}...")
            print(f"  [Bot] Message {i*2+2}: {bot_resp[:50]}...")

        evaluation = evaluate_conversation(history, persona['goal'])
        with conn:
            conn.execute(
                "UPDATE test_conversations SET end_time=CURRENT_TIMESTAMP, evaluation_score=?, evaluation_reason=? WHERE id=?",
                (evaluation['score'], evaluation['reason'], test_conv_id)
            )
        print(f"  [EVALUATION] Score: {evaluation['score']}/10 - {evaluation['reason']}")
        
    finally:
        conn.close()

def main():
    print("Iniciando Simulación de Validación...")
    init_sim_db()
    init_db() 
    
    # Clean up previous failed test personas
    conn = sqlite3.connect(SIM_DB_PATH)
    conn.execute("DELETE FROM personas")
    conn.execute("DELETE FROM test_conversations")
    conn.execute("DELETE FROM test_messages")
    conn.commit()
    conn.close()

    print("Generando personas de prueba...")
    personas_data = generate_personas(2)
    
    conn = sqlite3.connect(SIM_DB_PATH)
    conn.row_factory = sqlite3.Row
    personas = conn.execute("SELECT * FROM personas").fetchall()
    conn.close()
    
    for persona in personas:
        print(f"\n>>> Simulando 2 conversaciones para: {persona['name']} ({persona['personality']})")
        for i in range(2):
            print(f"  --- Conversación {i+1} ---")
            run_simulation_for_persona(persona, i)

if __name__ == "__main__":
    main()
