import os
import json
import requests
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity



# Load environment variables
load_dotenv()

app = Flask(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
WASENDER_API_TOKEN = os.getenv("WASENDER_API_TOKEN")
WASENDER_API_URL = "https://wasenderapi.com/api/send-message"

# Setup Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")


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



@app.route("/webhook", methods=["POST"])
def webhook():
    try:
        data = request.get_json()

        # Validate event type
        if data.get("event") != "messages.upsert":
            return jsonify({"status": "ignored"})

        message = data.get("data", {}).get("messages", {})

        # Skip messages from self
        if message.get("key", {}).get("fromMe"):
            return jsonify({"status": "self message"})

        sender = message.get("key", {}).get("remoteJid")

        # Extract message text
        incoming_text = None
        msg_content = message.get("message", {})

        if "conversation" in msg_content:
            incoming_text = msg_content["conversation"]
        elif "extendedTextMessage" in msg_content:
            incoming_text = msg_content["extendedTextMessage"].get("text")

        if not incoming_text:
            return jsonify({"status": "no text"})

        # Generate Gemini response
        gemini_response = responder(incoming_text)


        # Send reply via WaSender
        send_whatsapp_message(sender, gemini_response)

        return jsonify({"status": "success"})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

def send_whatsapp_message(recipient, text):
    if "@s.whatsapp.net" in recipient:
        recipient = recipient.split("@")[0]

    payload = {
        "to": recipient,
        "text": text
    }

    headers = {
        "Authorization": f"Bearer {WASENDER_API_TOKEN}",
        "Content-Type": "application/json"
    }

    response = requests.post(WASENDER_API_URL, json=payload, headers=headers)
    print(f"WaSender response: {response.status_code} {response.text}")
    return response.ok

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)

