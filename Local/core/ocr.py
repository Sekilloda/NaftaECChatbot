import os
import re
import cv2
import numpy as np
import pytesseract
import base64
import json
from PIL import Image
import configparser
from google import genai
from google.genai import types

DEFAULT_PARAMS = {
    'clahe_clip_limit': '2.0',
    'clahe_tile_grid_size_x': '8',
    'clahe_tile_grid_size_y': '8',
    'tesseract_psm': '6',
    'tesseract_lang': 'spa',
    'save_intermediate_images': 'False'
}

def load_params(config_file_path):
    config = configparser.ConfigParser()
    params = DEFAULT_PARAMS.copy()
    if not os.path.exists(config_file_path):
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
    except Exception as e:
        print(f"Error reading config: {e}")
    return params

def preprocess_image(cv_image, params):
    gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    clahe_clip = params.get('clahe_clip_limit', 2.0)
    clahe_tile_x = params.get('clahe_tile_grid_size_x', 8)
    clahe_tile_y = params.get('clahe_tile_grid_size_y', 8)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_tile_x, clahe_tile_y))
    processed_image = clahe.apply(gray_image)
    _, processed_image = cv2.threshold(processed_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return processed_image

def encode_image_to_base64(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None

def run_mistral_ocr_pipeline(image_path: str):
    """
    Uses Mistral for OCR and Gemini for structured extraction.
    Note: We use the mistralai library directly here.
    """
    from mistralai import Mistral # Local import to keep it contained
    
    print(f"[MISTRAL_OCR] Attempting Mistral OCR pipeline for {image_path}")
    mistral_api_key = os.getenv("MISTRAL_API_KEY")
    if not mistral_api_key:
        print("Error: MISTRAL_API_KEY not set.")
        return None

    base64_image = encode_image_to_base64(image_path)
    if not base64_image:
        return None

    try:
        client = Mistral(api_key=mistral_api_key)
        ocr_response = client.ocr.process(
            model="mistral-ocr-2503",
            document={
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{base64_image}"
            }
        )
        ocr_text = ocr_response.pages[0].markdown
    except Exception as e:
        print(f"Mistral OCR Error: {e}")
        return None

    try:
        # Use native Gemini for extraction with the new google-genai SDK
        gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        
        prompt = f"""
Extrae la siguiente información del texto de un comprobante bancario en formato JSON.
REGLAS:
1. Extrae: 'banco', 'monto', 'numero_transaccion', 'fecha'.
2. Si un campo no se encuentra, usa 'N/A'.
3. Para 'monto', usa punto decimal (ej: 123.45).
4. Para 'fecha', normaliza a DD/MM/YYYY.

Texto del comprobante:
{ocr_text}
"""
        response = gemini_client.models.generate_content(
            model="gemini-flash-lite-latest",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"Data extraction error: {e}")
        return None

def process_receipt_image(image_path: str, original_filename: str) -> bool:
    print(f"Processing OCR for: {original_filename}")
    params = load_params("ocr_params.txt")

    try:
        pil_image = Image.open(image_path)
        cv_image = np.array(pil_image.convert('RGB')) 
        cv_image = cv_image[:, :, ::-1].copy()
        preprocessed_cv_image = preprocess_image(cv_image, params)
        
        custom_tesseract_config = f'--psm {params["tesseract_psm"]}'
        raw_text = pytesseract.image_to_string(preprocessed_cv_image, lang=params['tesseract_lang'], config=custom_tesseract_config)
        
        banco, total, documento, fecha = "Not found", "Not found", "Not found", "Not found"
        lines = raw_text.split('\n')

        # Tesseract Basic Extraction
        for line in lines:
            if re.search(r"(?i)BANCO", line):
                match = re.search(r"(?i)BANCO\s*[:\-]?\s*(.*)", line)
                if match: banco = match.group(1).strip()
            if re.search(r"(?i)Documento|Referencia", line):
                nums = re.findall(r'\d+', line)
                if nums: documento = str(max(int(n) for n in nums))
            if re.search(r"(?i)Total|Monto", line):
                match = re.search(r"([\d]+[\.,][\d]{2})", line)
                if match: total = match.group(1).replace(',', '.')
            if re.search(r"(?i)Fecha", line):
                match = re.search(r"(\d{1,2}/\d{1,2}/\d{2,4})", line)
                if match: fecha = match.group(1)

        # Mistral Fallback
        if any(val == "Not found" for val in [banco, total, documento, fecha]):
            print("[OCR_FALLBACK] Attempting Mistral OCR.")
            mistral_data = run_mistral_ocr_pipeline(image_path)
            if mistral_data:
                if banco == "Not found" and mistral_data.get('banco') != "N/A": banco = mistral_data['banco']
                if total == "Not found" and mistral_data.get('monto') != "N/A": total = mistral_data['monto']
                if documento == "Not found" and mistral_data.get('numero_transaccion') != "N/A": documento = mistral_data['numero_transaccion']
                if fecha == "Not found" and mistral_data.get('fecha') != "N/A": fecha = mistral_data['fecha']

        parsed_info = [f"Banco: {banco}", f"Total: {total}", f"Documento: {documento}", f"Fecha: {fecha}"]
        output_txt_path = os.path.join(os.path.dirname(image_path), os.path.splitext(original_filename)[0] + ".txt")
        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(parsed_info))
        return True
    except Exception as e:
        print(f"OCR Error: {e}")
        return False
