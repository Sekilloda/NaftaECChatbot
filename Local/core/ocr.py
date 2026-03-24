import os
import re
import cv2
import numpy as np
import pytesseract
import base64
import json
import platform
import time
from PIL import Image
import configparser
from google import genai
from google.genai import types

# Initialize Gemini Client at module level
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
GEMINI_OCR_STRUCT_MODEL = os.getenv("GEMINI_OCR_STRUCT_MODEL", "gemini-2.5-flash")

DEFAULT_PARAMS = {
    'clahe_clip_limit': '2.0',
    'clahe_tile_grid_size_x': '8',
    'clahe_tile_grid_size_y': '8',
    'tesseract_psm': '6',
    'tesseract_lang': 'spa',
    'save_intermediate_images': 'False'
}

def get_mistral_api_key():
    return os.getenv("MISTRAL_API_KEY") or os.getenv("MISTRAL_API")

def load_params(config_file_path):
    config = configparser.ConfigParser()
    params = DEFAULT_PARAMS.copy()
    
    # Platform-specific default Tesseract paths
    if platform.system() == "Windows":
        possible_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            os.path.expanduser(r'~\AppData\Local\Tesseract-OCR\tesseract.exe')
        ]
        for path in possible_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                params['tesseract_cmd'] = path
                break
    elif platform.system() == "Darwin": # MacOS
        if os.path.exists('/opt/homebrew/bin/tesseract'):
            pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'
            params['tesseract_cmd'] = '/opt/homebrew/bin/tesseract'
        elif os.path.exists('/usr/local/bin/tesseract'):
            pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'
            params['tesseract_cmd'] = '/usr/local/bin/tesseract'

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
            if 'tesseract_cmd' in config_params:
                params['tesseract_cmd'] = config_params['tesseract_cmd']
                pytesseract.pytesseract.tesseract_cmd = params['tesseract_cmd']
    except Exception as e:
        print(f"[OCR] Error reading config: {e}")
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
        print(f"[OCR] Error encoding image: {e}")
        return None

def process_receipt_image(image_path: str, original_filename: str) -> dict:
    """
    Returns a dictionary of extracted fields or None if failed.
    """
    print(f"[OCR] Processing OCR for: {original_filename}")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, "ocr_params.txt")
    params = load_params(config_path)

    try:
        pil_image = Image.open(image_path)
        cv_image = np.array(pil_image.convert('RGB')) 
        cv_image = cv_image[:, :, ::-1].copy()
        preprocessed_cv_image = preprocess_image(cv_image, params)
        
        custom_tesseract_config = f'--psm {params["tesseract_psm"]}'
        raw_text = ""
        preferred_lang = params.get('tesseract_lang', 'spa')

        try:
            raw_text = pytesseract.image_to_string(preprocessed_cv_image, lang=preferred_lang, config=custom_tesseract_config)
        except Exception as e:
            print(f"[OCR] Tesseract OCR error: {e}")
        
        # Initial empty fields
        data = {
            "banco": "",
            "monto": "",
            "fecha": "",
            "numero_comprobante": "",
            "cuenta_origen": ""
        }

        # Tesseract Basic Extraction (RegEx)
        lines = raw_text.split('\n')
        for line in lines:
            line_clean = line.strip()
            if not line_clean: continue
            
            if not data["banco"] and re.search(r"(?i)BANCO", line_clean):
                match = re.search(r"(?i)BANCO\s*[:\-]?\s*(.*)", line_clean)
                if match: data["banco"] = match.group(1).strip()
            
            if not data["numero_comprobante"] and re.search(r"(?i)Documento|Referencia|Comprobante|Transacc", line_clean):
                nums = re.findall(r'\d{4,}', line_clean)
                if nums: data["numero_comprobante"] = str(max(nums, key=len))
            
            if not data["monto"] and re.search(r"(?i)Total|Monto|Valor|Importe", line_clean):
                match = re.search(r"([\d]+[\.,][\d]{2})", line_clean)
                if match: data["monto"] = match.group(1).replace(',', '.')
            
            if not data["fecha"] and re.search(r"(?i)Fecha", line_clean):
                match = re.search(r"(\d{1,2}/\d{1,2}/\d{2,4})", line_clean)
                if match: data["fecha"] = match.group(1)

        # Gemini-Refinement (Vision-capable model fallback/refinement)
        if gemini_client:
            print(f"[OCR] Refining data with Gemini ({GEMINI_OCR_STRUCT_MODEL})...")
            try:
                # We can send the raw text first for quick structured refinement
                prompt = f"""
                Analiza el siguiente texto extraído de un comprobante bancario y devuelve un JSON estructurado.
                Campos requeridos: banco, monto, numero_comprobante, fecha, cuenta_origen.
                
                REGLAS:
                1. 'monto' debe ser numérico con punto decimal.
                2. 'fecha' debe ser DD/MM/YYYY.
                3. 'numero_comprobante' es el número de referencia o transacción.
                4. 'cuenta_origen' es el número de cuenta de donde sale el dinero.
                5. Si no encuentras un campo, usa una cadena vacía "".
                
                Texto:
                {raw_text}
                """
                # Retry loop for Gemini refinement
                res = None
                for attempt in range(3):
                    try:
                        res = gemini_client.models.generate_content(
                            model=GEMINI_OCR_STRUCT_MODEL,
                            contents=prompt,
                            config=types.GenerateContentConfig(response_mime_type="application/json")
                        )
                        break
                    except Exception as e:
                        if "429" in str(e) and attempt < 2:
                            print(f"[OCR] Rate limited (429). Retrying in 20s... (Attempt {attempt+1}/3)")
                            time.sleep(20)
                        else:
                            raise e
                
                if res:
                    gemini_data = json.loads(res.text)
                    # Update if empty or refine
                    for key in data:
                        if not data[key] or data[key] == "":
                            val = gemini_data.get(key, "")
                            if val and val != "Not found":
                                data[key] = str(val)
            except Exception as ge:
                print(f"[OCR] Gemini refinement failed: {ge}")

        # Final sanity check for mandatory fields as per user request
        # (leave empty if not found, but we ensure they are present in the dict)
        return data
    except Exception as e:
        print(f"[OCR] Error in process_receipt_image: {e}")
        return None
