import cv2
import numpy as np
import pytesseract
from PIL import Image
import argparse
import os
import configparser
import sys

# Default parameters, to be used if config file is missing or keys are absent
DEFAULT_PARAMS = {
    'clahe_clip_limit': '2.0',
    'clahe_tile_grid_size_x': '8',
    'clahe_tile_grid_size_y': '8',
    'tesseract_psm': '6',
    'tesseract_lang': 'spa',
    'save_intermediate_images': 'True'
}

def load_params(config_file_path):
    config = configparser.ConfigParser()
    # Set default values before reading, so they are available if not in file
    # configparser needs section for defaults, but we'll handle it slightly differently
    # by directly using DEFAULT_PARAMS if section or key is missing.

    params = DEFAULT_PARAMS.copy() # Start with defaults

    if not os.path.exists(config_file_path):
        print(f"Warning: Config file '{config_file_path}' not found. Using default parameters.")
        # Convert string bools and numbers from defaults
        params['clahe_clip_limit'] = float(params['clahe_clip_limit'])
        params['clahe_tile_grid_size_x'] = int(params['clahe_tile_grid_size_x'])
        params['clahe_tile_grid_size_y'] = int(params['clahe_tile_grid_size_y'])
        params['tesseract_psm'] = int(params['tesseract_psm'])
        params['save_intermediate_images'] = params['save_intermediate_images'].lower() == 'true'
        return params

    try:
        config.read(config_file_path)
        if 'Parameters' in config:
            config_params = config['Parameters']
            params['clahe_clip_limit'] = float(config_params.get('clahe_clip_limit', DEFAULT_PARAMS['clahe_clip_limit']))
            params['clahe_tile_grid_size_x'] = int(config_params.get('clahe_tile_grid_size_x', DEFAULT_PARAMS['clahe_tile_grid_size_x']))
            params['clahe_tile_grid_size_y'] = int(config_params.get('clahe_tile_grid_size_y', DEFAULT_PARAMS['clahe_tile_grid_size_y']))
            params['tesseract_psm'] = str(config_params.get('tesseract_psm', DEFAULT_PARAMS['tesseract_psm'])) # Keep as string for Tesseract config
            params['tesseract_lang'] = config_params.get('tesseract_lang', DEFAULT_PARAMS['tesseract_lang'])
            params['save_intermediate_images'] = config_params.getboolean('save_intermediate_images', DEFAULT_PARAMS['save_intermediate_images'].lower() == 'true')
        else:
            print(f"Warning: '[Parameters]' section not found in '{config_file_path}'. Using default parameters for all.")
            # Re-apply type conversions for defaults if section was missing
            params['clahe_clip_limit'] = float(params['clahe_clip_limit'])
            params['clahe_tile_grid_size_x'] = int(params['clahe_tile_grid_size_x'])
            params['clahe_tile_grid_size_y'] = int(params['clahe_tile_grid_size_y'])
            params['tesseract_psm'] = str(params['tesseract_psm']) # Keep as string
            params['save_intermediate_images'] = params['save_intermediate_images'].lower() == 'true'

    except Exception as e:
        print(f"Error reading or parsing config file '{config_file_path}': {e}. Using default parameters.")
        # Re-apply type conversions for defaults if error occurred
        params['clahe_clip_limit'] = float(params['clahe_clip_limit'])
        params['clahe_tile_grid_size_x'] = int(params['clahe_tile_grid_size_x'])
        params['clahe_tile_grid_size_y'] = int(params['clahe_tile_grid_size_y'])
        params['tesseract_psm'] = str(params['tesseract_psm']) # Keep as string
        params['save_intermediate_images'] = params['save_intermediate_images'].lower() == 'true'
        
    return params

def preprocess_image(cv_image, params, output_dir, base_filename):
    print("[PREPROC] Starting preprocessing...")
    
    # Ensure output directory exists
    if params.get('save_intermediate_images', False): # Check if key exists and is True
        os.makedirs(output_dir, exist_ok=True)

    # 1. Grayscale
    gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    print("[PREPROC] Converted to grayscale.")
    if params.get('save_intermediate_images', False):
        cv2.imwrite(os.path.join(output_dir, f"{base_filename}_01_grayscale.png"), gray_image)

    # 2. CLAHE
    clahe_clip = params.get('clahe_clip_limit', 2.0)
    clahe_tile_x = params.get('clahe_tile_grid_size_x', 8)
    clahe_tile_y = params.get('clahe_tile_grid_size_y', 8)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_tile_x, clahe_tile_y))
    processed_image = clahe.apply(gray_image)
    print(f"[PREPROC] Applied CLAHE (Clip: {clahe_clip}, Tile: ({clahe_tile_x},{clahe_tile_y})).")
    if params.get('save_intermediate_images', False):
        cv2.imwrite(os.path.join(output_dir, f"{base_filename}_02_clahe.png"), processed_image)
        
    # 3. Binarization (Otsu)
    _, processed_image = cv2.threshold(processed_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print("[PREPROC] Applied Otsu's binarization.")
    if params.get('save_intermediate_images', False):
        cv2.imwrite(os.path.join(output_dir, f"{base_filename}_03_binary.png"), processed_image)
        
    return processed_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OCR an image using Tesseract with preprocessing.")
    parser.add_argument("image_path", help="Path to the image file for OCR.")
    parser.add_argument("--params_file", default="ocr_params.txt", help="Path to the parameters config file (default: ocr_params.txt in script dir).")
    args = parser.parse_args()

    if not os.path.isfile(args.image_path):
        print(f"Error: Image file not found at '{args.image_path}'")
        sys.exit(1)

    # Determine script directory to find default params_file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.params_file == "ocr_params.txt": # If default, assume it's in script_dir
        config_file_path = os.path.join(script_dir, args.params_file)
    else: # If user specified a path, use that
        config_file_path = args.params_file
        
    params = load_params(config_file_path)

    print("\n--- Current OCR Parameters ---")
    for key, value in params.items():
        print(f"{key}: {value}")
    print("----------------------------\n")

    try:
        pil_image = Image.open(args.image_path)
        # Convert PIL to OpenCV format (ensure it's BGR)
        cv_img = np.array(pil_image.convert('RGB'))
        cv_img = cv_img[:, :, ::-1].copy() 
    except FileNotFoundError:
        print(f"Error: Could not find image file at {args.image_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading image {args.image_path}: {e}")
        sys.exit(1)

    output_directory = os.path.dirname(args.image_path)
    base_fn = os.path.splitext(os.path.basename(args.image_path))[0]

    preprocessed_cv_image = preprocess_image(cv_img, params, output_directory, base_fn)

    # Perform OCR
    tess_lang = params.get('tesseract_lang', 'spa')
    tess_psm = str(params.get('tesseract_psm', '6')) # Ensure PSM is string for Tesseract config
    custom_tesseract_config = f'--psm {tess_psm}'
    
    print(f"\n[OCR_INFO] Running Tesseract with lang='{tess_lang}', config='{custom_tesseract_config}'...")
    try:
        raw_text = pytesseract.image_to_string(preprocessed_cv_image, lang=tess_lang, config=custom_tesseract_config)
        
        print(f"\n--- TESSERACT RAW OUTPUT (from ocr.py) ---")
        print(raw_text)
        print("--- END TESSERACT RAW OUTPUT ---")

    except pytesseract.TesseractNotFoundError:
        print("\nERROR: Tesseract executable not found or not in PATH.")
        print("Please ensure Tesseract OCR engine is installed correctly.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during Tesseract OCR: {e}")
        sys.exit(1)
