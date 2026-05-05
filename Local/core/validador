import pandas as pd
import os

# Ruta al Excel (asumiendo que está en la carpeta principal del proyecto)
EXCEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "precios_carreras.xlsx")

def cargar_precios():
    """Lee el Excel y lo convierte en un diccionario para consultas rápidas."""
    try:
        if not os.path.exists(EXCEL_PATH):
            print(f"⚠️ No se encontró el archivo Excel en: {EXCEL_PATH}")
            return {}
            
        df = pd.read_excel(EXCEL_PATH)
        precios = {}
        # Asumiendo columnas: id_carrera, nombre_carrera, distancia, precio_base
        for _, row in df.iterrows():
            carrera = str(row['nombre_carrera']).strip().lower()
            distancia = str(row['distancia']).strip().upper()
            if carrera not in precios:
                precios[carrera] = {}
            precios[carrera][distancia] = float(row['precio_base'])
        return precios
    except Exception as e:
        print(f"Error cargando Excel de precios: {e}")
        return {}

def calcular_precio_final(corredores):
    """
    Recibe una lista de diccionarios, ej:
    [{'cedula': '123', 'carrera': 'rio 21k', 'distancia': '10K'}, ...]
    """
    precios_db = cargar_precios()
    if not precios_db:
        return None, "Error del sistema: Base de precios no disponible."

    # 1. Contar carreras únicas para el descuento general
    carreras_unicas = set(c['carrera'].lower() for c in corredores)
    cantidad_carreras = len(carreras_unicas)
    
    descuento_multicarrera = 0.0
    if cantidad_carreras == 2:
        descuento_multicarrera = 0.10
    elif cantidad_carreras >= 3:
        descuento_multicarrera = 0.15

    # 2. Contar inscritos POR CARRERA para el descuento grupal
    conteo_por_carrera = {}
    for c in corredores:
        carr = c['carrera'].lower()
        conteo_por_carrera[carr] = conteo_por_carrera.get(carr, 0) + 1

    # 3. Calcular total corredor por corredor
    total_calculado = 0.0
    detalle_calculo = []

    for corredor in corredores:
        carrera_str = corredor['carrera'].lower()
        distancia = corredor['distancia'].upper()
        
        # Validar si existe la carrera y la distancia
        if carrera_str not in precios_db:
            return None, f"La carrera '{corredor['carrera']}' no existe en la base."
        
        precio_base = precios_db[carrera_str].get(distancia)
        if precio_base is None:
            return None, f"Distancia {distancia} no válida para la carrera {corredor['carrera']}."

        # Descuento grupal (>10 en la misma carrera)
        descuento_grupal = 0.10 if conteo_por_carrera[carrera_str] > 10 else 0.0

        # Regla: Se aplica el descuento mayor
        descuento_aplicar = max(descuento_multicarrera, descuento_grupal)
        
        precio_con_descuento = precio_base * (1 - descuento_aplicar)
        total_calculado += precio_con_descuento
        
        detalle_calculo.append({
            "cedula": corredor['cedula'],
            "carrera": corredor['carrera'],
            "distancia": distancia,
            "precio_base": precio_base,
            "descuento_aplicado": descuento_aplicar,
            "precio_final": round(precio_con_descuento, 2)
        })

    return round(total_calculado, 2), detalle_calculo
