import requests
import time 

API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.1"

# Prompt para medir tiempo con una frase de 10 palabras exactas
prompt_10_words = """
Tu tarea es generar frases que tengan exactamente 10 palabras, ni más ni menos.  
La frase debe ser coherente, gramaticalmente correcta y tener sentido completo.  

Ejemplos correctos (10 palabras):  
- La música en vivo transforma cualquier noche en una experiencia inolvidable.  
- Aprender un nuevo idioma requiere constancia, paciencia y mucha práctica.  
- El gato saltó sobre la mesa y rompió el jarrón.

Ejemplos incorrectos (por número de palabras):  
- Me gusta leer libros por las tardes. (7 palabras)  
- Siempre he querido viajar a Japón y aprender su cultura milenaria. (12 palabras)  

Ahora, genera una sola frase de exactamente 10 palabras.
"""

def measure_response_time(promot, n=10):
    tiempos = []
    for i in range(n):
        start = time.time()
        response = requests.post(API_URL, json={
            "model": MODEL_NAME,
            "prompt": promot,
            "stream": False
        })
        end = time.time()
        if response.status_code != 200:
            print(f"Error en la solicitud {i+1}: {response.status_code}")
            continue
        tiempos.append(end - start)
        print(f"Prueba {i+1}: {end - start:.3f} segundos")

    if tiempos:
        print(f"\nTiempo promedio de respuesta: {sum(tiempos)/len(tiempos):.3f} segundos")
    else:
        print("\nNo se pudo calcular el tiempo promedio debido a errores.")

if __name__ == "__main__":
    print("Iniciando medición de tiempo de respuesta con frase de 10 palabras...")
    measure_response_time(prompt_10_words, n=10)
    print("Medición completada.")
