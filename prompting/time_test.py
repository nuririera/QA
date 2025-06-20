import requests
import time 

API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.1"

# Prompt para medir tiempo con una frase de 10 palabras exactas
prompt_30_words = """
Tu tarea es generar frases que tengan exactamente 30 palabras, ni más ni menos.  
La frase debe ser coherente, gramaticalmente correcta y tener sentido completo.

Ejemplos correctos (30 palabras):  
- Aunque la tormenta era intensa, decidimos continuar la caminata, confiando en que el clima mejoraría pronto y podríamos llegar al refugio antes de que anocheciera por completo.  
- Los avances tecnológicos actuales permiten una comunicación instantánea entre personas de diferentes culturas, lo cual favorece el entendimiento mutuo, la colaboración internacional y el desarrollo de soluciones globales para problemas compartidos.  

Ejemplos incorrectos (por número de palabras):  
- Me encanta viajar, conocer nuevas personas y probar comidas diferentes. (10 palabras)  
- Siempre soñé con construir una casa en el bosque donde pudiera vivir tranquilo, rodeado de naturaleza, lejos del ruido de la ciudad y de la rutina diaria. (33 palabras)  

Ahora, genera una sola frase de exactamente 30 palabras.
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
    measure_response_time(prompt_30_words, n=10)
    print("Medición completada.")
