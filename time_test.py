import requests
import time 

API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.1"

prompt_simple = "Hola eres un medidor de tiempo, dime cuanto tiempo ha pasado desde que te lo dije"

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

    print(f"\nTiempo promedio de respuesta: {sum(tiempos)/len(tiempos):.3f} segundos")

if __name__ == "__main__":
    print("Iniciando medición de tiempo de respuesta...")
    measure_response_time(prompt_simple, n=10)
    print("Medición completada.")