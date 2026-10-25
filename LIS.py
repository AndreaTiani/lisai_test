# =======================================================================================================
# ------------------ Questo file rappresenta le chiamate che l'app principale può fare ------------------ 
# =======================================================================================================

import requests

# Questa stringa serve ad Andrea per avviare il file
# & "c:\Users\A1628apulia\Desktop\Progetti Python\.venv\Scripts\python.exe" "c:/Users/A1628apulia/Desktop/Progetti Python/LIS/LIS.py"


# ====================================== Servizio di test ======================================

# Chiamata all'endpoint dall'esterno
response = requests.get("http://127.0.0.1:8000/ai/test")

# Estrae la risposta JSON
dati = response.json()

print(dati["status"])   # Stampa: ok
print(dati["message"])  # Stampa: API di test funzionante!


# ====================================== Servizio di generazione risposta LLM ======================================

# L'URL del tuo nodo esposto
url = "http://127.0.0.1:8000/ai/generate"

# I dati da inviare
payload = {
    "prompt": "Ciao, sono Enrico Papi",
    "system_prompt": "Sei un llm gentile ed educato."
}

# Invia la richiesta al server
response = requests.post(url, json=payload)

# Legge la risposta
if response.status_code == 200:
    data = response.json()
    print(data["status"])
    print(data["result"])
else:
    print("Errore:", response.json())


# TODO:
# Verificare che il contesto funzioni