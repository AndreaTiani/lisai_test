# =======================================================================================================
# ------------------ Questo file rappresenta le chiamate che l'app principale può fare ------------------ 
# =======================================================================================================

import requests

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


# ====================================== Servizio RAG: domanda sul documento ======================================

url = "http://127.0.0.1:8000/ai/rag"

payload = {
    "domanda": "Buongiorno, sono una persona sorda e sto cercando lavoro, cosa mi consigli?",
    "k": 3,  # quanti chunk di contesto recuperare (1-10)
}

# Alla prima chiamata, se l'indice non esiste, il server lo crea:
# in quel caso la risposta arriva dopo alcuni minuti
response = requests.post(url, json=payload, timeout=600)

if response.status_code == 200:
    data = response.json()
    # print(data["status"])
    # print("Fonti:", data["fonti"])
    print(data["output"])

    # Il contesto recuperato serve a capire se sbaglia il retrieval o l'LLM
    # print(data["contesto"])
else:
    print("Errore:", response.json())
