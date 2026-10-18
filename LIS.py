# =======================================================================================================
# ------------------ Questo file rappresenta le chiamate che l'app principale può fare ------------------ 
# =======================================================================================================

import requests

# L'altro team fa la chiamata al tuo endpoint
response = requests.get("http://127.0.0.1:8000/LIS/test")

# Estrae la risposta JSON
dati = response.json()

print(dati["status"])   # Stampa: ok
print(dati["message"])  # Stampa: API di test funzionante!

# Questa stringa serve a me per avviare il file
# & "c:\Users\A1628apulia\Desktop\Progetti Python\.venv\Scripts\python.exe" "c:/Users/A1628apulia/Desktop/Progetti Python/LIS/LIS.py"