# LISAI — API LLM + RAG

Servizio FastAPI che espone un LLM locale (Ollama) e un RAG sul PDF `data/RAG_helpdesk.pdf`:
le risposte sono generate usando solo il contenuto del documento, con citazione della pagina.

---

## 1. Prerequisiti

| Cosa | Versione / note |
|---|---|
| Python | 3.10+ (testato con 3.12) |
| [Ollama](https://ollama.com/download) | in esecuzione su `http://localhost:11434` |
| Modello LLM | `Jadio/Qwen3_4b_instruct_q4km:latest` (~2,5 GB) |
| Modello di embedding | `embeddinggemma:300m` (~600 MB) |

## 2. Installazione

Tutti i comandi vanno eseguiti dalla **root del progetto** (la cartella con `main.py`).

```bash
# 1. Ollama: scaricalo dal sito e avvialo (opzionale: esegui all'avvio)
#    https://ollama.com/download

# 2. Modello LLM per la generazione
ollama pull Jadio/Qwen3_4b_instruct_q4km:latest

# 3. Modello di embedding per il RAG
ollama pull embeddinggemma:300m

# 4. Dipendenze Python
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt     # Windows
# source .venv/bin/activate && pip install -r requirements.txt  # Linux/macOS
```

Verifica che Ollama risponda e che i modelli ci siano:

```bash
ollama list
```

`requirements.txt` contiene solo il necessario per questo branch (API + RAG). Il primo
avvio scarica dalla rete il tokenizer del modello di embedding (~35 MB, una volta sola).

> Le dipendenze della parte LIS/vision (`mediapipe`, `opencv-contrib-python`, `torch`) **non**
> sono incluse: servono solo per l'elaborazione video e vanno installate a parte se e quando
> quella parte viene ripresa nello stesso ambiente.

## 3. Configurazione (`.env`)

Crea un file `.env` nella root (è già in `.gitignore`):

```env
OLLAMA_MODEL=Jadio/Qwen3_4b_instruct_q4km:latest
OLLAMA_HOST=http://localhost:11434
```

| Variabile | Default | Effetto |
|---|---|---|
| `OLLAMA_MODEL` | `llama3.2` | modello usato per generare |
| `OLLAMA_HOST` | `http://localhost:11434` | endpoint Ollama |
| `EMBEDDING_MODEL` | `embeddinggemma:300m` | modello di embedding |
| `EMBEDDING_TOKENIZER` | `unsloth/embeddinggemma-300m` | repo del tokenizer (il repo `google/...` è gated) |
| `EMBEDDING_DIMENSIONI` | *(nativa)* | troncamento vettori (es. `512`) |
| `HOST` / `PORT` | `127.0.0.1` / `8000` | indirizzo del server |

## 4. Avvio

```bash
.venv/Scripts/python.exe main.py
```

- API: `http://127.0.0.1:8000`
- Documentazione interattiva: `http://127.0.0.1:8000/docs`

## 5. API

| Metodo | Path | Body | Risposta |
|---|---|---|---|
| `GET` | `/ai/test` | — | `{"status": "OK", ...}` |
| `POST` | `/ai/generate` | `{"prompt", "system_prompt"}` | `{"status", "result": {"status", "output"}}` |
| `POST` | `/ai/rag` | `{"domanda", "k"}` | `{"status", "output", "fonti", "contesto"}` |

```bash
curl -X POST http://127.0.0.1:8000/ai/rag \
  -H "Content-Type: application/json" \
  -d "{\"domanda\": \"Come si apre un ticket di assistenza?\", \"k\": 3}"
```

In `/ai/rag` la risposta è in `output`, le pagine citate in `fonti`, i chunk recuperati in
`contesto` (utile per capire se sbaglia il retrieval o l'LLM). Client di esempio: `LIS.py`.

## 6. Indice del RAG (una volta sola)

```bash
# solo se l'indice non esiste (idempotente)
.venv/Scripts/python.exe -m src.handlers.rag_handler

# ricostruzione da zero
.venv/Scripts/python.exe -c "from src.handlers.rag_handler import indicizza_pdf; print(indicizza_pdf(force=True))"

# domanda da terminale
.venv/Scripts/python.exe -m src.handlers.rag_handler "Come si apre un ticket di assistenza?"
```

L'indice è persistente in `chroma_db/` (gitignored). Se non lo crei prima, la prima
`POST /ai/rag` lo costruisce da sola: in quel caso la richiesta resta aperta per minuti.

## 7. Problemi comuni

| Sintomo | Rimedio |
|---|---|
| `ConnectionError` in `LIS.py` | server non avviato (`main.py`) |
| `500 ... model required but not found` | manca un modello: `ollama pull ...` |
| `Read timed out` alla prima domanda | sta indicizzando: alza il timeout (600-3600 s) |
| `No module named 'src'` | esegui dalla root, con `python -m src...` (non `python src/...`) |
| risposte fuori tema | controlla `contesto` e `fonti`; prova a cambiare `k` |
| `pagine: 0` nell'output dell'indice | il PDF non ha testo estraibile (probabilmente scansionato) |
