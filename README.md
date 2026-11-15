# LISAI — API di test per progetto LISAI

Servizio FastAPI che espone un LLM locale (via Ollama) e un **RAG** su un documento PDF:
le domande vengono risposte usando esclusivamente il contenuto di `data/RAG_helpdesk.pdf`
(policy e regole operative di helpdesk Forma.Temp), con citazione della pagina.

---

## 1. Requisiti

| Requisito | Note |
|---|---|
| Python 3.10+ | sviluppato con 3.12 |
| [Ollama](https://ollama.com) in esecuzione | default `http://localhost:11434` |
| Modello di embedding `embeddinggemma:300m` | `ollama pull embeddinggemma:300m` (~600 MB) |
| Modello di generazione | nel `.env`: `OLLAMA_MODEL=...` (es. `Jadio/Qwen3_4b_instruct_q4km:latest`) |
| PDF del RAG | già incluso in `data/RAG_helpdesk.pdf` (818 pagine) |
| Connessione a Internet (solo la prima volta) | download del tokenizer del modello di embedding da HuggingFace (~35 MB) |

Nessuna chiave API è obbligatoria: `src/utils.py` usa Ollama. Il ramo DeepSeek è presente ma
**disattivato** (la riga `deepseek_key = os.getenv("DEEPSEEK_API_KEY")` è commentata e la variabile
è forzata a stringa vuota): per usarlo va scommentata quella riga.

## 2. Installazione

Tutti i comandi vanno eseguiti dalla **root del progetto** (la cartella che contiene `main.py`).

```bash
git clone URL_DEL_REPO LISAI
cd LISAI

# ambiente virtuale (se non esiste già)
python -m venv .venv

# dipendenze
.venv/Scripts/python.exe -m pip install -r requirements.txt        # Windows
# source .venv/bin/activate && pip install -r requirements.txt     # Linux/macOS

# modelli Ollama
ollama pull embeddinggemma:300m
ollama pull NOME_DEL_MODELLO_DI_GENERAZIONE
```

`requirements.txt` include anche le dipendenze della parte LIS (mediapipe, opencv, torch) che
**non servono al RAG**. Per un ambiente dedicato solo all'API bastano:

```bash
.venv/Scripts/python.exe -m pip install fastapi "uvicorn[standard]" pydantic python-dotenv requests "ollama>=0.6" openai "chromadb>=1.5,<2" "pypdf>=4" "tokenizers>=0.15" "huggingface-hub>=0.23" numpy
```

> Attenzione: `requirements.txt` pinna `numpy<2` per mediapipe. Se l'ambiente funziona già con
> numpy 2.x, non rilanciare l'installazione completa: il RAG gira con entrambe le versioni.

## 3. Configurazione (`.env`)

Il file `.env` nella root del progetto viene caricato automaticamente da `src/utils.py`.
È già in `.gitignore`.

```env
# Generazione (LLM)
OLLAMA_MODEL=Jadio/Qwen3_4b_instruct_q4km:latest
OLLAMA_HOST=http://localhost:11434

# RAG (tutti opzionali, questi sono i default)
# EMBEDDING_MODEL=embeddinggemma:300m
# EMBEDDING_TOKENIZER=unsloth/embeddinggemma-300m
# EMBEDDING_DIMENSIONI=512
# RAG_COLLECTION_NAME=helpdesk_embeddinggemma_300m

# Server
# HOST=127.0.0.1
# PORT=8000
```

| Variabile | Default | Effetto |
|---|---|---|
| `OLLAMA_MODEL` | `llama3.2` | modello usato per generare le risposte |
| `OLLAMA_HOST` | `http://localhost:11434` | endpoint Ollama |
| `EMBEDDING_MODEL` | `embeddinggemma:300m` | modello di embedding (deve essere scaricato in Ollama) |
| `EMBEDDING_TOKENIZER` | `unsloth/embeddinggemma-300m` | repo HuggingFace del tokenizer usato per il chunking |
| `EMBEDDING_DIMENSIONI` | *(nativa, 768)* | troncamento MRL dei vettori (es. 512 o 256) |
| `EMBEDDING_PREFISSO_DOCUMENTO` / `_QUERY` | prompt EmbeddingGemma | prefissi applicati a documenti e domande |
| `RAG_COLLECTION_NAME` | `helpdesk_<modello>` | nome della collezione ChromaDB |
| `HOST` / `PORT` | `127.0.0.1` / `8000` | indirizzo del server |

Cambiando `EMBEDDING_MODEL` cambia anche il nome della collezione, quindi non c'è conflitto con
gli indici costruiti con altri modelli.

## 4. Avvio

```bash
.venv/Scripts/python.exe main.py
```

- API: `http://127.0.0.1:8000`
- Documentazione interattiva (Swagger): **`http://127.0.0.1:8000/docs`** — è il modo più rapido
  per provare una domanda senza scrivere codice.

## 5. API disponibili

| Metodo | Path | Descrizione |
|---|---|---|
| `GET` | `/ai/test` | health check |
| `POST` | `/ai/generate` | risposta LLM libera, senza documento |
| `POST` | `/ai/rag` | domanda sul PDF (RAG) |

### `GET /ai/test`

```json
{"status": "OK", "message": "API di test funzionante!"}
```

### `POST /ai/generate`

Body:

```json
{"prompt": "Ciao, chi sei?", "system_prompt": "Sei un assistente utile."}
```

Risposta:

```json
{"status": "success", "result": {"status": "success", "output": "..."}}
```

> Il testo generato è in `result.output` (annidato: `chiama_llm_json` restituisce già
> `{status, output}` e l'endpoint lo incapsula in `result`).

### `POST /ai/rag`

Body:

| Campo | Tipo | Default | Note |
|---|---|---|---|
| `domanda` | string | *obbligatorio* | la domanda sul documento |
| `k` | int 1-10 | `4` | quanti chunk di contesto recuperare |

```json
{"domanda": "Come si apre un ticket di assistenza?", "k": 3}
```

Risposta:

```json
{
  "status": "success",
  "output": "testo della risposta, con la pagina citata",
  "fonti": [{"fonte": "RAG_helpdesk.pdf", "pagina": 4}],
  "contesto": "[Pagina 4]\nParte 1 — Il canale di assistenza Forma.Temp ..."
}
```

| Campo | Uso |
|---|---|
| `output` | la risposta da mostrare all'utente |
| `fonti` | pagine da cui proviene il contesto (per citazioni/verifica) |
| `contesto` | i chunk recuperati: serve a capire se sbaglia il retrieval o l'LLM |

Errori: `500` con `{"detail": "..."}` (Ollama spento, modello mancante, PDF assente, ecc.).

Esempi:

```bash
curl -X POST http://127.0.0.1:8000/ai/rag \
  -H "Content-Type: application/json" \
  -d "{\"domanda\": \"Come si apre un ticket di assistenza?\", \"k\": 3}"
```

```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/ai/rag",
    json={"domanda": "Come si apre un ticket di assistenza?", "k": 3},
    timeout=600,  # la prima chiamata può indicizzare: minuti
)
print(response.json()["output"])
```

Client di esempio completo: **`LIS.py`**.

## 6. Il RAG in pratica

### Come funziona

1. `pypdf` estrae il testo di ogni pagina del PDF;
2. ogni pagina è divisa in chunk da **500 token** con 60 token di sovrapposizione, usando il
   tokenizer del modello di embedding (`tokenizers` + `huggingface-hub`);
3. gli embedding sono calcolati da **Ollama** e salvati in **ChromaDB** persistente (`chroma_db/`,
   giù in `.gitignore`), spazio coseno;
4. per ogni domanda si recuperano i `k` chunk più simili e si passano come contesto a `chiama_llm`
   (`src/utils.py`), che genera la risposta.

Valori attuali sul PDF incluso: 818 pagine, ~492.000 token, **~1.454 chunk**.

### Costruire l'indice

```bash
# solo se manca (idempotente)
.venv/Scripts/python.exe -m src.handlers.rag_handler

# ricostruzione da zero (cancella l'indice esistente)
.venv/Scripts/python.exe -c "from src.handlers.rag_handler import indicizza_pdf; print(indicizza_pdf(force=True))"
```

Output atteso:

```json
{"status": "success", "messaggio": "PDF indicizzato", "collezione": "helpdesk_embeddinggemma_300m",
 "modello_embedding": "embeddinggemma:300m", "pagine": 818, "chunk": 1454}
```

Durata: alcuni minuti su CPU (è un'operazione una tantum). **Non** viene eseguita all'avvio del
server: la prima `POST /ai/rag` costruisce l'indice da sola se la collezione è vuota, e in quel caso
la richiesta resta aperta per minuti (usare `timeout` alto lato client).

### Domanda da terminale

```bash
.venv/Scripts/python.exe -m src.handlers.rag_handler "Come si apre un ticket di assistenza?"
```

### Ispezionare l'indice

```bash
.venv/Scripts/python.exe -c "from src.handlers.rag_handler import get_collection; c=get_collection(); print(c.name, c.count())"
```

### Quando va ricostruito

| Cambio | Ricostruire? |
|---|---|
| il PDF in `data/` | sì |
| `EMBEDDING_MODEL` (nuova collezione) | sì (automatico, nome diverso) |
| `CHUNK_TOKENS` / `CHUNK_OVERLAP_TOKENS` | sì |
| `EMBEDDING_DIMENSIONI` | sì (stessa collezione, dimensione diversa) |

`force=True` cancella **prima** di ricostruire: se Ollama è spento o il modello manca, si resta
senza indice.

## 7. Struttura del progetto

```
LISAI/
├── main.py                     # app FastAPI + uvicorn (HOST/PORT)
├── LIS.py                      # client di esempio (requests) verso l'API
├── requirements.txt
├── .env                        # configurazione locale (gitignored)
├── data/RAG_helpdesk.pdf       # documento indicizzato
├── chroma_db/                  # indice ChromaDB generato (gitignored)
└── src/
    ├── routes.py               # endpoint /ai/test, /ai/generate, /ai/rag
    ├── utils.py                # chiama_llm: Ollama (ramo DeepSeek commentato)
    └── handlers/
        ├── llm_handler.py      # wrapper {status, output}
        └── rag_handler.py      # estrazione PDF, chunking, ChromaDB, RAG
```

## 8. Problemi comuni

| Sintomo | Causa / rimedio |
|---|---|
| `ConnectionError` in `LIS.py` | server non avviato: lanciare `main.py` |
| `500 ... model required but not found` | manca il modello: `ollama pull embeddinggemma:300m` o quello in `OLLAMA_MODEL` |
| `Read timed out` alla prima domanda | sta indicizzando: alzare il timeout (600-3600 s) e riprovare dopo qualche minuto |
| `401 ... gated repo` sul tokenizer | `EMBEDDING_TOKENIZER` punta a un repo gated: usare `unsloth/embeddinggemma-300m` |
| risposte fuori tema | il retrieval non trova il passaggio: controllare `contesto`, alzare `k`, ridurre `CHUNK_TOKENS` |
| `dimension mismatch` su Chroma | cambiata la dimensione dei vettori: `indicizza_pdf(force=True)` |
| l'LLM ignora il documento | `OLLAMA_MODEL` non impostata nel `.env`: si sta usando `llama3.2` |

## 9. Limiti noti

- Non esiste ancora un endpoint per (ri)costruire l'indice: si usa la CLI (sezione 6).
- Nessuna soglia di distanza: se la domanda è fuori tema vengono passati comunque i `k` chunk più
  simili; il system prompt chiede al modello di ammettere quando non sa, ma non è una garanzia.
- Nessuna cronologia: ogni domanda è indipendente.
- `HOST` è `127.0.0.1`: il servizio è raggiungibile solo dalla macchina locale.
- L'indicizzazione è sincrona: una sola alla volta, e blocca la richiesta che l'ha avviata.
