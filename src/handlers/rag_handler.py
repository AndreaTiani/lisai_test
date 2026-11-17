import asyncio
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import chromadb
import numpy as np
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.api.types import (
    Documents,
    Embeddable,
    EmbeddingFunction,
    Embeddings,
    Metadata,
    Space,
)
from huggingface_hub import hf_hub_download
from ollama import Client
from pypdf import PdfReader
from tokenizers import SentencePieceUnigramTokenizer, Tokenizer

from src.utils import chiama_llm

# ============================== Configurazione ==============================

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PDF_PATH = BASE_DIR / "data" / "RAG_helpdesk.pdf"
CHROMA_PATH = BASE_DIR / "chroma_db"

# Va scaricato una volta con:  ollama pull embeddinggemma:300m
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "embeddinggemma:300m")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# Repo HuggingFace da cui prendere il tokenizer (quello del modello di embedding).
# Nota: google/embeddinggemma-300m e' un repo gated (401 senza login), quindi
# usiamo il mirror non gated di unsloth (stesso vocab).
TOKENIZER_REPO = os.getenv("EMBEDDING_TOKENIZER", "unsloth/embeddinggemma-300m")

# Troncamento MRL opzionale (es. 512 o 256): None = dimensione nativa (768)
DIMENSIONI = (
    int(os.environ["EMBEDDING_DIMENSIONI"])
    if os.getenv("EMBEDDING_DIMENSIONI")
    else None
)

# Il nome della collezione dipende dal modello: vettori di modelli diversi
# hanno dimensioni e spazi diversi e non sono confrontabili tra loro.
COLLECTION_NAME = os.getenv("RAG_COLLECTION_NAME") or (
    "helpdesk_" + re.sub(r"[^a-zA-Z0-9._-]", "_", EMBEDDING_MODEL)
)

CHUNK_TOKENS = 500          # token per chunk (contesto del modello: 2048)
CHUNK_OVERLAP_TOKENS = 60   # token di sovrapposizione tra chunk consecutivi
N_RISULTATI = 3             # chunk recuperati per ogni domanda
BATCH_SIZE = 32             # chunk inviati a Ollama per ogni add

# Prompt consigliati dal modello (facoltativi: migliorano il retrieval)
PREFISSO_DOCUMENTO = os.getenv("EMBEDDING_PREFISSO_DOCUMENTO", "title: none | text: ")
PREFISSO_QUERY = os.getenv(
    "EMBEDDING_PREFISSO_QUERY", "task: search result | query: "
)

SYSTEM_PROMPT_RAG = ("""
Sei un assistente esperto delle policy e delle regole operative di helpdesk
descritte nella documentazione Forma.Temp. Rispondi usando esclusivamente
le informazioni presenti nel contesto fornito: se la risposta non è nel
contesto, dillo chiaramente invece di inventare. Cita la pagina da cui
proviene l'informazione e rispondi in italiano.
La domanda ti verrà data dall'operatore e non dall'utente che sta
usufruendo del servizio quindi rispondi a modo.
""")

# Client, collezione e tokenizer sono costosi da creare: li teniamo in cache
_client: Optional[ClientAPI] = None
_collection: Optional[Collection] = None
_tokenizer: Optional[Tokenizer] = None


# ============================== Embedding via Ollama ==============================

class OllamaEmbeddingFunction(EmbeddingFunction[Documents]):
    """Calcola gli embedding con un modello servito da Ollama.

    Chroma usa questa classe sia in indicizzazione sia in ricerca, quindi il
    modello deve restare lo stesso per tutta la vita della collezione.
    """

    def __init__(self, model: str = EMBEDDING_MODEL, host: str = OLLAMA_HOST):
        self._model = model
        self._host = host
        self._client = Client(host=host)

    def __call__(self, input: Documents) -> Embeddings:
        # Il nome del parametro deve essere esattamente `input`:
        # Chroma valida la firma di __call__ all'avvio.
        return self._embed([PREFISSO_DOCUMENTO + testo for testo in input])

    def embed_query(self, input: Documents) -> Embeddings:
        # Chroma chiama questo metodo quando si usa query_texts
        return self._embed([PREFISSO_QUERY + testo for testo in input])

    def _embed(self, testi: Documents) -> Embeddings:
        testi = [testo if testo and testo.strip() else " " for testo in testi]
        if not testi:
            return []

        risposta = self._client.embed(
            model=self._model, input=testi, dimensions=DIMENSIONI
        )
        vettori = (
            risposta["embeddings"] if isinstance(risposta, dict) else risposta.embeddings
        )
        return [np.asarray(vettore, dtype=np.float32) for vettore in vettori]

    @staticmethod
    def name() -> str:
        return "ollama"

    def get_config(self) -> Dict[str, Any]:
        # Volutamente solo il modello: l'host è un dettaglio di deployment e
        # cambiarlo non deve invalidare la collezione già indicizzata.
        return {"model": self._model}

    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> "OllamaEmbeddingFunction":
        return OllamaEmbeddingFunction(model=config.get("model", EMBEDDING_MODEL))

    def default_space(self) -> Space:
        # embeddinggemma e simili: similarità coseno
        return "cosine"

    def max_tokens(self) -> int:
        return 2048


# ============================== ChromaDB ==============================

def get_client() -> ClientAPI:
    """Restituisce il client ChromaDB persistente (creato una sola volta)."""
    global _client
    if _client is None:
        CHROMA_PATH.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return _client


def get_collection() -> Collection:
    """Restituisce (o crea) la collezione con l'embedding function di Ollama."""
    global _collection
    if _collection is None:
        _collection = get_client().get_or_create_collection(
            name=COLLECTION_NAME,
            # cast: Chroma tipizza la EF su "Embeddable" (testo o immagini),
            # la nostra gestisce solo testo
            embedding_function=cast("EmbeddingFunction[Embeddable]", OllamaEmbeddingFunction()),
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


# ============================== Estrazione e chunking ==============================

def _carica_tokenizer() -> Tokenizer:
    """Scarica e carica il tokenizer, con fallback sul vocab sentencepiece."""
    try:
        percorso = hf_hub_download(repo_id=TOKENIZER_REPO, filename="tokenizer.json")
        return Tokenizer.from_file(percorso)
    except Exception as errore_json:
        try:
            percorso = hf_hub_download(
                repo_id=TOKENIZER_REPO, filename="tokenizer.model"
            )
            return SentencePieceUnigramTokenizer.from_spm(percorso)
        except Exception as errore_spm:
            raise RuntimeError(
                f"Impossibile scaricare il tokenizer da '{TOKENIZER_REPO}': "
                f"{errore_json} / {errore_spm}. Controlla la connessione o "
                "imposta EMBEDDING_TOKENIZER con un repo che contenga "
                "tokenizer.json o tokenizer.model."
            ) from errore_spm


def get_tokenizer() -> Tokenizer:
    """Carica (una sola volta) il tokenizer del modello di embedding.

    Il file viene scaricato dal repo HuggingFace e poi servito dalla cache
    locale di huggingface_hub. Alcuni modelli (es. la famiglia Gemma)
    pubblicano solo il vocab sentencepiece `tokenizer.model` e non
    `tokenizer.json`: in quel caso il tokenizer viene costruito da li'.
    """
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = _carica_tokenizer()
    return _tokenizer


def estrai_pagine_pdf(pdf_path: Path = PDF_PATH) -> List[Dict[str, Any]]:
    """Estrae il testo di ogni pagina del PDF.

    Ritorna una lista di dizionari ``{"pagina": int, "testo": str}``.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF non trovato: {pdf_path}")

    lettore = PdfReader(str(pdf_path))
    pagine = []
    for numero, pagina in enumerate(lettore.pages, start=1):
        testo = pagina.extract_text() or ""
        testo = normalizza_testo(testo)
        if testo:
            pagine.append({"pagina": numero, "testo": testo})
    return pagine


def normalizza_testo(testo: str) -> str:
    """Ripulisce gli spazi e le righe vuote lasciati dall'estrazione del PDF."""
    testo = testo.replace("\r", "\n").replace("\u00ad", "")  # soft hyphen
    testo = re.sub(r"[ \t]+", " ", testo)
    testo = re.sub(r"\n{3,}", "\n\n", testo)
    return testo.strip()


def dividi_in_chunk(
    testo: str,
    max_tokens: int = CHUNK_TOKENS,
    overlap_token: int = CHUNK_OVERLAP_TOKENS,
) -> List[str]:
    """Divide il testo in chunk di circa ``max_tokens`` token.

    Il testo viene tokenizzato una volta con il tokenizer del modello di
    embedding; i chunk sono finestre sugli id dei token, ricostruite sul testo
    originale tramite gli offset (così il chunk è testo vero, non una
    decodifica approssimata).

    La finestra si chiude preferibilmente su un confine di frase e i chunk
    consecutivi si sovrappongono di ``overlap_token`` token.
    """
    if not testo.strip():
        return []

    tokenizer = get_tokenizer()
    encoding = tokenizer.encode(testo, add_special_tokens=False)
    offsets = encoding.offsets
    totale = len(offsets)
    if totale == 0:
        return []

    chunks: List[str] = []
    inizio = 0
    minimo = max(max_tokens // 2, 1)  # non chiudere un chunk troppo corto

    while inizio < totale:
        fine = min(inizio + max_tokens, totale)

        # Arretra fino a un confine di frase, ma non oltre metà chunk
        if fine < totale:
            for i in range(fine - 1, inizio + minimo - 1, -1):
                inizio_char, fine_char = offsets[i]
                token = testo[inizio_char:fine_char]
                if "\n" in token or token.rstrip().endswith((".", "!", "?", ":", ";")):
                    fine = i + 1
                    break

        # Ricostruisce il testo del chunk dai confini dei token
        inizio_char = offsets[inizio][0]
        fine_char = offsets[fine - 1][1]

        # Non far cominciare il chunk a metà di una parola
        if inizio_char > 0 and not testo[inizio_char - 1].isspace():
            spazio = testo.find(" ", inizio_char, fine_char)
            if spazio != -1:
                inizio_char = spazio + 1

        chunk = testo[inizio_char:fine_char].strip()
        if chunk:
            chunks.append(chunk)

        if fine >= totale:
            break

        # Sovrapposizione: il prossimo chunk riparte `overlap_token` prima
        inizio = max(fine - overlap_token, inizio + 1)

    return chunks


# ============================== Indicizzazione ==============================


def indicizza_pdf(force: bool = False) -> Dict[str, Any]:
    """Indicizza il PDF nella collezione ChromaDB.

    Se la collezione contiene già dei documenti non fa nulla (a meno di
    ``force=True``, che la svuota e la ricostruisce).
    """
    global _collection
    if force:
        try:
            get_client().delete_collection(COLLECTION_NAME)
        except Exception:
            pass  # la collezione non esisteva
        _collection = None

    collection = get_collection()
    if collection.count() > 0 and not force:
        return {
            "status": "success",
            "messaggio": "Collezione già indicizzata",
            "collezione": COLLECTION_NAME,
            "modello_embedding": EMBEDDING_MODEL,
            "chunk": collection.count(),
        }

    pagine = estrai_pagine_pdf()
    ids: List[str] = []
    documenti: List[str] = []
    metadati: List[Metadata] = []

    for pagina in pagine:
        for indice, chunk in enumerate(dividi_in_chunk(pagina["testo"])):
            ids.append(f"p{pagina['pagina']}-c{indice}")
            documenti.append(chunk)
            metadati.append(
                {"fonte": PDF_PATH.name, "pagina": pagina["pagina"], "chunk": indice}
            )

    # Batch piccoli: ogni add() è una chiamata HTTP a Ollama con N testi
    for inizio in range(0, len(documenti), BATCH_SIZE):
        fine = inizio + BATCH_SIZE
        collection.add(
            ids=ids[inizio:fine],
            documents=documenti[inizio:fine],
            metadatas=metadati[inizio:fine],
        )

    return {
        "status": "success",
        "messaggio": "PDF indicizzato",
        "collezione": COLLECTION_NAME,
        "modello_embedding": EMBEDDING_MODEL,
        "pagine": len(pagine),
        "chunk": len(documenti),
    }


# ============================== Ricerca e risposta ==============================


def cerca(query: str, k: int = N_RISULTATI) -> List[Dict[str, Any]]:
    """Recupera i ``k`` chunk più simili alla query."""
    if not query.strip():
        return []

    collection = get_collection()
    if collection.count() == 0:
        indicizza_pdf()

    risultati = collection.query(
        query_texts=[query],
        n_results=min(k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    documenti = risultati.get("documents") or [[]]
    metadati = risultati.get("metadatas") or [[]]
    distanze = risultati.get("distances") or [[]]

    return [
        {
            "testo": documento,
            "pagina": (metadato or {}).get("pagina"),
            "fonte": (metadato or {}).get("fonte"),
            "distanza": distanza,
        }
        for documento, metadato, distanza in zip(documenti[0], metadati[0], distanze[0])
    ]


async def chiama_rag(domanda: str, k: int = N_RISULTATI) -> Dict[str, Any]:
    """Recupera il contesto dal PDF e genera la risposta con l'LLM.

    Ritorna un dizionario nello stesso formato di ``chiama_llm_json``:
    ``{"status": "success", "output": ..., "fonti": [...], "contesto": ...}``.
    """
    # Chroma e Ollama sono sincroni: li eseguiamo in un thread per non
    # bloccare l'event loop del server
    chunks = await asyncio.to_thread(cerca, domanda, k)

    if not chunks:
        return {
            "status": "error",
            "output": "Nessun contenuto indicizzato: impossibile rispondere.",
            "fonti": [],
            "contesto": "",
        }

    contesto = "\n\n---\n\n".join(
        f"[Pagina {chunk['pagina']}]\n{chunk['testo']}" for chunk in chunks
    )

    prompt = f"Contesto estratto dalla documentazione:\n\n{contesto}\n\nDomanda: {domanda}"

    risposta = await chiama_llm(prompt=prompt, system_prompt=SYSTEM_PROMPT_RAG)

    fonti = []
    for chunk in chunks:
        fonte = {"fonte": chunk["fonte"], "pagina": chunk["pagina"]}
        if fonte not in fonti:
            fonti.append(fonte)

    return {
        "status": "success",
        "output": risposta,
        "fonti": fonti,
        "contesto": contesto,
    }


# ============================== Riga di comando ==============================

# Uso (dalla root del progetto):
#   python -m src.handlers.rag_handler              -> indicizza il PDF
#   python -m src.handlers.rag_handler "domanda"    -> indicizza (se serve) e risponde

if __name__ == "__main__":
    import sys

    print(indicizza_pdf())

    if len(sys.argv) > 1:
        domanda = " ".join(sys.argv[1:])
        risultato = asyncio.run(chiama_rag(domanda))
        print(f"\nDomanda: {domanda}")
        print(f"Fonti: {risultato['fonti']}")
        print(f"\nRisposta:\n{risultato['output']}")
