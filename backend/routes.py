from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from backend.llm_handler import chiama_llm_json
from backend.rag_handler import chiama_rag


# Classe per richieste dell'utente per LLM
class PromptInput(BaseModel):
    prompt: str = Field(..., description="Il prompt dell'utente")
    system_prompt: str = Field(
        default="Sei un assistente utile.", description="System prompt"
    )


# Classe per richieste RAG (domande sul PDF in data/)
class RagInput(BaseModel):
    domanda: str = Field(..., description="La domanda dell'utente sul documento")
    k: int = Field(default=4, ge=1, le=10, description="Numero di chunk da recuperare")


# Creiamo il router
router = APIRouter(
    prefix="/ai",  # Prefisso dell'URL
    tags=["ai"],  # Cose di default
)


@router.get("/test")
def test_endpoint():
    return {"status": "OK", "message": "API di test funzionante!"}


@router.post("/generate")
async def generate_response(data: PromptInput):
    try:
        # Di default FastAPI assegna già il codice 200 ad esecuzioni senza errori, non serve formattare
        llm_text = await chiama_llm_json(
            prompt=data.prompt, system_prompt=data.system_prompt
        )
        return {"status": "success", "result": llm_text}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Errore durante l'esecuzione dell'LLM locale: {str(e)}",
        )


@router.post("/rag")
async def rag_response(data: RagInput):
    try:
        # Recupera i chunk dal PDF indicizzato e risponde con l'LLM
        return await chiama_rag(domanda=data.domanda, k=data.k)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Errore durante l'esecuzione del RAG: {str(e)}",
        )
