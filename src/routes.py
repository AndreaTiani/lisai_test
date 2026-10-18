from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from src.handlers.llm_handler import chiama_llm_json

# Classe per richieste dell'utente per LLM
class PromptInput(BaseModel):
    prompt: str = Field(..., description="Il prompt dell'utente")
    system_prompt: str = Field(default="Sei un assistente utile.", description="System prompt")

# Creiamo il router
router = APIRouter(
    prefix="/ai",     # Prefisso dell'URL
    tags=["ai"]       # Cose di default
)


@router.get("/test")
def test_endpoint():
    return {
        "status": "OK",
        "message": "API di test funzionante!"
        }

@router.post("/generate")
async def generate_response(data: PromptInput):
    try:
        # Di default FastAPI assegna già il codice 200 ad esecuzioni senza errori, non serve formattare
        llm_text = await chiama_llm_json(prompt=data.prompt, system_prompt=data.system_prompt)
        return {
            "status": "success",
            "result": llm_text
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Errore durante l'esecuzione dell'LLM locale: {str(e)}"
        )