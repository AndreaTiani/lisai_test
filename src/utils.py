import os
from ollama import AsyncClient

# Async ci serve per gestire funzioni che richiedono tempo
# Permette ad altre cose di essere eseguite mentre questa finisce di lavorare
async def chiama_llm(prompt: str, system_prompt: str = "") -> str:
    OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:latest")

    client = AsyncClient(host=OLLAMA_HOST)

    # Questa cosa impiega molto tempo, per questo usiamo Async
    response = await client.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    )

    return response["message"]["content"]