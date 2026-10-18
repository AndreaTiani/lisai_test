from src.utils import chiama_llm

async def chiama_llm_json(prompt: str, system_prompt: str = "") -> dict:
    # Esegue la chiamata tramite la funzione in utils.py
    resposta_testo = await chiama_llm(prompt=prompt, system_prompt=system_prompt)

    return {
        "status": "success",
        "output": resposta_testo
    }