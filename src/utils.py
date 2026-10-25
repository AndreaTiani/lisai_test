import os
from pathlib import Path
from typing import List, Dict, Optional
from dotenv import load_dotenv
from ollama import AsyncClient
from openai import AsyncOpenAI

# Individua la cartella genitore dell'env
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

async def chiama_llm(
    prompt: str, 
    system_prompt: str = "", 
    history: Optional[List[Dict[str, str]]] = None
) -> str:

    deepseek_key = os.getenv("DEEPSEEK_API_KEY")

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": prompt})

    # Se DEEPSEEK_API_KEY è presente, usa DeepSeek
    if deepseek_key:
        model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        client = AsyncOpenAI(
            api_key=deepseek_key,
            base_url="https://api.deepseek.com"
        )
        
        response = await client.chat.completions.create(
            model=model,
            messages=messages
        )
        return response.choices[0].message.content

    # Altrimenti usa Ollama
    else:
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        model = os.getenv("OLLAMA_MODEL", "llama3.2")

        client = AsyncClient(host=ollama_host)
        response = await client.chat(
            model=model,
            messages=messages
        )
        
        if isinstance(response, dict):
            return response["message"]["content"]
        return response.message.content