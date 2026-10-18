import uvicorn
import os
from fastapi import FastAPI
from src.routes import router as router

HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", 8000))

app = FastAPI(title="LISAI API")

# Includiamo il router
app.include_router(router)

# Esegue il Server (va sempre eseguito prima dell'app principale)
if __name__ == "__main__":
    uvicorn.run("main:app", host=HOST, port=PORT, reload=False)