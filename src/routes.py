from fastapi import APIRouter

# Creiamo il router
router = APIRouter(
    prefix="/LIS",     # Prefisso dell'URL
    tags=["LIS"]       # Cose di default
)

# "/" Perchè è il metodo default altrimenti "/metodo_a_caso..."
@router.get("/test")
def test_endpoint():
    return {"status": "ok", "message": "API di test funzionante!"}