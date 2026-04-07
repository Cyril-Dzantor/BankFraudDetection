from fastapi import APIRouter

router = APIRouter()

@router.get("/")
def get_health():
    return {"status": "ok", "version": "0.1.0"}
