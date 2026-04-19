from fastapi import APIRouter
from fastapi.responses import Response

router = APIRouter()


@router.get("/health")
def health_check():
    return Response("OK")
