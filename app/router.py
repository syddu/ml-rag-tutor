from fastapi import APIRouter
import endpoint

router = APIRouter()

router.include_router(endpoint.router, prefix = "/query", tags = ["query"])