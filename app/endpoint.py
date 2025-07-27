import json
from http import HTTPStatus
from fastapi import APIRouter
from pydantic import BaseModel
from starlette.responses import Response
from model import ask

router = APIRouter()


class QuerySchema(BaseModel):
    """Query Schema"""

    query: str

"""
Becuase of the router, every endpoint in this file is prefixed with /query/
"""


@router.post("/", dependencies=[])
def handle_event(data: QuerySchema) -> Response:
    query = data.query
    response = ask(query)
    return Response(
        content=json.dumps({"answer": response}),
        status_code=HTTPStatus.ACCEPTED,
    )