"""HTTP layer over `interface_api.Interface.ask`. It translates HTTP to `ask` and back and holds no other logic.

    POST /ask     Authorization: Bearer <token>, body {"question": "..."}  ->  {"answer", "stop_reason", "run_id"}
    GET  /health  ->  {"status": "ok"}

Every refusal has a fixed shape, {"error": "<word>"}: unauthorized (401), bad_request (400), not_found (404),
method_not_allowed (405), server_error (500). No message, no detail, no echo of what was sent.

Needs the optional `api` extra (fastapi, uvicorn). Nothing else in the package imports this file.
    python -m docflow.interface_api_http openapi > tests/fixtures/openapi.json
"""

from __future__ import annotations

try:
    from fastapi import FastAPI, Header
    from fastapi.exceptions import RequestValidationError
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, ConfigDict
    from starlette.exceptions import HTTPException
except ImportError as error:  # the rest of the package works without it
    raise ImportError("docflow.interface_api_http needs the optional 'api' extra (fastapi, uvicorn): pip install 'docflow[api]'") from error

from .authentication_directory import Unauthenticated
from .interface_api import BadQuestion


class AskBody(BaseModel):
    model_config = ConfigDict(extra="forbid")  # a field such as "user" is refused, not ignored
    question: str


class AskReply(BaseModel):
    answer: str | None
    stop_reason: str
    run_id: str


class Error(BaseModel):
    error: str


WORDS = {400: "bad_request", 401: "unauthorized", 404: "not_found", 405: "method_not_allowed"}


def refusal(status: int) -> JSONResponse:
    return JSONResponse({"error": WORDS.get(status, "server_error")}, status_code=status)


def create_app(interface) -> FastAPI:
    app = FastAPI(title="StateGate", version="0.1.0", docs_url=None, redoc_url=None)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/ask", response_model=AskReply, responses={400: {"model": Error}, 401: {"model": Error}})
    def ask(body: AskBody, authorization: str | None = Header(default=None)):
        scheme, _, token = (authorization or "").partition(" ")
        try:
            answer = interface.ask(token if scheme.lower() == "bearer" else None, body.question)
        except Unauthenticated:
            return refusal(401)
        except BadQuestion:
            return refusal(400)
        return AskReply(answer=answer.answer, stop_reason=answer.stop_reason, run_id=answer.run_id)

    @app.exception_handler(RequestValidationError)
    async def invalid(request, exc):
        return refusal(400)

    @app.exception_handler(HTTPException)
    async def http_error(request, exc):
        return refusal(exc.status_code)

    @app.exception_handler(Exception)
    async def failure(request, exc):
        return refusal(500)

    return app


if __name__ == "__main__":
    import json
    import sys

    if sys.argv[1:] == ["openapi"]:
        print(json.dumps(create_app(None).openapi(), indent=2, ensure_ascii=False))
