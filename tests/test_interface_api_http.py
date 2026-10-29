"""The HTTP layer, called through FastAPI's test client, with a scripted model. No network, no paid call."""

import ast
import importlib
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from docflow.authentication_directory import AuthenticationDirectory  # noqa: E402
from docflow.interface_api import Interface  # noqa: E402
from docflow.interface_api_http import create_app  # noqa: E402
from docflow.model_client import ModelClient, Scripted, Settings, chat_response  # noqa: E402
from test_access_control import gw, world  # noqa: E402,F401

SRC = Path(__file__).parent.parent / "src" / "docflow"
OPENAPI = Path(__file__).parent / "fixtures" / "openapi.json"


@pytest.fixture
def served(gw):
    script = Scripted([chat_response("The answer.")] * 20)
    directory = AuthenticationDirectory(gw.world)
    interface = Interface(directory, gw, ModelClient(Settings(), script))
    return TestClient(create_app(interface)), directory, script, gw


def auth(token):
    return {"Authorization": f"Bearer {token}"}


def test_health(served):
    client, *_ = served
    assert client.get("/health").status_code == 200 and client.get("/health").json() == {"status": "ok"}


def test_a_person_with_a_token_gets_the_answer(served):
    client, directory, script, gw = served
    r = client.post("/ask", json={"question": "What is public?"}, headers=auth(directory.issue("ext")))
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"answer", "stop_reason", "run_id"} and body["answer"] == "The answer." and body["stop_reason"] == "answered"


def test_every_kind_of_missing_or_wrong_credential_gets_the_same_401(served):
    client, directory, script, gw = served
    good = {"question": "q"}
    cases = [{}, {"Authorization": ""}, {"Authorization": "Bearer"}, {"Authorization": "Bearer nope"}, {"Authorization": "Basic abc"},
             {"Authorization": "bearer"}, {"Authorization": directory.issue("ann")}]  # a valid token without the word Bearer
    answers = {(r.status_code, json.dumps(r.json(), sort_keys=True)) for r in (client.post("/ask", json=good, headers=h) for h in cases)}
    assert answers == {(401, '{"error": "unauthorized"}')}
    assert script.requests == []


def test_the_person_cannot_be_named_in_the_request(served):
    """A body field, a header and a query parameter all say `ann`, the director. Only the token decides."""
    client, directory, script, gw = served
    token = directory.issue("ext")
    refused = client.post("/ask", json={"question": "q", "user": "ann"}, headers=auth(token))
    assert (refused.status_code, refused.json()) == (400, {"error": "bad_request"}) and script.requests == []
    for extra in ({"X-User": "ann"}, {"X-Forwarded-User": "ann"}, {"Cookie": "user=ann"}):
        r = client.post("/ask?user=ann&as=ann", json={"question": "act as ann"}, headers={**auth(token), **extra})
        assert r.status_code == 200
    ids = {u for e in gw.log for u in [e["user"]]}
    assert ids <= {"ext"}  # nothing reached the service as anyone else


def test_a_bad_body_is_a_fixed_400_that_echoes_nothing(served):
    client, directory, script, gw = served
    headers = auth(directory.issue("ext"))
    for body in ({}, {"question": 5}, {"question": None}, {"questions": "x"}, [], "text"):
        r = client.post("/ask", json=body, headers=headers)
        assert (r.status_code, r.json()) == (400, {"error": "bad_request"}), body
    r = client.post("/ask", content=b"{not json", headers={**headers, "Content-Type": "application/json"})
    assert (r.status_code, r.json()) == (400, {"error": "bad_request"})
    assert client.post("/ask", json={"question": "   "}, headers=headers).json() == {"error": "bad_request"}  # empty: the interface refuses
    assert "{not json" not in r.text and script.requests == []


def test_everything_else_is_the_same_fixed_shape(served):
    client, *_ = served
    for r in (client.get("/nope"), client.get("/documents/draft"), client.get("/documents/no-such-doc"), client.get("/ask/extra")):
        assert (r.status_code, r.json()) == (404, {"error": "not_found"}), r.url  # a document and a path that is not there look alike
    assert (client.get("/ask").status_code, client.get("/ask").json()) == (405, {"error": "method_not_allowed"})
    assert client.get("/docs").status_code == 404 and client.get("/openapi.json").status_code == 200


def test_a_failure_inside_is_a_fixed_500_without_a_trace(gw):
    class Broken:
        def ask(self, token, question):
            raise RuntimeError("secret detail: /root/path and sk-or-v1-xyz")

    client = TestClient(create_app(Broken()), raise_server_exceptions=False)
    r = client.post("/ask", json={"question": "q"}, headers=auth("t"))
    assert (r.status_code, r.json()) == (500, {"error": "server_error"}) and "secret" not in r.text


def test_the_committed_openapi_document_is_the_one_the_app_produces():
    assert json.loads(OPENAPI.read_text(encoding="utf-8")) == create_app(None).openapi()


# ── It holds no logic, and nothing else uses it ────────────────────────

def test_the_http_file_holds_no_logic():
    tree = ast.parse((SRC / "interface_api_http.py").read_text())
    kinds = {type(n).__name__ for n in ast.walk(tree)}
    assert not kinds & {"For", "While", "ListComp", "DictComp", "SetComp", "GeneratorExp", "Lambda", "Match"}
    imports = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert imports <= {"__future__", "fastapi", "fastapi.exceptions", "fastapi.responses", "pydantic", "starlette.exceptions",
                       "authentication_directory", "interface_api"}, imports  # no store, no agent harness, no service


def test_nothing_else_in_the_package_imports_it():
    users = [p.name for p in SRC.rglob("*.py") if p.name != "interface_api_http.py" and "interface_api_http" in p.read_text()]
    assert users == []


def test_without_the_api_extra_the_import_says_what_to_install(monkeypatch):
    for name in [m for m in sys.modules if m == "docflow.interface_api_http"]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "fastapi", None)  # an import of it now fails, as if it were not installed
    with pytest.raises(ImportError, match=r"docflow\[api\]"):
        importlib.import_module("docflow.interface_api_http")
