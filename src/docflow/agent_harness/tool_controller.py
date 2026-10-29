"""Tool controller (read only): what an agent acting for one person may call.

The same channels as the access control, already bound to the person, so a tool call cannot name another person.
Results are plain JSON values with a fixed set of fields. A document the person may not open comes back as
`{"error": "not_found"}`, the same for one that is hidden and one that does not exist.
"""

from __future__ import annotations

from ..access_control import enforcement
from ..access_control.enforcement import NotFound

NOT_FOUND = {"error": "not_found"}
UNKNOWN_TOOL = {"error": "unknown_tool"}
BAD_REQUEST = {"error": "bad_request"}

SCHEMAS = [
    {"type": "function", "function": {
        "name": "list_documents",
        "description": "List the documents you may open, each with its state and how to treat it. Optionally only one workspace.",
        "parameters": {"type": "object", "properties": {"workspace": {"type": "string"}}, "required": [], "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "get_document",
        "description": "Open one document by id and read its passages.",
        "parameters": {"type": "object", "properties": {"doc_id": {"type": "string"}, "version": {"type": "integer"}},
                       "required": ["doc_id"], "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "search",
        "description": "Search the passages of the documents you may read, by words.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer", "minimum": 1, "maximum": enforcement.MAX_SEARCH_LIMIT}},
                       "required": ["query"], "additionalProperties": False}}},
]

# argument -> (type, required), per tool: anything else is refused
_ARGS = {
    "list_documents": {"workspace": (str, False)},
    "get_document": {"doc_id": (str, True), "version": (int, False)},
    "search": {"query": (str, True), "limit": (int, False)},
}


def _passage(p) -> dict:
    return {"seq": p.seq, "pages": list(p.pages), "text": p.content[:enforcement.MAX_PASSAGE_CHARS]}


def tools(service, user: str) -> dict:
    def list_documents(workspace=None):
        docs = service.list_documents(user, workspace)
        return {"documents": [{"id": d.id, "version": d.version, "state": d.state, "workflow": d.workflow, "guidance": d.guidance}
                              for d in docs[:enforcement.MAX_LIST]],
                "truncated": len(docs) > enforcement.MAX_LIST}

    def get_document(doc_id, version=None):
        try:
            doc = service.get_document(user, doc_id, version)
        except NotFound:
            return dict(NOT_FOUND)
        passages = doc["passages"]
        return {"id": doc["id"], "version": doc["version"], "state": doc["state"], "guidance": doc["guidance"],
                "passages": [_passage(p) for p in passages[:enforcement.MAX_GET_PASSAGES]],
                "truncated": len(passages) > enforcement.MAX_GET_PASSAGES or any(len(p.content) > enforcement.MAX_PASSAGE_CHARS for p in passages[:enforcement.MAX_GET_PASSAGES])}

    def search(query, limit=5):
        hits = service.search(user, query, limit)
        return {"hits": [{"id": h.doc, "version": h.version, "state": h.state, "guidance": h.guidance, "score": h.score,
                          "passage": _passage(h.passage)} for h in hits]}

    return {"list_documents": list_documents, "get_document": get_document, "search": search}


def call(bound: dict, name: str, arguments: dict | None = None) -> dict:
    """Runs the tool the model named with the arguments it gave. Another name, an unknown argument or a wrong type is refused."""
    spec = _ARGS.get(name) if isinstance(name, str) else None
    if spec is None or name not in bound:
        return dict(UNKNOWN_TOOL)
    arguments = {} if arguments is None else arguments
    if not isinstance(arguments, dict) or set(arguments) - set(spec):
        return dict(BAD_REQUEST)
    for arg, (kind, required) in spec.items():
        if arg not in arguments:
            if required:
                return dict(BAD_REQUEST)
            continue
        value = arguments[arg]
        if arg in ("workspace", "version") and value is None:
            continue
        if not isinstance(value, kind) or isinstance(value, bool):
            return dict(BAD_REQUEST)
    return bound[name](**arguments)
