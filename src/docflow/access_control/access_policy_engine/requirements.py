"""Requirements: something an action waits for before the document moves (`requires:`).

A type is a class with `satisfied`, optionally `needed` and `problems`,
registered under the name workflow files use:

    requirements.register("signature", Signature)
"""

from __future__ import annotations

from typing import Any, Protocol

from ...support.model import Context, Refused


class Requirement(Protocol):
    def satisfied(self, settings: dict[str, Any], ctx: Context) -> bool: ...


class Approval:
    """Met once enough people have approved the current version.

    `pass: any` (one person), `pass: 2` (that many), `pass: all` (everyone
    the action's `by` allows). Reads `ctx.given` (people who have approved,
    counting the one approving now) and `ctx.asked` (for `all`).
    """

    settings = ("pass", "choices")

    def needed(self, settings: dict[str, Any], ctx: Context) -> int:
        match settings.get("pass", "any"):
            case "any":
                return 1
            case "all":
                return ctx.asked
            case int(count):
                return count
        return 1

    def satisfied(self, settings: dict[str, Any], ctx: Context) -> bool:
        return ctx.given >= self.needed(settings, ctx)

    def problems(self, settings: dict[str, Any]) -> list[str]:
        value = settings.get("pass", "any")
        if value in ("any", "all") or (isinstance(value, int) and not isinstance(value, bool) and value >= 1):
            return []
        return [f"`pass` must be `any`, `all` or a whole number of people, got {value!r}"]


Approval.counts_people = True  # an unmet approval records the actor's part and waits for more people


class Signature(Approval):
    """Met once enough people have signed the current version.

    Unlike an approval, a signature is bound to what was signed: the person gives `input["signature"]`, a
    mapping with `ref` (their signature's reference) and `digest` (of the version they signed), and the digest
    must be `ctx.digest`, the host's digest of the current version. The signature is kept as evidence
    (who, which version, the reference and digest). A new version invalidates earlier signatures, as it restarts
    any count. The engine checks the binding and keeps the record; whether the signature is genuine (a
    certificate, a signing service) is for the host to verify before it passes it in.
    """

    def validate(self, settings: dict[str, Any], ctx: Context) -> None:
        sig = ctx.input.get("signature")
        if not (isinstance(sig, dict) and isinstance(sig.get("ref"), str) and sig["ref"] and isinstance(sig.get("digest"), str)):
            raise Refused("signature_needed", needs="input signature: {ref, digest}")
        if ctx.digest is None or sig["digest"] != ctx.digest:
            raise Refused("signature_invalid", why="it is not a signature of the current version")

    def evidence(self, settings: dict[str, Any], ctx: Context) -> dict[str, Any]:
        sig = ctx.input["signature"]
        return {"ref": sig["ref"], "digest": sig["digest"]}


_types: dict[str, Any] = {"approval": Approval(), "signature": Signature()}


def register(name: str, requirement: Any) -> None:
    _types[name] = requirement


def types() -> dict[str, Any]:
    return dict(_types)


def satisfied(requires: dict[str, Any] | None, ctx: Context) -> bool:
    return requires is None or _types[requires["type"]].satisfied(requires, ctx)


def problems(requires: dict[str, Any]) -> list[str]:
    check = getattr(_types[requires["type"]], "problems", None)
    return check(requires) if check else []


def counts_people(requires: dict[str, Any]) -> bool:
    return bool(getattr(_types[requires["type"]], "counts_people", False))


def validate(requires: dict[str, Any] | None, ctx: Context) -> None:
    """Lets a requirement type refuse before anything is recorded (a signature that isn't of this version)."""
    check = requires and getattr(_types[requires["type"]], "validate", None)
    if check:
        check(requires, ctx)


def evidence(requires: dict[str, Any] | None, ctx: Context) -> dict[str, Any] | None:
    """What a requirement keeps about the person's part (a signature's reference and digest), if it keeps anything."""
    make = requires and getattr(_types[requires["type"]], "evidence", None)
    return make(requires, ctx) if make else None
