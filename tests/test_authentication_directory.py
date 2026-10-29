"""Authentication and directory. Who is in the ministry fixture comes from tests/conftest_ministry.py."""

import pytest

from docflow.authentication_directory import AuthenticationDirectory, Unauthenticated
from conftest_ministry import world  # noqa: F401


def test_a_token_belongs_to_the_one_person_it_was_issued_to(world):
    directory = AuthenticationDirectory(world)
    ann, ext = directory.issue("ann"), directory.issue("ext")
    assert directory.authenticate(ann) == "ann" and directory.authenticate(ext) == "ext"
    assert ann != ext and directory.issue("ann") != ann  # a new token each time
    assert directory.authenticate(directory.issue("ann")) == "ann"  # and a second one for the same person works too


def test_a_stranger_gets_the_same_refusal_whatever_is_wrong_with_the_token(world):
    directory = AuthenticationDirectory(world)
    real = directory.issue("ann")
    wrong = ["", " ", "nope", real[:-1], real + "x", real.upper(), real.strip("dfl_"), None, 5, b"x", ["ann"], "x" * 5000, "ann"]
    messages = set()
    for token in wrong:
        with pytest.raises(Unauthenticated) as refused:
            directory.authenticate(token)
        messages.add((type(refused.value), str(refused.value), refused.value.args))
    assert len(messages) == 1  # one type, one message: nothing says what was wrong


def test_a_revoked_token_and_a_removed_person_have_no_session(world):
    directory = AuthenticationDirectory(world)
    token, other = directory.issue("ann"), directory.issue("bob")
    directory.revoke(token)
    directory.revoke("never issued")
    directory.revoke(None)  # harmless
    with pytest.raises(Unauthenticated):
        directory.authenticate(token)
    assert directory.authenticate(other) == "bob"
    del world.users["bob"]
    with pytest.raises(Unauthenticated):
        directory.authenticate(other)


def test_only_a_hash_of_the_token_is_kept(world):
    directory = AuthenticationDirectory(world)
    token = directory.issue("ann")
    for text in (repr(directory), repr(vars(directory)), str(list(directory._sessions)), str(list(directory._sessions.values()))):
        assert token not in text and token[4:] not in text


def test_a_token_cannot_be_issued_for_someone_the_host_does_not_know(world):
    with pytest.raises(KeyError):
        AuthenticationDirectory(world).issue("nobody")


def test_the_directory_answers_what_the_policy_asks_about_people(world):
    d = AuthenticationDirectory(world)
    # from tests/conftest_ministry.py
    assert d.unit_of("ann") == "U1" and d.unit_of("cat") == "U2" and d.unit_of("ext") is None
    assert set(d.users_in_unit("U2")) == {"cat", "m2", "c1", "c2", "c3", "c4", "c5"}
    assert d.groups_of("km1") == ("knowledge management",) and d.groups_of("ann") == ()
    assert d.role_in("ann", "ws") == "contributor" and d.role_in("u1b", "ws") == "viewer" and d.role_in("u1x", "ws") is None
    assert d.role_in("km1", "ws") == "editor" and d.role_in("c1", "ws") == "viewer"  # through a group
    assert set(d.people()) == {"ann", "bob", "cat", "u1b", "u1x", "m2", "km1", "c1", "c2", "c3", "c4", "c5", "ext"}


def test_the_directory_hands_out_copies(world):
    d = AuthenticationDirectory(world)
    person = d.person("ann")
    person.unit = "U9"
    person.attrs["clearance"] = "high"
    assert world.users["ann"].unit == "U1" and world.users["ann"].attrs == {}
