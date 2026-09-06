"""Хранилище: просмотренные посты и настройки конкретного чата."""

from __future__ import annotations

import pytest
from conftest import run

from ttbot.state import State


@pytest.fixture
def state(tmp_path) -> State:
    return State(str(tmp_path / "state.db"))


# --- просмотренное -------------------------------------------------------


def test_new_profile_is_not_bootstrapped(state):
    assert run(state.is_bootstrapped("user")) is False


def test_seed_marks_everything_seen(state):
    run(state.seed("user", ["1", "2"]))
    assert run(state.is_bootstrapped("user")) is True
    assert run(state.is_seen("user", "1")) is True
    assert run(state.is_seen("user", "3")) is False


def test_mark_seen_is_idempotent(state):
    run(state.mark_seen("user", "1"))
    run(state.mark_seen("user", "1"))
    assert run(state.is_seen("user", "1")) is True


def test_profiles_do_not_share_history(state):
    run(state.seed("a", ["1"]))
    assert run(state.is_seen("b", "1")) is False


def test_state_survives_reopen(state, tmp_path):
    run(state.seed("user", ["1"]))
    again = State(str(tmp_path / "state.db"))
    assert run(again.is_seen("user", "1")) is True


# --- настройки чата ------------------------------------------------------


def test_prefs_start_empty(state):
    assert run(state.prefs(1)) == {}


def test_set_and_read_pref(state):
    run(state.set_pref(1, "prompt", "Ты пират"))
    assert run(state.prefs(1)) == {"prompt": "Ты пират"}


def test_pref_is_overwritten_not_duplicated(state):
    run(state.set_pref(1, "prompt", "первый"))
    run(state.set_pref(1, "prompt", "второй"))
    assert run(state.prefs(1)) == {"prompt": "второй"}


def test_empty_value_clears_the_pref(state):
    run(state.set_pref(1, "prompt", "Ты пират"))
    run(state.set_pref(1, "prompt", ""))
    assert run(state.prefs(1)) == {}


def test_prefs_are_per_chat(state):
    run(state.set_pref(1, "prompt", "первый чат"))
    run(state.set_pref(2, "prompt", "второй чат"))
    assert run(state.prefs(1)) == {"prompt": "первый чат"}
    assert run(state.prefs(2)) == {"prompt": "второй чат"}


def test_clear_prefs_wipes_only_this_chat(state):
    run(state.set_pref(1, "prompt", "a"))
    run(state.set_pref(1, "size", "1024x1024"))
    run(state.set_pref(2, "prompt", "b"))
    assert run(state.clear_prefs(1)) == 2
    assert run(state.prefs(1)) == {}
    assert run(state.prefs(2)) == {"prompt": "b"}


def test_clear_prefs_on_empty_chat(state):
    assert run(state.clear_prefs(999)) == 0
