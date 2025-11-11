"""
Comprehensive unit-tests for the authentication logic in `api.api`.
- Patch **`sqlitedict.SqliteDict`** before importing `api.api`, so no real DB files are touched.
- Patch **`os.getenv`** *and* **`api.constants.USE_USERTYPE_LOGIN`** before import so each conditional branch in the module is exercised deterministically.
- Every test stops its patchers explicitly (one statement per line) to guarantee isolation.
- Clear, verbose variable names improve readability.
"""
import importlib
import os
from typing import Callable
from unittest.mock import MagicMock, patch

import pytest

from types import ModuleType


# SqliteDict side-effect helpers
def sd_side_effect_with_test_user(*_a, **kwargs) -> MagicMock:
    """Return a mocked context-manager containing 'testuser' for full/edu/eval."""
    table_name = kwargs.get("tablename")
    context_mock = MagicMock()
    context_mock.__enter__.return_value = ({"testuser": "hashed_pw"} if table_name in {"full", "edu", "eval"} else {})
    return context_mock


def sd_side_effect_with_legacy_user(*_a, **_k) -> MagicMock:
    context_mock = MagicMock()
    context_mock.__enter__.return_value = {"legacyuser": "hashed_pw"}
    return context_mock


def sd_side_effect_empty_db(*_a, **_k) -> MagicMock:
    context_mock = MagicMock()
    context_mock.__enter__.return_value = {}
    return context_mock


# Fixture: reload api.api cleanly
@pytest.fixture
def load_fresh_api(monkeypatch):
    """
    Reload `api.api` with deterministic environment and patches.

    Parameters (via kwargs on call)
    --------------------------------
    env_vars : dict[str, str]        → environment variables for os.getenv
    sqlite_side_effect : Callable    → side-effect for each SqliteDict(...)
    use_usertype_login : bool        → value for constants.USE_USERTYPE_LOGIN
    """

    def _loader(
        *,
        env_vars: dict[str, str] | None = None,
        sqlite_side_effect: Callable | None = None,
        use_usertype_login: bool = False,
    ) -> tuple[ModuleType, patch, patch, patch]:  # type: ignore
        # patch os.getenv
        merged_env = dict(os.environ)
        if env_vars:
            merged_env.update(env_vars)
        getenv_patch = patch(
            "os.getenv",
            side_effect=lambda key, default=None: merged_env.get(key, default),
        )
        getenv_patch.start()

        # patch the constants flag before import
        constants_patch = patch(
            "api.constants.USE_USERTYPE_LOGIN",
            use_usertype_login,
        )
        constants_patch.start()

        # patch SqliteDict before import
        sqlite_dict_patch = patch("sqlitedict.SqliteDict", autospec=True)
        mocked_sqlite_dict_class = sqlite_dict_patch.start()
        if sqlite_side_effect:
            mocked_sqlite_dict_class.side_effect = sqlite_side_effect

        # import / reload target module
        import api.api as api_module  # type: ignore
        importlib.reload(api_module)

        return api_module, sqlite_dict_patch, getenv_patch, constants_patch

    return _loader


# Tests for find_user
def test_find_user_success_and_lowercase(load_fresh_api):
    (
        api_module,
        sqlite_dict_patch,
        getenv_patch,
        constants_patch,
    ) = load_fresh_api(sqlite_side_effect=sd_side_effect_with_test_user)

    found_user_record = api_module.find_user("TesTUsEr")

    sqlite_dict_patch.stop()
    getenv_patch.stop()
    constants_patch.stop()

    assert found_user_record == {
        "username": "testuser",
        "usertype": "full",
        "pwhash": "hashed_pw",
    }


def test_find_user_returns_none_for_unknown(load_fresh_api):
    (
        api_module,
        sqlite_dict_patch,
        getenv_patch,
        constants_patch,
    ) = load_fresh_api(sqlite_side_effect=sd_side_effect_empty_db)

    assert api_module.find_user("ghost") is None

    sqlite_dict_patch.stop()
    getenv_patch.stop()
    constants_patch.stop()


# Tests for USERTYPE login branch
def test_verify_password_success_sets_usertype(load_fresh_api):
    (
        api_module,
        sqlite_dict_patch,
        getenv_patch,
        constants_patch,
    ) = load_fresh_api(
        sqlite_side_effect=sd_side_effect_with_test_user,
        use_usertype_login=True,
    )

    with patch("api.api.check_password_hash", return_value=True):
        returned_username = api_module.verify_password("TestUser", "pwd")

    sqlite_dict_patch.stop()
    getenv_patch.stop()
    constants_patch.stop()

    assert returned_username == "testuser"
    assert api_module.auth.usertype == "full"


def test_verify_password_failure_wrong_password(load_fresh_api):
    (
        api_module,
        sqlite_dict_patch,
        getenv_patch,
        constants_patch,
    ) = load_fresh_api(
        sqlite_side_effect=sd_side_effect_with_test_user,
        use_usertype_login=True,
    )

    with patch("api.api.check_password_hash", return_value=False):
        result = api_module.verify_password("TestUser", "bad")

    sqlite_dict_patch.stop()
    getenv_patch.stop()
    constants_patch.stop()

    assert result is False


def test_verify_password_failure_user_missing(load_fresh_api):
    (
        api_module,
        sqlite_dict_patch,
        getenv_patch,
        constants_patch,
    ) = load_fresh_api(
        sqlite_side_effect=sd_side_effect_empty_db,
        use_usertype_login=True,
    )

    with patch("api.api.check_password_hash", return_value=True):
        result = api_module.verify_password("Missing", "pwd")

    sqlite_dict_patch.stop()
    getenv_patch.stop()
    constants_patch.stop()

    assert result is False


# Tests for COP_NO_LOGIN branch
def test_no_login_mode_success(load_fresh_api):
    (
        api_module,
        sqlite_dict_patch,
        getenv_patch,
        constants_patch,
    ) = load_fresh_api(env_vars={"COP_NO_LOGIN": "1"})

    returned_username = api_module.no_varification("MixedCaseUser", "ignored")

    sqlite_dict_patch.stop()
    getenv_patch.stop()
    constants_patch.stop()

    assert returned_username == "mixedcaseuser"
    assert api_module.auth.username == "mixedcaseuser"
    assert api_module.auth.usertype == "dev"


# Tests for COP_EXTERN_LOGIN branch
def test_external_login_success(load_fresh_api):
    (
        api_module,
        sqlite_dict_patch,
        getenv_patch,
        constants_patch,
    ) = load_fresh_api(env_vars={"COP_EXTERN_LOGIN": "1"})

    with patch("api.api.requests.post") as mocked_post:
        mocked_response = MagicMock(status_code=200)
        mocked_response.json.return_value = {
            "userType": "admin",
            "username": "externaluser",
        }
        mocked_post.return_value = mocked_response

        returned_username = api_module.extern_login("SomeUser", "pwd")

    sqlite_dict_patch.stop()
    getenv_patch.stop()
    constants_patch.stop()

    assert returned_username == "externaluser"
    assert api_module.auth.usertype == "admin"


def test_external_login_failure(load_fresh_api):
    (
        api_module,
        sqlite_dict_patch,
        getenv_patch,
        constants_patch,
    ) = load_fresh_api(env_vars={"COP_EXTERN_LOGIN": "1"})

    with patch("api.api.requests.post") as mocked_post:
        mocked_post.return_value = MagicMock(status_code=401)
        result = api_module.extern_login("SomeUser", "pwd")

    sqlite_dict_patch.stop()
    getenv_patch.stop()
    constants_patch.stop()

    assert result is False


# Tests for legacy login branch
def test_legacy_login_success(load_fresh_api):
    (
        api_module,
        sqlite_dict_patch,
        getenv_patch,
        constants_patch,
    ) = load_fresh_api(sqlite_side_effect=sd_side_effect_with_legacy_user)

    with patch("api.api.check_password_hash", return_value=True):
        username = api_module.verify_password("LegacyUser", "pwd")

    sqlite_dict_patch.stop()
    getenv_patch.stop()
    constants_patch.stop()

    assert username == "legacyuser"


def test_legacy_login_failure(load_fresh_api):
    (
        api_module,
        sqlite_dict_patch,
        getenv_patch,
        constants_patch,
    ) = load_fresh_api(sqlite_side_effect=sd_side_effect_empty_db)

    with patch("api.api.check_password_hash", return_value=False):
        result = api_module.verify_password("Ghost", "pwd")

    sqlite_dict_patch.stop()
    getenv_patch.stop()
    constants_patch.stop()

    assert result is False
