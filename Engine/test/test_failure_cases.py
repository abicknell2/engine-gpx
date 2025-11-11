# type: ignore
import json
import logging
import os
from test.shared_functions import (discover_test_cases, ensure_data_directory_exists)
from typing import Any, Generator
from unittest.mock import MagicMock, patch

import flask
from flask import Response
import flask.testing
import pytest

from api.api import app, auth  # Import the Flask app and authentication
from utils import logger
from utils.types.shared import AcceptedTypes

# isort: off
import gpx.custom_units  # noqa: F401
# isort: on

logging.basicConfig(level=logging.DEBUG)

# Get the absolute path of the current script's directory
current_dir: str = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT: str = os.path.dirname(current_dir)
DATA_DIR: str = os.path.join(REPO_ROOT, "test", "data", "failure_cases")


@pytest.fixture(scope="function", autouse=True)
def setup_data_directory() -> Generator[None, None, None]:
    """Fixture to ensure DATA_DIR exists before tests."""
    ensure_data_directory_exists(DATA_DIR)
    yield  # Run the test


@pytest.fixture(scope="module")
def test_client() -> Generator[flask.testing.FlaskClient, None, None] | None:
    """Create a test client for the Flask application with fully bypassed authentication."""
    mock_auth: MagicMock = MagicMock()
    mock_auth.username = "test_user"
    mock_auth.password = "test_password"

    def mock_verify_password(username: str, password: str) -> bool:
        return True

    def mock_authenticate(auth: Any, password: str) -> str:
        return "test_user"

    def mock_authorize(role: str, user: str, auth: Any) -> bool:
        return True

    with patch.dict(os.environ, {"COP_NO_LOGIN": "1"}):  # Mock login to bypass authentication
        with patch.object(auth, "verify_password", mock_verify_password):  # Always authenticate
            with patch.object(auth, "authenticate", mock_authenticate):  # Return mock user
                with patch.object(auth, "authorize", mock_authorize):  # Always authorize
                    with patch.object(auth, "get_auth", return_value=mock_auth):  # Disable auth enforcement
                        with patch.object(auth, "current_user", return_value="test_user"):  # Mock user context
                            with patch(
                                    "flask_httpauth.HTTPBasicAuth.login_required",
                                    lambda f: f,
                            ):  # Fully bypass login
                                with app.test_client() as client:
                                    client.environ_base["HTTP_AUTHORIZATION"
                                                        ] = "Basic dGVzdF91c2VyOnBhc3N3b3Jk"  # Inject mock auth header
                                    yield client
    return None


@pytest.mark.parametrize("input_json,expected_json", discover_test_cases(DATA_DIR))
def test_failure_cases_api(
    test_client: flask.testing.FlaskClient,
    input_json: str,
    expected_json: str,
) -> None:
    """Validate failure cases by making API requests to the Flask application."""
    with open(input_json, "r") as f:
        input_model: dict[str, AcceptedTypes] = json.load(f)

    with open(expected_json, "r") as f:
        expected_output: dict[str, AcceptedTypes] = json.load(f)
        print('expected_output: ', expected_output)

    response: Response = test_client.post("/solve", json=input_model)
    assert response.status_code == 200, f"Unexpected status code: {response.status_code}"

    # Fix: Extract JSON from streamed response
    response_text: str = response.data.decode("utf-8").strip()
    json_start: int = response_text.find("{")  # Find where JSON starts
    assert json_start != -1, "No JSON found in response"
    output: dict[str, AcceptedTypes] = json.loads(response_text[json_start:])  # Extract JSON part
    print('output: ', output)

    assert "errors" in output, "'errors' key missing in output"
    assert isinstance(output["errors"], list), "'errors' key must be a list"
    assert "errors" in expected_output, "'errors' key missing in expected output"
    assert isinstance(expected_output["errors"], list), "'errors' key must be a list in expected output"

    assert len(output["errors"]) == len(expected_output["errors"]), "Mismatch in number of errors"

    for i, (out_error, exp_error) in enumerate(zip(output["errors"], expected_output["errors"])):
        assert exp_error in out_error, f"Mismatch in error message at index {i}: Expected '{exp_error}', Got '{out_error}'"

    logger.info(f"✅ Test passed for failure case: {input_json}")


def test_missing_data_dir() -> None:
    """Test if FileNotFoundError is raised when DATA_DIR does not exist."""
    temp_path: str | None = None

    if os.path.exists(DATA_DIR):
        temp_path = DATA_DIR + "_backup"
        os.rename(DATA_DIR, temp_path)

    try:
        with pytest.raises(FileNotFoundError, match="Could not find the data directory"):
            ensure_data_directory_exists(DATA_DIR)
    finally:
        if temp_path:
            os.rename(temp_path, DATA_DIR)
