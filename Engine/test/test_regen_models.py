# type: ignore
import json
import logging
import os
from test.shared_functions import ensure_data_directory_exists
from typing import Any, Generator
from unittest.mock import MagicMock, patch

import flask
from flask import Response
import flask.testing
import pytest

from api.api import app, auth  # Import the Flask app and authentication
from utils.types.shared import AcceptedTypes

logging.basicConfig(level=logging.DEBUG)

# Get the absolute path of the current script's directory
current_dir: str = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT: str = os.path.dirname(current_dir)

MODELS_DIR: str = os.path.join(REPO_ROOT, "test", "data", "_models_to_regen")
RATE_RAMPS_DIR: str = os.path.join(REPO_ROOT, "test", "data", "rate_ramps")

DIR_TO_UPDATE = MODELS_DIR  # TODO: Change this to RATE_RAMPS_DIR etc. if needed


@pytest.fixture(scope="function", autouse=True)
def setup_data_directory() -> Generator[None, None, None]:
    """Fixture to ensure DATA_DIR exists before tests."""
    ensure_data_directory_exists(DIR_TO_UPDATE)
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


def discover_test_cases(data_dir: str) -> list[tuple[str, str]]:
    """Find valid input JSON and expected _results.json files across subfolders."""
    test_cases = []
    for root, _, files in os.walk(data_dir):
        json_files = set(f for f in files if f.endswith(".json"))

        for file in json_files:
            if file.endswith("_results.json"):
                input_file = file.replace("_results.json", ".json")
                if input_file in json_files:
                    test_cases.append(os.path.join(root, input_file))
    return test_cases


# @pytest.mark.skip("Skipping by default to avoid regenerating models when not necessary")
@pytest.mark.parametrize("input_json", discover_test_cases(DIR_TO_UPDATE))
def test_regen_models(
    test_client: flask.testing.FlaskClient,
    input_json: str,
) -> None:
    """Validate failure cases by making API requests to the Flask application."""
    # set the save model env var
    os.environ["SAVE_MODEL_DATA_FOR_TESTS"] = "1"
    with open(input_json, "r") as f:
        input_model: dict[str, AcceptedTypes] = json.load(f)

    response: Response = test_client.post("/solve?save_model_data_for_tests=1", json=input_model, buffered=True)
    assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}"
