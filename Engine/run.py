"""Run the Flask application."""

import os

# isort: off
import gpx.custom_units  # noqa: F401
# isort: on
from api.api import app

if __name__ == "__main__":
    host = os.getenv("APP_HOST", "127.0.0.1")
    port = int(os.getenv("APP_PORT", "5000"))
    debug = os.getenv("APP_DEBUG", "False").lower() == "true"

    app.run(host=host, port=port, debug=debug)
