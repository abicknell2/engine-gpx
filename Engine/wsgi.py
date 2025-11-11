"""Run the Flask application."""
# isort: off
import gpx.custom_units  # noqa: F401
# isort: on
from api.api import app

if __name__ == "__main__":
    app.run()
