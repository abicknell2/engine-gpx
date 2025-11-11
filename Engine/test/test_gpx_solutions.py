# import json
# import logging
# import math
# import os
# from api import uncertain_context
# from api.multiproduct import create_from_dict
# from test.shared_functions import (discover_test_cases,
#                                    ensure_data_directory_exists,
#                                    run_model_analysis)
# from typing import Any, Generator
# from unittest.mock import patch

# import api.context as context
# import pytest

# from api import logger
# import pickle
# # Get the absolute path of the current script's directory
# current_dir = os.path.abspath(os.path.dirname(__file__))

# # Move up one level to set the root folder
# REPO_ROOT = os.path.dirname(current_dir)

# DATA_DIR = os.path.join(REPO_ROOT, "test", "data", "gpx_solutions")

# # Define the data directory containing JSON files
# INPUT_JSON_PATH = os.path.join(DATA_DIR, "mixed_semiconductor", "mixed_semiconductor.json")
# SOLUTION_JSON_PATH = os.path.join(DATA_DIR, "mixed_semiconductor", "mixed_semiconductor_solution.pkl")

# def test_multiproduct_gpx_solutions() -> None:
#     """Validate rate ramp results against expected outputs."""
#     with open(INPUT_JSON_PATH, "r") as f:
#         input_model = json.load(f)

#     interaction = create_from_dict(inputdict=input_model["model"])

#     if interaction.has_uncertainty():
#         interaction.context = uncertain_context.Uncertain(interaction)

#     interaction.trade_study = "trade" in input_model
#     if interaction.trade_study:
#         interaction.trade_params = input_model["trade"]

#     if not interaction.has_uncertainty():
#         interaction.generate_gpx()

#     interaction.solve()

#     result_gens = []

#     with open(SOLUTION_JSON_PATH, "rb") as f:
#         sol = pickle.load(f)

#     # first collect all the results from the modules
#     for m in interaction.active_modules.values():
#         if hasattr(m, "get_results"):
#             result_gens.extend(m.get_results(sol))

#     print('result_gens: ', result_gens)
