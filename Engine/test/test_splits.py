"""
updated: 2025-07-29
validates that every shared MCell returns the expected x-split after the
new XFromSplits rewrite.

The fixture loader and expected-value logic are unchanged; only the way we
pull x-values from the engine output has been adapted to the new
`mCellResults` schema (which now carries an `x` list of
{product, value} tuples instead of the previous `xSplits` dict).
"""
import json
import math
import pathlib
from test.shared_functions import run_model_analysis

import pytest

from utils.types.shared import AcceptedTypes

# ------------------------------------------------------------------------
# paths / helpers
# ------------------------------------------------------------------------

DATA_DIR = (pathlib.Path(__file__).resolve().parents[1] / "test" / "data" / "splits_cases")

MODEL_PATHS = sorted(DATA_DIR.glob("*.json"))
TEST_IDS = [p.stem for p in MODEL_PATHS]


def _variants_from_products(model: dict) -> list[dict]:
    """create a minimal {name, split | partsPer | rate} list from a products block"""
    out = []
    for prod in model.get("products", []):
        src = prod.get("source", {})
        name = src.get("modelName") or prod.get("name") or "<unnamed>"
        entry: dict[str, float | str] = {"name": name}

        if "split" in prod:
            entry["split"] = prod["split"]

        mfg = src.get("manufacturing", {})
        if "partsPer" in mfg:
            entry["partsPer"] = mfg["partsPer"]
        if "rate" in mfg:
            entry["rate"] = mfg["rate"]

        out.append(entry)
    return out


def expected_x_splits(model_dict: dict) -> dict[str, float]:
    """design-time split for each product (unit-less)"""
    model = model_dict["model"]
    if model.get("variants"):
        variants = model["variants"]
    elif model.get("products"):
        variants = _variants_from_products(model)
    else:
        raise ValueError("No 'variants' or 'products' block found in model")

    # 1️⃣ explicit %
    if any("split" in v for v in variants):
        return {v["name"]: v["split"] / 100.0 for v in variants}

    # 2️⃣ parts-per kit
    if any("partsPer" in v for v in variants):
        total = sum(v["partsPer"] for v in variants)
        return {v["name"]: v["partsPer"] / total for v in variants}

    # 3️⃣ absolute rate
    total = sum(v["rate"] for v in variants)
    return {v["name"]: v["rate"] / total for v in variants}


def _extract_xdict(cell: dict) -> dict | None:
    """
    normalises the various ‘x’ shapes we get back:

    • 2024-era → {"xSplits": {"ProdA":0.5, …}}
    • 2025 XFromSplits → {"x": [{"product":"ProdA","value":0.5}, …]}
    """
    if "xSplits" in cell and isinstance(cell["xSplits"], dict):
        return cell["xSplits"]

    if "x" not in cell:
        return None

    xs = cell["x"]
    if isinstance(xs, dict):  # future proof
        return xs
    if isinstance(xs, list):  # newest schema
        return {d["product"]: d["value"] for d in xs}
    return None


# ------------------------------------------------------------------------
# main parameterised test
# ------------------------------------------------------------------------


@pytest.mark.parametrize("model_file", MODEL_PATHS, ids=TEST_IDS)
def test_x_splits(model_file: pathlib.Path) -> None:
    if "rate_split" in model_file.name:  # not yet supported
        pytest.skip("Rate-split support coming soon")

    with model_file.open() as fp:
        input_json: dict[str, AcceptedTypes] = json.load(fp)

    output = run_model_analysis(input_json)
    assert output, "run_model_analysis returned empty output"

    exp_split = expected_x_splits(input_json)

    # mCellResults is where shared-cell data now live
    for cell in output.get("mCellResults", []):
        xdict = _extract_xdict(cell)
        if not xdict:  # dedicated or non-shared cell
            continue

        cell_name = cell.get("name", cell.get("cellName", "<unknown>"))

        # must normalise to exactly 1.0
        assert math.isclose(sum(xdict.values()), 1.0, abs_tol=0.0), (f"{cell_name} x-sum != 1.0")

        # dedicated cell (len == 1) still needs x == 1.0
        if len(xdict) == 1:
            only_val = next(iter(xdict.values()))
            assert math.isclose(only_val, 1.0, abs_tol=0.0), (f"{cell_name} dedicated cell x != 1.0")
            continue

        # shared cell: each product’s split must equal the design split
        for prod, x_val in xdict.items():
            assert prod in exp_split, f"{cell_name} | '{prod}' not in design split"
            assert math.isclose(
                x_val, exp_split[prod], abs_tol=0.0
            ), f"{cell_name} | {prod} split {x_val:.6f} ≠ {exp_split[prod]:.6f}"
