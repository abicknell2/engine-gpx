import json

import pytest

from utils.types.data import Parameter


def test_parameter_repr_roundtrip():
    """`__repr__` should serialise to JSON without raising and
    that JSON should parse back to the same dict keys."""
    p = Parameter(
        name="Example parameter",
        key="ex_param",
        value=42,
        unit="USD",
        descr="just a test",
    )

    # Calling __repr__ must NOT raise
    repr_str = p.__repr__()

    # Ensure it is valid JSON
    loaded = json.loads(repr_str)
    print("\nREPR OUTPUT:\n", repr_str)
    #  REPR OUTPUT (if test fails):
    #  {
    #     "name": "Example parameter",
    #     "source": "",
    #     "unit": "USD",
    #     "value": 42,
    #     "min": null,
    #     "max": null,
    #     "key": "ex_param",
    #     "descr": "just a test",
    #     "tags": [],
    #     "category": "",
    #     "type": "",
    #     "property": "",
    #     "gpx_translator": "<function param_to_var at 0x7f09d2c0a0c0>",
    #     "infoLabels": {},
    #     "variables": {},
    #     "substitutions": {}
    # }

    # Basic round-trip checks
    assert loaded["name"] == "Example parameter"
    assert loaded["key"] == "ex_param"
    assert loaded["value"] == 42
    assert loaded["unit"] == "USD"
    assert loaded["descr"] == "just a test"
