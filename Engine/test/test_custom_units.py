import math

# isort: off
import gpx.custom_units  # noqa: F401
# isort: on
from gpkit import ureg


def test_percent_unit() -> None:
    # Test percent (pct) unit
    value_pct = 5 * ureg("pct")
    value_pct_base = value_pct.to_base_units()
    assert value_pct_base.magnitude == 0.05, f"Expected 0.05, got {value_pct_base.magnitude}"
    assert str(value_pct.units) == "percent", f"Expected 'percent', got '{value_pct.units}'"


def test_percent_conversion() -> None:
    # Test conversion of percent to dimensionless
    value_pct = 5 * ureg("pct")
    value_pct_dimless = value_pct.to('dimensionless')
    assert value_pct_dimless.magnitude == 0.05, f"Expected 0.05, got {value_pct_dimless.magnitude}"


def test_percent_string() -> None:
    # Confirm percent unit name
    assert str(ureg("pct").units) == "percent"


def test_ppm_unit() -> None:
    value_ppm = 100 * ureg("ppm")
    value_ppm_base = value_ppm.to_base_units()
    # Use math.isclose for floating-point comparison
    assert math.isclose(
        value_ppm_base.magnitude,
        1e-4,
        rel_tol=1e-6,
        abs_tol=1e-12,
    ), (f"Expected 1e-4, got {value_ppm_base.magnitude}")
    assert str(value_ppm.units) == "parts_per_million"


def test_ppm_conversion() -> None:
    # Test conversion of ppm to dimensionless
    value_ppm = 1 * ureg("ppm")
    value_ppm_dimless = value_ppm.to('dimensionless')
    assert value_ppm_dimless.magnitude == 1e-6, f"Expected 1e-6, got {value_ppm_dimless.magnitude}"


def test_ppm_string() -> None:
    # Confirm ppm unit name
    assert str(ureg("ppm").units) == "parts_per_million"
