"helpers with classifying units"

from gpkit import units
from pint.errors import DimensionalityError


def is_continuous_physical_quantity(input_units) -> bool:
    "determines if input is a physical (continuous) quantity."
    if is_mass(input_units):
        return True
    if is_length_dimension(input_units):
        return True


def is_mass(input_units) -> bool:
    "determines if input units define a mass"
    try:
        input_units.to('kilogram')  # convert only if dimensionally compatible
        return True
    except DimensionalityError:
        return False


def is_length_dimension(input_units) -> bool:
    "determines if length, area, or volume"
    for tgt in ('meter', 'meter**2', 'meter**3'):
        try:
            input_units.to(tgt)  # succeeds only for compatible dimensions
            return True
        except DimensionalityError:
            pass
    return False
