import datetime
import logging
import os
from pathlib import Path
import re
import traceback
import uuid

from gpkit.exceptions import (
    DualInfeasible, Infeasible, InvalidGPConstraint, InvalidLicense, InvalidPosynomial, InvalidSGPConstraint,
    MathematicallyInvalid, PrimalInfeasible, UnboundedGP, UnknownInfeasible, UnnecessarySGP
)
from pint.errors import (
    DefinitionError, DefinitionSyntaxError, DimensionalityError, LogarithmicUnitCalculusError, OffsetUnitCalculusError,
    PintTypeError, RedefinitionError, UndefinedUnitError
)

from api.errors import ApiModelError
from utils.constants.error_messages import KNOWN_ERROR_MESSAGES

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_ROOT = Path(PROJECT_ROOT / "logs")

# ALL end-user-safe messages as regexes
KNOWN_ERROR_REGEXES: list[str] = [
    r"need to specify a production rate",
    r"the production rate could not be found in the selected manufacturing module",
    r"could not locate a finance module", r"Model is infeasible. costs trending to 0.",
    r"Model is infeasible. Check rate requirement and constraints.",
    r"could not solve. one or more unbounded variables.", r"optimal solution not found.",
    r"solution found. could not generate results", r"inputdict must be a dictionary containing the key 'manufacturing'",
    r"designvars must be a dict",
    r"manufacturing module must have been gpx translated to get production resource variables",
    r"no resources to ramp were found", r"exceeded 24 hrs per day of production time", r"exceeded 7 days per week",
    r"exceeded days per year", r"inputdict must be a dictionary", r"finance must be a dictionary",
    r'input for "cost of capital" is required', r"duration units not allowed for",
    r"duration not entered in the correct units", r"hourly rate cannot be none", r"cannot repeat component name",
    r"too many rate rate_ramp to process", r"manufacturing data must be a dict",
    r"rates must be a list of dictionaries", r"each rate must be a dictionary", r"amortization must be a string",
    r"add an input for total manufacturing quantity", r"no input for duration or quantity",
    r"rate ramp requires a finance module", r"a finance module must be specified for off-shift production",
    r"manufacturing not defined", r"rate not defined", r"empty system. add at least one product to the system.",
    r"product names must be unique", r"multi product not available for", r"invalid type for rate:",
    r"invalid type for duration:", r"invalid type for quantity:", r"rates must be a list or tuple",
    r"rate ramp can only be used with system specified by \"split\"",
    r"a footprint is defined. specify at least one floor space cost.",
    r"headcounts found. define \"labor variable cost\"", r"missing cell", r"first pass yield greater than 1",
    r"off-shift cell and exit cell must allow queueing to operate off-shift",
    r"processlaborcost must be calculated as a variable cost", r"cannot estimate cost distribution",
    r"could not estimate costs from uncertainties", r"smooth only accepts 1 dimension arrays.",
    r"window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'",
    r"\"%\" cannot be used with a dynamic constraint. use \"unitless\" instead\.", r"error creating constraint",
    r"bad constraint found in building constraint set", r"Expected key to be str, but got",
    r"Selected Scenario .* is not valid", r"Key Error on variable: .* in module: .*", r".* must have an input value",
    r"Unit error on variable .* is not a unit", r"key not found:\s*.*\s*\|\s*varkeys:\s*.*",
    r"Could not convert finance module rate to", r"Duration units not allowed for", r"Cannot repeat component name",
    r"Unit conflict at constraint .*", r"Error creating constraint .*",
    r"log gradient calulation requires a solution input to return an object", r"class split must be less than 1",
    r"must specify a split for the class in the split context",
    r"must specify a class rate outside of the split context", r"parameters are not strictly increasing",
    r"distribution not found.*", r"solution case not found.*", r"currency not supported by this registry",
    r"not a valid input for a variable:.*", r"the model needs to have been solved as continuous",
    r"off-shift cell and exit cell must allow queueing to operate off-shift", r"first pass yield greater than 1",
    r"processlaborcost must be calculated as a variable cost", r"no matching keys found for filters category: .*",
    r"could not create constraint #.* for .*", r"cannot have an uncertain inequality for .*",
    r"issue creating .* as .* in constraint #.*", r".* must have an input value",
    r"could not load the selecrted manufacturing or design module",
    r"total program duration must be greater than the total rate ramp duration by at least one month",
    r'".*" must have strictly increasing uncertain input parameters', r".* has no upper bound",
    r".* has no lower bound", r".*cannot be shared across products"
]

KNOWN_ERROR_CLASSES = (
    # Pint errors
    DimensionalityError,
    DefinitionError,
    DefinitionSyntaxError,
    RedefinitionError,
    UndefinedUnitError,
    PintTypeError,
    OffsetUnitCalculusError,
    LogarithmicUnitCalculusError,
    # GPkit errors
    MathematicallyInvalid,
    InvalidPosynomial,
    InvalidGPConstraint,
    InvalidSGPConstraint,
    UnnecessarySGP,
    UnboundedGP,
    InvalidLicense,
    Infeasible,
    UnknownInfeasible,
    PrimalInfeasible,
    DualInfeasible,
)


def is_known_user_error(exc: Exception | str) -> bool:
    """
    Return True if the error message or its root cause matches any of the patterns
    that are safe to show directly to the end-user.
    """
    if isinstance(exc, Exception):
        known_exc = find_first_known_error(exc)
        if known_exc:
            if isinstance(known_exc, KNOWN_ERROR_CLASSES):
                return True
            if isinstance(known_exc, ApiModelError):
                msg = known_exc.message.lower()
            else:
                msg = str(known_exc).lower()
        else:
            msg = str(exc).lower()
    else:
        msg = str(exc).lower()

    for lit in KNOWN_ERROR_MESSAGES:
        if lit.lower() in msg:
            return True

    for pat in KNOWN_ERROR_REGEXES:
        if re.search(pat.lower(), msg.lower()):
            return True

    return False


def find_first_known_error(exc: Exception) -> Exception | None:
    """
    Walks the entire exception chain (starting with exc) and returns
    the first known (GPkit, Pint, or custom) exception.
    """
    seen: set[Exception] = set()
    current = exc

    while current and current not in seen:
        if isinstance(current, (ApiModelError, *KNOWN_ERROR_CLASSES)):
            return current
        seen.add(current)
        current = current.__cause__ if hasattr(current, "__cause__") else None

    return None


def should_mask_error(exc: Exception) -> bool:
    """
    Mask the error if it's neither a known user error by regex/literal,
    nor a domain-specific known error class.
    """
    return not is_known_user_error(exc)


def log_unexpected_error(exc: Exception, username: str | None = None) -> str:
    """
    Write full stack trace to a per-user daily log file.
    Logs all unexpected or domain-level exceptions unless they are totally known user errors.
    """
    root = get_root_exception(exc)

    print("EXC TYPE:", type(exc))
    print("EXC MESSAGE:", getattr(exc, "message", str(exc)))
    print("ROOT EXC:", getattr(exc, "message", str(root)))

    # Only skip logging for trivial user-facing literals/regexes
    if is_known_user_error(root) and not isinstance(root, KNOWN_ERROR_CLASSES):
        return ""

    error_uuid = str(uuid.uuid4())
    date_str = datetime.date.today().isoformat()

    user_folder = username or "anonymous"
    log_path = LOG_ROOT / user_folder
    log_path.mkdir(parents=True, exist_ok=True)

    full_log_file = log_path / f"errors_{date_str}.log"

    with open(full_log_file, "a") as f:
        timestamp = datetime.datetime.now().isoformat(sep=" ", timespec="seconds")
        f.write(f"\n[{timestamp}] ERROR ID: {error_uuid}\n")
        f.write(f"{type(exc).__name__}: {exc}\n")
        traceback.print_exception(exc, file=f)

    logging.getLogger(__name__).error("UNEXPECTED ERROR", exc_info=exc)
    return error_uuid


def get_root_exception(exc: Exception) -> Exception:
    while hasattr(exc, "__cause__") and exc.__cause__:
        exc = exc.__cause__
    return exc
