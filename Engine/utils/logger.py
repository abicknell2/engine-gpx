import inspect
import logging
import traceback

# Configure global logging
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s:%(name)s:%(message)s")


def _get_logger() -> logging.Logger:
    # Walk the stack to find the calling module
    frame = inspect.currentframe()
    caller_frame = frame.f_back.f_back
    module = inspect.getmodule(caller_frame)
    module_name = module.__name__ if module else "__main__"
    logger = logging.getLogger(module_name)

    # Ensure logger has at least one handler to prevent "No handlers" warning
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

    return logger


def info(msg: str) -> None:
    _get_logger().info(msg)


def debug(msg: str) -> None:
    _get_logger().debug(msg)


def warn(msg: str) -> None:
    _get_logger().warning(msg)


def error(msg: str) -> None:
    _get_logger().error(msg)


def exception(e: Exception, message: str = "An exception occurred:") -> None:
    if isinstance(e, KeyError):
        err = f"Missing required key: {e.args[0]}"
    else:
        err = repr(e)

    _get_logger().error(f"\n{message} {err}.\n\n{traceback.format_exc()}")
