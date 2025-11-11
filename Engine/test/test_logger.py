import io
import logging
import unittest

import utils.logger as logger


class TestLoggerWrappers(unittest.TestCase):

    def setUp(self):
        self.log_stream = io.StringIO()
        self.handler = logging.StreamHandler(self.log_stream)
        formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
        self.handler.setFormatter(formatter)

        self.root_logger = logging.getLogger()
        self.root_logger.setLevel(logging.DEBUG)
        self.root_logger.addHandler(self.handler)

    def tearDown(self):
        self.root_logger.removeHandler(self.handler)
        self.handler.close()

    def _get_log_output(self):
        self.handler.flush()
        return self.log_stream.getvalue()

    def test_debug_logs_module_name(self):
        logger.debug("debug test")
        output = self._get_log_output()
        self.assertIn(f"DEBUG:{__name__}:debug test", output)

    def test_info_logs_module_name(self):
        logger.info("info test")
        output = self._get_log_output()
        self.assertIn(f"INFO:{__name__}:info test", output)

    def test_warn_logs_module_name(self):
        logger.warn("warn test")
        output = self._get_log_output()
        self.assertIn(f"WARNING:{__name__}:warn test", output)

    def test_error_logs_module_name(self):
        logger.error("error test")
        output = self._get_log_output()
        self.assertIn(f"ERROR:{__name__}:error test", output)

    def test_exception_logs_module_name_and_trace(self):
        try:
            raise ValueError("uh oh")
        except Exception as e:
            logger.exception(e, "Something bad")

        output = self._get_log_output()

        # Match expected multiline logging format
        self.assertIn(f"ERROR:{__name__}:\nSomething bad", output)
        self.assertIn("ValueError: uh oh", output)


if __name__ == "__main__":
    unittest.main()
