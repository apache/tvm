import onnx.backend.test
from tvm.driver.onnx.backend import TVMBackend

import unittest
import warnings

pytest_plugins = 'onnx.backend.test.report',

backend_test = onnx.backend.test.runner.Runner(TVMBackend, __name__)

globals().update(backend_test.enable_report().test_cases)

if __name__ == '__main__':
    with warnings.catch_warnings():
        unittest.main()
