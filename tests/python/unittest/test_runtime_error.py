"""Test runtime error handling"""
import tvm

def test_op_translation():
    ferror = tvm._api_internal._test_raise_error_callback(
        "OpNotImplemented: myop")
    try:
        ferror()
        assert False
    except tvm.error.OpNotImplemented as e:
        msg = str(e)
        assert isinstance(e, NotImplementedError)
        assert msg.find("api_test.cc") != -1

    fchk_eq = tvm._api_internal._test_check_eq_callback(
        "InternalError: myop")
    try:
        fchk_eq(0, 1)
        assert False
    except tvm.error.InternalError as e:
        msg = str(e)
        assert msg.find("api_test.cc") != -1

    try:
        tvm._api_internal._ErrorTest(0, 1)
        assert False
    except ValueError as e:
        msg = str(e)
        assert msg.find("api_test.cc") != -1


def test_deep_callback():
    def error_callback():
        raise ValueError("callback error")
    wrap1 = tvm._api_internal._test_wrap_callback(error_callback)
    def flevel2():
        wrap1()
    wrap2 = tvm._api_internal._test_wrap_callback(flevel2)
    def flevel3():
        wrap2()
    wrap3 = tvm._api_internal._test_wrap_callback(flevel3)

    try:
        wrap3()
        assert False
    except ValueError as e:
        msg = str(e)
        idx2 = msg.find("in flevel2")
        idx3 = msg.find("in flevel3")
        assert idx2 != -1 and idx3 != -1
        assert idx2 > idx3


if __name__ == "__main__":
    test_op_translation()
    test_deep_callback()
