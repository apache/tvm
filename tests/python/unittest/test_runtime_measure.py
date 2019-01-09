import time
import ctypes

import tvm
from tvm.contrib.util import tempdir


def test_min_repeat_ms():
    tmp = tempdir()
    filename = tmp.relpath("log")

    @tvm.register_func
    def my_debug(filename):
        """one call lasts for 100 ms and writes one character to a file"""
        time.sleep(0.1)
        filename = ctypes.c_char_p(filename.value).value
        with open(filename, "a") as fout:
            fout.write("c")

    X = tvm.compute((), lambda : tvm.call_packed("my_debug", filename))
    s = tvm.create_schedule(X.op)
    func = tvm.build(s, [X])

    x = tvm.nd.empty((), dtype="int32")
    ftimer = func.time_evaluator(func.entry_name, tvm.cpu(),
                                 number=1, repeat=1)
    ftimer(x)

    with open(filename, "r") as fin:
        ct = len(fin.readline())
    
    assert ct == 2


    ftimer = func.time_evaluator(func.entry_name, tvm.cpu(),
                                 number=1, repeat=1, min_repeat_ms=1000)
    ftimer(x)

    # make sure we get more than 10 calls
    with open(filename, "r") as fin:
        ct = len(fin.readline())

    assert ct > 10 + 2
        

if __name__ == "__main__":
    test_min_repeat_ms()

