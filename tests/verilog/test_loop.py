import tvm
import os
from tvm.addon import verilog

def test_loop():
    sess = verilog.session([
        verilog.find_file("test_loop.v")
    ])
    # Get the handles by their names
    rst = sess.main.rst
    init = sess.main.init
    iter0 = sess.main.iter0
    iter1 = sess.main.iter1
    enable = sess.main.enable
    invalid = sess.main.done

    rst.put_int(1)
    # This will advance the cycle to next pos-edge of clk.
    sess.yield_until_posedge()
    rst.put_int(0)
    init.put_int(1)
    sess.yield_until_posedge()
    enable.put_int(1)
    init.put_int(0)

    for i in range(0, 3):
        for j in range(0, 4):
            while invalid.get_int():
                sess.yield_until_posedge()
            assert(iter1.get_int() == i)
            assert(iter0.get_int() == j)
            sess.yield_until_posedge()


if __name__ == "__main__":
    test_loop()
