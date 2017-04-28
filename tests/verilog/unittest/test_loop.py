import tvm
from tvm.contrib import verilog

def test_loop():
    sess = verilog.session([
        verilog.find_file("test_loop.v")
    ])
    # Get the handles by their names
    rst = sess.main.rst
    iter0 = sess.main.iter0
    iter1 = sess.main.iter1
    ready = sess.main.ready

    rst.put_int(1)
    ready.put_int(1)
    # This will advance the cycle to next pos-edge of clk.
    sess.yield_until_next_cycle()
    rst.put_int(0)
    sess.yield_until_next_cycle()

    for k in range(0, 1):
        for i in range(0, 3):
            for j in range(0, 4):
                assert(iter1.get_int() == i)
                assert(iter0.get_int() == j)
                sess.yield_until_next_cycle()


if __name__ == "__main__":
    test_loop()
