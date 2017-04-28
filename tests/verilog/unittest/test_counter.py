import tvm
from tvm.contrib import verilog

def test_counter():
    # Start a new session by run simulation on test_counter.v
    # Find file will search root/verilog and root/tests/verilog
    sess = verilog.session([
        verilog.find_file("test_counter.v"),
        verilog.find_file("example_counter.v")
    ])
    # Get the handles by their names
    rst = sess.main.rst
    counter = sess.main.counter
    cnt = sess.main["counter_unit1"]
    assert(counter.name == "main.counter")
    assert(counter.size == 4)
    rst.put_int(1)
    # This will advance the cycle to next pos-edge of clk.
    sess.yield_until_next_cycle()
    rst.put_int(0)
    sess.yield_until_next_cycle()

    for i in range(10):
        # get value of counter.
        assert(counter.get_int() == i)
        sess.yield_until_next_cycle()


def test_scratch():
    sess = verilog.session([
        verilog.find_file("test_counter.v"),
        verilog.find_file("example_counter.v")
    ])
    # Get the handles by their names
    rst = sess.main.rst
    counter = sess.main.counter
    rst.put_int(1)
    # This will advance the cycle to next pos-edge of clk.
    sess.yield_until_next_cycle()
    rst.put_int(0)
    temp = 0
    for i in range(10):
        if rst.get_int():
            rst.put_int(0)
            temp = counter.get_int()
        elif counter.get_int() == 3:
            rst.put_int(1)
        print("counter=%d, temp=%d" % (counter.get_int(), temp))
        sess.yield_until_next_cycle()

if __name__ == "__main__":
    test_scratch()
    test_counter()
