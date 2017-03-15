import tvm
from tvm.addon import verilog

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
    sess.yield_until_posedge()
    rst.put_int(0)

    for i in range(10):
        # get value of counter.
        assert(counter.get_int() == i)
        sess.yield_until_posedge()


if __name__ == "__main__":
    test_counter()
