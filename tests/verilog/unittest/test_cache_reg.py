import tvm
from tvm.contrib import verilog
from testing_util import FIFODelayedWriter, FIFODelayedReader

def run_with_lag(n, read_lag, write_lag):
    data = list(range(n))
    # head ptr of a
    sess = verilog.session([
        verilog.find_file("test_cache_reg.v")
    ])
    rst = sess.main.rst
    in_data = sess.main.in_data
    in_valid = sess.main.in_valid
    in_ready = sess.main.in_ready

    out_data = sess.main.out_data
    out_valid = sess.main.out_valid
    out_ready = sess.main.out_ready
    # hook up reader
    reader = FIFODelayedReader(out_data, out_valid, out_ready, read_lag)
    writer = FIFODelayedWriter(in_data, in_valid, in_ready, data, write_lag)
    rst.put_int(1)
    sess.yield_until_next_cycle()
    rst.put_int(0)
    sess.yield_until_next_cycle()
    sess.yield_callbacks.append(reader)
    sess.yield_callbacks.append(writer)
    timeout = sum(read_lag) + sum(write_lag) + n + 10
    for t in range(timeout):
        sess.yield_until_next_cycle()
        if len(reader.data) == n:
            break
    assert tuple(reader.data) == tuple(range(n))
    assert len(writer.data) == 0
    sess.shutdown()

def test_fifo():
    n = 20
    # slow reader
    run_with_lag(n, read_lag=[3,4,8], write_lag=[])
    # slow writer
    run_with_lag(n, read_lag=[0], write_lag=[0, 2, 10])
    # mix
    run_with_lag(n, read_lag=[3, 4, 8], write_lag=[0, 2, 10])


if __name__ == "__main__":
    test_fifo()
