import tvm
import numpy as np
from tvm.contrib import verilog

def test_buffer_fifo():
    # Test the tvm_buffer.v module as a fifo

    # Find file will search root/verilog and root/tests/verilog
    sess = verilog.session([
        verilog.find_file("test_buffer_fifo.v"),
        verilog.find_file("tvm_buffer.v")
    ])

    # Get the handles by their names
    rst = sess.main.rst
    enq = sess.main.enq
    write_data = sess.main.write_data
    read_data = sess.main.read_data
    read_data_valid = sess.main.read_data_valid

    # Simulation input data
    test_data = np.arange(16).astype('int8')

    # Initial state
    rst.put_int(1)
    enq.put_int(0)
    write_data.put_int(0)

    # De-assert reset
    sess.yield_until_next_cycle()
    rst.put_int(0)

    # Main simulation loop
    read_idx = 0
    write_idx = 0
    while read_idx < len(test_data):
        # write logic
        if (write_idx < len(test_data)):
            enq.put_int(1)
            write_data.put_int(write_idx)
            write_idx += 1
        else:
            enq.put_int(0)
        # read logic
        if (read_data_valid.get_int()):
            assert(read_data.get_int()==test_data[read_idx])
            read_idx += 1
        # step
        sess.yield_until_next_cycle()


if __name__ == "__main__":
    test_buffer_fifo()
