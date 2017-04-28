import tvm
import numpy as np
from tvm.contrib import verilog

def test_buffer_doublebuff():
    # Test the tvm_buffer.v module as a double buffer
    # Window size is 16, buffer size is 32
    window_width = 16
    set_size = 8

    # Find file will search root/verilog and root/tests/verilog
    sess = verilog.session([
        verilog.find_file("test_buffer_doublebuff.v"),
        verilog.find_file("tvm_buffer.v")
    ])

    # Get the handles by their names
    rst = sess.main.rst
    write_advance = sess.main.write_advance
    write_addr = sess.main.write_addr
    write_valid = sess.main.write_valid
    write_ready = sess.main.write_ready
    write_data = sess.main.write_data
    read_data = sess.main.read_data
    read_data_valid = sess.main.read_data_valid

    # Simulation input data
    test_data = np.arange(window_width*set_size).astype('int8')

    # Initial state
    rst.put_int(1)
    write_advance.put_int(0)
    write_addr.put_int(0)
    write_valid.put_int(0)
    write_data.put_int(0)

    # De-assert reset
    sess.yield_until_next_cycle()
    rst.put_int(0)

    # Leave the following signals set to true
    sess.yield_until_next_cycle()
    write_valid.put_int(1)

    # Main simulation loop
    write_idx = 0
    read_idx = 0
    while read_idx < len(test_data):
        # write logic
        if (write_idx < len(test_data)):
            write_advance.put_int(0)
            if (write_ready.get_int()):
                write_data.put_int(int(test_data[write_idx]))
                write_addr.put_int(write_idx % window_width)
                if (write_idx%window_width==window_width-1):
                    write_advance.put_int(1)
                write_idx += 1
        else:
            write_advance.put_int(0)
            write_valid.put_int(0)

        # correctness checks
        if (read_data_valid.get_int()):
            assert(read_data.get_int()==test_data[read_idx])
            # print "{} {}".format(read_data.get_int(), test_data[read_idx])
            read_idx += 1

        # step
        sess.yield_until_next_cycle()


if __name__ == "__main__":
    test_buffer_doublebuff()
