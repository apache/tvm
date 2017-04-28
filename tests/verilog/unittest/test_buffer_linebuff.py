import tvm
import numpy as np
from tvm.contrib import verilog

def test_buffer_linebuff():
    # Test the tvm_buffer.v module as a line buffer
    # Window is 8x8, kernel is 3x3
    window_width = 8
    kernel_width = 3

    # Find file will search root/verilog and root/tests/verilog
    sess = verilog.session([
        verilog.find_file("test_buffer_linebuff.v"),
        verilog.find_file("tvm_buffer.v")
    ])

    # Get the handles by their names
    rst = sess.main.rst
    write_advance = sess.main.write_advance
    write_valid = sess.main.write_valid
    write_ready = sess.main.write_ready
    write_data = sess.main.write_data
    read_data = sess.main.read_data
    read_data_valid = sess.main.read_data_valid

    # Simulation input data
    test_data = np.arange(window_width*window_width).astype('int8')

    # Initial state
    rst.put_int(1)
    write_advance.put_int(0)
    write_valid.put_int(0)
    write_data.put_int(0)

    # De-assert reset
    sess.yield_until_next_cycle()
    rst.put_int(0)

    # Leave the following signals set to true
    sess.yield_until_next_cycle()
    write_advance.put_int(1)
    write_valid.put_int(1)

    # Main simulation loop
    write_idx = 0
    read_idx = 0
    while read_idx < (window_width-kernel_width+1)*(window_width-kernel_width+1)*kernel_width*kernel_width:
        # write logic
        if (write_idx < len(test_data)):
            if (write_ready.get_int()):
                write_data.put_int(int(test_data[write_idx]))
                write_idx += 1
        else:
            write_advance.put_int(0)
            write_valid.put_int(0)

        # correctness checks
        if (read_data_valid.get_int()):
            # Derive convolution window indices
            baseIdx = read_idx // (kernel_width*kernel_width)
            offsetIdx = read_idx % (kernel_width*kernel_width)
            yOffset = offsetIdx // kernel_width
            xOffset = offsetIdx%kernel_width
            pixIndex = baseIdx + yOffset * window_width + xOffset
            assert(read_data.get_int()==test_data[pixIndex])
            # print "{} {}".format(read_data.get_int(), test_data[pixIndex])
            read_idx += 1

        # step
        sess.yield_until_next_cycle()


if __name__ == "__main__":
    test_buffer_linebuff()
