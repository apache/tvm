import tvm
import numpy as np
from tvm.addon import verilog

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
    wrAdvance = sess.main.write_advance
    wrValid = sess.main.write_valid
    wrReady = sess.main.write_ready
    wrData = sess.main.write_data
    rdData = sess.main.read_data
    rdDataValid = sess.main.read_data_valid

    # Simulation input data
    test_data = np.arange(window_width*window_width).astype('int8')

    # Initial state
    rst.put_int(1)
    wrAdvance.put_int(0)
    wrValid.put_int(0)
    wrData.put_int(0)

    # De-assert reset
    sess.yield_until_posedge()
    rst.put_int(0)

    # Leave the following signals set to true
    sess.yield_until_posedge()
    wrValid.put_int(1)
    wrAdvance.put_int(1)

    # Main simulation loop
    writeIdx = 0
    readIdx = 0
    while readIdx < (window_width-kernel_width+1)*(window_width-kernel_width+1)*kernel_width*kernel_width:
        # write logic
        if (writeIdx < window_width*window_width):
            if (wrReady.get_int()):
                wrData.put_int(test_data[writeIdx])
                writeIdx += 1
        else:
            wrAdvance.put_int(0)
            wrValid.put_int(0)
            
        # correctness checks
        if (rdDataValid.get_int()):
            # Derive convolution window indices
            baseIdx = readIdx/(kernel_width*kernel_width)
            offsetIdx = readIdx%(kernel_width*kernel_width)
            yOffset = offsetIdx/kernel_width
            xOffset = offsetIdx%kernel_width
            pixIndex = baseIdx + yOffset * window_width + xOffset
            assert(rdData.get_int()==test_data[pixIndex])
            # print "{} {}".format(rdData.get_int(), test_data[pixIndex])
            readIdx += 1

        # step
        sess.yield_until_posedge()


if __name__ == "__main__":
    test_buffer_linebuff()
