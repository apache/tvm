import tvm
import numpy as np
from tvm.addon import verilog

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
    wrAdvance = sess.main.write_advance
    wrAddr = sess.main.write_addr
    wrValid = sess.main.write_valid
    wrReady = sess.main.write_ready
    wrData = sess.main.write_data
    rdData = sess.main.read_data
    rdDataValid = sess.main.read_data_valid

    # Simulation input data
    test_data = np.arange(window_width*set_size).astype('int8')

    # Initial state
    rst.put_int(1)
    wrAdvance.put_int(0)
    wrAddr.put_int(0)
    wrValid.put_int(0)
    wrData.put_int(0)

    # De-assert reset
    sess.yield_until_posedge()
    rst.put_int(0)

    # Leave the following signals set to true
    sess.yield_until_posedge()
    wrValid.put_int(1)

    # Main simulation loop
    writeIdx = 0
    readIdx = 0
    while readIdx < len(test_data):
        # write logic
        if (writeIdx < len(test_data)):
            wrAdvance.put_int(0)
            if (wrReady.get_int()):
                wrData.put_int(test_data[writeIdx])
                wrAddr.put_int(writeIdx%window_width)
                if (writeIdx%window_width==window_width-1):
                    wrAdvance.put_int(1)
                writeIdx += 1
        else:
            wrAdvance.put_int(0)
            wrValid.put_int(0)
            
        # correctness checks
        if (rdDataValid.get_int()):
            assert(rdData.get_int()==test_data[readIdx])
            # print "{} {}".format(rdData.get_int(), test_data[readIdx])
            readIdx += 1

        # step
        sess.yield_until_posedge()


if __name__ == "__main__":
    test_buffer_doublebuff()
