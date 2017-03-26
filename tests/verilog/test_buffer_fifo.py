import tvm
import numpy as np
from tvm.addon import verilog

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
    valid = sess.main.read_data_valid
    dataIn = sess.main.write_data
    dataOut = sess.main.read_data

    # Simulation input data
    test_data = np.arange(16).astype('int8')

    # Initial state
    rst.put_int(1)
    enq.put_int(0)
    dataIn.put_int(0)

    # De-assert reset
    sess.yield_until_posedge()
    rst.put_int(0)

    # Main simulation loop
    readIdx = 0
    writeIdx = 0
    while readIdx < len(test_data):
        # write logic
        if (writeIdx < len(test_data)):
            enq.put_int(1)
            dataIn.put_int(writeIdx)
            writeIdx += 1
        else:
            enq.put_int(0)
        # read logic
        if (valid.get_int()):
            assert(dataOut.get_int()==test_data[readIdx])
            readIdx += 1
        sess.yield_until_posedge()


if __name__ == "__main__":
    test_buffer_fifo()
