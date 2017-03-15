import tvm
import numpy as np
from tvm.addon import verilog

class FIFOReader(object):
    """Auxiliary class to read from FIFO """
    def __init__(self, read_data, read_valid):
        self.read_data = read_data
        self.read_valid = read_valid
        self.data = []

    def __call__(self):
        if self.read_valid.get_int():
            self.data.append(self.read_data.get_int())

class FIFOWriter(object):
    """Auxiliary class to write to FIFO """
    def __init__(self, write_data, write_enable, write_full, data):
        self.write_data = write_data
        self.write_enable = write_enable
        self.write_full = write_full
        self.data = data

    def __call__(self):
        if self.data and not self.write_full.get_int():
            self.write_enable.put_int(1)
            self.write_data.put_int(int(self.data[0]))
            del self.data[0]
        else:
            self.write_enable.put_int(0)


def test_ram_read():
    n = 10
    # context for VPI RAM
    ctx = tvm.vpi(0)
    a_np = np.arange(n).astype('int8')
    a = tvm.nd.array(a_np, ctx)

    # head ptr of a
    a_ptr = int(a.handle[0].data)
    sess = verilog.session([
        verilog.find_file("test_vpi_ram.v"),
        verilog.find_file("tvm_vpi_ram.v")
    ])
    rst = sess.main.rst
    read_data = sess.main.read_data
    read_valid = sess.main.read_valid
    read_dequeue = sess.main.read_dequeue
    ctrl_read_req = sess.main.ctrl_read_req
    ctrl_read_addr = sess.main.ctrl_read_addr
    ctrl_read_size = sess.main.ctrl_read_size
    rst.put_int(1)
    sess.yield_until_posedge()
    rst.put_int(0)
    # hook up reader
    reader = FIFOReader(read_data, read_valid)
    sess.yield_callbacks.append(reader)
    # request read
    ctrl_read_req.put_int(1)
    ctrl_read_addr.put_int(a_ptr)
    ctrl_read_size.put_int(a.shape[0])

    sess.yield_until_posedge()
    # second read request
    ctrl_read_addr.put_int(a_ptr + 2)
    ctrl_read_size.put_int(a.shape[0] - 2)

    sess.yield_until_posedge()
    ctrl_read_req.put_int(0)
    read_dequeue.put_int(1)

    # yield until read is done
    for i in range(a.shape[0] * 2):
        sess.yield_until_posedge()
    # check if result matches
    r = np.concatenate((a_np, a_np[2:]))
    np.testing.assert_equal(np.array(reader.data), r)


def test_ram_write():
    n = 10
    # read from offset
    offset = 2
    # context for VPI RAM
    ctx = tvm.vpi(0)
    a_np = np.zeros(n).astype('int8')
    a = tvm.nd.array(a_np, ctx)
    w_data = range(2, n)
    r_data = np.array(w_data, dtype='int8')

    # head ptr of a
    a_ptr = int(a.handle[0].data)

    sess = verilog.session([
        verilog.find_file("test_vpi_ram.v"),
        verilog.find_file("tvm_vpi_ram.v")
    ])
    rst = sess.main.rst
    write_data = sess.main.write_data
    write_enable = sess.main.write_enable
    write_full = sess.main.write_full
    ctrl_write_req = sess.main.ctrl_write_req
    ctrl_write_addr = sess.main.ctrl_write_addr
    ctrl_write_size = sess.main.ctrl_write_size

    rst.put_int(1)
    sess.yield_until_posedge()
    rst.put_int(0)
    # hook up writeer
    writer = FIFOWriter(write_data, write_enable, write_full, w_data)

    sess.yield_callbacks.append(writer)
    # request write
    ctrl_write_req.put_int(1)
    ctrl_write_addr.put_int(a_ptr + offset)
    ctrl_write_size.put_int(a.shape[0] - offset)

    sess.yield_until_posedge()
    ctrl_write_req.put_int(0)

    # yield until write is done
    for i in range(a.shape[0]+2):
        sess.yield_until_posedge()

    # check if result matches
    np.testing.assert_equal(a.asnumpy()[2:],r_data)


if __name__ == "__main__":
    test_ram_write()
