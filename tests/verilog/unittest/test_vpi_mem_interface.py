import tvm
import numpy as np
from tvm.contrib import verilog

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
    def __init__(self, write_data, write_enable, write_pend, data):
        self.write_data = write_data
        self.write_enable = write_enable
        self.write_pend = write_pend
        self.data = data

    def __call__(self):
        if self.data and self.write_pend.get_int():
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
        verilog.find_file("test_vpi_mem_interface.v"),
        verilog.find_file("tvm_vpi_mem_interface.v")
    ])
    rst = sess.main.rst
    read_data = sess.main.read_data
    read_valid = sess.main.read_data_valid
    read_en = sess.main.read_en
    host_read_req = sess.main.read_req
    host_read_addr = sess.main.read_addr
    host_read_size = sess.main.read_size
    rst.put_int(1)
    sess.yield_until_next_cycle()
    rst.put_int(0)
    # hook up reader
    reader = FIFOReader(read_data, read_valid)
    sess.yield_callbacks.append(reader)
    # request read
    host_read_req.put_int(1)
    host_read_addr.put_int(a_ptr)
    host_read_size.put_int(a.shape[0])

    sess.yield_until_next_cycle()
    # second read request
    host_read_addr.put_int(a_ptr + 2)
    host_read_size.put_int(a.shape[0] - 2)

    sess.yield_until_next_cycle()
    host_read_req.put_int(0)
    read_en.put_int(1)

    # yield until read is done
    for i in range(a.shape[0] * 3):
        sess.yield_until_next_cycle()
    sess.shutdown()
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
    w_data = list(range(2, n))
    r_data = np.array(w_data, dtype='int8')

    # head ptr of a
    a_ptr = int(a.handle[0].data)

    sess = verilog.session([
        verilog.find_file("test_vpi_mem_interface.v"),
        verilog.find_file("tvm_vpi_mem_interface.v")
    ])
    rst = sess.main.rst
    write_data = sess.main.write_data
    write_en = sess.main.write_en
    write_ready = sess.main.write_data_ready
    host_write_req = sess.main.write_req
    host_write_addr = sess.main.write_addr
    host_write_size = sess.main.write_size

    rst.put_int(1)
    sess.yield_until_next_cycle()
    rst.put_int(0)
    # hook up writeer
    writer = FIFOWriter(write_data, write_en, write_ready, w_data)

    sess.yield_callbacks.append(writer)
    # request write
    host_write_req.put_int(1)
    host_write_addr.put_int(a_ptr + offset)
    host_write_size.put_int(a.shape[0] - offset)

    sess.yield_until_next_cycle()
    host_write_req.put_int(0)

    # yield until write is done
    for i in range(a.shape[0]+2):
        sess.yield_until_next_cycle()
    sess.shutdown()
    # check if result matches
    np.testing.assert_equal(a.asnumpy()[2:], r_data)


if __name__ == "__main__":
    test_ram_read()
    test_ram_write()
