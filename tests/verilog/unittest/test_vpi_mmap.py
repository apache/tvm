import tvm
import numpy as np
from tvm.contrib import verilog

def test_mmap():
    n = 10
    # context for VPI RAM
    ctx = tvm.vpi(0)
    a_np = np.arange(n).astype('int8')
    a = tvm.nd.array(a_np, ctx)

    # head ptr of a
    a_ptr = int(a.handle[0].data)
    sess = verilog.session([
        verilog.find_file("test_vpi_mmap.v"),
        verilog.find_file("tvm_vpi_mmap.v")
    ])
    rst = sess.main.rst
    read_addr = sess.main.read_addr
    read_data = sess.main.read_data
    write_addr = sess.main.write_addr
    write_data = sess.main.write_data
    write_en = sess.main.write_en
    mmap_addr = sess.main.mmap_addr

    # setup memory map.
    rst.put_int(1)
    sess.yield_until_next_cycle()
    rst.put_int(0)
    write_en.put_int(0)
    mmap_addr.put_int(a_ptr)
    sess.yield_until_next_cycle()

    # read test
    for i in range(n):
        read_addr.put_int(i)
        sess.yield_until_next_cycle()
        # read addr get set this cycle
        sess.yield_until_next_cycle()
        # get the data out
        assert(read_data.get_int() == i)

    # write test
    for i in reversed(range(n)):
        write_addr.put_int(i)
        write_en.put_int(1)
        write_data.put_int(i + 1)
        sess.yield_until_next_cycle()
        write_en.put_int(0)
        sess.yield_until_next_cycle()

    np.testing.assert_equal(a.asnumpy(), a_np + 1)


if __name__ == "__main__":
    test_mmap()
