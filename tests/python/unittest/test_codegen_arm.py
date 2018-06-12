import tvm
import re
import os
import ctypes

def test_popcount():
    target = 'llvm -target=armv7l-none-linux-gnueabihf -mcpu=cortex-a53 -mattr=+neon'

    def check_correct_assembly(type, elements, counts):
        n = tvm.convert(elements)
        A = tvm.placeholder(n, dtype=type, name='A')
        B = tvm.compute(A.shape, lambda i: tvm.popcount(A[i]), name='B')
        s = tvm.create_schedule(B.op)
        s[B].vectorize(s[B].op.axis[0])
        f = tvm.build(s, [A, B], target)

        # Verify we see the correct number of vpaddl and vcnt instructions in the assembly
        assembly = f.get_source('asm')
        matches = re.findall("vpaddl", assembly)
        assert (len(matches) == counts)
        matches = re.findall("vcnt", assembly)
        assert (len(matches) == 1)
    check_correct_assembly('uint16', 8, 1)
    check_correct_assembly('uint16', 4, 1)
    check_correct_assembly('uint32', 4, 2)
    check_correct_assembly('uint32', 2, 2)
    check_correct_assembly('uint64', 2, 3)

if __name__ == "__main__":
    test_popcount()
