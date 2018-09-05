"""Core kernel of dot product of 4 Int8 operations"""
#pylint: disable=invalid-name
import tvm


def _intrin_reduce4int8_common(vec_size, num_elements_intel):
    data = tvm.placeholder((num_elements_intel,), dtype='uint8', name='data')
    kernel = tvm.placeholder((vec_size, num_elements_intel), dtype='int8', name='kernel')
    k = tvm.reduce_axis((0, 4), name='k')
    C = tvm.compute((vec_size,),
                    lambda i: tvm.sum(data[k].astype('int32') *
                                      kernel[i, k].astype('int32'),
                                      axis=k),
                    name="C")

    a_buffer = tvm.decl_buffer(data.shape, dtype='uint8', name="a_buffer",
                               offset_factor=1,
                               strides=[1])
    b_buffer = tvm.decl_buffer(kernel.shape, dtype='int8', name="b_buffer",
                               offset_factor=1,
                               strides=[tvm.var('ldw'), 1])

    def _intrin_func(ins, outs):
        def _instr(index):
            ib = tvm.ir_builder.create()
            if index == 1:
                ib.emit(outs[0].vstore(0, tvm.const(0, 'int32x16')))
                return ib.get()

            a_int8 = ins[0].vload([0], "uint8x4")
            re_int32 = tvm.call_pure_intrin('int32', 'bitcast', a_int8)
            vec_ai32 = re_int32.astype('int32x16')
            vec_a = tvm.call_pure_intrin('int8x64', 'bitcast', vec_ai32)
            vec_b = ins[1].vload([0, 0], "int8x64")
            vec_one = tvm.const(1, "int16x32")
            pair_reduction = tvm.call_llvm_intrin('int16x32',
                                                  'llvm.x86.avx512.pmaddubs.w.512',
                                                  tvm.const(0, 'uint32'),
                                                  vec_a, vec_b)
            quad_reduction = tvm.call_llvm_intrin('int32x16',
                                                  'llvm.x86.avx512.pmaddw.d.512',
                                                  tvm.const(0, 'uint32'),
                                                  pair_reduction, vec_one)
            vec_c = outs[0].vload([0], "int32x16")
            out = quad_reduction + vec_c
            ib.emit(outs[0].vstore(0, out))
            return ib.get()

        # body, reset, update
        return _instr(0), _instr(1), _instr(2)

    with tvm.build_config(offset_factor=1, partition_const_loop=True):
        return tvm.decl_tensor_intrin(C.op, _intrin_func, binds={data:a_buffer, kernel:b_buffer})

def _intrin_reduce4int8_1x1(vec_size, num_elements_intel):
    data = tvm.placeholder((num_elements_intel,), dtype='uint8', name='data')
    kernel = tvm.placeholder((vec_size, num_elements_intel, 1, 1), dtype='int8', name='kernel')
    k = tvm.reduce_axis((0, 4), name='k')
    C = tvm.compute((vec_size,), \
                    lambda i: tvm.sum(data[k].astype('int32') *
                                      kernel[i, k, 0, 0].astype('int32'),
                                      axis=k),
                    name="C")

    a_buffer = tvm.decl_buffer(data.shape, dtype='uint8', name="a_buffer",
                               offset_factor=1,
                               strides=[1])
    b_buffer = tvm.decl_buffer(kernel.shape, dtype='int8', name="b_buffer",
                               offset_factor=1,
                               strides=[tvm.var('ldw'),
                                        tvm.var('ldw'),
                                        tvm.var('ldw'), 1]
                              )

    def _intrin_func(ins, outs):
        def _instr(index):
            ib = tvm.ir_builder.create()
            if index == 1:
                ib.emit(outs[0].vstore(0, tvm.const(0, 'int32x16')))
                return ib.get()

            a_int8 = ins[0].vload([0], "uint8x4")
            re_int32 = tvm.call_pure_intrin('int32', 'bitcast', a_int8)
            vec_ai32 = re_int32.astype('int32x16')
            vec_a = tvm.call_pure_intrin('int8x64', 'bitcast', vec_ai32)
            vec_b = ins[1].vload([0, 0, 0, 0], "int8x64")
            vec_one = tvm.const(1, "int16x32")
            pair_reduction = tvm.call_llvm_intrin('int16x32',
                                                  'llvm.x86.avx512.pmaddubs.w.512',
                                                  tvm.const(0, 'uint32'),
                                                  vec_a, vec_b)
            quad_reduction = tvm.call_llvm_intrin('int32x16',
                                                  'llvm.x86.avx512.pmaddw.d.512',
                                                  tvm.const(0, 'uint32'), \
                                                  pair_reduction, vec_one)
            vec_c = outs[0].vload([0], "int32x16")
            out = quad_reduction + vec_c
            ib.emit(outs[0].vstore(0, out))
            return ib.get()

        # body, reset, update
        return _instr(0), _instr(1), _instr(2)

    with tvm.build_config(offset_factor=1, partition_const_loop=True):
        return tvm.decl_tensor_intrin(C.op, _intrin_func, binds={data:a_buffer, kernel:b_buffer})
