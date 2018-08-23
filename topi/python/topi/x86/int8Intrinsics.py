"""Core kernel of dot product of 4 Int8 operations"""
import tvm


def _intrin_reduce4int8_common(vec_size, num_elements_intel):
  A = tvm.placeholder((num_elements_intel,), dtype='uint8', name='A')
  B = tvm.placeholder((vec_size, num_elements_intel), dtype='int8', name='B')
  k = tvm.reduce_axis((0, 4), name='k')
  C = tvm.compute((vec_size,), \
      lambda i:  tvm.sum(\
                    A[k].astype('int32') * B[i, k].astype('int32'), \
                    axis=k), name="C")
  s = tvm.create_schedule(C.op)

  Ab = tvm.decl_buffer(A.shape, dtype='uint8', name="Ab",
                       offset_factor=1,
                       strides=[1])
  Bb = tvm.decl_buffer(B.shape, dtype='int8', name="Bb",
                       offset_factor=1,
                       strides=[tvm.var('ldw'), 1])

  def _intrin_func(ins, outs):
    def _instr(index):
      ib = tvm.ir_builder.create()
      if index == 1:
          ib.emit(outs[0].vstore(0, tvm.const(0, 'int32x16')))
          return ib.get()

      A_int8 = ins[0].vload([0], "uint8x4")
      re_int32 = tvm.call_pure_intrin('int32', 'bitcast', A_int8)
      vecA_i32 = tvm.call_pure_intrin('int32x16', 'broadcast16', re_int32);
      vecA = tvm.call_pure_intrin('int8x64', 'bitcast', vecA_i32)
      vecB = ins[1].vload([0, 0], "int8x64")
      vecOne = tvm.const(1, "int16x32")
      pairReduction = tvm.call_llvm_intrin('int16x32', 'llvm.x86.avx512.pmaddubs.w.512', tvm.const(0, 'uint32'), vecA, vecB)
      quadReduction = tvm.call_llvm_intrin('int32x16',
                                            'llvm.x86.avx512.pmaddw.d.512',
                                            tvm.const(0, 'uint32'), \
                                            pairReduction, vecOne);
      vecC = outs[0].vload([0], "int32x16")
      out = quadReduction + vecC
      ib.emit(outs[0].vstore(0, out))
      return ib.get()

    # body, reset, update
    return _instr(0), _instr(1), _instr(2)

  with tvm.build_config(offset_factor=1, partition_const_loop=True):
    return tvm.decl_tensor_intrin(C.op, _intrin_func, binds={A:Ab, B:Bb})

def _intrin_reduce4int8_1x1(vec_size, num_elements_intel):
  A = tvm.placeholder((num_elements_intel,), dtype='uint8', name='A')
  B = tvm.placeholder((vec_size, num_elements_intel, 1, 1), dtype='int8', name='B')
  k = tvm.reduce_axis((0, 4), name='k')
  C = tvm.compute((vec_size,), \
      lambda i:  tvm.sum(\
                    A[k].astype('int32') * B[i, k, 0, 0].astype('int32'), \
                    axis=k), name="C")
  s = tvm.create_schedule(C.op)

  Ab = tvm.decl_buffer(A.shape, dtype='uint8', name="Ab",
                       offset_factor=1,
                       strides=[1])
  Bb = tvm.decl_buffer(B.shape, dtype='int8', name="Bb",
                       offset_factor=1,
                       strides=[tvm.var('ldw'), tvm.var('ldw'), tvm.var('ldw'), 1])

  def _intrin_func(ins, outs):
    def _instr(index):
      ib = tvm.ir_builder.create()
      if index == 1:
          ib.emit(outs[0].vstore(0, tvm.const(0, 'int32x16')))
          return ib.get()

      A_int8 = ins[0].vload([0], "uint8x4")
      re_int32 = tvm.call_pure_intrin('int32', 'bitcast', A_int8)
      vecA_i32 = tvm.call_pure_intrin('int32x16', 'broadcast16', re_int32);
      vecA = tvm.call_pure_intrin('int8x64', 'bitcast', vecA_i32)
      vecB = ins[1].vload([0, 0, 0, 0], "int8x64")
      vecOne = tvm.const(1, "int16x32")
      pairReduction = tvm.call_llvm_intrin('int16x32', 'llvm.x86.avx512.pmaddubs.w.512', tvm.const(0, 'uint32'), vecA, vecB)
      quadReduction = tvm.call_llvm_intrin('int32x16',
                                            'llvm.x86.avx512.pmaddw.d.512',
                                            tvm.const(0, 'uint32'), \
                                            pairReduction, vecOne);
      vecC = outs[0].vload([0], "int32x16")
      out = quadReduction + vecC
      ib.emit(outs[0].vstore(0, out))
      return ib.get()

    # body, reset, update
    return _instr(0), _instr(1), _instr(2)

  with tvm.build_config(offset_factor=1, partition_const_loop=True):
    return tvm.decl_tensor_intrin(C.op, _intrin_func, binds={A:Ab, B:Bb})
