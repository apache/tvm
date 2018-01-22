import tvm

local_buf = "local.a"

@tvm.register_func("tvm.info.mem.%s" % local_buf)
def meminfo_cache():
    return tvm.make.node(
            "MemoryInfo",
            unit_bits=8,
            max_simd_bits=32,
            max_num_bits=128 * 128 * 128,
            head_address = None
            )
def cast_intrin(src_dtype, dst_dtype):
    shape = (256,)
    a = tvm.placeholder(shape, name = "a", dtype = src_dtype)

    func = tvm.compute(shape, lambda i: a[i].astype(dst_dtype), name = "a_cast")
    in_buff = tvm.decl_buffer(shape, dtype=src_dtype, name='buffer_src', data=None, scope='local', data_alignment=-1, offset_factor=0)
    out_buff = tvm.decl_buffer(shape, dtype=dst_dtype, name='buffer_dst', data=None, scope='local', data_alignment=-1, offset_factor=0)
    
    def replace_intrin(ins, outs):
        i = ins[0]
        o = outs[0]
        ib = tvm.ir_builder.create()
        ib.emit(tvm.call_extern(dst_dtype, "cast", 
                                 i.access_ptr("r", "int32"),
                                 o.access_ptr('rw',"int32")))
        return ib.get()

    return tvm.decl_tensor_intrin(func.op, replace_intrin, name='a', binds={i:in_buff, o:out_buff})

src_dtype = "float16"
dst_dtype = "float16"

shape = (256,)
a = tvm.placeholder(shape, name = "a", dtype = src_dtype)
nocast = tvm.compute(shape, lambda i: a[i].astype(dst_dtype), name = "a_no_cast")
dst_dtype = "int32"
casts32 = tvm.compute(shape, lambda i: nocast[i].astype(dst_dtype), name = "a_cast_s32")
s = tvm.create_schedule(casts32.op)
print "shit"
s.cache_read(a, local_buf, [nocast])
s.cache_write(casts32, local_buf)
s.cache_write(nocast, local_buf)

s[nocast].compute_inline()
print tvm.lower(s, [a,casts32], simple_mode = True)
