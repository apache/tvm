import logging
import sys

import numpy as np
import tvm

from tvm import autotvm


def matmul_nn(batch, A, B, L, dtype='float16', layout='NN'):
    k = tvm.reduce_axis((0, L), name='k')
    if dtype == 'float16':
      out_type = 'float'
    elif dtype == 'int8':
      out_type = 'int'
    if (layout == 'NN'):
      return tvm.compute((batch, N, M), lambda b, i, j: tvm.sum((A[b, i, k] * B[b, k, j]).astype(out_type), axis=k))
    if (layout == 'NT'):
      return tvm.compute((batch, N, M), lambda b, i, j: tvm.sum((A[b, k, i] * B[b, k, j]).astype(out_type), axis=k))
    if (layout == 'TN'):
      return tvm.compute((batch, N, M), lambda b, i, j: tvm.sum((A[b, i, k] * B[b, j, k]).astype(out_type), axis=k))
    if (layout == 'TT'):
      return tvm.compute((batch, N, M), lambda b, i, j: tvm.sum((A[b, k, i] * B[b, j, k]).astype(out_type), axis=k))

@autotvm.template
def test_gemm_nn(batch, N, L, M, dtype, layout):
    if (layout == "NN"):
      shape_a = (batch, N, L)
      shape_b = (batch, L, M)
    elif (layout == "NT"):
      shape_a = (batch, L, N)
      shape_b = (batch, L, M)
    elif (layout == "TN"):
      shape_a = (batch, N, L)
      shape_b = (batch, M, L)
    elif (layout == "TT"):
      shape_a = (batch, L, N)
      shape_b = (batch, M, L)
    else:
      print ("Unsupported layout:", layout)
      sys.exit(1);
    A = tvm.placeholder(shape_a, name='A', dtype=dtype)
    B = tvm.placeholder(shape_b, name='B', dtype=dtype)
    C = matmul_nn(batch, A, B, L, dtype, layout)

    s = tvm.create_schedule(C.op)
    z, y, x = s[C].op.axis
    k = s[C].op.reduce_axis[0]

    AA = s.cache_read(A, "shared", [C])
    AL = s.cache_read(AA, "local", [C])
    BB = s.cache_read(B, "shared", [C])
    BL = s.cache_read(BB, "local", [C])
    CL = s.cache_write(C, "local")

    cfg = autotvm.get_config()

    cfg.define_knob("bx", [2, 4, 8])
    cfg.define_knob("by", [16, 32, 64])
    cfg.define_knob("step_k", [8, 16, 32])
    cfg.define_knob("v", [4, 8])
    bx = cfg['bx'].val
    by = cfg['by'].val
    step_k = cfg['step_k'].val
    v = cfg['v'].val

    TX = 8
    TY = 1
    tile_x = bx * TX
    tile_y = by * TY
    WX = min(16, tile_x)
    tile_k = 16
    vthread = 1

    yo, ty = s[C].split(y, tile_y*vthread)
    vy, ty = s[C].split(ty, tile_y)
    ty, yi = s[C].split(ty, TY)

    xo, xi = s[C].split(x, tile_x)
    tz, xi = s[C].split(xi, WX)
    tx, xi = s[C].split(xi, TX)
    ko, ki = s[CL].split(k, step_k * tile_k)
    kl, ki = s[CL].split(ki, tile_k)

    s[C].reorder(z, yo, xo, tz, ty, tx, yi, xi)
    s[C].bind(z, tvm.thread_axis("blockIdx.z"))
    s[C].bind(yo, tvm.thread_axis("blockIdx.y"))
    s[C].bind(xo, tvm.thread_axis("blockIdx.x"))
    s[C].bind(ty, tvm.thread_axis("threadIdx.y"))
    s[C].bind(tz, tvm.thread_axis("threadIdx.z"))
    s[C].bind(tx, tvm.thread_axis("threadIdx.x"))
    s[C].bind(vy, tvm.thread_axis((0, vthread), "vthread", name="vy"))
    s[CL].compute_at(s[C], tx)
    zo, yo, xo = CL.op.axis
    s[CL].reorder(ko, kl, ki, zo, yo, xo)

    s[AA].compute_at(s[CL], ko)
    xo, xi = s[AA].split(s[AA].op.axis[2], factor=bx*v)
    tz, tx = s[AA].split(xi, factor=(WX//TX)*v)
    tx, vec = s[AA].split(tx, factor=v)
    fused = s[AA].fuse(s[AA].op.axis[1], xo)
    _, ty = s[AA].split(fused, factor=by)
    s[AA].bind(ty, tvm.thread_axis("threadIdx.y"))
    s[AA].bind(tz, tvm.thread_axis("threadIdx.z"))
    s[AA].bind(tx, tvm.thread_axis("threadIdx.x"))
    s[AA].vectorize(vec)

    s[BB].compute_at(s[CL], ko)
    xo, xi = s[BB].split(s[BB].op.axis[2], factor=bx*v)
    tz, tx = s[BB].split(xi, factor=(WX//TX)*v)
    tx, vec = s[BB].split(tx, factor=v)
    fused = s[BB].fuse(s[BB].op.axis[1], xo)
    _, ty = s[BB].split(fused, factor=by)
    s[BB].bind(ty, tvm.thread_axis("threadIdx.y"))
    s[BB].bind(tz, tvm.thread_axis("threadIdx.z"))
    s[BB].bind(tx, tvm.thread_axis("threadIdx.x"))
    s[BB].vectorize(vec)

    s[AL].compute_at(s[CL], kl)
    s[BL].compute_at(s[CL], kl)

    s[CL].pragma(ko, 'tensor_core')

    return s, [A, B, C]

batch, M, N, L = 16, 512, 64, 512
dtype = 'float16'
layout = 'NN'
if len(sys.argv) >= 5:
  batch, M, N, L = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
if len(sys.argv) >= 6:
  dtype = sys.argv[5]
if len(sys.argv) >= 7:
  layout = sys.argv[6]

print ("batch=%d, M=%d, N=%d, K=%d, dtype=%s, layout=%s" % (batch, M, N, L, dtype, layout))

task = autotvm.task.create(test_gemm_nn, args=(batch, N, L, M, dtype, layout), target='cuda')
print(task.config_space)

logging.getLogger('autotvm').setLevel(logging.DEBUG)
logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))

measure_option = autotvm.measure_option(
    builder='local',
    runner=autotvm.LocalRunner(number=5))

tuner = autotvm.tuner.XGBTuner(task)
log_file = "batch_matmul_%d_%d_%d_%d_%s_%s.log" % (batch, M, N, L, dtype, layout)
with tvm.build_config():
    tuner.tune(n_trial=1000,
               measure_option=measure_option,
               callbacks=[autotvm.callback.log_to_file(log_file)])

dispatch_context = autotvm.apply_history_best(log_file)
best_config = dispatch_context.query(task.target, task.workload)
print("\nBest config:")
print(best_config)
with autotvm.apply_history_best(log_file):
    with tvm.target.create("cuda"):
        with tvm.build_config():
            s, arg_bufs = test_gemm_nn(batch, N, L, M, dtype, layout)
            print(tvm.lower(s, arg_bufs, simple_mode=True))
            func = tvm.build(s, arg_bufs)
dev_module = func.imported_modules[0]
print(dev_module.get_source())

# check correctness
if (layout == "NN"):
  shape_a = (batch, N, L)
  shape_b = (batch, L, M)
elif (layout == "NT"):
  shape_a = (batch, L, N)
  shape_b = (batch, L, M)
elif (layout == "TN"):
  shape_a = (batch, N, L)
  shape_b = (batch, M, L)
elif (layout == "TT"):
  shape_a = (batch, L, N)
  shape_b = (batch, M, L)

a_np = None
b_np = None
c_np = None
c_np_type = None
if dtype == 'float16':
  c_np_type = np.float32
  a_np = np.random.uniform(size=shape_a).astype(np.float16)
  b_np = np.random.uniform(size=shape_b).astype(np.float16)
  c_np = np.zeros((batch, N, M), dtype=np.float32)
  if (layout == "NN"):
    for bs in range(batch):
      c_np[bs, :, :] = np.dot(a_np[bs, :, :], b_np[bs, :, :])
  elif (layout == "NT"):
    for bs in range(batch):
      c_np[bs, :, :] = np.dot(a_np[bs, :, :].T, b_np[bs, :, :])
  elif (layout == "TN"):
    for bs in range(batch):
      c_np[bs, :, :] = np.dot(a_np[bs, :, :], b_np[bs, :, :].T)
  elif (layout == "TT"):
    for bs in range(batch):
      c_np[bs, :, :] = np.dot(a_np[bs, :, :].T, b_np[bs, :, :].T)
elif dtype == 'int8':
  c_np_type = np.int32
  a_np = np.random.randint(low=-128, high=127, size=shape_a).astype(np.int8)
  b_np = np.random.randint(low=-128, high=127, size=shape_b).astype(np.int8)
  c_np = np.zeros((batch, N, M), dtype=np.int32)
  if (layout == "NN"):
    for bs in range(batch):
      c_np[bs, :, :] = np.dot(a_np[bs, :, :].astype(int32), b_np[bs, :, :].astype(int32))
  elif (layout == "NT"):
    for bs in range(batch):
      c_np[bs, :, :] = np.dot(a_np[bs, :, :].astype(int32).T, b_np[bs, :, :].astype(int32))
  elif (layout == "TN"):
    for bs in range(batch):
      c_np[bs, :, :] = np.dot(a_np[bs, :, :].astype(int32), b_np[bs, :, :].astype(int32).T)
  elif (layout == "TT"):
    for bs in range(batch):
      c_np[bs, :, :] = np.dot(a_np[bs, :, :].astype(int32).T, b_np[bs, :, :].astype(int32).T)

ctx = tvm.gpu()
c_tvm = tvm.nd.array(np.zeros(c_np.shape, dtype=c_np_type), ctx=ctx)
a_tvm = tvm.nd.array(a_np, ctx=ctx)
b_tvm = tvm.nd.array(b_np, ctx=ctx)
func(a_tvm, b_tvm, c_tvm)

tvm.testing.assert_allclose(c_np, c_tvm.asnumpy(), rtol=1e-3)

evaluator = func.time_evaluator(func.entry_name, ctx, number=100)
print('Time cost of this operator: %f' % evaluator(a_tvm, b_tvm, c_tvm).mean)
