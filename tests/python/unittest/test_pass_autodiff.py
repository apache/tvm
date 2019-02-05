import tvm
import topi
import numpy as np
from tvm.testing import check_numerical_grads, estimate_performance, PerformanceEstimate
import time
import inspect
import sys

# Whether to dump the generated code
verbose = False

def get_shape(tensor, param_values=None):
    if param_values is None:
        param_values = {}
    return [tvm.ir_pass.Simplify(tvm.ir_pass.Substitute(s, param_values)).value
            for s in tensor.shape]

def check_equivalence(outputs1, outputs2, inputs, in_range=(-10, 10), iters=3):
    outputs1 = list(outputs1)
    outputs2 = list(outputs2)
    sched1 = tvm.create_schedule([o.op for o in outputs1])
    mout1 = tvm.build(sched1, outputs1 + inputs)

    sched2 = tvm.create_schedule([o.op for o in outputs2])
    mout2 = tvm.build(sched2, outputs2 + inputs)

    arguments1 = [tvm.nd.empty(get_shape(t), t.dtype) for t in outputs1 + inputs]
    arguments2 = [tvm.nd.empty(get_shape(t), t.dtype) for t in outputs1 + inputs]

    for i in range(iters):
        arguments1 = []
        arguments2 = []
        for a in outputs1 + inputs:
            val = np.random.uniform(in_range[0], in_range[1], size=get_shape(a)).astype(a.dtype)
            arguments1.append(tvm.nd.array(val))
            arguments2.append(tvm.nd.array(val))
        mout1(*arguments1)
        mout2(*arguments2)

        for j, _ in enumerate(outputs1):
            tvm.testing.assert_allclose(arguments1[j].asnumpy(), arguments2[j].asnumpy())

def check_grad(out, inputs, args=[], in_range=(-10,10), perf=None, param_values=None,
               acceptable_fail_fraction=None):
    line = inspect.getframeinfo(inspect.stack()[1][0]).lineno

    if not isinstance(inputs, (list, tuple)):
        inputs = [inputs]

    if param_values is None:
        param_values = {}

    if verbose:
        print("\n" + 80*"=" + "\n")
        print("Testing gradients, line {}\n".format(line))
        print("Original tensors:\n")
        print(tvm.PrintTensorRecursively(out))
        print()

    sout = tvm.create_schedule(out.op)
    mout = tvm.build(sout, [out] + inputs + args)

    ones = topi.full_like(out, 1.0)

    grads = list(tvm.differentiate(out, inputs, ones))

    if verbose:
        print("Gradients:\n")
        print(tvm.PrintTensorsRecursively(grads))
        print()

    grads_sched = tvm.create_schedule([g.op for g in grads])
    mgrad = tvm.build(grads_sched, grads + inputs + args)

    lowered = tvm.lower(grads_sched, grads + inputs + args, simple_mode=True)

    if verbose:
        print("Lowered gradients:\n")
        print(lowered)
        print()

    if perf != False:
        est = estimate_performance(grads, param_values=param_values)
        est_lowered = estimate_performance(lowered, param_values=param_values)

        if verbose:
            print("Note: performance tuples are (iterations, multiplications, memory)")
            print("Expected performance of grads: {}".format(perf))
            print("Estimated performance of grads: {}".format(est.as_tuple()))
            print("Estimated performance of lowered grads: {}".format(est_lowered.as_tuple()))
            print()

        if est_lowered.memory > est.memory:
            print("WARNING: Line {}: The estimated memory consumption increased after lowering, "
                  "this may indicate that tensor bounds have been expanded too much".format(line))
            print("before: {}  after: {}".format(est, est_lowered))

        (iters, mults, mem) = est.as_tuple()
        if perf is None or isinstance(perf, str):
            print("WARNING: Line {}: No performance information, you may set it to {}"
                  .format(line, est.as_tuple()))
            if isinstance(perf, str):
                print("0,/{!r}/{{s/{!r}/{}/}}".format(perf, perf, (iters, mults, mem)))
        elif perf != (iters, mults, mem):
            (ref_iters, ref_mults, ref_mem) = perf
            ref_est = PerformanceEstimate(*perf)

            if est <= ref_est:
                print("WARNING: Line {}: Estimated performance {} is better than {}. "
                      "Use this with sed:"
                      .format(line, est.as_tuple(), ref_est.as_tuple()))
                print("0,/{}/{{s/{}/{}/}}".format(perf, perf, (iters, mults, mem)))
            elif est >= ref_est:
                print("WARNING: Line {}: Estimated performance {} IS WORSE THAN {}"
                      .format(line, est.as_tuple(), ref_est.as_tuple()))
            else:
                print("WARNING: Line {}: Estimated performance {} does not match {}"
                      .format(line, est.as_tuple(), ref_est.as_tuple()))

            EST_RTOL = 1.5
            if iters > ref_iters*EST_RTOL or mults > ref_mults*EST_RTOL or mem > ref_mem*EST_RTOL:
                raise AssertionError("Line {}: Some of the estimated performance metrics are much "
                                     "worse than the reference ones (by {}): "
                                     "estimated {}, expected {}"
                                     .format(line, EST_RTOL, est.as_tuple(), ref_est.as_tuple()))

    input_vals = [tvm.nd.array(np.random.uniform(in_range[0], in_range[1],
                                                 size=get_shape(a, param_values)).astype(a.dtype))
                  for a in inputs]
    arg_vals = [tvm.nd.array(np.random.uniform(in_range[0], in_range[1],
                                               size=get_shape(a, param_values)).astype(a.dtype))
                for a in args]

    def fun(*arguments):
        arrays = [tvm.nd.empty(get_shape(out, param_values), out.dtype)] + \
            [tvm.nd.array(a) for a in list(arguments) + arg_vals]
        mout(*arrays)
        return arrays[0].asnumpy().sum()

    g_arg_vals = \
        [tvm.nd.empty(get_shape(i, param_values), g.dtype) for i, g in zip(inputs, grads)] + \
        input_vals + arg_vals
    mgrad(*g_arg_vals)
    g_res = [g_arg_vals[g].asnumpy() for g, _ in enumerate(grads)]

    check_numerical_grads(fun, [a.asnumpy() for a in input_vals], g_res,
                          acceptable_fail_fraction=acceptable_fail_fraction)

def test_differentiate_function():
    x = tvm.placeholder((32, 3, 28, 28), name='x')

    w = tvm.placeholder((10, 3, 3, 3), name='w')
    t1 = topi.nn.conv2d(x, w, 1, 0, 1)

    t2 = topi.nn.flatten(t1)
    t3 = topi.sum(t2)

    [dx1, dw1] = tvm.differentiate(t3, [x, w])
    [dx2, dw2] = tvm.differentiate(t2, [x, w], topi.full_like(t2, 1.0))

    check_equivalence([dx1, dw1], [dx2, dw2], [x, w])

    def mydiff(out, inp, head, t1=t1, t2=t2):
        assert out == t2 and inp == [t1]
        return [tvm.compute(t1.shape,
                            lambda ax0, ax1, ax2, ax3: head[ax0, ax3 + ax2*26 + ax1*676])]

    res = tvm.differentiate(t3, [x, w], override={t2: ([t1], mydiff)})
    check_equivalence(res.result, [dx1, dw1], [x, w])

    def mydiff2(out, inputs, head):
        return tvm.differentiate(out, inputs, head)

    res = tvm.differentiate(t3, [x, w], override={t1: ([x, w], mydiff2)})
    check_equivalence(res.result, [dx1, dw1], [x, w])

# Test some simple expressions
def test_autodiff():
    x = tvm.var("x", dtype='float32')
    k = tvm.reduce_axis((0, 10), name="k")
    l = tvm.reduce_axis((0, 10), name="l")
    A0 = tvm.placeholder((10, 10), name='A0')
    A1 = tvm.placeholder((10, 10), name='A1')

    B = tvm.compute((10, 10), lambda i, j: A0[i, j] + A0[j, i], name='B')
    check_grad(B, A0, perf=(10100, 10000, 200))

    B = tvm.compute((10, 10), lambda i, j: tvm.floor(A0[i, j]), name='B')
    check_grad(B, A0, perf=(100, 0, 100), acceptable_fail_fraction=0.05)

    B = tvm.compute((10, 10), lambda i, j: tvm.ceil(A0[i, j]), name='B')
    check_grad(B, A0, perf=(100, 0, 100), acceptable_fail_fraction=0.05)

    B = tvm.compute((10, 10), lambda i, j: tvm.trunc(A0[i, j]), name='B')
    check_grad(B, A0, perf=(100, 0, 100), acceptable_fail_fraction=0.05)

    B = tvm.compute((10, 10), lambda i, j: tvm.round(A0[i, j]), name='B')
    check_grad(B, A0, perf=(100, 0, 100), acceptable_fail_fraction=0.05)

    B = tvm.compute((10, 10), lambda i, j: A0[i, j] + tvm.exp(A0[j, i]), name='B')
    check_grad(B, A0, perf=(10100, 20000, 200))

    B = tvm.compute((10, 10), lambda i, j: tvm.log(tvm.abs(A0[i, j] + tvm.exp(A0[j, i]))), name='B')
    check_grad(B, A0, perf=(10100, 70000, 200))

    B = tvm.compute((10, 10), lambda i, j: tvm.sigmoid(A0[i, j]*A0[i, j]*A0[j, i]), name='B')
    check_grad(B, A0, perf=(10100, 120000, 200))

    B = tvm.compute((10, 10), lambda i, j: tvm.tanh(A0[i, j]*A0[i, j]*A0[j, i]), name='B')
    check_grad(B, A0, perf=(10100, 120000, 200))

    B = tvm.compute((10, 10), lambda i, j: tvm.sqrt(A0[i, j]*A0[i, j]*A0[j, i]), name='B')
    check_grad(B, A0, perf=(10100, 90000, 200), in_range=(0.1, 10))

    B = tvm.compute((10, 10), lambda i, j: tvm.power(tvm.abs(A0[i, j]), A0[j, i]), name='B')
    check_grad(B, A0, perf=(10100, 90000, 200))

    B = tvm.compute((10, 10), lambda i, j: A0[i, j] * A0[j, i], name='B')
    check_grad(B, A0, perf=(10100, 10000, 200))

    # TODO: This one needs transforming Sum(a + b) -> Sum(a) + Sum(b)
    B = tvm.compute((10,), lambda i: tvm.sum(A0[i, k]*A0[k, i], axis=k), name='B')
    check_grad(B, A0, perf=(11010, 1000, 1110))

    B = tvm.compute((10, 10), lambda i, j: tvm.sum(A0[i, k]*A0[k, i] + 5, axis=k), name='B')
    check_grad(B, A0, perf=(20100, 10000, 1200))

    B = tvm.compute((10, 10), lambda i, j: tvm.max(A0[i, k]*A0[k, j] + 5, axis=k), name='B')
    check_grad(B, A0, perf=(110100, 310000, 20200))

    B = tvm.compute((10, 10), lambda i, j: A0[i, j] * (A1[j, i] + A0[j, i]), name='B')
    check_grad(B, A0, [A1], perf=(10100, 10000, 200))

    B = tvm.compute((10, 10), lambda i, j: tvm.sum(A0[k, k] - A0[tvm.min(j + k, 9), j]*A0[i, k],
                                                   axis=k),
                    name='B')
    check_grad(B, A0, perf=(110100, 10000, 10200))

    def fcombine(x, y):
        return x*y

    def fidentity(t0):
        return tvm.const(1, t0)

    prod = tvm.comm_reducer(fcombine, fidentity, name='prod')
    B = tvm.compute((10, 10), lambda i, j: prod(A0[i, k] + A0[k, i], axis=k), name='B')
    check_grad(B, A0, perf=(20100, 40000, 2200))

    X = tvm.placeholder((10,), name='X')
    A = tvm.compute((10,), lambda i: X[i] + X[9 - i])
    B = tvm.compute((10,), lambda i: X[i] * X[9 - i])
    Y = topi.tensordot(A, B, 1)
    check_grad(Y, X, perf=(251, 230, 71))

def test_topi_autodiff():
    X = tvm.placeholder((1, 2, 4, 4), name='X')
    W = tvm.placeholder((5, 2, 3, 3), name='W')
    W1 = tvm.placeholder((2, 5, 3, 3), name='W1')
    W2 = tvm.placeholder((1,), name='W2')

    R = topi.nn.conv2d(X, W, 1, 1, 1)
    check_grad(R, [X, W], perf=(3410, 2880, 652))

    R1 = topi.nn.conv2d(topi.nn.relu(R), W1, 1, 0, 1)
    check_grad(R1, [X, W, W1], perf=(6198, 5320, 1250))

    R = topi.broadcast_to(W2, (5, 2, 3, 3))
    check_grad(R, [W2], perf=(180, 0, 91))

    R = topi.nn.conv2d(X, topi.broadcast_to(W2, (5, 2, 3, 3)), 1, 1, 1)
    check_grad(R, [X, W2], perf=(3590, 2880, 743))

    R = topi.nn.pool(X, [2, 2], [2, 2], [0, 0, 0, 0], 'avg')
    check_grad(R, X, perf=(40, 224, 40))

    R = topi.nn.pool(X, [2, 2], [2, 2], [0, 0, 0, 0], 'max')
    check_grad(R, X, perf=(168, 1248, 104))

    X = tvm.placeholder((1, 2, 5, 5), name='X')
    R = topi.reshape(X, (1, 32))
    check_grad(R, [X], perf=(82, 1200, 82))

    X = tvm.placeholder((1, 2, 5, 5), name='X')
    W = tvm.placeholder((2, 2, 3, 3), name='W')

    S = topi.reshape(X, (1, 50))
    check_grad(S, [X], perf=(100, 700, 100))

    R = X + topi.nn.conv2d(X + topi.nn.conv2d(X, W, 1, 1, 1), W, 1, 1, 1)
    check_grad(R, [X, W], perf=(6854, 5400, 1726))

    S = topi.nn.softmax(topi.reshape(R, (1, 50)))
    check_grad(S, [X, W], perf=(10956, 14201, 2333))

    S = topi.sigmoid(topi.reshape(R, (1, 50)))
    check_grad(S, [X, W], perf=(8004, 8350, 2026))

    S = topi.tanh(topi.reshape(R, (1, 50)))
    check_grad(S, [X, W], perf=(8004, 8350, 2026))

    S = topi.nn.log_softmax(topi.reshape(R, (1, 50)))
    check_grad(S, [X, W], perf=(10906, 13601, 2283))
    check_grad(S, [W], [X], perf=(8920, 11101, 1997))

    X = tvm.placeholder((1, 2, 3, 5), name='X')
    Y = tvm.placeholder((1, 2, 7, 5), name='Y')
    S = topi.concatenate((X, Y), 2)
    check_grad(S, [X, Y], perf=(100, 0, 100))

    X = tvm.placeholder((1, 2, 6, 5), name='X')
    (S, R) = topi.split(X, 2, 2)
    check_grad(S, [X], perf=(120, 0, 120))
    check_grad(R, [X], perf=(120, 0, 120))
    R1 = topi.concatenate((S, R), 2)
    check_grad(R1, [X], perf=(300, 0, 300))
    R2 = topi.concatenate((R, S), 2)
    check_grad(R2, [X], perf=(300, 0, 300))

    X = tvm.placeholder((4, 5), name='X')
    I = tvm.placeholder((100,), name='I', dtype='int32')
    R = topi.take(X, topi.abs(I))
    check_grad(R, [X], [I], perf=(2200, 6000, 220))

def test_stride_dilation():
    X = tvm.placeholder((1, 2, 10, 10), name='X')

    W = tvm.placeholder((2, 2, 1, 1), name='W')

    Y = topi.nn.conv2d(X, W, 1, 0, 1)
    check_grad(Y, [X, W], perf=(1404, 800, 808))
    Y = topi.nn.conv2d(X, W, 2, 0, 1)
    check_grad(Y, [X, W], perf=(928, 1572, 670))
    Y = topi.nn.conv2d(X, W, 3, 0, 1)
    check_grad(Y, [X, W], perf=(932, 1728, 672))
    Y = topi.nn.conv2d(X, W, 1, 0, 2)
    check_grad(Y, [X, W], perf=(1404, 800, 808))
    Y = topi.nn.conv2d(X, W, 2, 0, 2)
    check_grad(Y, [X, W], perf=(928, 1572, 670))
    Y = topi.nn.conv2d(X, W, 3, 0, 2)
    check_grad(Y, [X, W], perf=(932, 1728, 672))
    Y = topi.nn.conv2d(X, W, 1, 0, 3)
    check_grad(Y, [X, W], perf=(1404, 800, 808))
    Y = topi.nn.conv2d(X, W, 2, 0, 3)
    check_grad(Y, [X, W], perf=(928, 1572, 670))
    Y = topi.nn.conv2d(X, W, 3, 0, 3)
    check_grad(Y, [X, W], perf=(932, 1728, 672))

    W = tvm.placeholder((2, 2, 2, 2), name='W')

    Y = topi.nn.conv2d(X, W, 1, 0, 1)
    check_grad(Y, [X, W], perf=(3922, 2896, 1242))
    Y = topi.nn.conv2d(X, W, 2, 0, 1)
    check_grad(Y, [X, W], perf=(1650, 2800, 1066))
    Y = topi.nn.conv2d(X, W, 3, 0, 1)
    check_grad(Y, [X, W], perf=(1146, 2880, 890))
    Y = topi.nn.conv2d(X, W, 1, 0, 2)
    check_grad(Y, [X, W], perf=(3500, 19720, 1092))
    Y = topi.nn.conv2d(X, W, 2, 0, 2)
    check_grad(Y, [X, W], perf=(3408, 88848, 2034))
    Y = topi.nn.conv2d(X, W, 3, 0, 2)
    check_grad(Y, [X, W], perf=(2254, 65232, 992))
    Y = topi.nn.conv2d(X, W, 1, 0, 3)
    check_grad(Y, [X, W], perf=(3138, 17696, 970))
    Y = topi.nn.conv2d(X, W, 2, 0, 3)
    check_grad(Y, [X, W], perf=(3816, 82368, 2176))
    Y = topi.nn.conv2d(X, W, 3, 0, 3)
    check_grad(Y, [X, W], perf=(3834, 104432, 2306))

    W = tvm.placeholder((2, 2, 3, 3), name='W')

    Y = topi.nn.conv2d(X, W, 1, 0, 1)
    check_grad(Y, [X, W], perf=(7420, 5904, 1752))
    Y = topi.nn.conv2d(X, W, 2, 0, 1)
    check_grad(Y, [X, W], perf=(3888, 58592, 2214))
    Y = topi.nn.conv2d(X, W, 3, 0, 1)
    check_grad(Y, [X, W], perf=(1552, 2268, 1102))
    Y = topi.nn.conv2d(X, W, 1, 0, 2)
    check_grad(Y, [X, W], perf=(5916, 42392, 1256))
    Y = topi.nn.conv2d(X, W, 2, 0, 2)
    check_grad(Y, [X, W], perf=(6736, 21784, 3694))
    Y = topi.nn.conv2d(X, W, 3, 0, 2)
    check_grad(Y, [X, W], perf=(2672, 146096, 1668))
    Y = topi.nn.conv2d(X, W, 1, 0, 3)
    check_grad(Y, [X, W], perf=(2896, 89152, 956))
    Y = topi.nn.conv2d(X, W, 2, 0, 3)
    check_grad(Y, [X, W], perf=(2280, 12856, 1992))
    Y = topi.nn.conv2d(X, W, 3, 0, 3)
    check_grad(Y, [X, W], perf=(2224, 12032, 716))

    Y = topi.nn.pool(X, [1, 1], [1, 1], [0, 0, 0, 0], 'max')
    check_grad(Y, [X], perf=(200, 0, 200))
    Y = topi.nn.pool(X, [1, 1], [2, 2], [0, 0, 0, 0], 'max')
    check_grad(Y, [X], perf=(412, 1124, 412))
    Y = topi.nn.pool(X, [1, 1], [3, 3], [0, 0, 0, 0], 'max')
    check_grad(Y, [X], perf=(232, 1200, 232))
    Y = topi.nn.pool(X, [2, 2], [1, 1], [0, 0, 0, 0], 'max')
    check_grad(Y, [X], perf=(4162, 7200, 1962))
    Y = topi.nn.pool(X, [2, 2], [2, 2], [0, 0, 0, 0], 'max')
    check_grad(Y, [X], perf=(1050, 7800, 650))
    Y = topi.nn.pool(X, [2, 2], [3, 3], [0, 0, 0, 0], 'max')
    check_grad(Y, [X], perf=(858, 6304, 602))
    Y = topi.nn.pool(X, [3, 3], [1, 1], [0, 0, 0, 0], 'max')
    check_grad(Y, [X], perf=(18128, 34200, 3928))
    Y = topi.nn.pool(X, [3, 3], [2, 2], [0, 0, 0, 0], 'max')
    check_grad(Y, [X], perf=(6712, 131312, 1690))
    Y = topi.nn.pool(X, [3, 3], [3, 3], [0, 0, 0, 0], 'max')
    check_grad(Y, [X], perf=(1838, 12950, 704))

def test_some_conv2d_net():
    batch_size = 1
    num_classes = 10

    features = 4
    dense_units = 16

    x = tvm.placeholder((batch_size, 28, 14, 1))
    y = tvm.placeholder((batch_size, num_classes))

    w1 = tvm.placeholder((features, 1, 3, 5))
    b1 = tvm.placeholder((features,))
    w2 = tvm.placeholder((features, features, 3, 5))
    b2 = tvm.placeholder((features,))
    b3 = tvm.placeholder((dense_units,))
    w4 = tvm.placeholder((num_classes, dense_units))
    b4 = tvm.placeholder((num_classes,))

    t = topi.transpose(x, [0, 3, 1, 2])
    t = topi.nn.relu(topi.nn.conv2d(t, w1, 1, 0, 1) + topi.reshape(b1, (1, features, 1, 1)))
    t = topi.nn.relu(topi.nn.conv2d(t, w2, 1, 0, 1) + topi.reshape(b2, (1, features, 1, 1)))
    t = topi.nn.pool(t, [2, 2], [2, 2], [0, 0, 0, 0], 'avg')
    t = topi.transpose(t, [0, 2, 3, 1])
    t = topi.nn.flatten(t)
    w3 = tvm.placeholder((dense_units, get_shape(t)[1]))
    t = topi.nn.relu(topi.nn.dense(t, w3, b3))
    t = topi.nn.dense(t, w4, b4)

    t = - topi.sum(y * topi.nn.log_softmax(t)) / batch_size

    weights = [w1, b1, w2, b2, w3, b3, w4, b4]

    check_grad(t, weights, [x, y], in_range=(-1.0, 1.0), perf=(194865, 179089, 28194))

def test_free_vars():
    m = tvm.var('m')
    n = tvm.var('n')
    A = tvm.placeholder((m, n), name='A')
    B = tvm.placeholder((n,), name='B')

    Y = topi.add(A, B)
    check_grad(Y, [A, B], perf=(160, 0, 120), param_values={m: 5, n: 10})

    param_values = {m: 10}
    x = tvm.var("x", dtype='float32')
    k = tvm.reduce_axis((0, m), name="k")
    A0 = tvm.placeholder((m, m), name='A0')
    A1 = tvm.placeholder((m, m), name='A1')

    B = tvm.compute((m, m), lambda i, j: A0[i, j] + A0[j, i], name='B')
    check_grad(B, A0, perf=(10200, 10000, 300), param_values=param_values)

    B = tvm.compute((m,), lambda i: tvm.sum(A0[i, k]*A0[k, i], axis=k), name='B')
    check_grad(B, A0, perf=(11110, 1000, 1210), param_values=param_values)

    X = tvm.placeholder((m, n, 4, 4), name='X')
    param_values = {m: 1, n: 2}

    R = topi.nn.pool(X, [2, 2], [2, 2], [0, 0, 0, 0], 'avg')
    check_grad(R, X, perf=(72, 224, 72), param_values=param_values)

    R = topi.nn.pool(X, [2, 2], [2, 2], [0, 0, 0, 0], 'max')
    check_grad(R, X, perf=(200, 1248, 136), param_values=param_values)

if __name__ == "__main__":
    if "-v" in sys.argv:
        verbose = True

    test_differentiate_function()
    test_autodiff()
    test_topi_autodiff()
    test_stride_dilation()
    test_some_conv2d_net()
    test_free_vars()
