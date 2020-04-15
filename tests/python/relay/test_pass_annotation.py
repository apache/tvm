# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Unit tests for heterogeneous compilation and execution."""
import json
import numpy as np

import tvm
from tvm import relay
from tvm.contrib import graph_runtime
from tvm.relay.expr_functor import ExprMutator
from tvm.relay import transform


def run_opt_pass(expr, passes):
    passes = passes if isinstance(passes, list) else [passes]
    mod = tvm.IRModule.from_expr(expr)
    seq = tvm.transform.Sequential(passes)
    with tvm.transform.PassContext(opt_level=3):
        mod = seq(mod)
    return mod["main"]


def test_redundant_annotation():
    ctx1 = tvm.context(1)
    ctx2 = tvm.context(2)
    x = relay.var("x", shape=(3,))
    y = relay.var("y", shape=(3,))
    z = relay.var("z", shape=(3,))

    def annotated():
        add = relay.add(x, y)
        _add1 = relay.annotation.on_device(add, ctx2)
        _add2 = relay.annotation.on_device(add, ctx2)
        sub1 = relay.subtract(_add1, z)
        sub2 = relay.subtract(_add2, z)

        func = relay.Function([x, y, z], relay.Tuple([sub1, sub2]))
        func = run_opt_pass(func,
                            transform.RewriteAnnotatedOps(ctx1.device_type))
        return func

    def expected():
        add = relay.add(x, y)
        copy_add_sub1 = relay.device_copy(add, ctx2, ctx1)
        sub1 = relay.subtract(copy_add_sub1, z)
        copy_add_sub2 = relay.device_copy(add, ctx2, ctx1)
        sub2 = relay.subtract(copy_add_sub2, z)
        func = relay.Function([x, y, z], relay.Tuple([sub1, sub2]))
        return func

    annotated_func = annotated()
    expected_func = run_opt_pass(expected(), transform.InferType())
    assert tvm.ir.structural_equal(annotated_func, expected_func)


def test_annotate_expr():
    ctx1 = tvm.context(1)
    ctx2 = tvm.context(2)
    x = relay.var("x", shape=(3,))
    y = relay.var("y", shape=(3,))
    z = relay.var("z", shape=(3,))

    def annotated():
        add = relay.add(x, y)
        _add = relay.annotation.on_device(add, ctx1)
        sub = relay.subtract(_add, z)
        _sub = relay.annotation.on_device(sub, ctx2)
        expr = run_opt_pass(_sub,
                            transform.RewriteAnnotatedOps(ctx1.device_type))
        return expr

    def expected():
        add = relay.add(x, y)
        copy_add_sub = relay.device_copy(add, ctx1, ctx2)
        sub = relay.subtract(copy_add_sub, z)
        return sub

    annotated_expr = annotated()
    expected_expr = run_opt_pass(expected(), transform.InferType())
    assert tvm.ir.structural_equal(annotated_expr, expected_expr)


def test_annotate_all():
    ctx1 = tvm.context(1)
    ctx2 = tvm.context(2)
    x = relay.var("x", shape=(3,))
    y = relay.var("y", shape=(3,))
    z = relay.var("z", shape=(3,))

    def annotated():
        add = relay.add(x, y)
        _add = relay.annotation.on_device(add, ctx2)
        sub = relay.subtract(_add, z)
        _sub = relay.annotation.on_device(sub, ctx2)

        func = relay.Function([x, y, z], _sub)
        func = run_opt_pass(func,
                            transform.RewriteAnnotatedOps(ctx1.device_type))
        return func

    def expected():
        add = relay.add(x, y)
        sub = relay.subtract(add, z)
        func = relay.Function([x, y, z], sub)
        return func

    annotated_func = annotated()
    expected_func = run_opt_pass(expected(), transform.InferType())
    assert tvm.ir.structural_equal(annotated_func, expected_func)


def test_annotate_none():
    ctx1 = tvm.context(1)
    ctx2 = tvm.context(2)
    x = relay.var("x", shape=(3,))
    y = relay.var("y", shape=(3,))
    z = relay.var("z", shape=(3,))

    def annotated():
        add = relay.add(x, y)
        sub = relay.subtract(add, z)
        func = relay.Function([x, y, z], sub)
        func = run_opt_pass(func,
                            transform.RewriteAnnotatedOps(ctx1.device_type))
        return func

    def expected():
        add = relay.add(x, y)
        sub = relay.subtract(add, z)
        func = relay.Function([x, y, z], sub)
        return func

    annotated_func = annotated()
    expected_func = run_opt_pass(expected(), transform.InferType())
    assert tvm.ir.structural_equal(annotated_func, expected_func)


def check_annotated_graph(annotated_func, expected_func):
    annotated_func = run_opt_pass(annotated_func, transform.InferType())
    expected_func = run_opt_pass(expected_func, transform.InferType())
    assert tvm.ir.structural_equal(annotated_func, expected_func)


def test_conv_network():
    R""" The network is as following:
             data1     data2
               |         |
             conv2d    conv2d
                \       /
                   add
                    |
                  conv2d
    """
    batch_size = 1
    dshape = (batch_size, 64, 56, 56)
    weight = relay.var("weight", shape=(64, 64, 3, 3))
    data1 = relay.var("data1", shape=dshape)
    data2 = relay.var("data2", shape=dshape)
    dev1 = tvm.context(1)
    dev2 = tvm.context(2)

    def original():
        conv2d_1 = relay.nn.conv2d(
            data1,
            weight,
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1))
        conv2d_2 = relay.nn.conv2d(
            data2,
            weight,
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1))
        add = relay.add(conv2d_1, conv2d_2)
        conv2d_3 = relay.nn.conv2d(
            add,
            weight,
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1))

        func = relay.Function([data1, data2, weight], conv2d_3)
        func = run_opt_pass(
            func, transform.RewriteAnnotatedOps(tvm.context(3).device_type))
        return func


    def annotated():
        conv2d_1 = relay.nn.conv2d(
            data1,
            weight,
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1))
        _conv2d_1 = relay.annotation.on_device(conv2d_1, dev2)
        conv2d_2 = relay.nn.conv2d(
            data2,
            weight,
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1))
        _conv2d_2 = relay.annotation.on_device(conv2d_2, dev2)
        add = relay.add(_conv2d_1, _conv2d_2)
        _add = relay.annotation.on_device(add, dev1)
        conv2d_3 = relay.nn.conv2d(
            _add,
            weight,
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1))
        _conv2d_3 = relay.annotation.on_device(conv2d_3, dev2)

        func = relay.Function([data1, data2, weight], _conv2d_3)
        func = run_opt_pass(
            func, transform.RewriteAnnotatedOps(tvm.context(3).device_type))
        return func

    class ScheduleConv2d(ExprMutator):
        def __init__(self, device):
            self.device = device
            super().__init__()

        def visit_call(self, expr):
            visit = super().visit_call(expr)
            if expr.op == tvm.relay.op.get("nn.conv2d"):
                return relay.annotation.on_device(visit, self.device)
            else:
                return visit

    def annotate_with_visitor(func):
        sched = ScheduleConv2d(dev2)
        func = sched.visit(func)
        func = run_opt_pass(
            func, transform.RewriteAnnotatedOps(dev1.device_type))
        return func

    def expected():
        conv2d_1 = relay.nn.conv2d(
            data1,
            weight,
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1))
        device_copy1 = relay.device_copy(conv2d_1, dev2, dev1)
        conv2d_2 = relay.nn.conv2d(
            data2,
            weight,
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1))
        device_copy2 = relay.device_copy(conv2d_2, dev2, dev1)
        add = relay.add(device_copy1, device_copy2)
        device_copy3 = relay.device_copy(add, dev1, dev2)
        conv2d_3 = relay.nn.conv2d(
            device_copy3,
            weight,
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1))

        func = relay.Function([data1, data2, weight], conv2d_3)
        return func

    def check_storage_and_device_types():
        func = annotated()
        func = run_opt_pass(func, [transform.RewriteAnnotatedOps(3),
                                   transform.FuseOps(2)])
        smap = relay.backend._backend.GraphPlanMemory(func)
        storage_ids = []
        device_types = []
        for _, storage_dev_type in smap.items():
            assert len(storage_dev_type) == 2
            for sid in storage_dev_type[0]:
                storage_ids.append(sid.value)
            for did in storage_dev_type[1]:
                device_types.append(did.value)
        assert len(storage_ids) == 10
        assert len(set(storage_ids)) == 8
        assert len(set(device_types)) == 2
        assert set(device_types) == {1, 2}

    def test_manual_annotation():
        annotated_func = annotated()
        expected_func = expected()
        check_annotated_graph(annotated_func, expected_func)
        check_storage_and_device_types()

    def test_visitor_annotation():
        annotated_func = annotate_with_visitor(original())
        expected_func = expected()
        check_annotated_graph(annotated_func, expected_func)

    test_manual_annotation()
    test_visitor_annotation()


def run_fusible_network(dev, tgt):
    R""" The network is as following:
               x     y
                \   /
                 add
                /   \
             sqrt   log
                \   /
              subtract
                  |
                 exp
    """
    x = relay.var("x", shape=(1, 10))
    y = relay.var("y", shape=(10, 10))
    x_data = np.random.rand(1, 10).astype('float32')
    y_data = np.random.rand(10, 10).astype('float32')
    tmp_add = x_data + y_data
    tmp_sqrt = np.sqrt(tmp_add)
    tmp_log = np.log(tmp_add)
    tmp_sub = np.subtract(tmp_sqrt, tmp_log)
    ref_res = np.exp(tmp_sub)

    def get_func():
        add = relay.add(x, y)
        sqrt = relay.sqrt(add)
        log = relay.log(add)
        subtract = relay.subtract(sqrt, log)
        exp = relay.exp(subtract)

        func = relay.Function([x, y], exp)
        return func

    def test_runtime(target, device, func, fallback_device=None,
                     expected_index=None):
        params = {"x": x_data, "y": y_data}
        config = {"opt_level": 1}
        if fallback_device:
            config["fallback_device"] = fallback_device
        with relay.build_config(**config):
            graph, lib, params = relay.build(
                func,
                target,
                params=params)
            contexts = [tvm.cpu(0), tvm.context(device)]
            graph_json = json.loads(graph)
            if "device_index" in graph_json["attrs"]:
                device_index = graph_json["attrs"]["device_index"][1]
                assert device_index == expected_index
            mod = graph_runtime.create(graph, lib, contexts)
            mod.set_input(**params)
            mod.run()
            res = mod.get_output(0).asnumpy()
            tvm.testing.assert_allclose(res, ref_res, rtol=1e-5, atol=1e-5)

    def test_fuse_log_add(device, tgt):
        """ Only log and add are fused."""
        fallback_device = tvm.context("cpu")
        target = {"cpu": "llvm", device: tgt}
        cpu_ctx = fallback_device
        dev_ctx = tvm.context(device)

        def annotated():
            add = relay.add(x, y)
            sqrt = relay.sqrt(add)
            _sqrt = relay.annotation.on_device(sqrt, dev_ctx)
            log = relay.log(add)
            subtract = relay.subtract(_sqrt, log)
            exp = relay.exp(subtract)
            _exp = relay.annotation.on_device(exp, dev_ctx)

            func = relay.Function([x, y], _exp)
            func = run_opt_pass(
                func, transform.RewriteAnnotatedOps(cpu_ctx.device_type))
            return func

        def expected():
            add = relay.add(x, y)
            copy_add_sqrt = relay.device_copy(add, cpu_ctx, dev_ctx)
            sqrt = relay.sqrt(copy_add_sqrt)
            log = relay.log(add)
            copy_sqrt_subtract = relay.device_copy(sqrt, dev_ctx, cpu_ctx)
            subtract = relay.subtract(copy_sqrt_subtract, log)
            copy_sub_exp = relay.device_copy(subtract, cpu_ctx, dev_ctx)
            exp = relay.exp(copy_sub_exp)

            func = relay.Function([x, y], exp)
            return func

        annotated_func = annotated()
        expected_func = expected()
        ctx = tvm.context(device, 0)
        dev_idx = ctx.device_type
        expected_index = [1, 1, 1, dev_idx, dev_idx, 1, 1, dev_idx, dev_idx]
        check_annotated_graph(annotated_func, expected_func)
        test_runtime(target, device, annotated_func, fallback_device,
                     expected_index)

    def test_fuse_all(device, tgt):
        """Fuse all operators."""
        fallback_device = tvm.context("cpu")
        target = {"cpu": "llvm", device: tgt}
        cpu_ctx = fallback_device
        dev_ctx = tvm.context(device)

        def annotated():
            add = relay.add(x, y)
            _add = relay.annotation.on_device(add, dev_ctx)
            sqrt = relay.sqrt(_add)
            _sqrt = relay.annotation.on_device(sqrt, dev_ctx)
            log = relay.log(_add)
            _log = relay.annotation.on_device(log, dev_ctx)
            subtract = relay.subtract(_sqrt, _log)
            _subtract = relay.annotation.on_device(subtract, dev_ctx)
            exp = relay.exp(_subtract)
            _exp = relay.annotation.on_device(exp, dev_ctx)

            func = relay.Function([x, y], _exp)
            func = run_opt_pass(
                func, transform.RewriteAnnotatedOps(cpu_ctx.device_type))
            return func

        annotated_func = annotated()
        expected_func = get_func()
        check_annotated_graph(annotated_func, expected_func)
        test_runtime(target, device, annotated_func, fallback_device)

    def test_fallback_exp(device, tgt):
        fallback_device = tvm.context("cpu")
        target = {"cpu": "llvm", device: tgt}
        cpu_ctx = fallback_device
        dev_ctx = tvm.context(device)

        def annotated():
            add = relay.add(x, y)
            sqrt = relay.sqrt(add)
            log = relay.log(add)
            subtract = relay.subtract(sqrt, log)
            exp = relay.exp(subtract)
            _exp = relay.annotation.on_device(exp, cpu_ctx)

            func = relay.Function([x, y], _exp)
            func = run_opt_pass(
                func, transform.RewriteAnnotatedOps(dev_ctx.device_type))
            return func

        def expected():
            add = relay.add(x, y)
            sqrt = relay.sqrt(add)
            log = relay.log(add)
            subtract = relay.subtract(sqrt, log)
            copy_sub_exp = relay.device_copy(subtract, dev_ctx, cpu_ctx)
            exp = relay.exp(copy_sub_exp)

            func = relay.Function([x, y], exp)
            return func

        annotated_func = annotated()
        expected_func = expected()
        ctx = tvm.context(device, 0)
        dev_idx = ctx.device_type
        expected_index = [dev_idx, dev_idx, dev_idx, 1, 1]
        check_annotated_graph(annotated_func, expected_func)
        test_runtime(target, device, annotated_func, fallback_device,
                     expected_index)

    def test_fallback_all_operators(device, tgt):
        target = {device: tgt, "cpu": "llvm"}
        annotated_func = get_func()
        expected_func = get_func()
        check_annotated_graph(annotated_func, expected_func)
        test_runtime(target, device, annotated_func)


    test_fuse_log_add(dev, tgt)
    test_fuse_all(dev, tgt)
    test_fallback_exp(dev, tgt)
    test_fallback_all_operators(dev, tgt)

def run_unpropagatable_graph(dev, tgt):
    R""" The network is as following:
            a     b  c     d
             \   /    \   /
              add      mul
                \      /
                subtract
    """

    a = relay.var("a", shape=(10, 10))
    b = relay.var("b", shape=(10, 10))
    c = relay.var("c", shape=(10, 10))
    d = relay.var("d", shape=(10, 10))
    a_data = np.random.rand(10, 10).astype('float32')
    b_data = np.random.rand(10, 10).astype('float32')
    c_data = np.random.rand(10, 10).astype('float32')
    d_data = np.random.rand(10, 10).astype('float32')
    tmp_add = a_data + b_data
    tmp_mul = np.multiply(c_data, d_data)
    ref_res = np.subtract(tmp_add, tmp_mul)

    fallback_device = tvm.context("cpu")
    target = {"cpu": "llvm", dev: tgt}
    cpu_ctx = fallback_device
    dev_ctx = tvm.context(dev)

    def annotated():
        add = relay.add(a, b)
        _add = relay.annotation.on_device(add, dev_ctx)
        mul = relay.multiply(c, d)
        _mul = relay.annotation.on_device(mul, cpu_ctx)
        sub = relay.subtract(_add, _mul)
        _sub = relay.annotation.on_device(sub, dev_ctx)
        func = relay.Function([a, b, c, d], _sub)
        func = run_opt_pass(
            func, transform.RewriteAnnotatedOps(dev_ctx.device_type))
        return func

    def expected():
        add = relay.add(a, b)
        mul = relay.multiply(c, d)
        copy_mul_sub = relay.device_copy(mul, cpu_ctx, dev_ctx)
        sub = relay.subtract(add, copy_mul_sub)
        func = relay.Function([a, b, c, d], sub)
        return func

    annotated_func = annotated()
    expected_func = expected()
    expected_index = [2, 2, 2, 1, 1, 1, 2, 2]
    check_annotated_graph(annotated_func, expected_func)
    params = {"a": a_data, "b": b_data, "c": c_data, "d": d_data}
    config = {"opt_level": 0}
    config["fallback_device"] = fallback_device
    with relay.build_config(**config):
        graph, lib, params = relay.build(annotated_func, target, params=params)
        contexts = [tvm.cpu(0), tvm.context(dev)]
        graph_json = json.loads(graph)
        if "device_index" in graph_json["attrs"]:
            device_index = graph_json["attrs"]["device_index"][1]
            assert device_index == expected_index
        mod = graph_runtime.create(graph, lib, contexts)
        mod.set_input(**params)
        mod.run()
        res = mod.get_output(0).asnumpy()
        tvm.testing.assert_allclose(res, ref_res, rtol=1e-5, atol=1e-5)


def test_check_run():
    for dev, tgt in [("opencl", "opencl"), ("cuda", "cuda"),
                 ("opencl", str(tvm.target.intel_graphics()))]:
        if not tvm.runtime.enabled(dev):
            print("Skip test because %s is not enabled." % dev)
            continue
        run_fusible_network(dev, tgt)
        run_unpropagatable_graph(dev, tgt)


def test_tuple_get_item():
    dev = "cuda"
    if not tvm.runtime.enabled(dev):
        print("Skip test because %s is not enabled." % dev)
        return

    cpu_ctx = tvm.cpu(0)
    gpu_ctx = tvm.context(dev)

    def expected():
        x = relay.var("x", relay.ty.TensorType((3, 3, 4), "float32"))
        split = relay.op.split(x, 3)
        elem0 = relay.device_copy(split[0], gpu_ctx, cpu_ctx)
        elem1 = relay.device_copy(split[1], gpu_ctx, cpu_ctx)
        sub = elem0 - elem1
        func = relay.Function(relay.analysis.free_vars(sub), sub)
        return func

    def annotated():
        x = relay.var("x", relay.ty.TensorType((3, 3, 4), "float32"))
        split = relay.op.split(x, 3)
        split = split.astuple()
        split = relay.annotation.on_device(split, gpu_ctx)
        split = relay.TupleWrapper(split, 3)
        sub = split[0] - split[1]
        func = relay.Function(relay.analysis.free_vars(sub), sub)
        func = run_opt_pass(
            func, transform.RewriteAnnotatedOps(cpu_ctx.device_type))
        return func

    annotated_func = annotated()
    expected_func = run_opt_pass(expected(), transform.InferType())
    assert tvm.ir.structural_equal(annotated_func, expected_func)


if __name__ == "__main__":
    test_redundant_annotation()
    test_annotate_expr()
    test_annotate_all()
    test_annotate_none()
    test_conv_network()
    test_check_run()
    test_tuple_get_item()
