"""Unit tests for heterogeneous compilation and execution."""
import json
import numpy as np

import tvm
from tvm import relay
from tvm.contrib import graph_runtime


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
        sub = relay.subtract(add, z)

        func = relay.Function([x, y, z],
                              relay.Tuple(tvm.convert([_add1, _add2,
                                                       sub])))
        func = relay.ir_pass.infer_type(func)
        func = relay.ir_pass.rewrite_annotated_ops(func,
                                                   ctx1.device_type)
        func = relay.ir_pass.infer_type(func)
        return relay.Function(relay.ir_pass.free_vars(func.body[2]),
                              func.body[2])

    def expected():
        add = relay.add(x, y)
        copy_add_sub = relay.device_copy(add, ctx2, ctx1)
        sub = relay.subtract(copy_add_sub, z)
        func = relay.Function([x, y, z], sub)
        return func

    annotated_func = relay.ir_pass.infer_type(annotated())
    expected_func = relay.ir_pass.infer_type(expected())
    assert relay.ir_pass.alpha_equal(annotated_func, expected_func)


def test_annotate_all():
    ctx1 = tvm.context(1)
    ctx2 = tvm.context(2)
    x = relay.var("x", shape=(3,))
    y = relay.var("y", shape=(3,))
    z = relay.var("z", shape=(3,))

    def annotated():
        add = relay.add(x, y)
        _add = relay.annotation.on_device(add, ctx2)
        sub = relay.subtract(add, z)
        _sub = relay.annotation.on_device(sub, ctx2)

        func = relay.Function([x, y, z],
                              relay.Tuple(tvm.convert([_add, _sub,
                                                       sub])))
        func = relay.ir_pass.infer_type(func)
        func = relay.ir_pass.rewrite_annotated_ops(func,
                                                   ctx1.device_type)
        func = relay.ir_pass.infer_type(func)
        return relay.Function(relay.ir_pass.free_vars(func.body[2]),
                              func.body[2])

    def expected():
        add = relay.add(x, y)
        sub = relay.subtract(add, z)
        func = relay.Function([x, y, z], sub)
        return func

    annotated_func = relay.ir_pass.infer_type(annotated())
    expected_func = relay.ir_pass.infer_type(expected())
    assert relay.ir_pass.alpha_equal(annotated_func, expected_func)

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
        func = relay.ir_pass.infer_type(func)
        func = relay.ir_pass.rewrite_annotated_ops(func,
                                                   ctx1.device_type)
        return func

    def expected():
        add = relay.add(x, y)
        sub = relay.subtract(add, z)
        func = relay.Function([x, y, z], sub)
        return func

    annotated_func = relay.ir_pass.infer_type(annotated())
    expected_func = relay.ir_pass.infer_type(expected())
    assert relay.ir_pass.alpha_equal(annotated_func, expected_func)


def check_annotated_graph(annotated_func, expected_func):
    annotated_func = relay.ir_pass.infer_type(annotated_func)
    expected_func = relay.ir_pass.infer_type(expected_func)
    assert relay.ir_pass.alpha_equal(annotated_func, expected_func)


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
        add = relay.add(conv2d_1, conv2d_2)
        _add = relay.annotation.on_device(add, dev1)
        conv2d_3 = relay.nn.conv2d(
            add,
            weight,
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1))
        _conv2d_3 = relay.annotation.on_device(conv2d_3, dev2)

        func = relay.Function([data1, data2, weight],
                              relay.Tuple(tvm.convert([_conv2d_1, _conv2d_2,
                                                       _conv2d_3, _add,
                                                       conv2d_3])))
        func = relay.ir_pass.infer_type(func)
        func = relay.ir_pass.rewrite_annotated_ops(func,
                                                   tvm.context(3).device_type)
        func = relay.ir_pass.infer_type(func)
        return relay.Function(relay.ir_pass.free_vars(func.body[4]),
                              func.body[4])

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

        func = relay.Function([data1, weight, data2], conv2d_3)
        return func

    def check_storage_and_device_types():
        func = annotated()
        func = relay.ir_pass.rewrite_annotated_ops(func, 3)
        func = relay.ir_pass.infer_type(func)
        func = relay.ir_pass.fuse_ops(func, opt_level=2)
        func = relay.ir_pass.infer_type(func)
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

    annotated_func = annotated()
    expected_func = expected()
    check_annotated_graph(annotated_func, expected_func)
    check_storage_and_device_types()


def test_fusible_network():
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
            subtract = relay.subtract(sqrt, log)
            exp = relay.exp(subtract)
            _exp = relay.annotation.on_device(exp, dev_ctx)

            func = relay.Function([x, y],
                                  relay.Tuple(tvm.convert([_sqrt, _exp, exp])))
            func = relay.ir_pass.infer_type(func)
            func = relay.ir_pass.rewrite_annotated_ops(func,
                                                       cpu_ctx.device_type)
            func = relay.ir_pass.infer_type(func)
            return relay.Function(relay.ir_pass.free_vars(func.body[2]),
                                  func.body[2])

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
        expected_index = [1, 1, 1, 2, 2, 1, 1, 2, 2]
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
            sqrt = relay.sqrt(add)
            _sqrt = relay.annotation.on_device(sqrt, dev_ctx)
            log = relay.log(add)
            _log = relay.annotation.on_device(log, dev_ctx)
            subtract = relay.subtract(sqrt, log)
            _subtract = relay.annotation.on_device(subtract, dev_ctx)
            exp = relay.exp(subtract)
            _exp = relay.annotation.on_device(exp, dev_ctx)

            func = relay.Function([x, y],
                                  relay.Tuple(tvm.convert([_add, _sqrt, _log,
                                                           _subtract, _exp,
                                                           exp])))
            func = relay.ir_pass.infer_type(func)
            func = relay.ir_pass.rewrite_annotated_ops(func,
                                                       cpu_ctx.device_type)
            func = relay.ir_pass.infer_type(func)
            return relay.Function(relay.ir_pass.free_vars(func.body[5]),
                                  func.body[5])

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

            func = relay.Function([x, y],
                                  relay.Tuple(tvm.convert([_exp, exp])))
            func = relay.ir_pass.infer_type(func)
            func = relay.ir_pass.rewrite_annotated_ops(func,
                                                       dev_ctx.device_type)
            func = relay.ir_pass.infer_type(func)
            return relay.Function(relay.ir_pass.free_vars(func.body[1]),
                                  func.body[1])

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
        expected_index = [2, 2, 2, 1, 1]
        check_annotated_graph(annotated_func, expected_func)
        test_runtime(target, device, annotated_func, fallback_device,
                     expected_index)

    def test_fallback_all_operators(device, tgt):
        target = {device: tgt}
        annotated_func = get_func()
        expected_func = get_func()
        check_annotated_graph(annotated_func, expected_func)
        test_runtime(target, device, annotated_func)

    for dev, tgt in [("opencl", "opencl"), ("cuda", "cuda"),
                     ("opencl", str(tvm.target.intel_graphics()))]:
        if not tvm.module.enabled(dev):
            print("Skip test because %s is not enabled." % dev)
            continue
        test_fuse_log_add(dev, tgt)
        test_fuse_all(dev, tgt)
        test_fallback_exp(dev, tgt)
        test_fallback_all_operators(dev, tgt)


if __name__ == "__main__":
    test_redundant_annotation()
    test_annotate_all()
    test_annotate_none()
    test_conv_network()
    test_fusible_network()
