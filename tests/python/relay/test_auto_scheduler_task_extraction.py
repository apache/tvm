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
"""Test task extraction for auto-scheduler"""
import pytest

import tvm.relay.testing
import tvm.testing
from tvm import auto_scheduler, relay


def get_network(name, batch_size=1, layout="NHWC"):
    """Get the symbol definition and random weight of a network"""

    # auto-scheduler prefer NHWC layout
    if layout == "NHWC":
        image_shape = (224, 224, 3)
    elif layout == "NCHW":
        image_shape = (3, 224, 224)
    elif layout == "NCDHW":
        image_shape = (3, 16, 224, 224)
    elif layout == "NDHWC":
        image_shape = (3, 224, 224, 16)
    else:
        raise ValueError("Invalid layout: " + layout)

    if name == "resnet-18":
        mod, params = relay.testing.resnet.get_workload(
            num_layers=18, batch_size=batch_size, layout=layout, image_shape=image_shape
        )
    elif name == "resnet-50":
        mod, params = relay.testing.resnet.get_workload(
            num_layers=50, batch_size=batch_size, layout=layout, image_shape=image_shape
        )
    elif name == "winograd-test":
        input_shape = [1, 23, 40, 32]

        data = relay.var("data", shape=input_shape, dtype="float32")
        net = relay.testing.layers.conv2d(
            data=data,
            channels=128,
            kernel_size=3,
            strides=1,
            padding=1,
            data_layout="NHWC",
            kernel_layout="HWIO",
            name="",
        )
        bias = relay.var("conv1_bias")
        net = relay.nn.bias_add(net, bias, 3)
        net = relay.nn.relu(net)
        mod, params = relay.testing.create_workload(net)
    elif name == "resnet3d-18":
        mod, params = relay.testing.resnet_3d.get_workload(
            num_layers=18, batch_size=batch_size, layout=layout, image_shape=image_shape
        )
    elif name == "mobilenet":
        mod, params = relay.testing.mobilenet.get_workload(
            batch_size=batch_size, layout=layout, image_shape=image_shape
        )
    elif name == "resnet3d-18":
        mod, params = relay.testing.resnet_3d.get_workload(
            num_layers=18, batch_size=batch_size, layout=layout, image_shape=image_shape
        )
    elif name == "dcgan":
        mod, params = relay.testing.dcgan.get_workload(batch_size=batch_size, layout=layout)
    elif name == "mlp":
        data = relay.var("data", shape=(batch_size, 32))
        fc1 = relay.nn.dense(data, relay.var("fc1_weight"), units=32)
        fc1 = relay.nn.bias_add(fc1, relay.var("fc1_bias"), axis=-1)
        act1 = relay.nn.relu(fc1)
        fc2 = relay.nn.dense(act1, relay.var("fc2_weight"), units=32)
        fc2 = relay.nn.bias_add(fc2, relay.var("fc2_bias"), axis=-1)
        act2 = relay.nn.relu(fc2)
        mlp = act2
        args = relay.analysis.free_vars(act2)
        mlp = relay.Function(args, mlp)
        mod, params = relay.testing.init.create_workload(mlp)
    else:
        raise ValueError("Unsupported network: " + name)

    return mod, params


@tvm.testing.requires_cuda
def test_task_extraction_cuda():
    target = tvm.target.Target("cuda")

    mod, params = get_network("mlp")
    tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)

    assert len(tasks) == 1
    assert sum(task_weights) == 2

    for layout in ["NHWC", "NCHW"]:
        mod, params = get_network("resnet-18", layout=layout)
        tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)

        assert len(tasks) == 24
        assert sum(task_weights) == 25

        mod, params = get_network("mobilenet", layout=layout)
        tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)

        assert len(tasks) == 22
        assert sum(task_weights) == 30

    for layout in ["NCDHW", "NDHWC"]:
        mod, params = get_network("resnet3d-18", layout=layout)
        tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)

        assert len(tasks) == 23
        assert sum(task_weights) == 24, sum(task_weights)


def test_task_extraction():
    ishape = (1, 3, 224, 224)
    w1shape = (32, 3, 3, 3)
    w2shape = (32, 32, 3, 3)
    dtype = "float32"
    target = tvm.target.Target("llvm")

    def verify_task_extraction(func, expected_task, include_simple_tasks=False):
        mod = tvm.IRModule.from_expr(func)
        tasks, task_weights = auto_scheduler.extract_tasks(
            mod["main"], None, target, include_simple_tasks=include_simple_tasks
        )

        assert len(tasks) == expected_task
        assert len(task_weights) == expected_task

    def get_func():
        data = relay.var("data", shape=(ishape), dtype=dtype)
        weight1 = relay.var("weight1", shape=(w1shape), dtype=dtype)
        weight2 = relay.var("weight2", shape=(w2shape), dtype=dtype)

        conv2d = relay.nn.conv2d(data, weight1, kernel_size=(3, 3), padding=(1, 1))
        relu = relay.nn.relu(conv2d)
        conv2d = relay.nn.conv2d(relu, weight2, kernel_size=(3, 3), padding=(1, 1))
        out = relay.nn.relu(conv2d)
        return relay.Function([data, weight1, weight2], out)

    def get_fused_func():
        data = relay.var("data", shape=(ishape), dtype=dtype)
        weight1 = relay.var("weight1", shape=(w1shape), dtype=dtype)
        weight2 = relay.var("weight2", shape=(w2shape), dtype=dtype)

        fused_func = get_func()

        # Set to primitive to keep fuse_ops untouch.
        fused_func = fused_func.with_attr("Primitive", tvm.tir.IntImm("int32", 1))

        call = relay.Call(fused_func, [data, weight1, weight2])
        return relay.Function([data, weight1, weight2], call)

    def get_simple_func():
        data = relay.var("data", relay.TensorType((1, 2, 3), "float32"))
        out = relay.image.affine_grid(data, (150, 150))
        return relay.Function([data], out)

    def get_shape_of_func():
        data = relay.var("data", shape=(relay.Any(), 28, 28), dtype="float32")
        out = relay.shape_of(data)
        return relay.Function([data], out)

    def get_func_with_dynamic_shape():
        data = relay.var("data", shape=(relay.Any(), 32), dtype="float32")
        out = relay.max(data)
        return relay.Function(relay.analysis.free_vars(out), out)

    def get_func_with_control_flow():
        data = relay.var("data", shape=(1, 3, 224, 224))
        weight = relay.var("weight", shape=(32, 3, 3, 3))
        eq1 = relay.var("e1", shape=[], dtype="float32")
        eq2 = relay.var("e2", shape=[], dtype="float32")
        eq = relay.equal(eq1, eq2)

        true_branch = relay.zeros(shape=(1, 32, 222, 222), dtype="float32")
        false_branch = relay.nn.conv2d(data, weight, kernel_size=(3, 3), channels=32)
        ife = relay.If(eq, true_branch, false_branch)
        out = relay.erf(ife)
        return relay.Function([data, weight, eq1, eq2], out)

    def get_func_with_unsupported_op():
        def get_postproc_func():
            data = relay.var("data", shape=((1, 3, 6)), dtype=dtype)
            out = relay.nn.relu(data)
            func = relay.Function([data], out)
            func = func.with_attr("Primitive", tvm.tir.IntImm("int32", 1))
            return func

        cls_prob = relay.var("cls_prob", relay.ty.TensorType((1, 3, 3), "float32"))
        loc_pred = relay.var("loc_pred", relay.ty.TensorType((1, 3 * 4), "float32"))
        anchors = relay.var("anchors", relay.ty.TensorType((1, 3, 4), "float32"))

        mtl = relay.vision.multibox_transform_loc(
            cls_prob=cls_prob, loc_pred=loc_pred, anchor=anchors
        )
        nms = relay.vision.non_max_suppression(mtl[0], mtl[1], mtl[0], return_indices=False)
        out = relay.Call(get_postproc_func(), [nms])
        return relay.Function([cls_prob, loc_pred, anchors], out)

    # Relay FuseOps puts two conv2ds to separate functions and results in two tasks.
    verify_task_extraction(get_func(), 2)

    # By setting the function to primitive, Relay FuseOps will not break it and result in one task.
    verify_task_extraction(get_fused_func(), 1)

    # The Relay function without complex ops will not form a task by default.
    verify_task_extraction(get_simple_func(), 0)

    # Every Relay function becomes a task regardless what ops in its body.
    verify_task_extraction(get_simple_func(), 1, True)

    # The Relay function without any reduce op is considered as a simple task.
    verify_task_extraction(get_shape_of_func(), 0)
    verify_task_extraction(get_shape_of_func(), 1, True)

    # The Relay function with dynamic shape inputs/outputs will not be extracted.
    verify_task_extraction(get_func_with_dynamic_shape(), 0)

    # The Conv2D in the Relay function with control flow could still be a task.
    verify_task_extraction(get_func_with_control_flow(), 1)

    # Func1 (with NMS) -> Func2 (injective).
    verify_task_extraction(get_func_with_unsupported_op(), 1, True)


if __name__ == "__main__":
    test_task_extraction_cuda()
    test_task_extraction()
