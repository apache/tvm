#!/usr/bin/env python

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
"""Test script for tf op module"""
import tempfile
import os
import logging
import torch
import numpy as np
import tvm
import tvm.testing
from tvm import te, relay
from tvm.contrib import pt_op
from tvm.contrib import graph_runtime


def test_use_pt_graph_module():
    """main test function"""

    def build_export_graph(device):
        """relay build & export graph"""
        x = relay.var("x", shape=(10, 5))
        y = relay.var("y", shape=(1, 5))
        z = relay.add(x, y)
        z = relay.exp(z)
        func = relay.Function([x, y], z)
        x_data = np.random.rand(10, 5).astype("float32")
        y_data = np.random.rand(1, 5).astype("float32")
        params = {"y": y_data}

        pt_device = torch.device(device)
        if pt_device.type == 'cuda':
            target = 'cuda'
            ctx = tvm.gpu(pt_device.index)
        else:
            target = 'llvm'
            ctx = tvm.cpu(0)

        graph, lib, params = relay.build(tvm.IRModule.from_expr(func), target=target, params=params)
        mod = graph_runtime.create(graph, lib, ctx=ctx)
        mod.set_input(**params)
        mod.set_input(x=x_data)
        mod.run()
        res = mod.get_output(0).asnumpy()
        ref_res = np.exp(y_data + x_data)
        tvm.testing.assert_allclose(res, ref_res, atol=1e-5, rtol=1e-5)

        # export to tempdir
        tvm_assets = ["mod.so", "graph.json", "params"]
        export_dir = tempfile.mkdtemp("tvm_export")
        lib.export_library(os.path.join(export_dir, tvm_assets[0]))
        with open(os.path.join(export_dir, tvm_assets[1]), 'w') as fout:
            fout.write(graph)
        with open(os.path.join(export_dir, tvm_assets[2]), 'wb') as fout:
            fout.write(relay.save_param_dict(params))

        return export_dir

    def test_pt_run(device, trace=True, to_device=None):
        """test add lib with Pytorch wrapper"""
        print('############## Test on device:', device, '#################')
        export_dir = build_export_graph(device)
        engine = pt_op.module.GraphModule(num_inputs=2, num_outputs=1).to(device)

        x = np.random.rand(10, 5).astype("float32")
        y = np.random.rand(1, 5).astype("float32")

        expect = np.exp(y + x)

        tvm_assets = ["mod.so", "graph.json", "params"]
        assets = [os.path.join(export_dir, i) for i in tvm_assets]
        engine.init((x.shape, y.shape), *assets)
        outputs = engine.forward([torch.Tensor(x), torch.Tensor(y)])
        tvm.testing.assert_allclose(outputs[0].cpu(), expect, atol=1e-5, rtol=1e-5)

        if trace:
            print('################ Test trace and load #################')
            scripted = torch.jit.script(engine)
            scripted_dir = tempfile.mkdtemp("scripted")
            scripted_path = os.path.join(scripted_dir, 'model.pt')
            scripted.save(scripted_path)
            loaded = torch.jit.load(scripted_path)
            outputs = loaded.forward([torch.Tensor(x), torch.Tensor(y)])
            tvm.testing.assert_allclose(outputs[0].cpu(), expect, atol=1e-5, rtol=1e-5)
            del scripted
            del loaded

        if to_device:
            print('################ Test move from [{}] to [{}] #################'.format(device, to_device))
            engine = engine.to(to_device)
            outputs = engine.forward([torch.Tensor(x), torch.Tensor(y)])
            tvm.testing.assert_allclose(outputs[0].cpu(), expect, atol=1e-5, rtol=1e-5)
        del engine

    test_pt_run(device='cuda:0', trace=True, to_device='cuda:1')
    test_pt_run(device='cpu', trace=True)


if __name__ == "__main__":
    test_use_pt_graph_module()
