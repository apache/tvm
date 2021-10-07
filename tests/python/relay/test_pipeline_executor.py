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

import numpy as np
import tvm
import tvm.testing
from tvm import relay
from tvm.relay import transform
from tvm.contrib import graph_executor, pipeline_executor

"""
Split graph into a serial of sbgraph.
"""
def pipeline_graph(expr, indices):
    """Split Graph Into A Group Of Subgraph
    Parameters
    ----------
    expr : tvm.relay.Expr
    indices : Array[int]
    Returns
    -------
    ret : Array[tvm.relay.IRModule]
    """

    def run_opt_pass(expr, opt_pass):
        """Exectue a relay pass"""
        assert isinstance(opt_pass, tvm.transform.Pass)
        mod = tvm.IRModule.from_expr(expr)
        mod = tvm.relay.transform.InferType()(mod)
        mod = opt_pass(mod)
        entry = mod["main"]
        return entry if isinstance(expr, tvm.relay.Function) else entry.body

    def _operator_idx_inc(expr, operator_current_idx):
        """Increase operator index"""
        if not isinstance(expr, tvm.relay.expr.Constant):
            operator_current_idx = operator_current_idx + 1

        return operator_current_idx

    def merge_constant_expr(constant_expr, expr):
        # merge constant express with a express
        # Parameters
        # ----------
        # constant_expr:
        #     constant expression
        # expr:
        #     expression to merge with constant expression

        # If body not let, then reached end of the express
        if not isinstance(constant_expr.body, tvm.relay.expr.Let):
            return tvm.relay.expr.Let(constant_expr.var, constant_expr.value, expr)

        return tvm.relay.expr.Let(
            constant_expr.var, constant_expr.value, merge_constant_expr(constant_expr.body, expr)
        )

    def _recursion(anf, operator_indx, pipeline_mods, indices, constant_expr):
        # Enumrate all operator of compute graph then split the compute graph
        # into a group subgraph.
        # Parameters
        # ----------
        # anf:
        #     ANF format expression
        # operator_indx:
        #     current operator indice
        # pipeline_mods:
        #     the subgraph list get storage in this variable
        # indices:
        #     Array of indices use to define the subgraph scope
        # constant_expr:
        #     constant defined before current operator

        # Do the split work
        if isinstance(anf, tvm.relay.Function):
            return tvm.relay.Function(
                anf.params,
                _recursion(anf.body, operator_indx, pipeline_mods, indices, constant_expr),
                anf.ret_type,
                anf.type_params,
                anf.attrs,
            )
        if isinstance(anf, tvm.relay.expr.Let):
            value = anf.value
            operator_indx = _operator_idx_inc(value, operator_indx)

            # record constan expr to make sure all sugraph can find correct
            # constant.
            if isinstance(value, tvm.relay.expr.Constant):
                if not constant_expr:
                    constant_expr = tvm.relay.expr.Let(anf.var, value, anf.var)
                else:
                    constant_expr = tvm.relay.expr.Let(anf.var, value, constant_expr)

            if isinstance(value, tvm.relay.expr.Call):
                if isinstance(value.op, tvm.ir.Op):

                    # if have expr a(b(c(d(e)))) and indexes are [1,2,3]
                    # then would get separate modules for a(b),c,d(e).
                    # the split area is a(b)[0,1] c[2,2] d(e)[2,3]
                    if indices and operator_indx == indices[0]:
                        indices.pop(0)
                        ann = _recursion(
                            anf.body, operator_indx, pipeline_mods, indices, constant_expr
                        )

                        # when current subgraph use previous subgraph constant,
                        # such constant may become free varaible due to the constant
                        # not exist, merge the previous constant with current subgraph
                        # to avoid such issue.
                        if constant_expr:
                            ann = merge_constant_expr(constant_expr, ann)

                        ann = run_opt_pass(ann, transform.ToGraphNormalForm())
                        mod = tvm.IRModule.from_expr(ann)
                        pipeline_mods.insert(0, mod)
                        return tvm.relay.expr.Let(anf.var, value, anf.var)
            return tvm.relay.expr.Let(
                anf.var,
                value,
                _recursion(anf.body, operator_indx, pipeline_mods, indices, constant_expr),
            )
        else:
            return anf

    pipeline_mods = []

    # operator count start from 0, then initial value get set into -1
    operator_indx = -1
    constant_expr = None
    subgraph_indices = indices.copy()
    anf = run_opt_pass(expr, transform.ToANormalForm())
    anf = run_opt_pass(anf, transform.InferType())
    ann = _recursion(anf, operator_indx, pipeline_mods, subgraph_indices, constant_expr)
    ann = run_opt_pass(ann.body, transform.ToGraphNormalForm())
    mod = tvm.IRModule.from_expr(ann)
    pipeline_mods.insert(0, mod)
    return pipeline_mods

def run_modules(mod_configs, dev, target, dname, data, iMod, iName, iData):
    mod_input = {}
    final_output = {}
    indx = 0
    for mod in mod_configs:
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target)

        m = graph_executor.GraphModule(lib["default"](dev))
        # Get input information
        mod_key = indx
        if mod_key in mod_input:
            for input in mod_input[mod_key]:
                input = mod_input[mod_key][input]
                m.set_input(input["index"], input["data"])
        else:
            m.set_input(dname, data)

        # set input for specify module
        if mod == iMod:
            m.set_input(iName, iData)

        m.run()
        n = m.get_num_outputs()
        # parse mod_config and set current output as next mod input data
        mconfig = mod_configs[mod]
        for output in mconfig["pipeline"]["output"]:
            output_data = m.get_output(output["output_indx"]).asnumpy()
            for dep in output["dependent"]:
                # currnet output use as dependent input,
                # input_name indicate the input index number.
                mod_indx = dep["mod_indx"]
                input_name = dep["input_name"]
                if mod_indx == 0:
                    final_output[input_name] = output_data
                else:
                    if mod_indx in mod_input:
                        mod_input[mod_indx][input_name] = {"index": input_name, "data": output_data}
                    else:
                        mod_input[mod_indx] = {
                            input_name: {"index": input_name, "data": output_data}
                        }
        indx = indx + 1

    return final_output

def get_network():
    dshape = (3, 3)
    data = relay.var("data", relay.TensorType(dshape, "float32"))
    data21 = relay.var("data_1", relay.TensorType(dshape, "float32"))
    mvalue1 = np.full((1), 1).astype("float32")
    mvalue2 = np.full((1), 2).astype("float32")
    mvalue3 = np.full((1), 3).astype("float32")
    mv1 = relay.Constant(tvm.nd.array(mvalue1))
    mv2 = relay.Constant(tvm.nd.array(mvalue2))
    mv3 = relay.Constant(tvm.nd.array(mvalue3))
    data = relay.var("data", relay.TensorType(dshape, "float32"))
    net = relay.add(data, mv1)
    net = relay.multiply(net, mv3)

    net = relay.add(net, mv2)
    net = relay.add(net, data21)
    net = relay.add(net, mv3)

    net = relay.multiply(net, mv3)
    net_output2 = relay.subtract(net, mv2)
    net = relay.add(net, net)
    func = relay.Function([data, data21], net)
    mod = tvm.IRModule.from_expr(func)
    return mod, dshape

def get_split_mod():
    mod, dshape = get_network()
    """
    #split compute graph into 4 subgraph
    """
    pl = [2, 5]
    mods = pipeline_graph(mod["main"], pl)
    return mods, dshape

def run_pipeline(target):
    """
    #Get 4 pipeline module.
    """
    mods, dshape = get_split_mod()
    """
    #Prepare batch data for pipeline feeding
    """
    datas = []
    for i in range(len(mods) + 1):
        datas.append(np.full(dshape, 3 + i).astype("float32"))

    # set configure
    indx = 0
    mod_config = {}
    mconfig = {"target_host": None, "mod_name": "default", "build": None, "params": None}
    mconfig1 = mconfig.copy()
    mconfig1["target"] = target[0]
    mconfig1["dev"] = target[1]
    # third output is final output, second output for mod3, first for mod2
    # input
    mconfig1["pipeline"] = {
        "mod_indx": 0,
        "output": [
            {"output_indx": 0, "dependent": [{"mod_indx": 1, "input_name": "x"}]},
        ],
    }
    mod_config[mods[0]] = mconfig1

    mconfig2 = mconfig.copy()
    mconfig2["target"] = "llvm"
    mconfig2["dev"] = tvm.cpu(0)
    mconfig2["pipeline"] = {
        "mod_indx": 1,
        "output": [
            {"output_indx": 0, "dependent": [{"mod_indx": 2, "input_name": "x"}]},
        ],
    }
    mod_config[mods[1]] = mconfig2

    mconfig3 = mconfig.copy()
    mconfig3["target"] = "llvm"
    mconfig3["dev"] = tvm.cpu(0)

    mconfig3["pipeline"] = {
        "mod_indx": 2,
        "output": [{"output_indx": 0, "dependent": [{"mod_indx": -1, "input_name": "1"}]}],
    }
    mod_config[mods[2]] = mconfig3

    """
    #Run with graph executor for verification purpose
    """
    outs = [
        run_modules(mod_config, tvm.cpu(), "llvm", "data", data, mods[1], "data_1", data)
        for data in datas
    ]
    """


    #build and create pipeline module
    """
    with relay.build_config(opt_level=3):
        pipeline_mods, string_config = pipeline_executor.build_pipeline(
            mod_config, "/scratch/hj/data/"
        )

    pipeline_module = pipeline_executor.create(pipeline_mods, string_config)

    """
    #Use pipeline executor to pipeline the said pipeline which use different backend
    """
    d3 = np.full(dshape, 10).astype("float32")
    for data in datas:
        pipeline_module.set_input("data", data)
        pipeline_module.set_input("data_1", data, mod_idx=2)
        pipeline_module.run()

    """
    Get result
    """
    pipeline_outputs = []
    for i in range(len(datas)):
        curOutputs = [output.asnumpy() for output in pipeline_module.get_output()]
        pipeline_outputs.append(curOutputs)

    """
    #Stop pipeline execution.
    """
    pipeline_module.stop()
    """

    #Verify result
    """
    for ref_out, out in zip(outs, pipeline_outputs):
        for ref in ref_out:
            tvm.testing.assert_allclose(ref_out[ref], out[int(ref) - 1])
            print(ref_out[ref])


def test_pipeline():
    if pipeline_executor.pipeline_executor_enabled():
        target_list = tvm.testing.enabled_targets()
        for target in target_list:
            run_pipeline(target)


if __name__ == "__main__":
    test_pipeline()
