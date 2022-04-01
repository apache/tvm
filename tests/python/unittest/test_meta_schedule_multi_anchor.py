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
from tvm import relay
from tvm.meta_schedule.tune import Parse, extract_task_from_relay
from tvm.meta_schedule.database import TuningRecord, JSONDatabase
from tvm.meta_schedule.integration import ApplyHistoryBest


def get_dense_dense(data_shape, weight_shape):
    def multi_dense():
        p_data = relay.var("p_data", shape=data_shape, dtype="uint8")
        p_weight1 = relay.var("p_weight1", shape=weight_shape, dtype="int8")
        p_weight2 = relay.var("p_weight2", shape=weight_shape, dtype="int8")

        dense1 = relay.nn.dense(p_data, p_weight1, out_dtype="int32")
        dense2 = relay.nn.dense(relay.cast(dense1, "uint8"), p_weight2, out_dtype="int32")

        f = relay.Function([p_data, p_weight1, p_weight2], dense2)
        f = f.with_attr("Primitive", tvm.tir.IntImm("int32", 1))
        return f

    data = relay.var("data", shape=data_shape, dtype="uint8")
    weight1 = relay.var("weight1", shape=weight_shape, dtype="int8")
    weight2 = relay.var("weight2", shape=weight_shape, dtype="int8")

    out = relay.Call(multi_dense(), [data, weight1, weight2])
    return relay.Function([data, weight1, weight2], out)


def get_ref(data_np, weight1_np, weight2_np):
    dense1 = np.dot(data_np, np.transpose(weight1_np))
    return np.dot(dense1.astype("uint8"), np.transpose(weight2_np))


def test():
    M, N, K = 128, 128, 128
    data_shape = (M, K)
    weight_shape = (N, K)

    relay_mod = tvm.IRModule.from_expr(get_dense_dense(data_shape, weight_shape))

    # print(relay.transform.InferType()(relay_mod))

    target = "llvm"

    data_np = np.random.uniform(1, 10, size=data_shape).astype("uint8")
    weight1_np = np.random.uniform(1, 10, size=weight_shape).astype("int8")
    weight2_np = np.random.uniform(1, 10, size=weight_shape).astype("int8")

    params = {"weight1": weight1_np, "weight2": weight2_np}

    extracted_tasks = extract_task_from_relay(relay_mod, target, params)

    assert len(extracted_tasks) == 1

    task = extracted_tasks[0]

    database = JSONDatabase(
        path_workload="database_workload.json",
        path_tuning_record="database_tuning_record.json",
    )

    mod = Parse._mod(task.dispatched[0])

    workload = database.commit_workload(mod)

    sch = tvm.tir.Schedule(mod)
    print(sch.mod.script())

    tune_rec = TuningRecord(sch.trace, [0.0], workload, tvm.target.Target(target), [])

    database.commit_tuning_record(tune_rec)

    with ApplyHistoryBest(database):
        with tvm.transform.PassContext(
            opt_level=3,
            config={"relay.backend.use_meta_schedule": True},
        ):
            lib = relay.build(relay_mod, target=target, params=params)

    dev = tvm.device(target, 0)

    runtime = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))

    runtime.set_input("data", data_np)
    runtime.run()

    out = runtime.get_output(0).numpy()

    ref = get_ref(data_np, weight1_np, weight2_np)

    np.testing.assert_equal(out, ref)


if __name__ == "__main__":
    test()
