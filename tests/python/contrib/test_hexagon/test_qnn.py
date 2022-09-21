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
from tvm.contrib.hexagon.session import Session
from tvm.meta_schedule import postproc, schedule_rule
from tvm.tir.tensor_intrin.hexagon import VRMPY_u8i8i32_INTRIN
from tvm.contrib.hexagon.meta_schedule import get_hexagon_local_builder, get_hexagon_rpc_runner
from tvm import meta_schedule as ms


executor = relay.backend.Executor("graph", {"link-params": True})
target_hexagon = tvm.target.hexagon("v68")
target_llvm = tvm.target.Target("llvm")


@tvm.testing.requires_hexagon
def test_resnet50(hexagon_session: Session):
    with open("qresnet50.json", "r") as fi:
        mod = tvm.ir.load_json(fi.read())

    with open("qresnet50.params", "rb") as fi:
        params = relay.load_param_dict(fi.read())

    # print(relay.transform.InferType()(mod))
    # return

    inp = np.random.randn(1, 3, 224, 224).astype("float32")
    input_name = "image"

    with tvm.transform.PassContext(
        opt_level=3,
    ):
        # opt_mod, _ = relay.optimize(
        #     mod,
        #     tvm.target.Target(target_hexagon, host=target_hexagon),
        #     params=params,
        # )

        # print(opt_mod)

        # return

        hexagon_lowered = relay.build(
            mod,
            tvm.target.Target(target_hexagon, host=target_hexagon),
            params=params,
            executor=executor,
        )

    with tvm.transform.PassContext(opt_level=3):
        llvm_lowered = tvm.relay.build(
            mod,
            tvm.target.Target(target_llvm, host=target_llvm),
            params=params,
        )

    # assert "vrmpy" in hexagon_lowered.lib.get_source("asm")
    # print(hexagon_lowered.lib.get_source("asm"))

    # debug_ex = hexagon_session.get_graph_debug_executor(hexagon_lowered.get_graph_json(), hexagon_lowered.lib)
    # print(debug_ex.profile(input_name=inp))

    # return

    graph_mod = hexagon_session.get_executor_from_factory(hexagon_lowered)
    graph_mod.set_input(input_name, inp.copy())

    llvm_graph_mod = tvm.contrib.graph_executor.GraphModule(llvm_lowered["default"](tvm.cpu(0)))
    llvm_graph_mod.set_input(input_name, inp.copy())

    import time

    t0 = time.time()
    graph_mod.run()
    hexagon_output = graph_mod.get_output(0).numpy()
    print("run finished in ", time.time() - t0)

    llvm_graph_mod.run()
    ref_result = llvm_graph_mod.get_output(0).numpy()
    print(np.max(np.abs(ref_result - hexagon_output)), np.mean(np.abs(ref_result - hexagon_output)))

    time_ms = graph_mod.benchmark(hexagon_session.device, number=1, repeat=20).mean * 1e3

    print("time elapsed: ", time_ms)

    debug_ex = hexagon_session.get_graph_debug_executor(hexagon_lowered.get_graph_json(), hexagon_lowered.lib)
    print(debug_ex.profile(input_name=inp.copy()))


def tune_ms(mod, params, hexagon_launcher):
    sch_rules = [
        schedule_rule.AutoInline(
            into_producer=False,
            into_consumer=True,
            inline_const_tensor=True,
            disallow_if_then_else=True,
            require_injective=True,
            require_ordered=True,
            disallow_op=["tir.exp"],
        ),
        schedule_rule.MultiLevelTilingWithIntrin(
            VRMPY_u8i8i32_INTRIN,
            structure="SRSRS",
            tile_binds=None,
            max_innermost_factor=64,
            vector_load_lens=None,
            reuse_read=None,
            reuse_write=schedule_rule.ReuseType(
                req="may",
                levels=[1, 2],
                scope="global",
            ),
        ),
        schedule_rule.ParallelizeVectorizeUnroll(
            max_jobs_per_core=16,
            max_vectorize_extent=128,
            unroll_max_steps=[0, 16, 64, 512],
            unroll_explicit=True,
        ),
    ]

    postprocs = [
        postproc.DisallowDynamicLoop(),
        postproc.RewriteParallelVectorizeUnroll(),
        postproc.RewriteReductionBlock(),
        postproc.RewriteTensorize(vectorize_init_loop=True),
    ]

    work_dir = "work_auto_tensorize"
    config = ms.TuneConfig(
        strategy="replay_trace",
        # strategy="evolutionary",
        num_trials_per_iter=8,
        max_trials_per_task=8,
        max_trials_global=20000,
    )

    target = tvm.target.Target(target_hexagon, host=target_hexagon)

    if False:
        return ms.tune_relay(
            mod=mod,
            params=params,
            target=target,
            config=config,
            work_dir=work_dir,
            builder=get_hexagon_local_builder(),
            runner=get_hexagon_rpc_runner(hexagon_launcher, number=20),
            executor=executor,
            sch_rules=lambda: sch_rules,
            postprocs=lambda: postprocs,
        )
    else:
        pass_config = {"relay.FuseOps.link_params": True,
                       "relay.backend.use_meta_schedule": True,
                       "relay.backend.tir_converter": "default"
                       }

        from tvm.meta_schedule.tune import tune_extracted_tasks
        from tvm.meta_schedule.relay_integration import extract_task_from_relay

        extracted_tasks = extract_task_from_relay(mod, target, params, pass_config=pass_config)

        tune_tasks = []

        for task in extracted_tasks:
            # if "conv2d" in task.task_name:
            if True:
                tune_tasks.append(task)

        database = tune_extracted_tasks(
            tune_tasks,
            config,
            work_dir,
            builder=get_hexagon_local_builder(),
            runner=get_hexagon_rpc_runner(hexagon_launcher, number=20),
            num_threads=32,
            sch_rules=lambda: sch_rules,
            postprocs=lambda: postprocs,
        )

        with target, database:
            with tvm.transform.PassContext(
                opt_level=3,
                config={
                    "relay.backend.use_meta_schedule": True,
                    "relay.backend.use_meta_schedule_dispatch": target.kind.name != "cuda",
                    "relay.backend.tir_converter": "default",
                },
            ):
                return relay.build(mod, target=target, params=params, executor=executor)


@tvm.testing.requires_hexagon
def test_resnet50_auto_tensorize(hexagon_launcher):
    with open("qresnet50.json", "r") as fi:
        mod = tvm.ir.load_json(fi.read())

    with open("qresnet50.params", "rb") as fi:
        params = relay.load_param_dict(fi.read())

    inp = np.random.randn(1, 3, 224, 224).astype("float32")
    input_name = "image"

    hexagon_lowered = tune_ms(mod, params, hexagon_launcher)

    with tvm.transform.PassContext(opt_level=3):
        llvm_lowered = tvm.relay.build(
            mod,
            tvm.target.Target(target_llvm, host=target_llvm),
            params=params,
        )

    with hexagon_launcher.start_session() as session:
        graph_mod = session.get_executor_from_factory(hexagon_lowered)
        graph_mod.set_input(input_name, inp.copy())

        llvm_graph_mod = tvm.contrib.graph_executor.GraphModule(llvm_lowered["default"](tvm.cpu(0)))
        llvm_graph_mod.set_input(input_name, inp.copy())

        import time

        t0 = time.time()
        graph_mod.run()
        hexagon_output = graph_mod.get_output(0).numpy()
        print("run finished in ", time.time() - t0)

        llvm_graph_mod.run()
        ref_result = llvm_graph_mod.get_output(0).numpy()
        print(np.max(np.abs(ref_result - hexagon_output)), np.mean(np.abs(ref_result - hexagon_output)))

        time_ms = graph_mod.benchmark(session.device, number=1, repeat=20).mean * 1e3

        print("time elapsed: ", time_ms)

        debug_ex = session.get_graph_debug_executor(hexagon_lowered.get_graph_json(), hexagon_lowered.lib)
        print(debug_ex.profile(input_name=inp.copy()))


@tvm.testing.requires_hexagon
def test_qnn_conv2d(hexagon_launcher):
    with open("qnn_conv2d.json", "r") as fi:
        mod = tvm.ir.load_json(fi.read())

    with open("qnn_conv2d.params", "rb") as fi:
        params = relay.load_param_dict(fi.read())

    if True:
        hexagon_lowered = tune_ms(mod, params ,hexagon_launcher)
    else:
        with tvm.transform.PassContext(
            opt_level=3,
        ):
            hexagon_lowered = relay.build(
                mod,
                tvm.target.Target(target_hexagon, host=target_hexagon),
                params=params,
                executor=executor,
            )

    inp = np.load("qconv2d_input.npy")
    input_name = "input"

    with hexagon_launcher.start_session() as session:
        graph_mod = session.get_executor_from_factory(hexagon_lowered)
        graph_mod.set_input(input_name, inp.copy())
        # graph_mod.set_input(**params)

        import time

        t0 = time.time()
        graph_mod.run()
        hexagon_output = graph_mod.get_output(0).numpy()
        print("run finished in ", time.time() - t0)

        pt_result = np.load("qconv2d_output.npy")
        print(np.max(np.abs(pt_result - hexagon_output)), np.mean(np.abs(pt_result - hexagon_output)))

        # time_ms = graph_mod.benchmark(hexagon_session.device, number=1, repeat=20).mean * 1e3

        # print("time elapsed: ", time_ms)


@tvm.testing.requires_hexagon
def test_qconv2d_subgraph(hexagon_session: Session):
    mod = tvm.parser.fromtext(
"""
#[version = "0.0.5"]
def @main(%p070: Tensor[(1, 2, 56, 56, 32), uint8], %p150: Tensor[(8, 2, 1, 1, 8, 32, 4), int8], %p250: Tensor[(1, 8, 1, 1, 32), int32] , %p350: Tensor[(1, 8, 1, 1, 32), int64], %p450: Tensor[(1, 8, 1, 1, 32), int64], %p550: Tensor[(1, 8, 1, 1, 32), int64], %p617: Tensor[(1), int32] ) -> Tensor[(1, 8, 56, 56, 32), int32] {
    %546 = nn.contrib_conv2d_NCHWc(%p070, %p150, padding=[0, 0, 0, 0], channels=256, kernel_size=[1, 1], data_layout="NCHW32c", kernel_layout="OIHW8i32o4i", out_layout="NCHW32c", out_dtype="int32") /* ty=Tensor[(1, 8, 56, 56, 32), int32] */;
    %547 = add(%546, %p250) /* ty=Tensor[(1, 8, 56, 56, 32), int32] */;
    %548 = cast(%547, dtype="int64") /* ty=Tensor[(1, 8, 56, 56, 32), int64] */;
    %549 = multiply(%548, %p350) /* ty=Tensor[(1, 8, 56, 56, 32), int64] */;
    %550 = add(%549, %p450) /* ty=Tensor[(1, 8, 56, 56, 32), int64] */;
    %551 = right_shift(%550, %p550) /* ty=Tensor[(1, 8, 56, 56, 32), int64] */;
    %552 = cast(%551, dtype="int32") /* ty=Tensor[(1, 8, 56, 56, 32), int32] */;
    %553 = add(77 /* ty=int32 */, %552) /* ty=Tensor[(1, 8, 56, 56, 32), int32] */;
    %554 = clip(%553, a_min=0f, a_max=255f) /* ty=Tensor[(1, 8, 56, 56, 32), int32] */;
    %555 = subtract(%554, %p617) /* ty=Tensor[(1, 8, 56, 56, 32), int32] */;
    fixed_point_multiply(%555, multiplier=1147032118, shift=2) /* ty=Tensor[(1, 8, 56, 56, 32), int32] */
  }
""")

    mod2 = tvm.parser.fromtext(
"""
#[version = "0.0.5"]
def @main(%p070: Tensor[(1, 2, 56, 56, 32), uint8] /* ty=Tensor[(1, 2, 56, 56, 32), uint8] */, %p150: Tensor[(8, 2, 1, 1, 8, 32, 4), int8] /* ty=Tensor[(8, 2, 1, 1, 8, 32, 4), int8] */, %p250: Tensor[(1, 8, 1, 1, 32), int32] /* ty=Tensor[(1, 8, 1, 1, 32), int32] */, %p350: Tensor[(1, 8, 1, 1, 32), int64] /* ty=Tensor[(1, 8, 1, 1, 32), int64] */, %p450: Tensor[(1, 8, 1, 1, 32), int64] /* ty=Tensor[(1, 8, 1, 1, 32), int64] */, %p550: Tensor[(1, 8, 1, 1, 32), int64] /* ty=Tensor[(1, 8, 1, 1, 32), int64] */, %p617: Tensor[(1), int32] /* ty=Tensor[(1), int32] */, kernel_layout="OIHW8i32o4i", data_layout="NCHW32c", out_layout="NCHW32c") {
    nn.contrib_conv2d_NCHWc(%p070, %p150, padding=[0, 0, 0, 0], channels=256, kernel_size=[1, 1], data_layout="NCHW32c", kernel_layout="OIHW8i32o4i", out_layout="NCHW32c", out_dtype="int32")
  }
""")

    params = {}

    with tvm.transform.PassContext(opt_level=3):
        hexagon_lowered = relay.build(
            mod2,
            tvm.target.Target(target_hexagon, host=target_hexagon),
            params=params,
        )

    # print(hexagon_lowered.lib.get_source("asm"))
    graph_mod = hexagon_session.get_executor_from_factory(hexagon_lowered)
    graph_mod.run()
    time_ms = graph_mod.benchmark(hexagon_session.device, number=1, repeat=20).mean * 1e3

    print("time elapsed: ", time_ms)

    # debug_ex = hexagon_session.get_graph_debug_executor(hexagon_lowered.get_graph_json(), hexagon_lowered.lib)
    # print(debug_ex.profile())
