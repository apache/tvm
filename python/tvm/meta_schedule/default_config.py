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
# pylint: disable=import-outside-toplevel
"""Pre-configured Defaults for MetaSchedule search rules"""
import logging
from os import path as osp
from typing import Callable, Dict, List, Optional, Union

from tvm.ir import IRModule
from tvm.target import Target
from tvm.tir import PrimFunc

from .builder import Builder, LocalBuilder
from .cost_model import CostModel, XGBModel
from .database import Database, JSONDatabase
from .feature_extractor import PerStoreFeature
from .measure_callback import MeasureCallback
from .mutator import Mutator
from .postproc import Postproc
from .runner import LocalRunner, Runner
from .schedule_rule import ScheduleRule
from .space_generator import PostOrderApply, SpaceGenerator

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

FnSpaceGenerator = Callable[[], SpaceGenerator]
FnScheduleRule = Callable[[], List[ScheduleRule]]
FnPostproc = Callable[[], List[Postproc]]
FnMutatorProb = Callable[[], Dict[Mutator, float]]


def mod(mod: Union[PrimFunc, IRModule]) -> IRModule:  # pylint: disable=redefined-outer-name
    """Normalize the input to an IRModule"""
    if isinstance(mod, PrimFunc):
        mod = mod.with_attr("global_symbol", "main")
        mod = mod.with_attr("tir.noalias", True)
        mod = IRModule({"main": mod})
    if not isinstance(mod, IRModule):
        raise TypeError(f"Expected `mod` to be PrimFunc or IRModule, but gets: {mod}")
    func_names = mod.get_global_vars()
    (func_name,) = func_names
    if len(func_names) == 1 and func_name.name_hint != "main":
        mod = IRModule({"main": mod[func_name]})
    return mod


def target(target: Union[str, Target]) -> Target:  # pylint: disable=redefined-outer-name
    """Normalize the input to tvm.target.Target"""
    if isinstance(target, str):
        target = Target(target)
    if not isinstance(target, Target):
        raise TypeError(f"Expected `target` to be str or Target, but gets: {target}")
    return target


def builder(builder: Optional[Builder]) -> Builder:  # pylint: disable=redefined-outer-name
    """Normalize the input to tvm.meta_schedule.Builder"""
    if builder is None:
        builder = LocalBuilder()  # type: ignore
    if not isinstance(builder, Builder):
        raise TypeError(f"Expected `builder` to be Builder, but gets: {builder}")
    return builder


def runner(runner: Optional[Runner]) -> Runner:  # pylint: disable=redefined-outer-name
    """Normalize the input to tvm.meta_schedule.Runner"""
    if runner is None:
        runner = LocalRunner()  # type: ignore
    if not isinstance(runner, Runner):
        raise TypeError(f"Expected `runner` to be Runner, but gets: {runner}")
    return runner


def database(
    database: Union[None, Database],  # pylint: disable=redefined-outer-name
    path: str,
) -> Database:
    """Normalize the input to tvm.meta_schedule.Database"""
    if database is None:
        path_workload = osp.join(path, "database_workload.json")
        path_tuning_record = osp.join(path, "database_tuning_record.json")
        logger.info(
            "Creating JSONDatabase. Workload at: %s. Tuning records at: %s",
            path_workload,
            path_tuning_record,
        )
        database = JSONDatabase(
            path_workload=path_workload,
            path_tuning_record=path_tuning_record,
        )
    if not isinstance(database, Database):
        raise TypeError(f"Expected `database` to be Database, but gets: {database}")
    return database


def callbacks(  # pylint: disable=redefined-outer-name
    measure_callbacks: Optional[List[MeasureCallback]],
) -> List[MeasureCallback]:
    """Normalize the input to List[tvm.meta_schedule.MeasureCallback]"""
    if measure_callbacks is None:
        from tvm.meta_schedule import measure_callback as M

        return [
            M.AddToDatabase(),
            M.RemoveBuildArtifact(),
            M.EchoStatistics(),
            M.UpdateCostModel(),
        ]
    if not isinstance(measure_callbacks, (list, tuple)):
        raise TypeError(
            f"Expected `measure_callbacks` to be List[MeasureCallback], "
            f"but gets: {measure_callbacks}"
        )
    measure_callbacks = list(measure_callbacks)
    for i, callback in enumerate(measure_callbacks):
        if not isinstance(callback, MeasureCallback):
            raise TypeError(
                f"Expected `measure_callbacks` to be List[MeasureCallback], "
                f"but measure_callbacks[{i}] is: {callback}"
            )
    return measure_callbacks


def cost_model(
    cost_model: Optional[CostModel],  # pylint: disable=redefined-outer-name
    adpative_training: Optional[bool],
) -> CostModel:
    """Normalize the input to tvm.meta_schedule.CostModel"""
    if cost_model is None:
        return XGBModel(  # type: ignore
            extractor=PerStoreFeature(),
            adaptive_training=adpative_training is None or adpative_training,
        )
    if not isinstance(cost_model, CostModel):
        raise TypeError(f"Expected `cost_model` to be CostModel, but gets: {cost_model}")
    return cost_model


def space_generator(
    space_generator: Optional[FnSpaceGenerator],  # pylint: disable=redefined-outer-name
) -> SpaceGenerator:
    """Normalize the input to tvm.meta_schedule.SpaceGenerator"""
    if space_generator is None:
        return PostOrderApply()
    if callable(space_generator):
        space_generator = space_generator()
    if not isinstance(space_generator, SpaceGenerator):
        raise TypeError(
            f"Expected `space_generator` to return SpaceGenerator, " f"but gets: {space_generator}"
        )
    return space_generator


def schedule_rules(  # pylint: disable=redefined-outer-name
    sch_rules: Optional[FnScheduleRule],
    target: Target,
) -> List[ScheduleRule]:
    """Normalize the input to List[tvm.meta_schedule.ScheduleRule]"""
    if callable(sch_rules):
        return sch_rules()
    if sch_rules is not None:
        raise TypeError(f"Expected `sch_rules` to be None or callable, but gets: {sch_rules}")
    if target.kind.name in ["llvm", "hexagon"]:
        return _DefaultLLVM.schedule_rules()
    if target.kind.name in ["cuda", "rocm", "vulkan"]:
        return _DefaultCUDA.schedule_rules()
    raise ValueError(f"Unsupported target: {target}")


def postproc(  # pylint: disable=redefined-outer-name
    postproc: Optional[FnPostproc],
    target: Target,
) -> List[Postproc]:
    """Normalize the input to List[tvm.meta_schedule.Postproc]"""
    if callable(postproc):
        return postproc()
    if postproc is not None:
        raise TypeError(f"Expected `postproc` to be None or callable, but gets: {postproc}")
    if target.kind.name in ["llvm", "hexagon"]:
        return _DefaultLLVM.postprocs()
    if target.kind.name in ["cuda", "rocm", "vulkan"]:
        return _DefaultCUDA.postprocs()
    raise ValueError(f"Unsupported target: {target}")


def mutator_probs(  # pylint: disable=redefined-outer-name
    mutator_probs: Optional[FnMutatorProb],
    target: Target,
) -> Dict[Mutator, float]:
    """Normalize the input to Dict[tvm.meta_schedule.Mutator, float]"""
    if callable(mutator_probs):
        return mutator_probs()
    if mutator_probs is not None:
        raise TypeError(
            f"Expected `mutator_probs` to be None or callable, but gets: {mutator_probs}"
        )
    if target.kind.name in ["llvm", "hexagon"]:
        return _DefaultLLVM.mutator_probs()
    if target.kind.name in ["cuda", "rocm", "vulkan"]:
        return _DefaultCUDA.mutator_probs()
    raise ValueError(f"Unsupported target: {target}")


class _DefaultLLVM:
    """Default tuning configuration for LLVM."""

    @staticmethod
    def schedule_rules() -> List[ScheduleRule]:
        from tvm.meta_schedule import schedule_rule as M

        return [
            M.AutoInline(
                into_producer=False,
                into_consumer=True,
                inline_const_tensor=True,
                disallow_if_then_else=True,
                require_injective=True,
                require_ordered=True,
                disallow_op=["tir.exp"],
            ),
            M.AddRFactor(max_jobs_per_core=16, max_innermost_factor=64),
            M.MultiLevelTiling(
                structure="SSRSRS",
                tile_binds=None,
                max_innermost_factor=64,
                vector_load_lens=None,
                reuse_read=None,
                reuse_write=M.ReuseType(
                    req="may",
                    levels=[1, 2],
                    scope="global",
                ),
            ),
            M.ParallelizeVectorizeUnroll(
                max_jobs_per_core=16,
                max_vectorize_extent=64,
                unroll_max_steps=[0, 16, 64, 512],
                unroll_explicit=True,
            ),
            M.RandomComputeLocation(),
        ]

    @staticmethod
    def postprocs() -> List[Postproc]:
        from tvm.meta_schedule import postproc as M

        return [
            M.DisallowDynamicLoop(),
            M.RewriteParallelVectorizeUnroll(),
            M.RewriteReductionBlock(),
            M.RewriteLayout(),
        ]

    @staticmethod
    def mutator_probs() -> Dict[Mutator, float]:
        from tvm.meta_schedule import mutator as M

        return {
            M.MutateTileSize(): 0.9,
            M.MutateComputeLocation(): 0.05,
            M.MutateUnroll(): 0.03,
            M.MutateParallel(max_jobs_per_core=16): 0.02,
        }


class _DefaultCUDA:
    """Default tuning configuration for CUDA."""

    @staticmethod
    def schedule_rules() -> List[ScheduleRule]:
        from tvm.meta_schedule import schedule_rule as M

        return [
            M.MultiLevelTiling(
                structure="SSSRRSRS",
                tile_binds=["blockIdx.x", "vthread.x", "threadIdx.x"],
                max_innermost_factor=64,
                vector_load_lens=[1, 2, 3, 4, 8, 16],
                reuse_read=M.ReuseType(
                    req="must",
                    levels=[4],
                    scope="shared",
                ),
                reuse_write=M.ReuseType(
                    req="must",
                    levels=[3],
                    scope="local",
                ),
            ),
            M.AutoInline(
                into_producer=True,
                into_consumer=True,
                inline_const_tensor=True,
                disallow_if_then_else=False,
                require_injective=False,
                require_ordered=False,
                disallow_op=None,
            ),
            M.CrossThreadReduction(thread_extents=[4, 8, 16, 32, 64, 128, 256, 512]),
            M.ParallelizeVectorizeUnroll(
                max_jobs_per_core=-1,  # disable parallelize
                max_vectorize_extent=-1,  # disable vectorize
                unroll_max_steps=[0, 16, 64, 512, 1024],
                unroll_explicit=True,
            ),
            M.AutoBind(
                max_threadblocks=256,
                thread_extents=[32, 64, 128, 256, 512, 1024],
            ),
        ]

    @staticmethod
    def postprocs() -> List[Postproc]:
        from tvm.meta_schedule import postproc as M

        return [
            M.DisallowDynamicLoop(),
            M.RewriteCooperativeFetch(),
            M.RewriteUnboundBlock(),
            M.RewriteParallelVectorizeUnroll(),
            M.RewriteReductionBlock(),
            M.VerifyGPUCode(),
        ]

    @staticmethod
    def mutator_probs() -> Dict[Mutator, float]:
        from tvm.meta_schedule import mutator as M

        return {
            M.MutateTileSize(): 0.9,
            M.MutateUnroll(): 0.08,
            M.MutateThreadBinding(): 0.02,
        }


class _DefaultCUDATensorCore:
    """Default tuning configuration for CUDA TensorCore."""

    @staticmethod
    def schedule_rules():
        from tvm.meta_schedule import schedule_rule as M
        from tvm.tir.tensor_intrin.cuda import get_wmma_intrin_group

        return [
            M.MultiLevelTilingTensorCore(
                intrin_groups=[
                    get_wmma_intrin_group(
                        store_scope="shared",
                        in_dtype=in_dtype,
                        out_dtype=out_dtype,
                        trans_b=trans_b,
                    )
                    for (in_dtype, out_dtype) in [("float16", "float16"), ("int8", "int32")]
                    for trans_b in [False, True]
                ],
                structure="SSSRRSRS",
                tile_binds=["blockIdx.y", "blockIdx.x", "threadIdx.y"],
                max_innermost_factor=4,
                vector_load_lens=[1, 2, 3, 4, 8, 16],
                reuse_read=M.ReuseType(req="must", levels=[4], scope="shared"),
                reuse_write=M.ReuseType(
                    req="must",
                    levels=[2],
                    scope="shared",
                ),
                use_software_pipeline=False,
            ),
            *_DefaultCUDA.schedule_rules(),
        ]

    @staticmethod
    def postprocs() -> List[Postproc]:
        from tvm.meta_schedule import postproc as M

        return [
            M.DisallowDynamicLoop(),
            M.RewriteCooperativeFetch(),
            M.RewriteUnboundBlock(),
            M.RewriteParallelVectorizeUnroll(),
            M.RewriteReductionBlock(),
            M.RewriteTensorize(),
            M.VerifyGPUCode(),
        ]

    @staticmethod
    def mutator_probs() -> Dict[Mutator, float]:
        return _DefaultCUDA.mutator_probs()
