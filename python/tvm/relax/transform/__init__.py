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
"""Relax transformations. """

from .transform import (
    AllocateWorkspace,
    AlterOpImpl,
    AnnotateTIROpPattern,
    AttachGlobalSymbol,
    BindParams,
    BindSymbolicVars,
    BundleModelParams,
    CallTIRRewrite,
    CanonicalizeBindings,
    CombineParallelMatmul,
    ConvertLayout,
    DataflowBlockPass,
    DeadCodeElimination,
    DecomposeOpsForInference,
    DecomposeOpsForTraining,
    EliminateCommonSubexpr,
    FewShotTuning,
    FoldConstant,
    FunctionPass,
    FuseOps,
    FuseOpsByPattern,
    FuseTIR,
    FusionPattern,
    Gradient,
    KillAfterLastUse,
    LambdaLift,
    LegalizeOps,
    LiftTransformParams,
    LowerAllocTensor,
    MergeCompositeFunctions,
    MetaScheduleApplyDatabase,
    MetaScheduleTuneIRMod,
    MetaScheduleTuneTIR,
    Normalize,
    NormalizeGlobalVar,
    PatternCheckContext,
    RealizeVDevice,
    RemovePurityChecking,
    RemoveUnusedParameters,
    RemoveUnusedOutputs,
    RewriteCUDAGraph,
    RewriteDataflowReshape,
    RunCodegen,
    SplitCallTIRByPattern,
    StaticPlanBlockMemory,
    ToMixedPrecision,
    ToNonDataflow,
    UpdateVDevice,
    VMBuiltinLower,
    VMShapeLower,
    dataflowblock_pass,
    function_pass,
)

from .lazy_transform_params import LazyTransformParams
from .optimize_layout_transform import OptimizeLayoutTransform
from .remove_redundant_reshape import RemoveRedundantReshape
from .fast_math import FastMathTransform

# Import to register the legalization functions.
from . import legalize_ops, tuning_api
