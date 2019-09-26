..  Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

..    http://www.apache.org/licenses/LICENSE-2.0

..  Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.

tvm.relay.transform
----------------------

.. automodule:: tvm.relay.transform

.. autofunction:: tvm.relay.transform.build_config

.. autofunction:: tvm.relay.transform.module_pass

.. autofunction:: tvm.relay.transform.function_pass

.. autofunction:: tvm.relay.transform.InferType

.. autofunction:: tvm.relay.transform.FoldScaleAxis

.. autofunction:: tvm.relay.transform.BackwardFoldScaleAxis

.. autofunction:: tvm.relay.transform.ForwardFoldScaleAxis

.. autofunction:: tvm.relay.transform.SimplifyInference

.. autofunction:: tvm.relay.transform.CanonicalizeOps

.. autofunction:: tvm.relay.transform.DeadCodeElimination

.. autofunction:: tvm.relay.transform.FoldConstant

.. autofunction:: tvm.relay.transform.FuseOps

.. autofunction:: tvm.relay.transform.CombineParallelConv2D

.. autofunction:: tvm.relay.transform.CombineParallelDense

.. autofunction:: tvm.relay.transform.AlterOpLayout

.. autofunction:: tvm.relay.transform.Legalize

.. autofunction:: tvm.relay.transform.RewriteAnnotatedOps

.. autofunction:: tvm.relay.transform.ToANormalForm

.. autofunction:: tvm.relay.transform.ToCPS

.. autofunction:: tvm.relay.transform.EtaExpand

.. autofunction:: tvm.relay.transform.ToGraphNormalForm

.. autofunction:: tvm.relay.transform.EliminateCommonSubexpr

.. autofunction:: tvm.relay.transform.PartialEvaluate

.. autofunction:: tvm.relay.transform.CanonicalizeCast

.. autofunction:: tvm.relay.transform.LambdaLift

.. autofunction:: tvm.relay.transform.PrintIR

.. autoclass:: tvm.relay.transform.Pass
    :members:

.. autoclass:: tvm.relay.transform.PassInfo
    :members:

.. autoclass:: tvm.relay.transform.PassContext
    :members:

.. autoclass:: tvm.relay.transform.ModulePass
    :members:

.. autoclass:: tvm.relay.transform.FunctionPass
    :members:

.. autoclass:: tvm.relay.transform.Sequential
    :members:
