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
"""The tvm.meta_schedule.postproc package."""
from .disallow_dynamic_loop import DisallowDynamicLoop
from .postproc import Postproc, PyPostproc
from .rewrite_cooperative_fetch import RewriteCooperativeFetch
from .rewrite_layout import RewriteLayout
from .rewrite_parallel_vectorize_unroll import RewriteParallelVectorizeUnroll
from .rewrite_reduction_block import RewriteReductionBlock
from .rewrite_tensorize import RewriteTensorize
from .rewrite_unbound_block import RewriteUnboundBlock
from .verify_gpu_code import VerifyGPUCode
