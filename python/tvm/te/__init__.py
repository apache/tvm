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
# pylint: disable=unused-import, redefined-builtin, wildcard-import
"""Namespace for Tensor-level IR"""
# expose all operators in tvm tir.op
from tvm.tir.op import *

from .schedule import Schedule, create_schedule
from .tensor import TensorSlice, Tensor
from .tensor_intrin import decl_tensor_intrin
from .tag import tag_scope
from .operation import placeholder, compute, scan, extern, var, size_var
from .operation import thread_axis, reduce_axis
