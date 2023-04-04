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
"""Default legalization function for unary operators."""
from tvm import topi
from .common import _call_topi_without_attr, register_legalize

# To avoid conflict of IRModule function name and libc function name, we add
# "tir_" as the prefix of the generated PrimFunc name.
register_legalize("relax.abs", _call_topi_without_attr(topi.abs, "tir_abs"))
register_legalize("relax.ceil", _call_topi_without_attr(topi.ceil, "tir_ceil"))
register_legalize("relax.cos", _call_topi_without_attr(topi.cos, "tir_cos"))
register_legalize("relax.log", _call_topi_without_attr(topi.log, "tir_log"))
register_legalize("relax.exp", _call_topi_without_attr(topi.exp, "tir_exp"))
register_legalize("relax.floor", _call_topi_without_attr(topi.floor, "tir_floor"))
register_legalize("relax.negative", _call_topi_without_attr(topi.negative, "tir_negative"))
register_legalize("relax.round", _call_topi_without_attr(topi.round, "tir_round"))
register_legalize("relax.rsqrt", _call_topi_without_attr(topi.rsqrt, "tir_rsqrt"))
register_legalize("relax.sigmoid", _call_topi_without_attr(topi.sigmoid, "tir_sigmoid"))
register_legalize("relax.sign", _call_topi_without_attr(topi.sign, "tir_sign"))
register_legalize("relax.sinh", _call_topi_without_attr(topi.sinh, "tir_sinh"))
register_legalize("relax.sin", _call_topi_without_attr(topi.sin, "tir_sin"))
register_legalize("relax.sqrt", _call_topi_without_attr(topi.sqrt, "tir_sqrt"))
register_legalize("relax.tanh", _call_topi_without_attr(topi.tanh, "tir_tanh"))
register_legalize("relax.clip", _call_topi_without_attr(topi.clip, "tir_clip"))
