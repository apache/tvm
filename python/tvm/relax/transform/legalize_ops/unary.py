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

register_legalize("relax.abs", _call_topi_without_attr(topi.abs))
register_legalize("relax.cos", _call_topi_without_attr(topi.cos))
register_legalize("relax.log", _call_topi_without_attr(topi.log))
register_legalize("relax.exp", _call_topi_without_attr(topi.exp))
register_legalize("relax.negative", _call_topi_without_attr(topi.negative))
register_legalize("relax.sigmoid", _call_topi_without_attr(topi.sigmoid))
register_legalize("relax.sin", _call_topi_without_attr(topi.sin))
register_legalize("relax.sqrt", _call_topi_without_attr(topi.sqrt))
register_legalize("relax.tanh", _call_topi_without_attr(topi.tanh))
register_legalize("relax.clip", _call_topi_without_attr(topi.clip))
