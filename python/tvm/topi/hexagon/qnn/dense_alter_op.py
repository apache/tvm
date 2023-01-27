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
"""QNN Dense alter op functions for Hexagon"""

from tvm import relay
from ..dense_alter_op import check_vrmpy_applicable
from ...nn import qnn_dense_alter_layout


@qnn_dense_alter_layout.register("hexagon")
def _alter_qnn_dense_layout(_attrs, inputs, tinfos, out_type):
    data_tensor = tinfos[0]
    weight_tensor = tinfos[1]

    if check_vrmpy_applicable(data_tensor, weight_tensor):
        weight_layout = "NC32n4c"
        return relay.qnn.op.contrib_dense_pack(*inputs, weight_layout, None, out_type.dtype)
    else:
        return None
