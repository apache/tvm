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

"""TVM TOPI connector, eventually most of these should go to TVM repo"""

from . import bitpack
from .graphpack import graph_pack
from . import op
from .vta_conv2d import conv2d_packed, schedule_conv2d_packed
from .vta_conv2d_transpose import conv2d_transpose_packed, schedule_conv2d_transpose_packed
from .vta_group_conv2d import group_conv2d_packed, schedule_group_conv2d_packed
from .vta_dense import dense_packed, schedule_dense_packed
from . import utils
