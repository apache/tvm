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

""" Hexagon testing fixtures used to deduce testing argument
    values from testing parameters """

import tvm
from .infrastructure import get_packed_filter_layout


@tvm.testing.fixture
def shape_nhwc(batch, in_channel, in_size):
    return (batch, in_size, in_size, in_channel)


@tvm.testing.fixture
def shape_oihw(out_channel, in_channel, kernel):
    return (out_channel, in_channel, kernel, kernel)


@tvm.testing.fixture
def shape_oihw8i32o4i(out_channel, in_channel, kernel):
    return get_packed_filter_layout(out_channel, in_channel, kernel, kernel)
