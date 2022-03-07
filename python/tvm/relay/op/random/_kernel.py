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
"""Splittable and parallelizable PRNG kernels."""
# pylint: disable=invalid-name,unused-argument
from __future__ import absolute_import

from .. import strategy
from ..op import register_strategy, register_pattern, OpPattern


# Threefry
register_strategy("random.threefry_generate", strategy.threefry_generate_strategy)
register_pattern("random.threefry_generate", OpPattern.OPAQUE)
register_strategy("random.threefry_split", strategy.threefry_split_strategy)
register_pattern("random.threefry_split", OpPattern.OPAQUE)

# Distribution
register_strategy("random.uniform", strategy.uniform_strategy)
register_pattern("random.uniform", OpPattern.OPAQUE)
register_strategy("random.normal", strategy.normal_strategy)
register_pattern("random.normal", OpPattern.OPAQUE)
