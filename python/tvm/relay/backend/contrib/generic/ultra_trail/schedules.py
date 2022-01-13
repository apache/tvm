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
"""TIR schedule functions for the UltraTrail accelerator"""

def example_sch_func(sch):
    n, k, c, f_x, x = sch.get_loops(sch.get_block("conv1d_ncw"))
    k_0, k_1 = sch.split(k, factors=[2, None])
    sch.reorder(n, k_0, c, f_x, x, k_1)
    return sch
