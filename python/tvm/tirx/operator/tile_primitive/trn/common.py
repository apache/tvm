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

"""Common utilities for TRN operator scheduling."""

from tvm.arith.analyzer import Analyzer
from tvm.tirx.operator.tile_primitive import DispatchContext

# Used to generate the correct [:, None] for mask/predicate
nki_dim = "nki_dim"


def init_analyzer(sctx: DispatchContext):
    """Initialize an analyzer with the dispatch context.

    Parameters
    ----------
    sctx : DispatchContext
        The dispatch context

    Returns
    -------
    Analyzer :
        The initialized analyzer
    """
    analyzer = Analyzer()
    for v, r in sctx.var_range_map.items():
        analyzer.bind(v, r)
    return analyzer
