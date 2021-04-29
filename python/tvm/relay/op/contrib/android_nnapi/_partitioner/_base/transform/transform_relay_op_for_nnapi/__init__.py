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
"""Convert Relay operators into mathematically equivalent forms
so that Android NNAPI supports it
"""
import tvm
from .expand_batch_norm import ExpandBatchNorm as _ExpandBatchNorm
from .expand_split import ExpandSplit as _ExpandSplit


class TransformRelayOpForNnapi:
    """Convert Relay operators into mathematically equivalent forms so
    that Android NNAPI supports it
    """

    def __call__(self, mod):
        """Convert Relay operators into mathematically equivalent forms
        so that Android NNAPI supports it

        Parameters
        ----------
        mod: tvm.IRModule
            The module to be transformed

        Returns
        -------
        mod: tvm.IRModule
            The transformed module

        """
        assert isinstance(mod, tvm.IRModule)
        for pazz in [_ExpandBatchNorm(), _ExpandSplit()]:
            mod = pazz(mod)
        return mod
