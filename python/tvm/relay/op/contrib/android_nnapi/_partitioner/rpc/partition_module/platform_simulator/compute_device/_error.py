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
"""Error encountered during RPC profiling."""
from tvm.contrib.target.android_nnapi.relayir_to_nnapi_converter.error import (
    AndroidNNAPICompilerError,
)


class AndroidNNAPICompilerProfilingError(AndroidNNAPICompilerError):
    """Error caused by profiling failure

    Parameters
    ----------
    msg: str
        An optional error message

    Notes
    -----
    This error is used internally in the partitioner and does not intend to be
    handled by other modules.
    """
