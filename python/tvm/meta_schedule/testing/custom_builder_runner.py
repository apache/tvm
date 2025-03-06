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
"""Customized builder and runner methods"""
# pylint: disable=import-outside-toplevel

from typing import Dict, Union, Callable

import numpy as np  # type: ignore
from tvm.meta_schedule.runner import RPCConfig
from tvm.runtime import Module, Executable


def run_module_via_rpc(
    rpc_config: RPCConfig,
    lib: Union[Module, Executable],
    dev_type: str,
    args: Union[Dict[int, np.ndarray], Dict[str, np.ndarray]],
    continuation: Callable,
):
    """Execute a tvm.runtime.Module on RPC remote"""
    # pylint: disable=import-outside-toplevel
    import os
    import tempfile

    from tvm.contrib.tar import tar
    from tvm.runtime import ndarray

    # pylint: enable=import-outside-toplevel

    with tempfile.TemporaryDirectory() as tmp_dir:
        filename = os.path.join(tmp_dir, "tvm_tmp_mod." + tar.output_format)
        lib.export_library(filename, fcompile=tar)
        session = rpc_config.connect_server()
        session.upload(filename)
        _, filename = os.path.split(filename)
        rt_mod = session.load_module(filename)
        dev = session.device(dev_type=dev_type, dev_id=0)
        nd_args = {k: ndarray.array(v, dev) for k, v in args.items()}
        return continuation(rt_mod, dev, nd_args)
