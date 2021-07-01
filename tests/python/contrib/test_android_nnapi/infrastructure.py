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

import re
import tvm


def annotate_for_android_nnapi(mod, android_api_level):
    """Annotate Relay IR Function with attrs required by the Android NNAPI compiler.

    Parameters
    ----------
    mod: tvm.IRModule
        The module to be annotated.

    android_api_level: int
        The target Android API level.

    Returns
    -------
    mod: tvm.IRModule
        The annotated module.
    """
    ret = tvm.IRModule()
    gvs = mod.get_global_vars()
    for gv in gvs:
        func = mod[gv]
        func = func.with_attr("NnapiTargetVersion", tvm.tir.IntImm("int32", android_api_level))
        ret[gv] = func
    return ret


def is_compilable(mod, android_api_level):
    """Check if a module is compilable.

    Parameters
    ----------
    mod: runtime.Module
        The module to be checked for compilability.

    android_api_level: int
        The targeting Android API level for testing of compilability.

    Returns
    -------
    result: bool
        Whether the module is compilable.
    """
    tempdir = tvm.contrib.utils.tempdir()
    temp_lib_path = tempdir.relpath("lib.so")
    kwargs = {}
    kwargs["options"] = [
        "--target={}".format(
            f"aarch64-linux-android{android_api_level}"
        ),  # use aarch64 for testing
        "-O0",  # disable opt for testing
        "-lneuralnetworks",
        "-shared",
        "-fPIC",
    ]
    mod.export_library(temp_lib_path, fcompile=tvm.contrib.ndk.create_shared, **kwargs)
    return True
