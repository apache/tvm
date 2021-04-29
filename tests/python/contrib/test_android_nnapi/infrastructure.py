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
    """Annotate Relay IR Function with attrs required by the Android NNAPI converter

    Parameters
    ----------
    mod: tvm.IRModule
        The module to be annotated

    android_api_level: int
        The target Android API level

    Returns
    -------
    mod: tvm.IRModule
        The annotated module

    """
    ret = tvm.IRModule()
    gvs = mod.get_global_vars()
    for gv in gvs:
        func = mod[gv]
        func = func.with_attr("NnapiTargetVersion", tvm.tir.IntImm("int32", android_api_level))
        ret[gv] = func
    return ret


def _minify_c(src):
    ret = src
    # strip comments
    ret = re.sub(r"//.*", "", ret)
    ret = re.sub(r"/\*.*\*/", "", ret)

    # strip meaning less spaces. assumes no here docs
    ret = re.sub(r"^[\t ]+", "", ret, 0, re.M)
    ret = re.sub(r" +$", "", ret, 0, re.M)
    ret = re.sub(r"[\t ]+", " ", ret, 0)
    ret = re.sub(r" *([;,{}()=]) *", r"\1", ret)

    ret = re.sub(r"\n", "", ret)
    return ret


def verify_codegen_eq(res, ans):
    """Verify generated source code res equals to ans

    Parameters
    ----------
    res: str
        The generated source code

    ans: str
        The answer

    """
    assert _minify_c(res) == _minify_c(ans)
