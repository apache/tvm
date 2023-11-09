#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

import os
import sys
import numpy as np

import tvm
from tvm import relay
from tvm.driver import tvmc
from tvm.relay.op.contrib import clml
from tvm.contrib import utils
from string import Template


def main():
    print("CLML Codegen")
    if len(sys.argv) != 2:
        print("Usage: python clml_codegen.py <model_path>")
        return

    tvmc_model = tvmc.load(sys.argv[1])
    mod = tvmc_model.mod
    params = tvmc_model.params
    with tvm.transform.PassContext(opt_level=3):
        mod = tvmc.transform.convert_graph_layout(mod, "NCHW")
    with tvm.transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
        clml_mod = clml.partition_for_clml(mod, params)
        libm = relay.build(
            clml_mod,
            target="opencl",
            target_host="llvm -mtriple=aarch64-linux-gnu",
            params=params,
        )

        # Extract CLML related params
        (clml_params_save, gen_src) = clml.CLMLGenSrc(libm).get_artifacts()
        np.savez("clml_params.npz", **clml_params_save)

        f_src = open("../clml_models.cc", "w")
        f_src.write("\n".join(gen_src))
        f_src.close()
        os.popen("clang-format-15 -i ../clml_models.cc")


if __name__ == "__main__":
    main()
