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
import json
import numpy as np

import tvm
from tvm import relay
from tvm.driver import tvmc
from tvm.relay.op.contrib import clml
from tvm.contrib import utils
from string import Template


def main():
    print("CLML Codegen From JSON")
    if len(sys.argv) != 3:
        print("Usage: python clml_codegen_json.py <json path> <outfile path>")
        return

    with open(sys.argv[1], "r") as file:
        codegen = json.load(file)
        (_, gen_src) = clml.CLMLGenSrc(codegen).get_artifacts()

        f_src = open(sys.argv[2], "w")
        f_src.write("\n".join(gen_src))
        f_src.close()
        os.popen("clang-format-15 -i " + sys.argv[2])


if __name__ == "__main__":
    main()
