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
import tempfile
import subprocess


class CompileEngine(object):
    def __init__(self, cuda_arch, cutlass_path, binary_prefix):
        self.cuda_arch = cuda_arch
        self.binary_prefix = binary_prefix
        self.cutlass = cutlass_path
        self.cflags = "-I{cutlass}/include -I{cutlass}/tools/util/include -O3 -std=c++11".format(
            cutlass=cutlass_path
        )
        self.cflags += " -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1"
        self.cflags += " -gencode=arch=compute_{arch},code=[sm_{arch},compute_{arch}]".format(
            arch=cuda_arch
        )
        self.cflags += " -Xcompiler=-Wconversion -Xcompiler=-fno-strict-aliasing"
        self.cmd = "nvcc {cflags} {src} -o {output}"

    def _compile(self, op_name, src):
        os.makedirs(self.binary_prefix, exist_ok=True)
        opath = os.path.join(self.binary_prefix, op_name)
        if os.path.exists(opath):
            return
        fi = tempfile.NamedTemporaryFile("w", delete=False, suffix=".cu")
        fi.write(src)
        fi.close()
        cmd = self.cmd.format(cflags=self.cflags, src=fi.name, output=opath)
        os.system(cmd)
        os.unlink(fi.name)

    def _execute(self, op_name, args):
        opath = os.path.join(self.binary_prefix, op_name)
        cmd = [opath]
        if args is not None:
            cmd.append(str(args[0]))
            cmd.append(str(args[1]))
            cmd.append(str(args[2]))
            if len(args) > 3:
                cmd.append(str(args[3]))
        sp = subprocess.run(cmd, capture_output=True)
        try:
            print("command: " + str(cmd))
            print("time: " + str(sp.stdout))
            rt = float(sp.stdout)
        except:
            rt = 9999999
        return rt

    def evaluate(self, op_name, src, args=None):
        self._compile(op_name, src)
        return self._execute(op_name, args)
