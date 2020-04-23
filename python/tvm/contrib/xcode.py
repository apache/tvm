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
# pylint: disable=invalid-name
"""Utility to invoke Xcode compiler toolchain"""
from __future__ import absolute_import as _abs

import os
import sys
import subprocess
from .._ffi.base import py_str
from . import util

def xcrun(cmd):
    """Run xcrun and return the output.

    Parameters
    ----------
    cmd : list of str
        The command sequence.

    Returns
    -------
    out : str
        The output string.
    """
    cmd = ["xcrun"] + cmd
    proc = subprocess.Popen(cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    (out, _) = proc.communicate()
    return out.strip()


def codesign(lib):
    """Codesign the shared libary

    This is an required step for library to be loaded in
    the app.

    Parameters
    ----------
    lib : The path to the library.
    """
    if "TVM_IOS_CODESIGN" not in os.environ:
        raise RuntimeError("Require environment variable TVM_IOS_CODESIGN "
                           " to be the signature")
    signature = os.environ["TVM_IOS_CODESIGN"]
    cmd = ["codesign", "--force", "--sign", signature]
    cmd += [lib]
    proc = subprocess.Popen(cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    (out, _) = proc.communicate()
    if proc.returncode != 0:
        msg = "Codesign error:\n"
        msg += py_str(out)
        raise RuntimeError(msg)


def create_dylib(output, objects, arch, sdk="macosx"):
    """Create dynamic library.

    Parameters
    ----------
    output : str
        The target shared library.

    objects : list
        List of object files.

    options : str
        The additional options.

    arch : str
        Target major architectures

    sdk : str
        The sdk to be used.
    """
    clang = xcrun(["-sdk", sdk, "-find", "clang"])
    sdk_path = xcrun(["-sdk", sdk, "--show-sdk-path"])
    cmd = [clang]
    cmd += ["-dynamiclib"]
    cmd += ["-arch", arch]
    cmd += ["-isysroot", sdk_path]
    cmd += ["-o", output]
    if isinstance(objects, str):
        cmd += [objects]
    else:
        cmd += objects

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    (out, _) = proc.communicate()

    if proc.returncode != 0:
        msg = "Compilation error:\n"
        msg += py_str(out)
        raise RuntimeError(msg)


# assign so as default output format
create_dylib.output_format = "dylib"

def compile_metal(code, path_target=None, sdk="macosx"):
    """Compile metal with CLI tool from env.

    Parameters
    ----------
    code : str
        The cuda code.

    path_target : str, optional
        Output file.

    sdk : str, optional
        The target platform SDK.

    Return
    ------
    metallib : bytearray
        The bytearray of the metallib
    """
    temp = util.tempdir()
    temp_code = temp.relpath("my_lib.metal")
    temp_ir = temp.relpath("my_lib.air")
    temp_target = temp.relpath("my_lib.metallib")

    with open(temp_code, "w") as out_file:
        out_file.write(code)
    file_target = path_target if path_target else temp_target

    # See:
    # - https://developer.apple.com/documentation/metal/gpu_functions_libraries/building_a_library_with_metal_s_command-line_tools#overview # pylint: disable=line-too-long
    #
    #   xcrun -sdk macosx metal -c MyLibrary.metal -o MyLibrary.air
    #   xcrun -sdk macosx metallib MyLibrary.air -o MyLibrary.metallib
    cmd1 = ["xcrun", "-sdk", sdk, "metal", "-O3"]
    cmd1 += ["-c", temp_code, "-o", temp_ir]
    cmd2 = ["xcrun", "-sdk", sdk, "metallib"]
    cmd2 += [temp_ir, "-o", file_target]
    proc = subprocess.Popen(
        ' '.join(cmd1) + ";" + ' '.join(cmd2),
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    (out, _) = proc.communicate()
    if proc.returncode != 0:
        sys.stderr.write("Compilation error:\n")
        sys.stderr.write(py_str(out))
        sys.stderr.flush()
        libbin = None
    else:
        libbin = bytearray(open(file_target, "rb").read())
    return libbin


def compile_coreml(model, out_dir="."):
    """Compile coreml model and return the compiled model path.
    """
    mlmodel_path = os.path.join(out_dir, "tmp.mlmodel")
    model.save(mlmodel_path)

    xcrun(["coremlcompiler", "compile", mlmodel_path, out_dir])

    return os.path.join(out_dir, "tmp.mlmodelc")


class XCodeRPCServer(object):
    """Wrapper for RPC server

    Parameters
    ----------
    cmd : list of str
       The command to run

    lock: FileLock
       Lock on the path
    """
    def __init__(self, cmd, lock):
        self.proc = subprocess.Popen(cmd)
        self.lock = lock

    def join(self):
        """Wait server to finish and release its resource
        """
        self.proc.wait()
        self.lock.release()


def popen_test_rpc(host,
                   port,
                   key,
                   destination,
                   libs=None,
                   options=None):
    """Launch rpc server via xcodebuild test through another process.

    Parameters
    ----------
    host : str
        The address of RPC proxy host.

    port : int
        The port of RPC proxy host

    key : str
        The key of the RPC server

    destination : str
        Destination device of deployment, as in xcodebuild

    libs : list of str
        List of files to be packed into app/Frameworks/tvm
        These can be dylibs that can be loaed remoted by RPC.

    options : list of str
        Additional options to xcodebuild

    Returns
    -------
    proc : Popen
        The test rpc server process.
        Don't do wait() on proc, since it can terminate normally.
    """
    if "TVM_IOS_RPC_ROOT" in os.environ:
        rpc_root = os.environ["TVM_IOS_RPC_ROOT"]
    else:
        curr_path = os.path.dirname(os.path.realpath(os.path.expanduser(__file__)))
        rpc_root = os.path.join(curr_path, "../../../apps/ios_rpc")
    proj_path = os.path.realpath(os.path.join(rpc_root, "tvmrpc.xcodeproj"))
    if not os.path.exists(proj_path):
        raise RuntimeError("Cannot find tvmrpc.xcodeproj in %s," +
                           (" please set env TVM_IOS_RPC_ROOT correctly" % rpc_root))

    # Lock the path so only one file can run
    lock = util.filelock(os.path.join(rpc_root, "ios_rpc.lock"))

    with open(os.path.join(rpc_root, "rpc_config.txt"), "w") as fo:
        fo.write("%s %d %s\n" % (host, port, key))
        libs = libs if libs else []
        for file_name in libs:
            fo.write("%s\n" % file_name)

    cmd = ["xcrun", "xcodebuild",
           "-scheme", "tvmrpc",
           "-project", proj_path,
           "-destination", destination]
    if options:
        cmd += options
    cmd += ["test"]

    return XCodeRPCServer(cmd, lock)
