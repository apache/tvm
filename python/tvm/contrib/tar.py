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

"""Util to invoke tarball in the system."""
# pylint: disable=invalid-name
from __future__ import absolute_import as _abs
import os
import shutil
import subprocess
from . import utils
from .._ffi.base import py_str


def tar(output, files):
    """Create tarball containing all files in root.

    Parameters
    ----------
    output : str
        The target shared library.

    files : list
        List of files to be bundled.
    """
    cmd = ["tar"]
    cmd += ["-czf"]
    temp = utils.tempdir()
    fset = set()
    for fname in files:
        base = os.path.basename(fname)
        if base in fset:
            raise ValueError(f"duplicate file name {base}")
        fset.add(base)
        shutil.copy(fname, temp.relpath(base))
    cmd += [output]
    cmd += ["-C", temp.temp_dir]
    cmd += temp.listdir()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out, _) = proc.communicate()

    if proc.returncode != 0:
        msg = "Tar error:\n"
        msg += py_str(out)
        raise RuntimeError(msg)


# assign output format
tar.output_format = "tar"


def untar(tar_file, directory):
    """Unpack all tar files into the directory

    Parameters
    ----------
    tar_file : str
        The source tar file.

    directory : str
        The target directory
    """
    cmd = ["tar"]
    cmd += ["-xf"]
    cmd += [tar_file]
    cmd += ["-C", directory]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out, _) = proc.communicate()

    if proc.returncode != 0:
        msg = "Tar error:\n"
        msg += py_str(out)
        raise RuntimeError(msg)


def normalize_file_list_by_unpacking_tars(temp, file_list):
    """Normalize the file list by unpacking tars in list.

    When a filename is a tar, it will untar it into an unique dir
    in temp and return the list of files in the tar.
    When a filename is a normal file, it will be simply added to the list.

    This is useful to untar objects in tar and then turn
    them into a library.

    Parameters
    ----------
    temp: tvm.contrib.utils.TempDirectory
        A temp dir to hold the untared files.

    file_list: List[str]
        List of path

    Returns
    -------
    ret_list: List[str]
        An updated list of files
    """
    temp_count = 0
    ret_list = []
    for file_path in file_list:
        # enable extracting a tarball
        if file_path.endswith(".tar"):
            temp_dir = temp.relpath(f"temp{temp_count}")
            temp_count += 1
            os.mkdir(temp_dir)
            untar(file_path, temp_dir)
            # append all files inside
            for root, _, files in os.walk(temp_dir):
                for file in files:
                    ret_list.append(os.path.join(root, file))
        else:
            ret_list.append(file_path)
    return ret_list
