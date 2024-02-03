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
"""tvm.contrib.msc.core.utils.file"""

import os
import shutil
import tempfile
import types
import subprocess
from functools import partial
from typing import List, Any, Union
from importlib.machinery import SourceFileLoader

from .namespace import MSCMap, MSCKey, MSCFramework
from .register import get_registered_func


def is_callable(name: str, framework: str = MSCFramework.MSC) -> bool:
    """Check if name is callable.

    Parameters
    ----------
    name: string
        The name of the registered func or path:f_name str.
    framework: string
        Should be from MSCFramework.

    Returns
    -------
    is_callable: bool
        Whether the name is callable
    """

    func = get_registered_func(name, framework)
    if func:
        return True
    if ".py:" in name:
        path, _ = name.split(":")
        return os.path.isfile(path)
    return False


def load_callable(name: str, framework: str = MSCFramework.MSC) -> callable:
    """Load a callable  object.

    Parameters
    ----------
    name: string
        The name of the registered func or path:f_name str.
    framework: string
        Should be from MSCFramework.

    Returns
    -------
    func: callable
        The function.
    """

    func = get_registered_func(name, framework)
    if func:
        return func
    if ".py:" in name:
        path, func_name = name.split(":")
        loader = SourceFileLoader(path.replace(".py", ""), path)
        mod = types.ModuleType(loader.name)
        loader.exec_module(mod)
        return getattr(mod, func_name)
    raise Exception("Func {} is neighter registered nor path.py:name string")


class MSCDirectory(object):
    """Create a directory manager for MSC"""

    def __init__(self, path: str = None, keep_history: bool = True, cleanup: bool = False):
        self._path = os.path.abspath(path or tempfile.mkdtemp())
        self._cleanup = cleanup
        self._cwd = os.getcwd()
        if os.path.isdir(self._path) and not keep_history:
            shutil.rmtree(self._path)
        if not os.path.isdir(self._path):
            os.mkdir(self._path)

    def __str__(self):
        return "{}(Cleanup: {}): {} Files".format(self._path, self._cleanup, len(self.listdir()))

    def __enter__(self):
        if not os.path.isdir(self._path):
            os.mkdir(self._path)
        os.chdir(self._path)
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        os.chdir(self._cwd)
        self.clean_up()

    def __del__(self):
        self.clean_up()

    def clean_up(self):
        """Clean up the dir"""

        if self._cleanup and os.path.isdir(self._path):
            shutil.rmtree(self._path)

    def add_file(self, name: str, contains: str) -> str:
        """Add a file under the folder

        Parameters
        ----------
        name: str
            The name of the file.
        contains: str
            The contains of the file.

        Returns
        -------
        path: str
            The abs file path.
        """

        file_path = self.relpath(name)
        base_dir = os.path.dirname(name)
        if base_dir and not os.path.isdir(base_dir):
            os.makedirs(base_dir)
        with open(file_path, "w") as f:
            f.write(contains)
        return file_path

    def move(self, src_path: str, dst_path: str = None):
        """Move a file or folder to another folder

        Parameters
        ----------
        src_path: str
            The name of the source file or folder.
        dst_path: str
            The target file name or folder path.

        Returns
        -------
        path: str
            The abs file path.
        """

        if src_path != os.path.abspath(src_path):
            src_path = os.path.join(self.relpath(src_path))
        assert os.path.isfile(src_path), "Source path {} not exist".format(src_path)
        if not dst_path:
            dst_path = self.relpath(os.path.basename(src_path))
        if dst_path != os.path.abspath(dst_path):
            dst_path = self.relpath(dst_path)
        os.rename(src_path, dst_path)
        return dst_path

    def copy(self, src_path: str, dst_path: str = None):
        """Copy a file to another folder

        Parameters
        ----------
        src_path: str
            The name of the source file or folder.
        dst_path: str
            The target file name or folder path.

        Returns
        -------
        path: str
            The abs file path.
        """

        if src_path != os.path.abspath(src_path):
            src_path = os.path.join(self.relpath(src_path))
        assert os.path.exists(src_path), "Source path {} not exist".format(src_path)
        if not dst_path:
            dst_path = self.relpath(os.path.basename(src_path))
        if dst_path != os.path.abspath(dst_path):
            dst_path = self.relpath(dst_path)
        if os.path.isfile(src_path):
            shutil.copy2(src_path, dst_path)
        else:
            if os.path.isdir(dst_path):
                os.remove(dst_path)
            shutil.copytree(src_path, dst_path)
        return dst_path

    def create_dir(self, name: str, keep_history: bool = True, cleanup: bool = False) -> Any:
        """Add a dir under the folder

        Parameters
        ----------
        name: str
            The name of the file.
        keep_history: bol
            Whether to keep history.
        cleanup: bool
            Whether to clean up before exit.


        Returns
        -------
        dir: MSCDirectory
            The created dir.
        """

        dir_path = self.relpath(name)
        if os.path.isfile(dir_path):
            os.remove(dir_path)
        return self.__class__(dir_path, keep_history=keep_history, cleanup=cleanup)

    def relpath(self, name: str, keep_history: bool = True) -> str:
        """Relative path in dir

        Parameters
        ----------
        name: str
            The name of the file.

        Returns
        -------
        path: str
            The concatenated path.
        """

        f_path = os.path.join(self._path, name)
        if os.path.isfile(f_path) and not keep_history:
            os.remove(f_path)
        if os.path.isdir(f_path) and not keep_history:
            shutil.rmtree(f_path)
        return f_path

    def listdir(self, as_abs: bool = False) -> List[str]:
        """List contents in the dir.

        Parameters
        ----------
        as_abs: bool
            Whether to show abs path.

        Returns
        -------
        names: list
            The content of directory
        """

        if not os.path.isdir(self._path):
            return []
        if as_abs:
            return [os.path.join(self._path, f) for f in os.listdir(self._path)]
        return os.listdir(self._path)

    def destory(self):
        """Destory the dir."""

        if os.path.isdir(self._path):
            shutil.rmtree(self._path)

    @property
    def path(self):
        return self._path


def msc_dir(path: str = None, keep_history: bool = True, cleanup: bool = False) -> MSCDirectory:
    """Create MSCDirectory

    Parameters
    ----------
    path: str
        The path of the dir.
    keep_history: bool
        Whether to remove files before start.
    cleanup: bool
        Whether to clean up before exit.

    Returns
    -------
    dir: MSCDirectory
        The created dir.
    """

    return MSCDirectory(path, keep_history, cleanup)


def set_workspace(
    path: Union[str, MSCDirectory] = None, keep_history: bool = True, cleanup: bool = False
) -> MSCDirectory:
    """Create MSCDirectory as worksapce and set to map

    Parameters
    ----------
    path: str
        The path of the dir.
    keep_history: bool
        Whether to remove files before start.
    cleanup: bool
        Whether to clean up before exit.

    Returns
    -------
    dir: MSCDirectory
        The created dir.
    """

    if isinstance(path, MSCDirectory):
        MSCMap.set(MSCKey.WORKSPACE, path)
        return path
    path = path or "msc_workspace"
    workspace = MSCDirectory(path, keep_history, cleanup)
    MSCMap.set(MSCKey.WORKSPACE, workspace)
    return workspace


def get_workspace() -> MSCDirectory:
    """Get workspace from MSCMap

    Returns
    -------
    dir: MSCDirectory
        The worksapce dir.
    """

    workspace = MSCMap.get(MSCKey.WORKSPACE)
    assert workspace, "Can not find workspace, please call set_workspace"
    return workspace


def get_workspace_subdir(
    name: str = None, keep_history: bool = True, cleanup: bool = False
) -> MSCDirectory:
    """Create sub dir for workspace

    Parameters
    ----------
    name: str
        The sub dir name under workspace.
    keep_history: bool
        Whether to remove files before start.
    cleanup: bool
        Whether to clean up before exit.

    Returns
    -------
    dir: MSCDirectory
        The created dir.
    """

    return get_workspace().create_dir(name, keep_history, cleanup)


def to_abs_path(path: str, root_dir: MSCDirectory = None, keep_history: bool = True) -> str:
    """Change path to abs path

    Parameters
    ----------
    path: str
        The path of the file.
    root_dir: MSCDirectory
        Root dir to save the file.
    keep_history: bool
        Whether to remove files before start.

    Returns
    -------
    abs_path: str
        The abspath.
    """

    root_dir = root_dir or get_workspace()
    if os.path.abspath(path) == path:
        return path
    return root_dir.relpath(path, keep_history)


def pack_folder(path: str, style="tar"):
    """Pack the folder

    Parameters
    ----------
    path: str
        The path of the folder.
    style: str
        The pack style.

    Returns
    -------
    pack_path: str
        The packed path.
    """

    root = os.path.dirname(path)
    if style == "tar":
        cmd = "tar --exculde={0}.tar.gz -zcvf {0}.tar.gz {0} && rm -rf {0}".format(path)
    else:
        raise NotImplementedError("Pack style {} is not supported".format(style))
    if root:
        with msc_dir(root):
            retcode = subprocess.call(cmd, shell=True)
    else:
        retcode = subprocess.call(cmd, shell=True)
    assert retcode == 0, "Failed to pack the folder {}({}): {}".format(path, style, retcode)


get_build_dir = partial(get_workspace_subdir, name="Build")
get_cache_dir = partial(get_workspace_subdir, name="Cache")
get_config_dir = partial(get_workspace_subdir, name="Config")
get_dataset_dir = partial(get_workspace_subdir, name="Dataset")
get_gym_dir = partial(get_workspace_subdir, name="Gym")
get_output_dir = partial(get_workspace_subdir, name="Output")
get_visual_dir = partial(get_workspace_subdir, name="Visual")
get_weights_dir = partial(get_workspace_subdir, name="Weights")
