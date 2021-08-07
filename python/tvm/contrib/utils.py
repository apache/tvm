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
"""Common system utilities"""
import atexit
import contextlib
import datetime
import os
import pathlib
import tempfile
import threading
import shutil

try:
    import fcntl
except ImportError:
    fcntl = None


class DirectoryCreatedPastAtExit(Exception):
    """Raised when a TempDirectory is created after the atexit hook runs."""


class TempDirectory(object):
    """Helper object to manage temp directory during testing.

    Automatically removes the directory when it went out of scope.
    """

    # When True, all TempDirectory are *NOT* deleted and instead live inside a predicable directory
    # tree.
    _KEEP_FOR_DEBUG = False

    # In debug mode, each tempdir is named after the sequence
    _NUM_TEMPDIR_CREATED = 0
    _NUM_TEMPDIR_CREATED_LOCK = threading.Lock()

    @classmethod
    def _increment_num_tempdir_created(cls):
        with cls._NUM_TEMPDIR_CREATED_LOCK:
            to_return = cls._NUM_TEMPDIR_CREATED
            cls._NUM_TEMPDIR_CREATED += 1

        return to_return

    _DEBUG_PARENT_DIR = None

    @classmethod
    def _get_debug_parent_dir(cls):
        if cls._DEBUG_PARENT_DIR is None:
            all_parents = f"{tempfile.gettempdir()}/tvm-debug-mode-tempdirs"
            if not os.path.isdir(all_parents):
                os.makedirs(all_parents)
            cls._DEBUG_PARENT_DIR = tempfile.mkdtemp(
                prefix=datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S___"), dir=all_parents
            )
        return cls._DEBUG_PARENT_DIR

    TEMPDIRS = set()

    @classmethod
    def remove_tempdirs(cls):
        temp_dirs = getattr(cls, "TEMPDIRS", None)
        if temp_dirs is None:
            return

        for path in temp_dirs:
            shutil.rmtree(path, ignore_errors=True)

        cls.TEMPDIRS = None

    @classmethod
    @contextlib.contextmanager
    def set_keep_for_debug(cls, set_to=True):
        """Keep temporary directories past program exit for debugging."""
        old_keep_for_debug = cls._KEEP_FOR_DEBUG
        try:
            cls._KEEP_FOR_DEBUG = set_to
            yield
        finally:
            cls._KEEP_FOR_DEBUG = old_keep_for_debug

    def __init__(self, custom_path=None):
        if self.TEMPDIRS is None:
            raise DirectoryCreatedPastAtExit()

        self._created_with_keep_for_debug = self._KEEP_FOR_DEBUG
        if custom_path:
            os.mkdir(custom_path)
            self.temp_dir = custom_path
        else:
            if self._created_with_keep_for_debug:
                parent_dir = self._get_debug_parent_dir()
                self.temp_dir = f"{parent_dir}/{self._increment_num_tempdir_created():05d}"
                os.mkdir(self.temp_dir)
            else:
                self.temp_dir = tempfile.mkdtemp()

        if not self._created_with_keep_for_debug:
            self.TEMPDIRS.add(self.temp_dir)

    def remove(self):
        """Remove the tmp dir"""
        if self.temp_dir:
            if not self._created_with_keep_for_debug:
                shutil.rmtree(self.temp_dir, ignore_errors=True)
                self.TEMPDIRS.remove(self.temp_dir)
            self.temp_dir = None

    @property
    def path(self):
        return pathlib.Path(self.temp_dir)

    def __truediv__(self, other):
        if not isinstance(other, (str, pathlib.Path)):
            raise TypeError(
                "TempDirectory / operator: must supply str or pathlib.Path; got %r" % (other,)
            )

        return self.path / other

    def __del__(self):
        temp_dirs = getattr(self, "TEMPDIRS", None)
        if temp_dirs is None:
            # Do nothing if the atexit hook has already run.
            return

        self.remove()

    def relpath(self, name):
        """Relative path in temp dir

        Parameters
        ----------
        name : str
            The name of the file.

        Returns
        -------
        path : str
            The concatenated path.
        """
        return os.path.join(self.temp_dir, name)

    def listdir(self):
        """List contents in the dir.

        Returns
        -------
        names : list
            The content of directory
        """
        return os.listdir(self.temp_dir)


atexit.register(TempDirectory.remove_tempdirs)


def tempdir(custom_path=None):
    """Create temp dir which deletes the contents when exit.

    Parameters
    ----------
    custom_path : str, optional
        Manually specify the exact temp dir path

    Returns
    -------
    temp : TempDirectory
        The temp directory object
    """
    return TempDirectory(custom_path)


class FileLock(object):
    """File lock object

    Parameters
    ----------
    path : str
        The path to the lock
    """

    def __init__(self, path):
        self.lock_file = open(path, "w")
        if fcntl:
            fcntl.lockf(self.lock_file, fcntl.LOCK_EX)

    def release(self):
        """Release the lock"""
        if self.lock_file:
            if fcntl:
                fcntl.lockf(self.lock_file, fcntl.LOCK_UN)
            self.lock_file.close()
            self.lock_file = None


def filelock(path):
    """Create a file lock which locks on path

    Parameters
    ----------
    path : str
        The path to the lock

    Returns
    -------
    lock : File lock object
    """
    return FileLock(path)


def is_source_path(path):
    """Check if path is source code path.

    Parameters
    ----------
    path : str
        A possible path

    Returns
    -------
    valid : bool
        Whether path is a possible source path
    """
    if os.path.exists(path):
        return True
    if path.find("\n") != -1:
        return False
    spath = path.rsplit(".", 1)
    return len(spath) == 2 and spath[1].strip() == spath[1]


def which(exec_name):
    """Try to find full path of exec_name

    Parameters
    ----------
    exec_name : str
        The executable name

    Returns
    -------
    path : str
        The full path of executable if found, otherwise returns None
    """
    base_list = ["", "/bin"] + os.environ.get("PATH", "").split(os.pathsep)
    for path in base_list:
        full_path = os.path.join(path, exec_name)
        if os.path.isfile(full_path) and os.access(full_path, os.X_OK):
            return full_path
    return None
