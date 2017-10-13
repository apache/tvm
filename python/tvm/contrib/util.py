"""Common system utilities"""
from __future__ import absolute_import as _abs
import os
import tempfile
import shutil

class TempDirectory(object):
    """Helper object to manage temp directory during testing.

    Automatically removes the directory when it went out of scope.
    """
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        self._rmtree = shutil.rmtree

    def remove(self):
        """Remote the tmp dir"""
        if self.temp_dir:
            self._rmtree(self.temp_dir, ignore_errors=True)
            self.temp_dir = None

    def __del__(self):
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
        """"List contents in the dir.

        Returns
        -------
        names : list
            The content of directory
        """
        return os.listdir(self.temp_dir)

def tempdir():
    """Create temp dir which deletes the contents when exit.

    Returns
    -------
    temp : TempDirectory
        The temp directory object
    """
    return TempDirectory()
