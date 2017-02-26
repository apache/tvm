"""Utilities to make tempdir"""
from __future__ import absolute_import as _abs
import os
import tempfile
import shutil

class TempDirectory(object):
    """Helper object to manage temp directory during testing"""
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()

    def __del__(self):
        shutil.rmtree(self.temp_dir)

    def relpath(self, name):
        """Relative path in temp dir

        Parameters
        ----------
        name : str
            The name of the file.
        """
        return os.path.join(self.temp_dir, name)


def tempdir():
    """Return a new temp dir which deletes the contents when exit

    Returns
    -------
    temp : TempDirectory
        The temp directory object
    """
    return TempDirectory()
