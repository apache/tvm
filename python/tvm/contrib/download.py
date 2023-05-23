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
# pylint: disable=missing-timeout
"""Helper utility for downloading"""

import logging
import os
from pathlib import Path
import shutil
import tempfile
import time

LOG = logging.getLogger("download")


def download(url, path, overwrite=False, size_compare=False, retries=3):
    """Downloads the file from the internet.
    Set the input options correctly to overwrite or do the size comparison

    Parameters
    ----------
    url : str
        Download url.

    path : str
        Local file path to save downloaded file.

    overwrite : bool, optional
        Whether to overwrite existing file, defaults to False.

    size_compare : bool, optional
        Whether to do size compare to check downloaded file, defaults
        to False

    retries: int, optional
        Number of time to retry download, defaults to 3.

    """
    # pylint: disable=import-outside-toplevel
    import urllib.request as urllib2

    path = Path(path).resolve()
    if path.exists() and path.is_file() and not overwrite:
        if size_compare:
            import requests

            file_size = path.stat().st_size
            res_head = requests.head(url)
            res_get = requests.get(url, stream=True)
            if "Content-Length" not in res_head.headers:
                res_get = urllib2.urlopen(url)
            url_file_size = int(res_get.headers["Content-Length"])
            if url_file_size != file_size:
                LOG.warning("Existing file %s has incorrect size, downloading fresh copy", path)
                download(url, path, overwrite=True, size_compare=False, retries=retries)
                return

        LOG.info("File %s exists, skipping.", path)
        return

    LOG.info("Downloading from url %s to %s", url, path)

    # Stateful start time
    start_time = time.time()
    dirpath = path.parent
    dirpath.mkdir(parents=True, exist_ok=True)

    def _download_progress(count, block_size, total_size):
        # pylint: disable=unused-argument
        """Show the download progress."""
        if count == 0:
            return
        duration = time.time() - start_time
        progress_bytes = int(count * block_size)
        progress_megabytes = progress_bytes / (1024.0 * 1024)
        speed_kbps = int(progress_bytes / (1024 * duration))
        percent = min(int(count * block_size * 100 / total_size), 100)

        # Temporarily suppress newlines on the output stream.
        prev_terminator = logging.StreamHandler.terminator
        logging.StreamHandler.terminator = ""
        LOG.debug(
            "\r...%d%%, %.2f MB, %d KB/s, %d seconds passed",
            percent,
            progress_megabytes,
            speed_kbps,
            duration,
        )
        logging.StreamHandler.terminator = prev_terminator

    with tempfile.TemporaryDirectory() as tempdir:
        tempdir = Path(tempdir)
        download_loc = tempdir.joinpath(path.name)

        for i_retry in range(retries):
            # pylint: disable=broad-except
            try:

                urllib2.urlretrieve(url, download_loc, reporthook=_download_progress)
                LOG.debug("")
                try:
                    download_loc.rename(path)
                except OSError:
                    # Prefer a move, but if the tempdir and final
                    # location are in different drives, fall back to a
                    # copy.
                    shutil.copy2(download_loc, path)
                return

            except Exception as err:
                if i_retry == retries - 1:
                    raise err

                LOG.warning(
                    "%s\nDownload attempt %d/%d failed, retrying.", repr(err), i_retry, retries
                )


if "TEST_DATA_ROOT_PATH" in os.environ:
    TEST_DATA_ROOT_PATH = Path(os.environ.get("TEST_DATA_ROOT_PATH"))
else:
    TEST_DATA_ROOT_PATH = Path(Path("~").expanduser(), ".tvm_test_data")
TEST_DATA_ROOT_PATH.mkdir(parents=True, exist_ok=True)


def download_testdata(url, relpath, module=None, overwrite=False):
    """Downloads the test data from the internet.

    Parameters
    ----------
    url : str
        Download url.

    relpath : str
        Relative file path.

    module : Union[str, list, tuple], optional
        Subdirectory paths under test data folder.

    overwrite : bool, defaults to False
        If True, will download a fresh copy of the file regardless of
        the cache.  If False, will only download the file if a cached
        version is missing.

    Returns
    -------
    abspath : str
        Absolute file path of downloaded file

    """
    global TEST_DATA_ROOT_PATH
    if module is None:
        module_path = ""
    elif isinstance(module, str):
        module_path = module
    elif isinstance(module, (list, tuple)):
        module_path = Path(*module)
    else:
        raise ValueError("Unsupported module: " + module)
    abspath = Path(TEST_DATA_ROOT_PATH, module_path, relpath)
    download(url, abspath, overwrite=overwrite, size_compare=False)
    return str(abspath)
