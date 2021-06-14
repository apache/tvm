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
"""Helper utility for downloading"""
from pathlib import Path
from os import environ
import sys
import time
import uuid
import shutil


def download(url, path, overwrite=False, size_compare=False, verbose=1, retries=3):
    """Downloads the file from the internet.
    Set the input options correctly to overwrite or do the size comparison

    Parameters
    ----------
    url : str
        Download url.

    path : str
        Local file path to save downloaded file

    overwrite : bool, optional
        Whether to overwrite existing file

    size_compare : bool, optional
        Whether to do size compare to check downloaded file.

    verbose: int, optional
        Verbose level

    retries: int, optional
        Number of time to retry download, default at 3.
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
                print("exist file got corrupted, downloading %s file freshly..." % path)
                download(url, path, True, False)
                return
        print("File {} exists, skip.".format(path))
        return

    if verbose >= 1:
        print("Downloading from url {} to {}".format(url, path))

    # Stateful start time
    start_time = time.time()
    dirpath = path.parent
    dirpath.mkdir(parents=True, exist_ok=True)
    random_uuid = str(uuid.uuid4())
    tempfile = Path(dirpath, random_uuid)

    def _download_progress(count, block_size, total_size):
        # pylint: disable=unused-argument
        """Show the download progress."""
        if count == 0:
            return
        duration = time.time() - start_time
        progress_size = int(count * block_size)
        speed = int(progress_size / (1024 * duration))
        percent = min(int(count * block_size * 100 / total_size), 100)
        sys.stdout.write(
            "\r...%d%%, %.2f MB, %d KB/s, %d seconds passed"
            % (percent, progress_size / (1024.0 * 1024), speed, duration)
        )
        sys.stdout.flush()

    while retries >= 0:
        # Disable pyling too broad Exception
        # pylint: disable=W0703
        try:
            if sys.version_info >= (3,):
                urllib2.urlretrieve(url, tempfile, reporthook=_download_progress)
                print("")
            else:
                f = urllib2.urlopen(url)
                data = f.read()
                with open(tempfile, "wb") as code:
                    code.write(data)
            shutil.move(tempfile, path)
            break
        except Exception as err:
            retries -= 1
            if retries == 0:
                if tempfile.exists():
                    tempfile.unlink()
                raise err
            print(
                "download failed due to {}, retrying, {} attempt{} left".format(
                    repr(err), retries, "s" if retries > 1 else ""
                )
            )


if "TEST_DATA_ROOT_PATH" in environ:
    TEST_DATA_ROOT_PATH = Path(environ.get("TEST_DATA_ROOT_PATH"))
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
