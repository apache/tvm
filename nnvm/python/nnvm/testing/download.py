# pylint: disable=invalid-name, no-member, import-error, no-name-in-module, global-variable-undefined, bare-except
"""Helper utility for downloading"""
from __future__ import print_function
from __future__ import absolute_import as _abs

import os
import sys
import time
import urllib
import requests

if sys.version_info >= (3,):
    import urllib.request as urllib2
else:
    import urllib2

def _download_progress(count, block_size, total_size):
    """Show the download progress.
    """
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                     (percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()

def download(url, path, overwrite=False, size_compare=False):
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
    """
    if os.path.isfile(path) and not overwrite:
        if size_compare:
            file_size = os.path.getsize(path)
            res_head = requests.head(url)
            res_get = requests.get(url, stream=True)
            if 'Content-Length' not in res_head.headers:
                res_get = urllib2.urlopen(url)
            url_file_size = int(res_get.headers['Content-Length'])
            if url_file_size != file_size:
                print("exist file got corrupted, downloading %s file freshly..." % path)
                download(url, path, True, False)
                return
        print('File {} exists, skip.'.format(path))
        return
    print('Downloading from url {} to {}'.format(url, path))
    try:
        urllib.request.urlretrieve(url, path, reporthook=_download_progress)
        print('')
    except:
        urllib.urlretrieve(url, path, reporthook=_download_progress)
