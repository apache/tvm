"""Helper utility for downloading"""
from __future__ import print_function
from __future__ import absolute_import as _abs

import os
import sys
import time

def download(url, path, overwrite=False, size_compare=False, verbose=1):
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
    """
    if sys.version_info >= (3,):
        import urllib.request as urllib2
    else:
        import urllib2

    if os.path.isfile(path) and not overwrite:
        if size_compare:
            import requests
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

    if verbose >= 1:
        print('Downloading from url {} to {}'.format(url, path))

    # Stateful start time
    start_time = time.time()

    def _download_progress(count, block_size, total_size):
        #pylint: disable=unused-argument
        """Show the download progress.
        """
        if count == 0:
            return
        duration = time.time() - start_time
        progress_size = int(count * block_size)
        speed = int(progress_size / (1024 * duration))
        percent = min(int(count * block_size * 100 / total_size), 100)
        sys.stdout.write("\r...%d%%, %.2f MB, %d KB/s, %d seconds passed" %
                         (percent, progress_size / (1024.0 * 1024), speed, duration))
        sys.stdout.flush()

    if sys.version_info >= (3,):
        urllib2.urlretrieve(url, path, reporthook=_download_progress)
        print("")
    else:
        f = urllib2.urlopen(url)
        data = f.read()
        with open(path, "wb") as code:
            code.write(data)
