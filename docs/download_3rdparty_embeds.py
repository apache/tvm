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

# pylint: disable=redefined-outer-name, missing-module-docstring
import argparse
import hashlib
import os
import re
from html.parser import HTMLParser
from typing import Callable, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urlparse

import requests

# NOTE: This script is called by the Makefile via `make htmldepoly`.
# It's not called every time the docs are built on CI. However, it's
# can be only called during deployment stage, instead of building the docs.
# Also, we can download the resources manually before running this script to
# avoid the overhead of downloading the resources every time the docs are built.

# Set to store unique external URLs found during processing
BASE_URL = "https://tvm.apache.org"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HTML_DIR = os.path.join(SCRIPT_DIR, "_build/html")


class ExternalURLParser(HTMLParser):
    """HTML Parser to find external URLs in HTML content."""

    def __init__(self):
        super().__init__()
        self.external_urls: List[str] = []
        self.base_domain = urlparse(BASE_URL).netloc
        # Tags and their attributes that might contain external resources
        self.tags_to_check = {
            "img": "src",
            "script": "src",
            "iframe": "src",
            "video": "src",
            "audio": "src",
            "link": "href",
            "source": "src",
            "embed": "src",
        }

    def handle_starttag(self, tag: str, attrs: List[tuple[str, Union[str, None]]]) -> None:
        """Handle HTML start tags to find external URLs."""
        if tag not in self.tags_to_check:
            return

        attr_name = self.tags_to_check[tag]
        for name, value in attrs:
            if name != attr_name or not value:
                continue

            if value.startswith(("http://", "https://")):
                domain = urlparse(value).netloc
                if domain and domain != self.base_domain:
                    self.external_urls.append(value)


def detect_html_external_urls(html_content: str) -> List[str]:
    """
    Detect third-party embedded resources in HTML content.

    Parameters
    ----------
    html_content : str
        The HTML content to analyze

    Returns
    -------
    List[str]
        List of external URLs found in the HTML content
    """
    parser = ExternalURLParser()
    parser.feed(html_content)
    return parser.external_urls


def detect_css_external_urls(css_content: str) -> List[str]:
    """
    Detect external URLs in CSS content.

    Parameters
    ----------
    css_content : str
        The CSS content to analyze

    Returns
    -------
    List[str]
        List of external URLs found in the CSS content
    """
    external_urls: List[str] = []
    # Regex to find URLs in CSS
    url_pattern = re.compile(r'url\(["\']?(.*?)["\']?\)')
    matches = url_pattern.findall(css_content)
    for match in matches:
        if match.startswith(("http://", "https://")) and not match.startswith(BASE_URL):
            external_urls.append(match)
    return external_urls


def all_files_in_dir(path: str) -> List[str]:
    """
    Get a list of all files in a directory and its subdirectories.

    Parameters
    ----------
    path : str
        The root directory path to search

    Returns
    -------
    List[str]
        List of full paths to all files found
    """
    return [os.path.join(root, file) for root, _, files in os.walk(path) for file in files]


def detect_urls(files: List[str], verbose: bool = False) -> List[str]:
    """
    Detect external URLs in the given HTML and CSS files.

    Parameters
    ----------
    files : List[str]
        List of file paths to check for external URLs
    verbose : bool, optional
        Whether to print verbose output, by default False

    Returns
    -------
    List[str]
        List of external URLs found in the files
    """

    external_urls: Set[str] = set()
    for file in files:
        f_detect: Union[Callable[[str, str], List[str]], None] = None
        if file.endswith(".html"):
            f_detect = detect_html_external_urls
        elif file.endswith(".css"):
            f_detect = detect_css_external_urls
        else:
            continue
        with open(file, "r", encoding="utf-8") as f:
            content = f.read()
            urls = f_detect(content)
        if verbose:
            print(f"Processing {file}")
            exist_urls, new_urls = 0, 0
            for url in urls:
                if url in external_urls:
                    exist_urls += 1
                else:
                    new_urls += 1
                    if verbose:
                        print(f"Found new {url}")
            print(f"Found {exist_urls} existing resources and {new_urls} new resources")
        external_urls.update(urls)
    if verbose:
        print(f"Total {len(external_urls)} external resources")
        print("External resources:")
        print("\n".join(external_urls))

    return list(external_urls)


def download_external_urls(
    external_urls: List[str], verbose: bool = False
) -> Tuple[Dict[str, str], List[str]]:
    """
    Download external URLs and save them to docs/_static/downloads.

    Parameters
    ----------
    external_urls : List[str]
        List of external URLs to download
    verbose : bool, optional
        Whether to print verbose output, by default False

    Returns
    -------
    Tuple[Dict[str, str], List[str]]
        A tuple containing:
        - Dictionary mapping original URLs to their downloaded file paths
        - List of paths to all downloaded files (including source maps)
    """
    download_dir = os.path.join(HTML_DIR, "_static/downloads")
    os.makedirs(download_dir, exist_ok=True)
    used_file_names: Set[str] = set()
    downloaded_files: List[str] = []
    remap_urls: Dict[str, str] = {}
    for url in external_urls:
        query = urlparse(url).query
        if url.startswith("https://fonts.googleapis.com/css2"):
            file_name = f"{hashlib.md5(url.encode()).hexdigest()}.css"
        elif query:
            raise ValueError(f"Unsupported URL with query: {url}")
        else:
            file_name = urlparse(url).path.split("/")[-1]
        if verbose:
            print(f"remapping {url} to {file_name}")
        if file_name in used_file_names:
            raise ValueError(f"File name {file_name} already exists")
        used_file_names.add(file_name)
        response = requests.get(url, timeout=30)
        body = response.content
        with open(os.path.join(download_dir, file_name), "wb") as f:
            f.write(body)
        remap_urls[url] = os.path.join(download_dir, file_name)
        downloaded_files.append(os.path.join(download_dir, file_name))

        # Also download the sourceMappingURL
        if not url.startswith("https://fonts.googleapis.com/css2"):
            map_file_name = f"{file_name}.map"
            response = requests.get(f"{url}.map", timeout=30)
            if response.status_code == 200:
                body = response.content
                with open(os.path.join(download_dir, map_file_name), "wb") as f:
                    f.write(body)
                    if verbose:
                        print(f"Downloaded {map_file_name} for {url}")
                downloaded_files.append(os.path.join(download_dir, map_file_name))

    return remap_urls, downloaded_files


def replace_urls_in_files(remap_urls: Dict[str, str], verbose: bool = False):
    """
    Replace external URLs with their downloaded versions in HTML/CSS files.

    Parameters
    ----------
    remap_urls : Dict[str, str]
        Dictionary mapping original URLs to their downloaded file paths
    verbose : bool, optional
        Whether to print verbose output, by default False
    """
    for root, _, files in os.walk(HTML_DIR):
        for file in files:
            if not (file.endswith(".html") or file.endswith(".css")):
                continue

            file_path = os.path.join(root, file)
            if verbose:
                print(f"Processing {file_path}")

            # Calculate relative path from current file to _static/downloads
            rel_path = os.path.relpath(
                os.path.join(HTML_DIR, "_static/downloads"), os.path.dirname(file_path)
            )

            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            new_content = content
            for original_url, new_path in remap_urls.items():
                relative_url = os.path.join(rel_path, os.path.basename(new_path))
                new_content = new_content.replace(original_url, relative_url)

            if new_content != content:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(new_content)
                if verbose:
                    print(f"Updated {file_path}")


def download_and_replace_urls(files: Optional[List[str]] = None, verbose: bool = False):
    """
    Download external URLs found in files and replace them with local copies.
    Recursively processes any new external URLs found in downloaded content.

    Parameters
    ----------
    files : Optional[List[str]], optional
        List of files to check for external URLs. If None, checks all files under HTML_DIR
    verbose : bool, optional
        Whether to print verbose output, by default False
    """
    if files is None:
        files = all_files_in_dir(HTML_DIR)
    remap_urls = {}
    while True:
        external_urls = detect_urls(files, verbose=verbose)
        if not external_urls:
            break
        round_remap_urls, files = download_external_urls(external_urls, verbose=verbose)
        remap_urls.update(round_remap_urls)

    replace_urls_in_files(remap_urls, verbose=verbose)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-v", "--verbose", action="store_true")
    args = args.parse_args()
    download_and_replace_urls(verbose=args.verbose)
