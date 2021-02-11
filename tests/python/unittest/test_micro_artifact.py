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

"""Unit tests for the artifact module."""

import pytest
import json
import os
import shutil
import tvm

from tvm.contrib import utils

pytest.importorskip("tvm.micro")
from tvm.micro import artifact

FILE_LIST = ["label1", "label2", "label12", "unlabelled"]


TEST_METADATA = {"foo": "bar"}


TEST_LABELS = {"label1": ["label1", "label12"], "label2": ["label2", "label12"]}


def build_artifact(artifact_path, immobile=False):
    os.mkdir(artifact_path)

    for f in FILE_LIST:
        with open(os.path.join(artifact_path, f), "w") as lib_f:
            lib_f.write(f"{f}\n")

    sub_dir = os.path.join(artifact_path, "sub_dir")
    os.mkdir(sub_dir)
    os.symlink("label1", os.path.join(artifact_path, "rel_symlink"))
    os.symlink("label2", os.path.join(artifact_path, "abs_symlink"), "label2")
    os.symlink(
        os.path.join(artifact_path, "sub_dir"), os.path.join(artifact_path, "abs_dir_symlink")
    )

    from tvm.micro import artifact

    art = artifact.Artifact(artifact_path, TEST_LABELS, TEST_METADATA, immobile=immobile)

    return art


@tvm.testing.requires_micro
def test_basic_functionality():
    temp_dir = utils.tempdir()
    artifact_path = temp_dir.relpath("foo")
    art = build_artifact(artifact_path)

    assert art.abspath("bar") == os.path.join(artifact_path, "bar")

    for label, paths in TEST_LABELS.items():
        assert art.label(label) == paths
        assert art.label_abspath(label) == [os.path.join(artifact_path, p) for p in paths]


@tvm.testing.requires_micro
def test_archive():
    from tvm.micro import artifact

    temp_dir = utils.tempdir()
    art = build_artifact(temp_dir.relpath("foo"))

    # Create archive
    archive_path = art.archive(temp_dir.temp_dir)
    assert archive_path == temp_dir.relpath("foo.tar")

    # Inspect created archive
    unpack_dir = temp_dir.relpath("unpack")
    os.mkdir(unpack_dir)
    shutil.unpack_archive(archive_path, unpack_dir)

    for path in FILE_LIST:
        with open(os.path.join(unpack_dir, "foo", path)) as f:
            assert f.read() == f"{path}\n"

    with open(os.path.join(unpack_dir, "foo", "metadata.json")) as metadata_f:
        metadata = json.load(metadata_f)

    assert metadata["version"] == 2
    assert metadata["labelled_files"] == TEST_LABELS
    assert metadata["metadata"] == TEST_METADATA

    # Unarchive and verify basic functionality
    unarchive_base_dir = temp_dir.relpath("unarchive")
    unarch = artifact.Artifact.unarchive(archive_path, unarchive_base_dir)

    assert unarch.metadata == TEST_METADATA
    assert unarch.labelled_files == TEST_LABELS
    for f in FILE_LIST:
        assert os.path.exists(os.path.join(unarchive_base_dir, f))


@tvm.testing.requires_micro
def test_metadata_only():
    from tvm.micro import artifact

    temp_dir = utils.tempdir()
    base_dir = temp_dir.relpath("foo")
    art = build_artifact(base_dir)

    artifact_path = art.archive(temp_dir.relpath("foo.artifact"), metadata_only=True)
    unarch_base_dir = temp_dir.relpath("bar")
    unarch = artifact.Artifact.unarchive(artifact_path, unarch_base_dir)
    assert unarch.base_dir == base_dir

    for p in unarch.label_abspath("label1") + unarch.label_abspath("label2"):
        assert os.path.exists(p)

    os.unlink(art.abspath("label1"))
    with open(art.abspath("label2"), "w+") as f:
        f.write("changed line\n")

    try:
        artifact.Artifact.unarchive(artifact_path, os.path.join(temp_dir.temp_dir, "bar2"))
        assert False, "unarchive should raise error"
    except artifact.ArchiveModifiedError as err:
        assert str(err) == (
            "Files in metadata-only archive have been modified:\n"
            " * label1: original file not found\n"
            " * label2: sha256 mismatch: expected "
            "6aa3c5668c8794c791400e19ecd7123949ded1616eafb0395acdd2d896354e83, got "
            "ed87db21670a81819d65eccde87c5ae0243b2b61783bf77e9b27993be9a3eca0"
        )


if __name__ == "__main__":
    test_basic_functionality()
    test_archive()
    test_metadata_only()
    # TODO: tests for dir symlinks, symlinks out of bounds, loading malformed artifact tars.
