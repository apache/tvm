#!/usr/bin/env python3
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

"""TVM Python requirements.txt generator.

This script generates a set of requirements.txt files (stored in `./requirements`) that describe
TVM's Python dependencies.

## Pieces

TVM can be roughly broken into these named pieces along the lines of Python dependencies:

- "core": A core piece, which is intended to be buildable with very few external dependencies. Users
  can use Relay, compile models, and run autotuning with this part.
- "importer-<tool>": Model importers, which convert models defined in various other tools (i.e.
  TensorFlow, PyTorch, etc) into Relay models.
- Extra features (i.e. XGBoost in AutoTVM). These enhance TVM's functionality, but aren't required
  for basic operation.

## What this tool does

From these pieces, this tool builds:
 - requirements/<name>.txt - Python dependencies for each named piece above, `<name>` is the same as
   the quoted piece name.
 - requirements/all.txt - Consolidated Python dependencies for all pieces, excluding dev below.
 - requirements/dev.txt - Python dependencies needed to develop TVM, such as lint and test tools.

The data representing each piece is contained in the two maps below.
"""

import argparse
import collections
import os
import re
import sys
import textwrap
import typing

RequirementsByPieceType = typing.List[typing.Tuple[str, typing.Tuple[str, typing.List[str]]]]


# Maps named TVM piece (see description above) to a list of names of Python packages. Please use
# alphabetical order for each package list, and do not add version constraints here!
REQUIREMENTS_BY_PIECE: RequirementsByPieceType = [
    # Base requirements needed to install tvm.
    (
        "core",
        (
            "Base requirements needed to install tvm",
            [
                "attrs",
                "cloudpickle",
                "decorator",
                "ml_dtypes",
                "numpy",
                "packaging",
                "psutil",
                "scipy",
                "tornado",
                "typing_extensions",
            ],
        ),
    ),
    # Provide support for Arm(R) Ethos(TM)-U NPU.
    (
        "ethosu",
        (
            "Requirements for using Arm(R) Ethos(TM)-U NPU",
            [
                "ethos-u-vela",
            ],
        ),
    ),
    # Relay frontends.
    (
        "importer-caffe",
        (
            "Requirements for the Caffe importer",
            [
                "numpy",
                "protobuf",
                "scikit-image",
                "six",
            ],
        ),
    ),
    (
        "importer-caffe2",
        (
            "Requirements for the Caffe2 importer",
            [
                "future",  # Hidden dependency of torch.
                "torch",
            ],
        ),
    ),
    ("importer-coreml", ("Requirements for the CoreML importer", ["coremltools"])),
    ("importer-darknet", ("Requirements for the DarkNet importer", ["opencv-python"])),
    (
        "importer-keras",
        ("Requirements for the Keras importer", ["tensorflow", "tensorflow-estimator"]),
    ),
    (
        "importer-onnx",
        (
            "Requirements for the ONNX importer",
            [
                "future",  # Hidden dependency of torch.
                "onnx",
                "onnxoptimizer",
                "onnxruntime",
                "torch",
                "torchvision",
            ],
        ),
    ),
    (
        "importer-paddle",
        ("Requirements for the PaddlePaddle importer", ["paddlepaddle"]),
    ),
    (
        "importer-pytorch",
        (
            "Requirements for the PyTorch importer",
            [
                "future",  # Hidden dependency of torch.
                "torch",
                "torchvision",
            ],
        ),
    ),
    (
        "importer-tensorflow",
        ("Requirements for the TensorFlow importer", ["tensorflow", "tensorflow-estimator"]),
    ),
    (
        "importer-tflite",
        ("Requirements for the TFLite importer", ["tensorflow", "tensorflow-estimator", "tflite"]),
    ),
    (
        "tvmc",
        (
            "Requirements for the tvmc command-line tool",
            [
                "ethos-u-vela",
                "future",  # Hidden dependency of torch.
                "onnx",
                "onnxoptimizer",
                "onnxruntime",
                "paddlepaddle",
                "tensorflow",
                "tflite",
                "torch",
                "torchvision",
                "xgboost",
            ],
        ),
    ),
    # Vitis AI requirements
    (
        "vitis-ai",
        (
            "Requirements for the Vitis AI codegen",
            [
                "h5py",
                "progressbar",
            ],
        ),
    ),
    # XGBoost, useful for autotuning on some targets.
    (
        "xgboost",
        (
            "Requirements for XGBoost autotuning",
            [
                "future",  # Hidden dependency of torch.
                "torch",
                "xgboost",
            ],
        ),
    ),
    # Development requirements
    (
        "dev",
        (
            "Requirements to develop TVM -- lint, docs, testing, etc.",
            [
                "astroid",  # pylint requirement, listed so a hard constraint can be included.
                "autodocsumm",
                "black",
                "commonmark",
                "cpplint",
                "docutils",
                "image",
                "matplotlib",
                "pillow",
                "pylint",
                "sphinx",
                "sphinx_autodoc_annotation",
                "sphinx_gallery",
                "sphinx_rtd_theme",
                "types-psutil",
            ],
        ),
    ),
]

ConstraintsType = typing.List[typing.Tuple[str, typing.Union[None, str]]]

# Maps a named Python package (which should appear in REQUIREMENTS_BY_PIECE above) to a
# semver or pip version constraint. Semver constraints are translated into requirements.txt-friendly
# constraints.
#
# These constraints serve only to record technical reasons why a particular version can't be used.
# They are the default install_requires used in setup.py. These can be further narrowed to restrict
# dependencies to those tested or used in CI; however, that process is not done here.
#
# Policy for constraints listed here:
# 1. Each package specified in REQUIREMENTS_BY_PIECE must be included here.
# 2. If TVM will functionally break against an old version of a dependency, specify a >= relation
#    here. Include a comment linking to context or explaining why the constraint is in place.
CONSTRAINTS = [
    ("astroid", None),
    ("attrs", None),
    ("autodocsumm", None),
    ("black", "==20.8b1"),
    ("cloudpickle", None),
    ("commonmark", ">=0.7.3"),  # From PR #213.
    ("coremltools", None),
    ("cpplint", None),
    ("decorator", None),
    (
        "docutils",
        "<0.17",
    ),  # Work around https://github.com/readthedocs/sphinx_rtd_theme/issues/1115
    ("ethos-u-vela", "==3.8.0"),
    ("future", None),
    ("h5py", "==2.10.0"),
    ("image", None),
    ("matplotlib", None),
    ("numpy", None),
    ("onnx", None),
    ("onnxoptimizer", None),
    ("onnxruntime", None),
    ("opencv-python", None),
    ("paddlepaddle", None),
    ("pillow", None),
    ("progressbar", None),
    ("protobuf", None),
    ("psutil", None),
    ("pylint", None),
    ("scikit-image", None),
    ("scipy", None),
    ("six", None),
    ("sphinx", None),
    ("sphinx_autodoc_annotation", None),
    ("sphinx_gallery", None),
    ("sphinx_rtd_theme", None),
    ("tensorflow", None),
    ("tensorflow-estimator", None),
    ("tflite", None),
    ("torch", None),
    ("torchvision", None),
    ("tornado", None),
    ("typing_extensions", None),
    ("xgboost", ">=1.1.0"),  # From PR #4953 & Issue #12009
]

################################################################################
# End of configuration options.
################################################################################


# Required keys in REQUIREMENTS_BY_PIECE.
REQUIRED_PIECES: typing.List[str] = ["core", "dev"]

# Regex to validates piece names.
PIECE_REGEX: typing.Pattern = re.compile(r"^[a-z0-9][a-z0-9-]*", re.IGNORECASE)

# Regex to match a constraint specification. Multiple constraints are not supported.
CONSTRAINT_REGEX: typing.Pattern = re.compile(r"(?:\^|\<|(?:~=)|(?:<=)|(?:==)|(?:>=)|\>)[^<>=\^,]+")

# Regex for parsing semantic versions. See
# https://semver.org/#is-there-a-suggested-regular-expression-regex-to-check-a-semver-string
SEMVER_REGEX: typing.Pattern = re.compile(
    r"^(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)(?:-(?P<prerelease>(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?(?:\+(?P<buildmetadata>[0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"
)


def validate_requirements_by_piece() -> typing.List[str]:
    """Validate REQUIREMENTS_BY_PIECE, returning a list of problems.

    Returns
    -------
    list[str] :
        A list of strings, each one describing a distinct problem with REQUIREMENTS_BY_PIECE.
    """
    problems = []

    unseen_required_pieces = set(REQUIRED_PIECES)
    seen_pieces = set()

    # Ensure that core is listed first and dev is listed last.
    saw_core = False
    saw_dev = False

    if not isinstance(REQUIREMENTS_BY_PIECE, (list, tuple)):
        problems.append(f"must be list or tuple, see {REQUIREMENTS_BY_PIECE!r}")
        return problems

    for piece, value in REQUIREMENTS_BY_PIECE:
        if not isinstance(piece, str):
            problems.append(f"piece {piece!r}: must be str")
            continue

        if piece in unseen_required_pieces:
            unseen_required_pieces.remove(piece)

        piece_lower = piece.lower()
        if piece_lower in seen_pieces:
            problems.append(f"piece {piece}: listed twice")

        seen_pieces.add(piece_lower)

        if not saw_core and piece != "core":
            problems.append(f'piece {piece}: must list after "core" (core must be first)')
        elif piece == "core":
            saw_core = True

        if saw_dev:
            problems.append(f'piece {piece}: must list before "dev" (dev must be last)')
        elif piece == "dev":
            saw_dev = True

        if not isinstance(value, (tuple, list)) or len(value) != 2:
            problems.append(
                f'piece {piece}: should be formatted like ("{piece}", ("<requirements.txt comment>", ["dep1", "dep2", ...])). got: {value!r}'
            )
            continue

        description, deps = value

        if not isinstance(description, str):
            problems.append(f"piece {piece}: description should be a string, got {description!r}")

        if not isinstance(deps, (list, tuple)) or any(not isinstance(d, str) for d in deps):
            problems.append(f"piece {piece}: deps should be a list of strings, got {deps!r}")
            continue

        if list(sorted(deps)) != list(deps):
            problems.append(
                f"piece {piece}: deps must be sorted. Correct order:\n  {list(sorted(deps))!r}"
            )

        piece_deps = set()
        for d in deps:
            if CONSTRAINT_REGEX.search(d):
                problems.append(
                    f"piece {piece}: dependency {d} should not specify a version. "
                    "Add it to CONSTRAINTS instead."
                )

            if d.lower() in piece_deps:
                problems.append(f"piece {piece}: dependency {d} listed twice")

            piece_deps.add(d.lower())

    extras_pieces = [
        k for (k, _) in REQUIREMENTS_BY_PIECE if k not in ("dev", "core") if isinstance(k, str)
    ]
    sorted_extras_pieces = list(sorted(extras_pieces))
    if sorted_extras_pieces != list(extras_pieces):
        problems.append(
            'pieces other than "core" and "dev" must appear in alphabetical order: '
            f"{sorted_extras_pieces}"
        )

    return problems


def parse_semver(
    package: str, constraint: str, problems: typing.List[str]
) -> typing.Tuple[typing.List[str], int, int]:
    """Parse a semantic versioning constraint of the form "^X.[.Y[.Z[...]]]]"

    Parameters
    ----------
    package : str
        Name of the package specifying this constraint, for reporting problems.
    constraint : str
        The semver constraint. Must start with "^"
    problems : List[str]
        A list of strings describing problems that have occurred validating the configuration.
        Problems encountered while validating constraint are appended to this list.

    Returns
    -------
    tuple[list[str], int, int] :
        A 3-tuple. The first element is a list containing an entry for each component in the
        semver string (components separated by "."). The second element is the index of the
        component in the list which must not change to meet the semver constraint. The third element
        is an integer, the numeric value of the changing component (this can be non-trivial when
        the patch is the changing part but pre-, post-release, or build metadta.

        See "Caret requirements" at https://python-poetry.org/docs/versions/.
    """
    m = SEMVER_REGEX.match(constraint[1:])
    if not m:
        problems.append(f"{package}: invalid semver constraint {constraint}")
        return [], 0, 0

    min_ver_parts = [
        m.group("major"),
        m.group("minor"),
        m.group("patch")
        + (f"-{m.group('prerelease')}" if m.group("prerelease") else "")
        + (f"+{m.group('buildmetadata')}" if m.group("buildmetadata") else ""),
    ]

    # Major/minor version handling is simple
    for i, p in enumerate(min_ver_parts[:2]):
        x = int(p.strip())
        if x:
            return min_ver_parts, i, x

    # For patch version, consult only the numeric patch
    if m.group("patch"):
        patch_int = int(m.group("patch"))
        if patch_int or min_ver_parts[2] != m.group("patch"):
            return min_ver_parts, 2, patch_int

    # All 0's
    return min_ver_parts, 0, 0


def validate_constraints() -> typing.List[str]:
    """Validate CONSTRAINTS, returning a list of problems found.

    Returns
    -------
    list[str] :
        A list of strings, each one describing a distinct problem found in CONSTRAINTS.
    """
    problems = []

    if not isinstance(CONSTRAINTS, (list, tuple)):
        problems.append(f"must be list or tuple, see: {CONSTRAINTS!r}")

    seen_packages = set()
    all_deps = set()
    for _, (_, deps) in REQUIREMENTS_BY_PIECE:
        for d in deps:
            all_deps.add(d.lower())

    for package, constraint in CONSTRAINTS:
        if package in seen_packages:
            problems.append(f"{package}: specified twice")
        seen_packages.add(package)

        if package.lower() not in all_deps:
            problems.append(f"{package}: not specified in REQUIREMENTS_BY_PIECE")

        if constraint is None:  # None is just a placeholder that allows for comments.
            continue

        if not CONSTRAINT_REGEX.match(constraint):
            problems.append(
                f'{package}: constraint "{constraint}" does not look like a valid constraint'
            )

        if constraint.startswith("^"):
            parse_semver(package, constraint, problems)

    all_constrained_packages = [p for (p, _) in CONSTRAINTS]
    sorted_constrained_packages = list(sorted(all_constrained_packages))
    if sorted_constrained_packages != all_constrained_packages:
        problems.append(
            "CONSTRAINTS entries should be in this sorted order: " f"{sorted_constrained_packages}"
        )

    return problems


class ValidationError(Exception):
    """Raised when a validation error occurs."""

    @staticmethod
    def format_problems(config: str, problems: typing.List[str]) -> str:
        """Format a list of problems with a global config variable into human-readable output.

        Parameters
        ----------
        config : str
            Name of the global configuration variable of concern. Prepended to the output.
        problems: list[str]
            A list of strings, each one a distinct problem with that config variable.

        Returns
        -------
        str :
            A human-readable string suitable for console, listing the problems as bullet points.
        """
        formatted = []
        for p in problems:
            assert isinstance(p, str), f"problems element not a str: {p}"
            formatted.append(
                "\n".join(
                    textwrap.wrap(
                        f"{config}: {p}", width=80, initial_indent=" * ", subsequent_indent="   "
                    )
                )
            )

        return "\n".join(formatted)

    def __init__(self, config: str, problems: typing.List[str]):
        """Describes an error that occurs validating one of the global config variables.

        Parameters
        ----------
        config : str
            Name of the global configuration variable of concern. Prepended to the output.
        problems: list[str]
            A list of strings, each one a distinct problem with that config variable.
        """
        super(ValidationError, self).__init__(self.format_problems(config, problems))
        self.problems = problems


def validate_or_raise():
    problems = validate_requirements_by_piece()
    if problems:
        raise ValidationError("REQUIREMENTS_BY_PIECE", problems)

    problems = validate_constraints()
    if problems:
        raise ValidationError("CONSTRAINTS", problems)


def semver_to_requirements(dep: str, constraint: str, joined_deps: typing.List[str]):
    """Convert a SemVer-style constraint to a setuptools-compatible constraint.

    Parameters
    ----------
    dep : str
        Name of the PyPI package to depend on.
    constraint : str
        The SemVer constraint, of the form "^<semver constraint>"
    joined_deps : list[str]
        A list of strings, each a setuptools-compatible constraint which could be written to
        a line in requirements.txt. The converted constraint is appended to this list.
    """
    problems: typing.List[str] = []
    min_ver_parts, fixed_index, fixed_part = parse_semver(dep, constraint, problems)
    text_problems = "\n" + "\n".join(f" * {p}" for p in problems)
    assert (
        not problems
    ), f"should not happen: validated semver {constraint} parses with problems:{text_problems}"

    max_ver_parts = (
        min_ver_parts[:fixed_index]
        + [str(fixed_part + 1)]
        + ["0" for _ in min_ver_parts[fixed_index + 1 :]]
    )
    joined_deps.append(f'{dep}>={".".join(min_ver_parts)},<{".".join(max_ver_parts)}')


def join_requirements() -> typing.Dict[str, typing.Tuple[str, typing.List[str]]]:
    """Validate, then join REQUIRMENTS_BY_PIECE against CONSTRAINTS and return the result.

    Returns
    -------
    An OrderedDict containing REQUIREMENTS_BY_PIECE, except any dependency mentioned in CONSTRAINTS
    is replaced by a setuptools-compatible constraint.
    """
    validate_or_raise()

    constraints_map = collections.OrderedDict([(p.lower(), c) for (p, c) in CONSTRAINTS])

    to_return = collections.OrderedDict()
    all_deps = set()
    for piece, (description, deps) in REQUIREMENTS_BY_PIECE:
        joined_deps = []
        for d in deps:
            constraint = constraints_map.get(d.lower())
            if constraint is None:
                joined_deps.append(d)
                continue

            if constraint[0] == "^":
                semver_to_requirements(d, constraint, joined_deps)
            else:
                joined_deps.append(f"{d}{constraint}")

        if piece != "dev":
            all_deps.update(joined_deps)

        to_return[piece] = (description, joined_deps)

    to_return["all-prod"] = (
        "Combined dependencies for all TVM pieces, excluding dev",
        list(sorted(all_deps)),
    )

    return to_return


def join_and_write_requirements(args: argparse.Namespace):
    try:
        joined_deps = join_requirements()
    except ValidationError as e:
        print(f"ERROR: invalid requirements configuration in {__file__}:", file=sys.stderr)
        print(str(e), file=sys.stderr)
        sys.exit(2)

    if args.lint:
        sys.exit(0)

    output_dir = os.path.join(os.path.dirname(__file__), "requirements")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    elif not os.path.isdir(output_dir):
        print(
            f"ERROR: output directory {output_dir} exists but is not a dir. Delete it",
            file=sys.stderr,
        )
        sys.exit(2)

    for piece, (description, deps) in joined_deps.items():
        with open(os.path.join(output_dir, f"{piece}.txt"), "w") as f:
            f.write(
                f"# AUTOGENERATED by python/gen_requirements.py{os.linesep}"
                f"#{os.linesep}"
                f"# {description}{os.linesep}"
            )
            for d in deps:
                f.write(f"{d}{os.linesep}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lint", action="store_true", help="Just lint dependencies, don't generate anything"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    join_and_write_requirements(args)


if __name__ == "__main__":
    main()
