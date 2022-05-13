#!/usr/bin/env python3
import argparse
import importlib
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import typing


SECTION_RE = re.compile(r'\[([^]]+)\].*')


def remove_sections(lines : typing.List[str], section_names : typing.List[str]) -> typing.List[str]:
  """Remove .toml sections from a list of lines.

  Parameters
  ----------
  lines : list[str]
      A list containing the lines of the toml file.
  section_names : list[str]
      A list of names of sections which should be removed.

  Returns
  -------
  (removed, insert_points):
      A 2-tuple. `removed` is a new list of strings with those sections removed. `insert_points` is
      a dict containing an entry for each section removed; key is the section name and value is the
      index into `removed` where that section would have been.
  """
  removed = []
  insert_points = {}
  drop_line = False
  for line in lines:
    m = SECTION_RE.match(line)
    if m:
      drop_line = m.group(1) in section_names
      insert_points[m.group(1)] = len(removed)

    if not drop_line:
      removed.append(line)

  return removed, insert_points


def write_dependencies(requirements_by_piece : dict, constraints : dict, output_f):
  """Write the [tool.poetry.dependencies] section of pyproject.toml.

  Parameters
  ----------
  requirements_by_piece : dict
      The REQUIREMENTS_BY_PIECE dict from gen_requirements.py module.
  constraints : dict
      The CONSTRAINTS dict from gen_requirements.py module, updated with additional constraints from
      ci-constraints.txt.
  output_f : File
      A file-like object where the section should be written.
  """
  output_f.write("[tool.poetry.dependencies]\n"
                 'python = ">=3.7, <3.9"\n')
  core_packages = set(requirements_by_piece["core"][1])
  dev_packages = set(requirements_by_piece["dev"][1])

  for package, constraint in constraints.items():
    if package in dev_packages:
      continue

    optional = package not in core_packages
    marker = f', markers = "{constraint.environment_marker}"' if constraint.environment_marker else ''
    output_f.write(
        f"{package} = {{ version = \"{constraint.constraint or '*'}\", optional = {str(optional).lower()}{marker} }}\n")

  output_f.write("\n")


def write_dev_dependencies(requirements_by_piece : dict, constraints : dict, output_f):
  """Write the [tool.poetry.dev-dependencies] section of pyproject.toml.

  Parameters
  ----------
  requirements_by_piece : dict
      The REQUIREMENTS_BY_PIECE dict from gen_requirements.py module.
  constraints : dict
      The CONSTRAINTS dict from gen_requirements.py module, updated with additional constraints from
      ci-constraints.txt.
  output_f : File
      A file-like object where the section should be written.
  """
  output_f.write("[tool.poetry.dev-dependencies]\n")
  dev_packages = set(requirements_by_piece["dev"][1])

  for package, constraint in constraints.items():
    if package not in dev_packages:
      continue

    output_f.write(f"{package} = \"{constraint.constraint or '*'}\"\n")

  output_f.write("\n")


def write_extras(requirements_by_piece : dict, constraints : dict, output_f):
  """Write the [tool.poetry.extras] section of pyproject.toml.

  Parameters
  ----------
  requirements_by_piece : dict
      The REQUIREMENTS_BY_PIECE dict from gen_requirements.py module.
  constraints : dict
      The CONSTRAINTS dict from gen_requirements.py module, updated with additional constraints from
      ci-constraints.txt.
  output_f : File
      A file-like object where the section should be written.
  """
  output_f.write("[tool.poetry.extras]\n")

  for piece, (description, packages) in requirements_by_piece.items():
    if piece in ("core", "dev"):
      # These pieces do not need an extras declaration.
      continue

    output_f.write(f"# {description}\n")
    package_list = ", ".join(f'"{p}"' for p in sorted(packages))
    output_f.write(f"{piece} = [{package_list}]\n\n")

  output_f.write("\n")


# List of all the emitted sections in order they are to be emitted.
SECTION_ORDER = ("tool.poetry.dependencies", "tool.poetry.dev-dependencies", "tool.poetry.extras")


CI_CONSTRAINTS_RE = re.compile(r'(?P<package_name>[a-zA-Z0-9_-]+) = "(?P<version>[^"]+)".*')


def generate_pyproject_toml(ci_constraints_txt : pathlib.Path, gen_requirements_py : pathlib.Path,
                            template_pyproject_toml : pathlib.Path,
                            output_pyproject_toml : pathlib.Path):
  """Generate poetry dependencies sections in pyproject.toml from gen_requirements.py.

  Existing [tool.poetry.dev-dependencies], [tool.poetry.dependencies], and [tool.poetry.extras]
  sections are overwritten.

  Parameters
  ----------
  ci_constraints_txt : pathlib.Path
      Path to ci-constraints.txt.
  gen_requirements_py : pathlib.Path
      Path to the python/gen_requirements.py file in TVM.
  template_pyproject_toml : pathlib.Path
      Path to a pyproject.toml whose [{dev-,}dependencies] sections should be replaced with those from
      gen_requirements.py. In production, this is expected to be the checked-in pyproject.toml at
      the root of the TVM repo.
  output_pyproject_toml : pathlib.Path
      Non-existent path to the revised pyproject.toml.
  """
  with open(template_pyproject_toml) as template_f:
    pyproject_toml, insert_points = remove_sections(template_f, SECTION_ORDER)

  insert_points = {s: insert_points.get(s, len(pyproject_toml)) for s in SECTION_ORDER}

  sys.path.insert(0, str(gen_requirements_py.resolve().parent))
  gen_requirements = importlib.import_module(gen_requirements_py.stem)
  sys.path.pop(0)

  constraints_list = []
  for pkg, constraint in gen_requirements.CONSTRAINTS:
    gen_requirements.parse_constraint_entry(pkg, constraint, None, constraints_list)

  constraints = {r.package: r for r in constraints_list}
  with open(ci_constraints_txt) as ci_constraints_f:
    for i, line in enumerate(ci_constraints_f):
      if not line.strip():
        continue

      m = CI_CONSTRAINTS_RE.match(line)
      if not m:
        if line.startswith("#"):
          continue
        print(f"{ci_constraints_txt}: {i}: Malformed line {line}")
        sys.exit(2)

      package_name = m.group("package_name")
      if package_name not in constraints:
        print(f"{ci_constraints_txt}: {i}: Package {package_name} not listed in gen_requirements.py")
        sys.exit(2)

      constraint = constraints[package_name]
      if constraint.constraint != "==*":
        print(f"{ci_constraints_txt}: {i}: Package {package_name} already functionally constrained in gen_requirements.py")
        sys.exit(2)

      constraints[package_name] = gen_requirements.Requirement(constraint.package, m.group("version"), constraint.environment_marker)

  stop_points = list(sorted([(v, k) for k, v in insert_points.items()], key=lambda x: (x[0], SECTION_ORDER.index(x[1]))))
  next_stop = stop_points.pop(0)
  with open(output_pyproject_toml, "w") as output_f:
    def _write(next_stop, i):
      while next_stop[0] == i:
        writer_function_name = f"write_{next_stop[1][len('tool.poetry.'):].replace('-', '_')}"
        globals()[writer_function_name](dict(gen_requirements.REQUIREMENTS_BY_PIECE), constraints, output_f)
        next_stop = stop_points.pop(0) if stop_points else (None, "")

      return next_stop

    for i, line in enumerate(pyproject_toml):
      next_stop = _write(next_stop, i)
      output_f.write(line)

    next_stop = _write(next_stop, len(pyproject_toml))
    assert next_stop[0] is None, f"Did not write all sections. Remaining: {next_stop}"


def freeze_deps(output_pyproject_toml):
  with open(output_pyproject_toml.parent / "poetry-lock.log", "w") as f:
    # Disable parallel fetching which tends to result in "Connection aborted" errors.
    # https://github.com/python-poetry/poetry/issues/3219
    subprocess.check_call(["poetry", "config", "installer.parallel", "false"], cwd=output_pyproject_toml.parent)
    subprocess.check_call(["poetry", "lock", "-vv"], stdout=f, stderr=subprocess.STDOUT, cwd=output_pyproject_toml.parent)


REPO_ROOT = pathlib.Path(__file__).parent.parent


def parse_args(argv : typing.List[str]) -> argparse.Namespace:
  parser = argparse.ArgumentParser(argv[0], usage="Create a pyproject.toml containing the information in python/gen_requirements.py")
  parser.add_argument("--ci-constraints",
                      type=pathlib.Path, default=REPO_ROOT / "docker/ci-constraints.txt",
                      help=("Path to a file describing packages held back in "
                            "CI to make routine package updates possible."))
  parser.add_argument("--gen-requirements-py",
                      type=pathlib.Path, default=REPO_ROOT / "python" / "gen_requirements.py",
                      help="Path to python/gen_requirements.py in the TVM repo")
  parser.add_argument("--template-pyproject-toml",
                      type=pathlib.Path,
                      help="Path to the pyproject.toml to use as a basis for the updated pyproject.toml.")
  parser.add_argument("--output-base",
                      type=pathlib.Path,
                      help="Path where the updated pyproject.toml and poetry.lock should be written.")

  return parser.parse_args(argv[1:])


def main(argv : typing.List[str]):
  args = parse_args(argv)

  if args.output_base.exists():
    shutil.rmtree(args.output_base)
  args.output_base.mkdir(parents=True)

  pyproject_toml = pathlib.Path(args.output_base) / "pyproject.toml"
  generate_pyproject_toml(args.ci_constraints, args.gen_requirements_py, args.template_pyproject_toml, pyproject_toml)
  with open(pyproject_toml) as f:
    print(f.read())
  freeze_deps(pyproject_toml)


if __name__ == "__main__":
  main(sys.argv)
