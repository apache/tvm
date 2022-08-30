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
"""
Provides support for micro targets (microTVM).
"""
import argparse
import os
from pathlib import Path
import shutil
import sys

from . import TVMCException
from .main import register_parser
from .arguments import TVMCSuppressedArgumentParser
from .project import (
    get_project_options,
    get_and_check_options,
    get_project_dir,
)

try:
    import tvm.micro.project as project
    from tvm.micro import get_microtvm_template_projects
    from tvm.micro.build import MicroTVMTemplateProjectNotFoundError
    from tvm.micro.project_api.server import ServerError
    from tvm.micro.project_api.client import ProjectAPIServerNotFoundError

    SUPPORT_MICRO = True
except (ImportError, NameError):
    SUPPORT_MICRO = False


@register_parser
def add_micro_parser(subparsers, main_parser, json_params):
    """Includes parser for 'micro' context and associated subcommands:
    create-project (create), build, and flash.
    """

    if SUPPORT_MICRO is False:
        # Don't create 'tvmc micro' parser.
        return

    # Probe available default platform templates.
    templates = {}
    for p in ("zephyr", "arduino"):
        try:
            templates[p] = get_microtvm_template_projects(p)
        except MicroTVMTemplateProjectNotFoundError:
            pass

    micro = subparsers.add_parser("micro", help="select micro context.")
    micro.set_defaults(func=drive_micro)

    micro_parser = micro.add_subparsers(title="subcommands")
    # Selecting a subcommand under 'micro' is mandatory
    micro_parser.required = True
    micro_parser.dest = "subcommand"

    # 'create_project' subcommand
    create_project_parser = micro_parser.add_parser(
        "create-project",
        aliases=["create"],
        help="create a project template of a given type or given a template dir.",
    )
    create_project_parser.set_defaults(subcommand_handler=create_project_handler)
    create_project_parser.add_argument(
        "project_dir",
        help="project dir where the new project based on the template dir will be created.",
    )
    create_project_parser.add_argument("MLF", help="Model Library Format (MLF) .tar archive.")
    create_project_parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="force project creating even if the specified project directory already exists.",
    )

    # 'build' subcommand
    build_parser = micro_parser.add_parser(
        "build",
        help="build a project dir, generally creating an image to be flashed, e.g. zephyr.elf.",
    )
    build_parser.set_defaults(subcommand_handler=build_handler)
    build_parser.add_argument("project_dir", help="project dir to build.")
    build_parser.add_argument("-f", "--force", action="store_true", help="Force rebuild.")

    # 'flash' subcommand
    flash_parser = micro_parser.add_parser(
        "flash", help="flash the built image on a given micro target."
    )
    flash_parser.set_defaults(subcommand_handler=flash_handler)
    flash_parser.add_argument("project_dir", help="project dir where the built image is.")

    # For each platform add arguments detected automatically using Project API info query.

    # Create subparsers for the platforms under 'create-project', 'build', and 'flash' subcommands.
    help_msg = (
        "you must select a platform from the list. You can pass '-h' for a selected "
        "platform to list its options."
    )
    create_project_platforms_parser = create_project_parser.add_subparsers(
        title="platforms", help=help_msg, dest="platform"
    )
    build_platforms_parser = build_parser.add_subparsers(
        title="platforms", help=help_msg, dest="platform"
    )
    flash_platforms_parser = flash_parser.add_subparsers(
        title="platforms", help=help_msg, dest="platform"
    )

    subcmds = {
        # API method name    Parser associated to method      Handler func to call after parsing
        "generate_project": [create_project_platforms_parser, create_project_handler],
        "build": [build_platforms_parser, build_handler],
        "flash": [flash_platforms_parser, flash_handler],
    }

    # Helper to add a platform parser to a subcmd parser.
    def _add_parser(parser, platform):
        platform_name = platform[0].upper() + platform[1:] + " platform"
        platform_parser = parser.add_parser(
            platform, add_help=False, help=f"select {platform_name}."
        )
        platform_parser.set_defaults(platform=platform)
        return platform_parser

    parser_by_subcmd = {}
    for subcmd, subcmd_parser_handler in subcmds.items():
        subcmd_parser = subcmd_parser_handler[0]
        subcmd_parser.required = True  # Selecting a platform or template is mandatory
        parser_by_platform = {}
        for platform in templates:
            new_parser = _add_parser(subcmd_parser, platform)
            parser_by_platform[platform] = new_parser

        # Besides adding the parsers for each default platform (like Zephyr and Arduino), add a
        # parser for 'template' to deal with adhoc projects/platforms.
        new_parser = subcmd_parser.add_parser(
            "template", add_help=False, help="select an adhoc template."
        )
        new_parser.add_argument(
            "--template-dir", required=True, help="Project API template directory."
        )
        new_parser.set_defaults(platform="template")
        parser_by_platform["template"] = new_parser

        parser_by_subcmd[subcmd] = parser_by_platform

    disposable_parser = TVMCSuppressedArgumentParser(main_parser)
    try:
        known_args, _ = disposable_parser.parse_known_args()
    except TVMCException:
        return

    try:
        subcmd = known_args.subcommand
        platform = known_args.platform
    except AttributeError:
        # No subcommand or platform, hence no need to augment the parser for micro targets.
        return

    # Augment parser with project options.

    if platform == "template":
        # adhoc template
        template_dir = str(Path(known_args.template_dir).resolve())
    else:
        # default template
        template_dir = templates[platform]

    try:
        template = project.TemplateProject.from_directory(template_dir)
    except ProjectAPIServerNotFoundError:
        sys.exit(f"Error: Project API server not found in {template_dir}!")

    template_info = template.info()

    options_by_method = get_project_options(template_info)

    # TODO(gromero): refactor to remove this map.
    subcmd_to_method = {
        "create-project": "generate_project",
        "create": "generate_project",
        "build": "build",
        "flash": "flash",
    }

    method = subcmd_to_method[subcmd]
    parser_by_subcmd_n_platform = parser_by_subcmd[method][platform]
    _, handler = subcmds[method]

    parser_by_subcmd_n_platform.formatter_class = (
        # Set raw help text so help_text format works
        argparse.RawTextHelpFormatter
    )
    parser_by_subcmd_n_platform.set_defaults(
        subcommand_handler=handler,
        valid_options=options_by_method[method],
        template_dir=template_dir,
    )

    required = any([opt["required"] for opt in options_by_method[method]])
    nargs = "+" if required else "*"

    help_text_by_option = [opt["help_text"] for opt in options_by_method[method]]
    help_text = "\n\n".join(help_text_by_option) + "\n\n"

    parser_by_subcmd_n_platform.add_argument(
        "--project-option", required=required, metavar="OPTION=VALUE", nargs=nargs, help=help_text
    )

    parser_by_subcmd_n_platform.add_argument(
        "-h",
        "--help",
        "--list-options",
        action="help",
        help="show this help message which includes platform-specific options and exit.",
    )

    for one_entry in json_params:
        micro.set_defaults(**one_entry)


def drive_micro(args):
    # Call proper handler based on subcommand parsed.
    args.subcommand_handler(args)


def create_project_handler(args):
    """Creates a new project dir."""
    project_dir = get_project_dir(args.project_dir)

    if os.path.exists(project_dir):
        if args.force:
            shutil.rmtree(project_dir)
        else:
            raise TVMCException(
                "The specified project dir already exists. "
                "To force overwriting it use '-f' or '--force'."
            )

    template_dir = str(Path(args.template_dir).resolve())
    if not os.path.exists(template_dir):
        raise TVMCException(f"Template directory {template_dir} does not exist!")

    mlf_path = str(Path(args.MLF).resolve())
    if not os.path.exists(mlf_path):
        raise TVMCException(f"MLF file {mlf_path} does not exist!")

    options = get_and_check_options(args.project_option, args.valid_options)

    try:
        project.generate_project_from_mlf(template_dir, project_dir, mlf_path, options)
    except ServerError as error:
        print("The following error occurred on the Project API server side: \n", error)
        sys.exit(1)


def build_handler(args):
    """Builds a firmware image given a project dir."""
    project_dir = get_project_dir(args.project_dir)

    if not os.path.exists(project_dir):
        raise TVMCException(f"{project_dir} doesn't exist.")

    if os.path.exists(project_dir + "/build"):
        if args.force:
            shutil.rmtree(project_dir + "/build")
        else:
            raise TVMCException(
                f"There is already a build in {project_dir}. "
                "To force rebuild it use '-f' or '--force'."
            )

    options = get_and_check_options(args.project_option, args.valid_options)

    try:
        prj = project.GeneratedProject.from_directory(project_dir, options=options)
        prj.build()
    except ServerError as error:
        print("The following error occurred on the Project API server side: ", error)
        sys.exit(1)


def flash_handler(args):
    """Flashes a firmware image to a target device given a project dir."""

    project_dir = get_project_dir(args.project_dir)

    if not os.path.exists(project_dir + "/build"):
        raise TVMCException(f"Could not find a build in {project_dir}")

    options = get_and_check_options(args.project_option, args.valid_options)

    try:
        prj = project.GeneratedProject.from_directory(project_dir, options=options)
        prj.flash()
    except ServerError as error:
        print("The following error occurred on the Project API server side: ", error)
        sys.exit(1)
