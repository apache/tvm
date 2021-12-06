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

"""Defines glue wrappers around the Project API which mate to TVM interfaces."""

import pathlib
from typing import Union

from .. import __version__
from ..contrib import utils
from .build import get_standalone_crt_dir
from .model_library_format import ExportableModule, export_model_library_format
from .project_api import client
from .transport import Transport, TransportTimeouts


class ProjectTransport(Transport):
    """A Transport implementation that uses the Project API client."""

    def __init__(self, api_client, options):
        self._api_client = api_client
        self._options = options
        self._timeouts = None

    def timeouts(self):
        assert self._timeouts is not None, "Transport not yet opened"
        return self._timeouts

    def open(self):
        reply = self._api_client.open_transport(self._options)
        self._timeouts = TransportTimeouts(**reply["timeouts"])

    def close(self):
        if not self._api_client.is_shutdown:
            self._api_client.close_transport()
            self._api_client.shutdown()

    def write(self, data, timeout_sec):
        self._api_client.write_transport(data, timeout_sec)

    def read(self, n, timeout_sec):
        return self._api_client.read_transport(n, timeout_sec)["data"]


class TemplateProjectError(Exception):
    """Raised when the Project API server given to GeneratedProject reports is_template=True."""


class GeneratedProject:
    """Defines a glue interface to interact with a generated project through the API server."""

    @classmethod
    def from_directory(cls, project_dir: Union[pathlib.Path, str], options: dict):
        return cls(client.instantiate_from_dir(project_dir), options)

    def __init__(self, api_client, options):
        self._api_client = api_client
        self._options = options
        self._info = self._api_client.server_info_query(__version__)
        if self._info["is_template"]:
            raise TemplateProjectError()

    def build(self):
        self._api_client.build(self._options)

    def flash(self):
        self._api_client.flash(self._options)

    def transport(self):
        return ProjectTransport(self._api_client, self._options)

    def info(self):
        return self._info

    @property
    def options(self):
        return self._options

    @options.setter
    def options(self, options):
        self._options = options


class NotATemplateProjectError(Exception):
    """Raised when the API server given to TemplateProject reports is_template=false."""


class TemplateProject:
    """Defines a glue interface to interact with a template project through the API Server."""

    @classmethod
    def from_directory(cls, template_project_dir):
        return cls(client.instantiate_from_dir(template_project_dir))

    def __init__(self, api_client):
        self._api_client = api_client
        self._info = self._api_client.server_info_query(__version__)
        if not self._info["is_template"]:
            raise NotATemplateProjectError()

    def _check_project_options(self, options: dict):
        """Check if options are valid ProjectOptions"""
        available_options = [option["name"] for option in self.info()["project_options"]]
        if options and not set(options.keys()).issubset(available_options):
            raise ValueError(
                f"""options:{list(options)} include non valid ProjectOptions.
                        Here is a list of available options:{list(available_options)}."""
            )

    def generate_project_from_mlf(self, model_library_format_path, project_dir, options: dict):
        """Generate a project from MLF file."""
        self._check_project_options(options)
        self._api_client.generate_project(
            model_library_format_path=str(model_library_format_path),
            standalone_crt_dir=get_standalone_crt_dir(),
            project_dir=project_dir,
            options=options,
        )

        return GeneratedProject.from_directory(project_dir, options)

    def info(self):
        return self._info

    def generate_project(self, graph_executor_factory, project_dir, options):
        """Generate a project given GraphRuntimeFactory."""
        model_library_dir = utils.tempdir()
        model_library_format_path = model_library_dir.relpath("model.tar")
        export_model_library_format(graph_executor_factory, model_library_format_path)

        return self.generate_project_from_mlf(model_library_format_path, project_dir, options)


def generate_project(
    template_project_dir: Union[pathlib.Path, str],
    module: ExportableModule,
    generated_project_dir: Union[pathlib.Path, str],
    options: dict = None,
):
    """Generate a project for an embedded platform that contains the given model.

    Parameters
    ----------
    template_project_path : pathlib.Path or str
        Path to a template project containing a microTVM Project API server.

    generated_project_path : pathlib.Path or str
        Path to a directory to be created and filled with the built project.

    module : ExportableModule
        A runtime.Module exportable as Model Library Format. The value returned from tvm.relay.build
        or tvm.build.

    options : dict
        If given, Project API options given to the microTVM API server found in both
        template_project_path and generated_project_path.

    Returns
    -------
    GeneratedProject :
        A class that wraps the generated project and which can be used to further interact with it.
    """
    template = TemplateProject.from_directory(str(template_project_dir))
    return template.generate_project(module, str(generated_project_dir), options)


def generate_project_from_mlf(
    template_project_dir: Union[pathlib.Path, str],
    project_dir: Union[pathlib.Path, str],
    mlf_path: Union[pathlib.Path, str],
    options: dict,
):
    """Generate a project from a platform template and an existing Model Library Format archive.

    Parameters
    ----------
    template_project_path : pathlib.Path or str
        Path to a template project containing a microTVM Project API server.

    project_dir : pathlib.Path or str
        Path to a directory where the project will be created.

    mlf_path : pathlib.Path or str
        Path to the Model Library Format archive that will be used when creating
        the new project. The archive file will be copied to project_dir.

    options : dict
        Project API options given to the microTVM API server for the specified platform.

    Returns
    -------
    GeneratedProject :
        A class that wraps the generated project and which can be used to further interact with it.
    """

    template = TemplateProject.from_directory(str(template_project_dir))
    return template.generate_project_from_mlf(str(mlf_path), str(project_dir), options)
