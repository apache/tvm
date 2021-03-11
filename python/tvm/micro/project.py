"""Defines glue wrappers around the Project API which mate to TVM interfaces."""

from .. import __version__
from ..contrib import utils
from .build import get_standalone_crt_dir
from .model_library_format import export_model_library_format
from .project_api import client
from .transport import Transport, TransportTimeouts


class ProjectTransport(Transport):

    def __init__(self, client, options):
        self._client = client
        self._options = options
        self._timeouts = None

    def timeouts(self):
        assert self._timeouts is not None, "Transport not yet opened"
        return self._timeouts

    def open(self):
        reply = self._client.open_transport(self._options)
        self._timeouts = TransportTimeouts(**reply["timeouts"])

    def close(self):
        if not self._client.is_shutdown:
            self._client.close_transport()
            # NOTE: assume caller is tvm.micro.Session.
            self._client.shutdown()

    def write(self, data, timeout_sec):
        self._client.write_transport(data, timeout_sec)

    def read(self, n, timeout_sec):
        return self._client.read_transport(n, timeout_sec)["data"]


class TemplateProjectError(Exception):
    """Raised when the Project API server given to GeneratedProject reports is_template=True."""


class GeneratedProject:
    """Defines a glue interface to interact with a generated project through the API server."""

    @classmethod
    def from_directory(cls, project_dir, options):
        return cls(client.instantiate_from_dir(project_dir), options)

    def __init__(self, client, options):
        self._client = client
        self._options = options
        self._info = self._api_client.server_info_query(__version__)
        if self._info["is_template"]:
            raise TemplateProjectError()

    def build(self):
        self._client.build(self._options)

    def flash(self):
        self._client.flash(self._options)

    def transport(self):
        return ProjectTransport(self._client, self._options)


class NotATemplateProjectError(Exception):
    """Raised when the Project API server given to TemplateProject reports is_template=false."""


class TemplateProject:

    @classmethod
    def from_directory(cls, template_project_dir, options):
        return cls(client.instantiate_from_dir(template_project_dir), options)

    def __init__(self, client, options):
        self._client = client
        self._options = options
        self._info = self._api_client.server_info_query(__version__)
        if not self._info["is_template"]:
            raise NotATemplateProjectError()

    def generate_project(self, graph_executor_factory, project_dir):
        """Generate a project given GraphRuntimeFactory."""
        model_library_dir = utils.tempdir()
        model_library_format_path = model_library_dir.relpath('model.tar')
        export_model_library_format(
            graph_executor_factory, model_library_format_path)

        self._client.generate_project(
            model_library_format_path=model_library_format_path,
            standalone_crt_dir=get_standalone_crt_dir(),
            project_dir=project_dir,
            options=self._options)

        return GeneratedProject.from_directory(project_dir, self._options)


def generate_project(template_project_dir : str, graph_executor_factory, project_dir : str, options : dict = None):
    template = TemplateProject.from_directory(template_project_dir, options)
    return template.generate_project(graph_executor_factory, project_dir)
