import tarfile

from tvm.contrib import util
from . import artifact
from . import compiler


class MicroLibrary(artifact.Artifact):

  ARTIFACT_TYPE = 'micro_library'

  @classmethod
  def from_unarchived(cls, base_dir, labelled_files, metadata):
    library_files = labelled_files['library_files']
    del labelled_files['library_files']

    debug_files = None
    if 'debug_files' in labelled_files:
      debug_files = labelled_files['debug_files']
      del labelled_files['debug_files']

    return cls(base_dir, library_files, debug_files=debug_files, labelled_fiels=labelled_files,
               metadata=metadata)

  def __init__(self, base_dir, library_files, debug_files=None, labelled_files=None, metadata=None):
    labelled_files = {} if labelled_files is None else dict(labelled_files)
    metadata = {} if metadata is None else dict(metadata)
    labelled_files['library_files'] = library_files
    if debug_files is not None:
      labelled_files['debug_files'] = debug_files

    super(MicroLibrary, self).__init__(base_dir, labelled_files, metadata)

    self.library_files = library_files
    self.debug_file = debug_files



def create_micro_library(output, objects, options=None):
  """Create a MicroLibrary using the default compiler options.

  Parameters
  ----------
  output : str
      Path to the output file, expected to end in .tar.
  objects : List[str]
      Paths to the source files to include in the library.
  options : Optional[List[str]]
      If given, additional command-line flags for the compiler.
  """
  temp_dir = util.tempdir()
  cc = compiler.DefaultCompiler()
  output = temp_dir.relpath('micro-library.o')
  cc.Library(output, objects, options=options)

  with open(output, 'rb') as output_f:
    elf_data = output_f.read()

  # TODO(areusch): Define a mechanism to determine compiler and linker flags for each lib
  # enabled by the target str, and embed here.
  micro_lib = MicroLibrary('', elf_data, {'target': cc.target.str()})
  micro_lib.save(output)
