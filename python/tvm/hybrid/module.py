"""Methods and data structures to support dumping HalideIR to Hybrid Script.
This allows users to do quick hack to generated HalideIR and cast it back to
TVM modules.

To enable this feature, you need to build with -DUSE_HYBRID_DUMP=ON.
"""

import imp

from ..contrib import util
from .util import _internal_assert
from .util import _is_tvm_arg_types
from .parser import source_to_op


class HybridModule(object):
    """The usage of Hybrid Module is very similar to conventional TVM module,
    but conventional TVM module requires a function body which is already fully
    lowered. This contradicts to the fact that Hybrid Module is originally a text
    format for Phase 0 HalideIR. Thus, a totally separated module is defined."""


    def __init__(self, src, name):
        temp = util.tempdir()
        dst = temp.relpath(name)
        self.src_ = src
        with open(dst, 'w') as f:
            f.write("import tvm\n@tvm.hybrid.script\n%s" % src)
        py_module = imp.load_source(name, dst)
        _internal_assert(hasattr(py_module, name), \
                         "The loaded source has no given function!")
        self.func_ = getattr(py_module, name)
        _internal_assert(callable(self.func_), "This should be a function! At least callable!")


    def __call__(self, *args):
        if _is_tvm_arg_types(args):
            return source_to_op(self.src_, globals(), args)
        return self.func_(*args)


    def get_source(self):
        return self.src_


    def save(self, path):
        if not path.endswith('.py'):
            path = path + '.py'
        with open(path, 'w') as f:
            f.write(self.src_)
