"""Methods and data structures to support dumping HalideIR to Hybrid Script.
This allows users to do quick hack to generated HalideIR and cast it back to
TVM modules.
"""

import imp
from ..contrib import util
from .util import _internal_assert


class HybridModule(object):
    """The usage of Hybrid Module is very similar to conventional TVM module,
    but conventional TVM module requires a function body which is already fully
    lowered. This contradicts to the fact that Hybrid Module is originally a text
    format for Phase 0 HalideIR. Thus, a totally separated module is defined."""


    def __init__(self, src, name):
        temp = util.tempdir()
        self.name_ = name
        self.dst_ = temp.relpath(name)
        self.src_ = 'import tvm\ntvm.hybrid.script\n%s' % src
        with open(self.dst_, 'w') as f:
            f.write(self.src_)
        self.py_module_ = imp.load_source(name, self.dst_)
        _internal_assert(hasattr(self.py_module_, name), \
                         "The loaded source has no given function!")


    def __call__(self, *args):
        return getattr(self.py_module_, self.name_)(*args)


    def get_source(self):
        return self.src_
