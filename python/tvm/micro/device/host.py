import sys

from . import MicroBinutil, register_binutil

class HostBinutil(MicroBinutil):
    def __init__(self):
        super(HostBinutil, self).__init__('')

    def create_lib(self, obj_path, src_path, lib_type, options=None):
        if options is None:
            options = []
        if sys.maxsize > 2**32 and sys.platform.startswith('linux'):
            options += ['-mcmodel=large']
        super(HostBinutil, self).create_lib(obj_path, src_path, lib_type, options=options)

    def device_id():
        return 'host'


register_binutil(HostBinutil)

def default_config():
    return {
        'binutil': 'host',
        'mem_layout': {
            'text': {
                'size': 20480,
            },
            'rodata': {
                'size': 20480,
            },
            'data': {
                'size': 768,
            },
            'bss': {
                'size': 768,
            },
            'args': {
                'size': 1280,
            },
            'heap': {
                'size': 262144,
            },
            'workspace': {
                'size': 20480,
            },
            'stack': {
                'size': 80,
            },
        },
        'word_size': 8 if sys.maxsize > 2**32 else 4,
        'thumb_mode': False,
        'comms_method': 'host',
    }
