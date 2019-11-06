from .. import MicroBinutil

#
# [Device Memory Layout]
#   RAM   (rwx) : START = 0x20000000, LENGTH = 320K
#   FLASH (rx)  : START = 0x8000000,  LENGTH = 1024K
#

class Stm32F746XXBinutil(MicroBinutil):
    def __init__(self):
        super(Stm32F746XXBinutil, self).__init__('arm-none-eabi-')

    def create_lib(self, obj_path, src_path, lib_type, options=None):
        if options is None:
            options = []
        options += [
            '-mcpu=cortex-m7',
            '-mlittle-endian',
            '-mfloat-abi=hard',
            '-mfpu=fpv5-sp-d16',
            '-mthumb',
            '-gdwarf-5',
            ]
        super(Stm32F746XXBinutil, self).create_lib(obj_path, src_path, lib_type, options=options)

    def device_id(self):
        return 'stm32f746xx'


def default_config(server_addr, server_port):
    return {
        'binutil': 'stm32f746xx',
        'mem_layout': {
            'text': {
                'start': 0x20000180,
                'size': 20480,
            },
            'rodata': {
                'start': 0x20005180,
                'size': 20480,
            },
            'data': {
                'start': 0x2000a180,
                'size': 768,
            },
            'bss': {
                'start': 0x2000a480,
                'size': 768,
            },
            'args': {
                'start': 0x2000a780,
                'size': 1280,
            },
            'heap': {
                'start': 0x2000ac80,
                'size': 262144,
            },
            'workspace': {
                'start': 0x2004ac80,
                'size': 20480,
            },
            'stack': {
                'start': 0x2004fc80,
                'size': 80,
            },
        },
        'word_size': 4,
        'thumb_mode': True,
        'comms_method': 'openocd',
        'server_addr': server_addr,
        'server_port': server_port,
    }
