from . import MicroBinutil, OpenOcdComm

class ArmBinutil(MicroBinutil):
    def __init__(self):
        super(ArmBinutil, self).__init__('arm-none-eabi-')

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
            '-DSTM32F746xx'
            ]
        super(ArmBinutil, self).create_lib(obj_path, src_path, lib_type, options=options)

    def device_id(self):
        return 'stm32f746xx'


def get_config(server_addr, server_port):
    return {
        'binutil': ArmBinutil(),
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
        'comms_method': OpenOcdComm(server_addr, server_port),
    }
