# Slot, Latency
Config = {
    'x220': {
        'Slot': ['Scalar', 'Memory', 'Vector'],
        'Latency': # [default, optimze0, optimze1, ...]
        {
            'Scalar':
            {
                'Int Imm': [1],
                'Int Var': [1],
                'Add': [1],
                'Mul': [1],
            },
            'Memory':
            {
                'Store': [6, 3], #
                'Load': [6, 1],
            },
            'Vector':
            {
                'Float Imm': [1],
                'Float Var': [1],
                'Add': [1],
                'Max': [3, 1],
                'Mul': [3],
                'Div': [1], # Shifting
                'LT': [3],
                'Bit Shift': [5],
                'Ch Concat': [1],
                'Mac': [5, 1],
                'LUT': [8],
                'LUT0': [3], 'LUT1': [1], 'LUT2': [1], 'LUT3': [3],
            },
            'Control':
            {
                'For Start': [1],
                'For End': [3], # iter++, cmpr(iter, extent), jump SP
                'Seq': [1],
            }
        },
        'Type':
        {
            'Scalar' : ['Int Var', 'Int Imm'],
            'Memory' : ['Store', 'Load'],
            'Vector' : ['Float Var', 'Float Imm', 'Bit Shift'],
            'Control': ['For Start', 'For End', 'Seq'],
        }
    }
}
