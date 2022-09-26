# Slot, Latency
Config = {
    'x220': {
        'Slot': ['Scalar', 'Memory', 'Vector'],
        'Latency':
        {
            'Scalar':
            {
                'Int Imm': 1,
                'Int Var': 1,
                'Add': 1,
                'Mul': 1,
            },
            'Memory':
            {
                'Store': 6,
                'Load': 6,
            },
            'Vector':
            {
                'Float Imm': 1,
                'Float Var': 1,
                'Add': 1,
                'Max': 3,
                'Mul': 3,
                'LT': 3,
            },
            'Control':
            {
                'For Start': 1,
                'For End': 1,
                'Seq': 1,
            }
        },
        'Type':
        {
            'Scalar' : ['Int Var', 'Int Imm'],
            'Memory' : ['Store', 'Load'],
            'Vector' : ['Float Var', 'Float Imm'],
            'Control': ['For Start', 'For End', 'Seq'],
        }
    }
}