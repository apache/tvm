import os

if os.name == 'nt':
    os.system('node_modules\\.bin\\jest')
else:
    os.system('node node_modules/.bin/jest')
